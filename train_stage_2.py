import argparse
import copy
import logging
import math
import os
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
import h5py
import numpy as np
import pickle

from src.dataset.operation_action_rotation import OperationVideoDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_plus import UNet3DConditionModel
from src.models.mlp_policy import ZHMLP
from src.utils.rotation_embed import EulerRotationEmbed

# from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.pipelines.pipeline_human2robot_vid_plus import Pose2VideoPipeline
from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):

    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        pose_guider2: PoseGuider,
        action_head: ZHMLP,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.pose_guider2 = pose_guider2
        self.action_head = action_head
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        ref_pose_img,
        pose_img,
        ref_action,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        ref_pose_cond_tensor = ref_pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea += self.pose_guider2(ref_pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        result = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        )

        model_pred = result.sample
        # 取第1次upsample之后的结果
        int_fea_list = result.intermediate_feature_list[1]

        action_pred = self.action_head(int_fea_list, ref_action)

        return model_pred, action_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=24,
    generator=None,
    test_cases=[],
    data_stats="/home/ubuntu/project/Moore-AnimateAnyone/data/dataset_stats.pkl",
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider
    pose_guider2 = ori_net.pose_guider2
    action_head = ori_net.action_head

    if generator is None:
        generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=torch.float16)

    tmp_action_head = copy.deepcopy(action_head)
    tmp_action_head = tmp_action_head.to(dtype=torch.float16)

    rotation_embed = EulerRotationEmbed()

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        pose_guider=pose_guider,
        pose_guider2=pose_guider2,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)
    tmp_action_head = tmp_action_head.to(accelerator.device)

    # for normalization
    with open(data_stats, "rb") as f:
        stats = pickle.load(f)

    def pre_process(s_end_state):
        new_shape = s_end_state.shape[:-1] + (10,)
        result = np.zeros(new_shape)

        s_end_state[..., :3] = (
            s_end_state[..., :3] - stats["end_state_mean"][:3]
        ) / stats["end_state_std"][:3]
        # angle
        result[..., 3:9] = rotation_embed.angles_to_embed(s_end_state[..., 3:6])
        # gripper not change
        result[..., 9] = s_end_state[..., 6]
        return result

    results = []
    for source_path in test_cases:
        file_name = Path(source_path).stem

        with h5py.File(source_path, "r") as f:
            human_video_raw = np.uint8(f["/cam_data/human_camera"])
            robot_video_raw = np.uint8(f["/cam_data/robot_camera"])

            end_position = f["/end_position"][()]
            gripper_state = f["/gripper_state"][()]
            end_state = np.concatenate([end_position, gripper_state[:, None]], axis=1)
            end_state = pre_process(end_state)
            # BGR -> RGB
            if human_video_raw.shape[3] == 3:
                human_video_raw = human_video_raw[:, :, :, ::-1]
            if robot_video_raw.shape[3] == 3:
                robot_video_raw = robot_video_raw[:, :, :, ::-1]

        # 降频
        factor = 4
        human_video = []
        for idx in range(0, len(human_video_raw), factor):
            vid = human_video_raw[idx]
            human_video.append(vid)
        human_video = np.uint8(human_video)

        robot_video = []
        for idx in range(0, len(robot_video_raw), factor):
            vid = robot_video_raw[idx]
            robot_video.append(vid)
        robot_video = np.uint8(robot_video)

        video_length = len(human_video)

        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        video_result = None

        total_loss = 0

        for i in range(0, video_length, clip_length):
            start = i
            end = i + clip_length
            if video_length < end:
                break

            # GT end state in this loop
            cur_end_state = torch.tensor(end_state[start:end, :])
            cur_end_state = cur_end_state.to(
                device=accelerator.device, dtype=torch.float16
            )
            ref_action = cur_end_state[0, :]
            ref_action = ref_action

            ref_image_pil = Image.fromarray(robot_video[i])
            ref_pose_image_pil = Image.fromarray(human_video[i])

            pose_list = []
            pose_tensor_list = []

            ref_tensor_list = []

            ref_pose_list = []
            ref_pose_tensor_list = []

            for j in range(start, end):
                pose_pil = Image.fromarray(human_video[j])
                pose_tensor_list.append(pose_transform(pose_pil))
                pose_list.append(pose_pil)

                ref_pose_tensor_list.append(pose_transform(ref_pose_image_pil))
                ref_pose_list.append(ref_pose_image_pil)

                ref_tensor_list.append(pose_transform(ref_image_pil))

            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)

            ref_pose_tensor = torch.stack(ref_pose_tensor_list, dim=0)  # (f, c, h, w)
            ref_pose_tensor = ref_pose_tensor.transpose(0, 1)

            ref_tensor = torch.stack(ref_tensor_list, dim=0)
            ref_tensor = ref_tensor.transpose(0, 1)

            pipeline_output = pipe(
                ref_image_pil,
                ref_pose_list,
                pose_list,
                width,
                height,
                clip_length,
                20,
                3.5,
                generator=generator,
            )
            video = pipeline_output.videos
            features = pipeline_output.record_features

            # generate video
            pose_tensor = pose_tensor.unsqueeze(0)
            ref_pose_tensor = ref_pose_tensor.unsqueeze(0)
            ref_tensor = ref_tensor.unsqueeze(0)

            video = torch.cat(
                [ref_tensor, ref_pose_tensor, video, pose_tensor], dim=0
            )  # (2, c, f, h, w)

            if video_result is None:
                video_result = video
            else:
                video_result = torch.cat([video_result, video], dim=2)

            fea = features[0][1].squeeze(0)
            action_pred = tmp_action_head(fea, ref_action)

            # calculate loss
            position_loss = F.mse_loss(
                action_pred[..., :3].float(), cur_end_state[..., :3].float()
            )

            # angle loss
            angle_loss = F.mse_loss(
                action_pred[..., 3:9].float(), cur_end_state[..., 3:9].float()
            )

            # gripper loss
            gripper_loss = F.binary_cross_entropy_with_logits(
                action_pred[..., -1].float(), cur_end_state[..., -1].float()
            )

            angle_loss_weight = 1
            gripper_loss_weight = 1

            action_loss = (
                position_loss
                + angle_loss_weight * angle_loss
                + gripper_loss_weight * gripper_loss
            )
            total_loss += action_loss

        results.append(
            {
                "name": f"{file_name}",
                "loss": total_loss,
                "vid": video_result,
            }
        )

    del tmp_denoising_unet
    del tmp_action_head
    del pipe
    torch.cuda.empty_cache()

    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    inference_config_path = "./configs/inference/inference_v2.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        # cfg.motion_module_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)

    pose_guider2 = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)

    # action_head = ZHMLP(1280, 28, 512, 7).to(device="cuda", dtype=weight_dtype)
    # denoise unt is float32 so here should be float32
    # action_head = ZHMLP(1280, 112, 512, 7).to(device="cuda")
    action_head = ZHMLP(1280, 112, 512, 10).to(device="cuda")

    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider2.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider2-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)
    pose_guider2.requires_grad_(False)

    # set action head learnable
    action_head.requires_grad_(True)

    # Set motion module learnable
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        pose_guider2,
        action_head,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = OperationVideoDataset(
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        img_scale=(1.0, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        stats_path=cfg.stats_path,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        train_action_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # b f 7
                end_state_list = batch["end_state_list"].to(weight_dtype)
                # b 7
                ref_action = end_state_list[:, 0, :]

                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
                pixel_values_pose = pixel_values_pose.transpose(
                    1, 2
                )  # (bs, c, f, H, W)

                pixel_values_pose_ref = batch[
                    "pixel_values_pose_ref"
                ]  # (bs, f, c, H, W)
                pixel_values_pose_ref = pixel_values_pose_ref.transpose(
                    1, 2
                )  # (bs, c, f, H, W)

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"],
                        batch["clip_ref_img"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred, action_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    clip_image_embeds,
                    pixel_values_pose_ref,
                    pixel_values_pose,
                    ref_action=ref_action,
                    uncond_fwd=uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # add action loss
                # position loss
                position_loss = F.mse_loss(
                    action_pred[..., :3].float(), end_state_list[..., :3].float()
                )

                # angle loss
                angle_loss = F.mse_loss(
                    action_pred[..., 3:9].float(), end_state_list[..., 3:9].float()
                )

                # gripper loss
                gripper_loss = F.binary_cross_entropy_with_logits(
                    action_pred[..., -1].float(), end_state_list[..., -1].float()
                )

                angle_loss_weight = 1
                gripper_loss_weight = 1

                action_loss = (
                    position_loss
                    + angle_loss_weight * angle_loss
                    + gripper_loss_weight * gripper_loss
                )

                # action_loss = F.l1_loss(action_pred, end_state_list)
                action_loss = action_loss.mean(
                    dim=list(range(1, len(action_loss.shape)))
                )

                total_loss = loss + cfg.action_loss_weight * action_loss
                # loss = action_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                avg_action_loss = accelerator.gather(
                    action_loss.repeat(cfg.data.train_bs)
                ).mean()
                train_action_loss += (
                    avg_action_loss.item() / cfg.solver.gradient_accumulation_steps
                )

                # Backpropagate
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1

                accelerator.log(
                    {"train_loss": train_loss, "train_action_loss": train_action_loss},
                    step=global_step,
                )
                train_loss = 0.0
                train_action_loss = 0.0

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            clip_length=cfg.data.n_sample_frames,
                            generator=generator,
                            test_cases=cfg.test_cases,
                            data_stats=cfg.stats_path,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            val_loss = sample_dict["loss"]

                            # 记录 val_loss 到 MLflow
                            mlflow.log_metric(
                                f"val_loss_{sample_name}", val_loss, step=global_step
                            )

                            with TemporaryDirectory() as temp_dir:
                                out_file = Path(
                                    f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                                )
                                save_videos_grid(vid, out_file, n_rows=2)
                                mlflow.log_artifact(out_file)

            logs = {
                "step_loss": total_loss.detach().item(),
                "action_loss": action_loss.detach().item(),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        # save model after each epoch
        if accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)
            # save action head only
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "motion_module",
                global_step,
                total_limit=3,
            )
            save_checkpoint2(
                unwrap_net.action_head,
                save_dir,
                "action_head",
                global_step,
                total_limit=3,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint2(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, save_path)


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage3.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
