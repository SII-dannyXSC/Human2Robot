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

from src.dataset.operation_action import OperationVideoDataset
from src.models.edm_diffusion.score_wrappers import GCDenoiser
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_plus import UNet3DConditionModel

# from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.pipelines.pipeline_vpp import Pose2VideoPipeline
from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)
from src.models.h2r_policy import H2R_Policy

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

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

    print(cfg.motion_module_path)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.motion_module_path,
        # cfg.mm_path,
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
    
    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        pose_guider2=pose_guider2,
        scheduler=train_noise_scheduler,
    )
    pipe = pipe.to(device="cuda", dtype=weight_dtype)

    # TODO: fill it
    net = H2R_Policy(
        pipe,
        latent_dim=cfg.model.latent_dim,
        num_sampling_steps=cfg.model.num_sampling_steps,
        sigma_data=cfg.model.sigma_data,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        sigma_sample_density_type=cfg.model.sigma_sample_density_type,
        seed=cfg.model.seed,
        Former_depth=cfg.model.Former_depth,
        Former_heads=cfg.model.Former_heads,
        Former_dim_head=cfg.model.Former_dim_head,
        Former_num_time_embeds=cfg.model.Former_num_time_embeds,
        num_latents=cfg.model.num_latents,
        use_Former=cfg.model.use_Former,
        extract_layer_idx=cfg.model.extract_layer_idx,
        use_all_layer=cfg.model.use_all_layer,
        obs_seq_len=cfg.model.obs_seq_len,
        action_dim=cfg.model.action_dim,
        action_seq_len=cfg.model.action_seq_len,
    )

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

    # TODO: check paras here
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
    
    num_inference_steps = 30
    guidance_scale = 3.5
    width = 424
    height = 240
    generator = torch.manual_seed(cfg.seed)
    
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
                
                ref_img = batch["pixel_values_ref_img"].to(weight_dtype)
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
                
                clip_ref_img = batch["clip_ref_img"].to(weight_dtype)
                
                # ---- Forward!!! -----
                loss = net(
                    end_state_list,
                    ref_img,
                    clip_ref_img,
                    pixel_values_pose_ref,
                    pixel_values_pose,
                    width,
                    height,
                    end_state_list.shape[1],
                    num_inference_steps,
                    guidance_scale,
                    generator=generator,
                    stop_step = 0
                )
                
            
                total_loss = loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

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
                # reference_control_reader.clear()
                # reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1

                accelerator.log(
                    {"train_loss": train_loss},
                    step=global_step,
                )
                train_loss = 0.0

            logs = {
                "step_loss": total_loss.detach().item(),
                # "action_loss": action_loss.detach().item(),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)
            # save action head only
            unwrap_net = accelerator.unwrap_model(net)
            # TODO: check here
            save_checkpoint2(
                unwrap_net.Video_Former,
                save_dir,
                "video_former",
                global_step,
                total_limit=3,
            )
            save_checkpoint2(
                unwrap_net.model,
                save_dir,
                "model",
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
