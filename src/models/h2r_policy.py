import logging
import math
from typing import Dict, Optional, Tuple
from functools import partial
from torch import einsum, nn
import torch
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import einops
from src.models.edm_diffusion.score_wrappers import GCDenoiser

from src.models.video_former import Video_Former_2D,Video_Former_3D
from diffusers import StableVideoDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from src.models.edm_diffusion.gc_sampling import *
from src.pipelines.pipeline_vpp import Pose2VideoPipeline


logger = logging.getLogger(__name__)

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.video_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


class H2R_Policy(nn.Module):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            pipeline,
            latent_dim: int = 512,
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            sigma_sample_density_type: str = 'loglogistic',
            seed: int = 42,
            Former_depth: int = 3,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 1,
            num_latents: int = 3,
            use_Former: str = '3d',
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 7,
            action_seq_len: int = 10,
            noise_scheduler: str = 'exponential',
            act_window_size: int = 10,
            sampler_type: str = 'ddim',
    ):  
        super().__init__()
        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer

        self.action_dim = action_dim
        self.act_window_size = act_window_size
        self.extract_layer_idx = extract_layer_idx
        self.use_Former = use_Former
        self.Former_num_time_embeds = Former_num_time_embeds


        # TODO: calculate condition_dim
        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim

        if use_Former=='3d':
            self.Video_Former = Video_Former_3D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
                use_temporal=True,
             )
        elif use_Former == '2d':
            self.Video_Former = Video_Former_2D(
                    dim=latent_dim,
                    depth=Former_depth,
                    dim_head=Former_dim_head,
                    heads=Former_heads,
                    num_time_embeds=Former_num_time_embeds,
                    num_latents=num_latents,
                    condition_dim=condition_dim,
                 )
        else:
            self.Video_Former = nn.Linear(condition_dim,latent_dim)

        print('use_Former:', self.use_Former)
        print('use_all_layer',self.use_all_layer)

        self.seed = seed

        # 移除手动设备管理，让 Accelerate 处理
        # pipeline = pipeline.to(self.device)  # ❌ 删除这行

        # policy network
        self.model = GCDenoiser(action_dim = action_dim,
                                obs_dim=latent_dim,
                                goal_dim=512,
                                num_tokens=num_latents,
                                goal_window_size = 1,
                                obs_seq_len = obs_seq_len,
                                act_seq_len = action_seq_len,
                                sigma_data=0.5)  # 移除 .to(self.device)

        self.pipeline: Pose2VideoPipeline = pipeline
        # 移除 self.save_hyperparameters() 因为不再是 LightningModule
        # diffusion stuff
        # 移除未定义的变量引用
        self.sampler_type = sampler_type
        # self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        # self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        # self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.ema_callback_idx = None

        for param in self.model.inner_model.proprio_emb.parameters():
            param.requires_grad = False
        # for param in self.model.inner_model.goal_emb.parameters():
        #     param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def process_device(self):
        pass

    # def configure_optimizers(self):
    #     """
    #     Initialize optimizers and learning rate schedulers based on model configuration.
    #     """
    #     # Configuration for models using transformer weight decay
    #     '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
    #         weight_decay=self.optimizer_config.transformer_weight_decay
    #     )'''
    #     optim_groups = [
    #         {"params": self.model.inner_model.parameters(),
    #          "weight_decay": self.optimizer_config.transformer_weight_decay},
    #         {"params": self.Video_Former.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
    #     ]

    #     # TODO: edit here
    #     optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
    #                                   betas=self.optimizer_config.betas)

    #     # Optionally initialize the scheduler
    #     if self.use_lr_scheduler:
    #         lr_configs = OmegaConf.create(self.lr_scheduler)
    #         scheduler = TriStageLRScheduler(optimizer, lr_configs)
    #         lr_scheduler = {
    #             "scheduler": scheduler,
    #             "interval": 'step',
    #             "frequency": 1,
    #         }
    #         return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    #     else:
    #         return optimizer

    # def on_before_zero_grad(self, optimizer=None):
    #     total_grad_norm = 0.0
    #     total_param_norm = 0.0
    #     for p in self.model.parameters():
    #         if p.grad is not None:
    #             total_grad_norm += p.grad.norm().item() ** 2
    #         total_param_norm += p.norm().item() ** 2
    #     total_grad_norm = total_grad_norm ** 0.5
    #     total_param_norm = total_param_norm ** 0.5

    #     self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
    #     self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)


    def training_step(
                    self,
                    actions,                     
                    ref_img,
                    clip_ref_img,
                    pixel_values_pose_ref,
                    pixel_values_pose,
                    width,
                    height,
                    video_length,
                    num_inference_steps,
                    guidance_scale,
                    generator,
                    stop_step) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        total_loss, action_loss = (
            torch.tensor(0.0).to(ref_img.device),
            torch.tensor(0.0).to(ref_img.device),
        )
        num_frames = self.Former_num_time_embeds

        # TODO: edit here
        features = self.pipeline(
                    ref_image = ref_img,
                    clip_ref_img = clip_ref_img,
                    ref_pose_images = pixel_values_pose_ref,
                    pose_images = pixel_values_pose,
                    width = width,
                    height = height,
                    video_length = video_length,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    generator=generator,
                    stop_step=stop_step,
                    extract_layer_idx=self.extract_layer_idx)

        # b c f h w -> b f c h w
        features = rearrange(features, "b c f h w -> b f c h w")
        features = rearrange(features, "b f c h w-> b f c (h w)")
        features = rearrange(features, "b f c l-> b f l c")
        features = features[:, :num_frames, :, :]
        
        features = self.Video_Former(features)
        
        # TODO: edit here
        act_loss, sigmas, noise = self.diffusion_loss(
            features,
            actions,
        )

        action_loss += act_loss
        total_loss += act_loss

        # total_bs = actions.shape[0]

        # self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(self, dataset_batch: Dict[str, Dict]) -> Dict[
        str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal module, as well as the
            episode indices.
        """
        raise NotImplementedError("Validation step is not implemented")
        # output = {}
        # val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        #     # Compute the required embeddings
        # predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)

        # # predict the next action sequence
        # action_pred = self.denoise_actions(
        #     torch.zeros_like(latent_goal).to(latent_goal.device),
        #     predictive_feature,
        #     latent_goal,
        #     inference=True,
        # )
        # dataset_batch["actions"] = dataset_batch["actions"].to(action_pred.device)
        # # compute the mse action loss
        # pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
        # val_total_act_loss_pp += pred_loss

        # output[f"idx:"] = dataset_batch["idx"]
        # output["validation_loss"] = val_total_act_loss_pp
        # return output

    # TODO: edit here
    def diffusion_loss(
            self,
            perceptual_emb: torch.Tensor,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=actions.device).to(actions.device)
        noise = torch.randn_like(actions).to(actions.device)
        loss, _ = self.model.loss(perceptual_emb, actions, noise, sigmas)
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            # latent_goal: torch.Tensor,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        # if len(latent_goal.shape) < len(
        #         perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
        #     latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler,device=perceptual_emb.device)
        sigmas = sigmas.to(device = perceptual_emb.device, dtype = perceptual_emb.dtype)
        B = perceptual_emb.shape[0]
        x = torch.randn((B, self.act_window_size, self.action_dim), device=perceptual_emb.device) * self.sigma_max
        actions = self.sample_loop(sigmas, x, input_state, None, latent_plan, self.sampler_type, extra_args)
        return actions

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, sigmas, scaler=scaler, disable=True)
            # x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type, device="cuda"):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self,
                actions,
                ref_img,
                clip_ref_img,
                pixel_values_pose_ref,
                pixel_values_pose,
                width,
                height,
                video_length,
                num_inference_steps,
                guidance_scale,
                generator,
                stop_step):
        return self.training_step(actions, ref_img,clip_ref_img, pixel_values_pose_ref, pixel_values_pose, width, height, video_length, num_inference_steps, guidance_scale, generator, stop_step)
        #def training_step(self, batch: Dict[str, Dict], batch_idx: int,
        #                  dataloader_idx: int = 0) -> torch.Tensor

    def eval_forward(
                    self, 
                    ref_img,
                    clip_ref_img,
                    pixel_values_pose_ref,
                    pixel_values_pose,
                    width,
                    height,
                    video_length,
                    num_inference_steps,
                    guidance_scale,
                    generator,
                    stop_step):
        """
        Method for doing inference with the model.
        """
        # rgb_static = obs["rgb_obs"]['rgb_static']
        # rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        # num_frames = self.Former_num_time_embeds
        # rgb_static = rgb_static.to(self.device)
        # rgb_gripper = rgb_gripper.to(self.device)
        # batch = rgb_static.shape[0]

        with torch.no_grad():
            # input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
            # language = [language] + [language]
            features = self.pipeline(
                    ref_image = ref_img,
                    clip_ref_img = clip_ref_img,
                    ref_pose_images = pixel_values_pose_ref,
                    pose_images = pixel_values_pose,
                    width = width,
                    height = height,
                    video_length = video_length,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    generator = generator,
                    stop_step = stop_step,
                    extract_layer_idx=self.extract_layer_idx)
            # b c f h w -> b f c h w
            num_frames = self.Former_num_time_embeds
            features = rearrange(features, "b c f h w -> b f c h w")
            features = rearrange(features, "b f c h w-> b f c (h w)")
            features = rearrange(features, "b f c l-> b f l c")
            features = features[:, :num_frames, :, :]
            # do something
            features = self.Video_Former(features)
            act_seq = self.denoise_actions(
                None,
                features,
                # inference=True,
            )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        raise NotImplementedError("Step is not implemented")
        # if self.rollout_step_counter % self.multistep == 0:
        #     pred_action_seq = self.eval_forward(obs, goal)

        #     self.pred_action_seq = pred_action_seq

        # current_action = self.pred_action_seq[0, self.rollout_step_counter]
        # if len(current_action.shape) == 2:
        #     current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        # self.rollout_step_counter += 1
        # if self.rollout_step_counter == self.multistep:
        #     self.rollout_step_counter = 0

        # return current_action

    def from_pretrained(self, video_former_path, model_path, device, dtype):
        self.Video_Former.load_state_dict(torch.load(video_former_path, map_location="cpu"))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        def set_dtype(module):
            return module.to(device, dtype=dtype)
        self.model.apply(lambda m: set_dtype(m))
        self.Video_Former.apply(lambda m: set_dtype(m))


    # 移除 on_train_start 方法，因为不再是 LightningModule
    # def on_train_start(self) -> None:
    #     self.model.to(dtype=self.dtype)
    #     self.Video_Former.to(dtype=self.dtype)
    #     self.pipeline.to(dtype=self.dtype)

#     @rank_zero_only
#     def on_train_epoch_start(self) -> None:
#         logger.info(f"Start training epoch {self.current_epoch}")

#     @rank_zero_only
#     def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
#         logger.info(f"Finished training epoch {self.current_epoch}")

#     @rank_zero_only
#     def on_validation_epoch_end(self) -> None:
#         logger.info(f"Finished validation epoch {self.current_epoch}")

#     def on_validation_epoch_start(self) -> None:
#         log_rank_0(f"Start validation epoch {self.current_epoch}")

#     @rank_zero_only
#     def on_train_epoch_start(self) -> None:
#         logger.info(f"Start training epoch {self.current_epoch}")

#     @rank_zero_only
#     def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
#         logger.info(f"Finished training epoch {self.current_epoch}")

#     @rank_zero_only
#     def on_validation_epoch_end(self) -> None:
#         logger.info(f"Finished validation epoch {self.current_epoch}")

#     def on_validation_epoch_start(self) -> None:
#         log_rank_0(f"Start validation epoch {self.current_epoch}")


# @rank_zero_only
# def log_rank_0(*args, **kwargs):
#     # when using ddp, only log with rank 0 process
#     logger.info(*args, **kwargs)