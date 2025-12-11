from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import random
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_plus import UNet3DConditionModel
from src.pipelines.pipeline_human2robot_vid_plus import Pose2VideoPipeline

class Diffusion_feature_extractor(nn.Module):
    def __init__(
        self,
        pipeline,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.num_frames = 16

    @torch.no_grad()
    def forward(
            self,
            ref_image_pil: torch.Tensor,
            ref_pose_list: torch.Tensor,
            pose_list: torch.Tensor,
            width,
            height,
            timestep: Union[torch.Tensor, float, int],
            extract_layer_idx: Union[torch.Tensor, float, int],
            use_latent = False,
            all_layer = False,
            step_time = 1,
            max_length = 20,
    ):
        height = height or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor


        device = self.pipeline._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        
        
        
        
        
        
        
        
        
        
        
        # height = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor //3
        # width = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor //3
        self.pipeline.vae.eval()
        self.pipeline.image_encoder.eval()
        device = self.pipeline.unet.device
        dtype = self.pipeline.vae.dtype
        #print('dtype:',dtype)
        vae = self.pipeline.vae

        num_videos_per_prompt=1

        batch_size = pixel_values.shape[0]

        pixel_values = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)

        with torch.no_grad():
            # texts, tokenizer, text_encoder, img_cond=None, img_cond_mask=None, img_encoder=None, position_encode=True, use_clip=False, max_length=20
            encoder_hidden_states = self.encode_text(texts, self.tokenizer, self.text_encoder, position_encode=self.position_encoding, use_clip=True, max_length=max_length)
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        image_embeddings = encoder_hidden_states

        needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast
        #if needs_upcasting:
        #    self.pipeline.vae.to(dtype=torch.float32)
        #    pixel_values.to(dtype=torch.float32)
        if pixel_values.shape[-3] == 4:
            image_latents = pixel_values/vae.config.scaling_factor
        else:
            image_latents = self.pipeline._encode_vae_image(pixel_values, device, num_videos_per_prompt, False)
        image_latents = image_latents.to(image_embeddings.dtype)

        #print('dtype:', image_latents.dtype)

        #if needs_upcasting:
        #    self.pipeline.vae.to(dtype=torch.float16)

        #num_frames = self.pipeline.unet.config.num_frames
        num_frames = 16
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        fps=4
        motion_bucket_id=127
        added_time_ids = self.pipeline._get_add_time_ids(
            fps,
            motion_bucket_id,
            0,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            False,
        )
        added_time_ids = added_time_ids.to(device)

        self.pipeline.scheduler.set_timesteps(timestep, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            None,
            None,
        )

        for i, t in enumerate(timesteps):
            #print('step:',i)
            if i == step_time - 1:
                complete = False
            else:
                complete = True
            #print('complete:',complete)

            latent_model_input = latents
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention
            # latent_model_input = torch.cat([mask, latent_model_input, image_latents], dim=2)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            #print('latent_model_input_shape:',latent_model_input.shape)
            #print('image_embeddings_shape:',image_embeddings.shape)

            # predict the noise residual
            # print('extract_layer_idx:',extract_layer_idx)
            # print('latent_model_input_shape:',latent_model_input.shape)
            # print('encoder_hidden_states:',image_embeddings.shape)
            feature_pred = self.step_unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                use_layer_idx=extract_layer_idx,
                all_layer = all_layer,
                complete = complete,
            )[0]
            # feature_pred = self.pipeline.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=image_embeddings,
            #     added_time_ids=added_time_ids,
            #     return_dict=False,
            # )[0]

            # print('feature_pred_shape:',feature_pred.shape)

            if not complete:
                break

            latents = self.pipeline.scheduler.step(feature_pred, t, latents).prev_sample

        return feature_pred