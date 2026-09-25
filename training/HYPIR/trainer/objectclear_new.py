from typing import Dict
import gc
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import loguru
from accelerate.logging import get_logger
from diffusers.training_utils import compute_snr

from HYPIR.trainer.objectclear import ObjectClearTrainer

logger = get_logger(__name__, log_level="INFO")


def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

    unet_layer_names = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(unet_layer_names) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = unet_layer_names[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(query, key, attention_mask)
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.new_get_attention_scores = types.MethodType(make_new_get_attention_scores_fn(name), module)
            module.get_attention_scores = module.new_get_attention_scores

    return unet


def clear_cross_attention_scores(cross_attention_scores):
    keys = list(cross_attention_scores.keys())
    for key in keys:
        del cross_attention_scores[key]
    gc.collect()

def resize_attn_map_divide2(attn_map, mask, fuse_index):
    """Extract one token attention map and resize it to latent mask resolution."""
    bxh, num_noise_latents, _ = attn_map.shape
    batch_size = mask.shape[0]

    if bxh % batch_size != 0:
        raise ValueError(f"Unexpected attention shape {attn_map.shape} for batch size {batch_size}.")

    num_heads = bxh // batch_size
    size = int(num_noise_latents**0.5)
    if size * size != num_noise_latents:
        raise ValueError(f"num_noise_latents={num_noise_latents} is not a square number.")

    attn_map = attn_map.view(batch_size, num_heads, num_noise_latents, -1)
    index_tensor = torch.full(
        (batch_size, num_heads, num_noise_latents, 1),
        fuse_index,
        dtype=torch.long,
        device=attn_map.device,
    )
    attn_map = torch.gather(attn_map, dim=3, index=index_tensor).squeeze(-1)
    attn_map = attn_map.view(batch_size, num_heads, size, size)
    attn_map = F.interpolate(attn_map, size=mask.shape[-2:], mode="bilinear", antialias=True)

    attn_min = attn_map.amin(dim=(-2, -1), keepdim=True)
    attn_max = attn_map.amax(dim=(-2, -1), keepdim=True)
    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    return attn_map

class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=0.1, normalize=False, background_loss_weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize
        self.background_loss_weight = background_loss_weight

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )

        object_segmaps = (object_segmaps > self.threshold).to(object_segmaps.dtype)
        background_segmaps = 1 - object_segmaps

        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(dim=2) / background_segmaps_sum
        object_loss = (object_token_attn_prob * object_segmaps).sum(dim=2) / object_segmaps_sum

        return self.background_loss_weight * background_loss - object_loss + 1


def get_object_localization_loss_for_one_layer(cross_attention_scores, object_segmaps, loss_fn, fuse_index):
    bxh, num_noise_latents, _ = cross_attention_scores.shape
    batch_size, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    object_segmaps = F.interpolate(object_segmaps, size=(size, size), mode="bilinear", antialias=True)
    object_segmaps = object_segmaps.view(batch_size, max_num_objects, -1)

    num_heads = bxh // batch_size
    cross_attention_scores = cross_attention_scores.view(batch_size, num_heads, num_noise_latents, -1)

    index_tensor = torch.full(
        (batch_size, num_heads, num_noise_latents, 1),
        fuse_index,
        dtype=torch.long,
        device=cross_attention_scores.device,
    )
    object_token_attn_prob = torch.gather(cross_attention_scores, dim=3, index=index_tensor)

    object_segmaps = object_segmaps.permute(0, 2, 1).unsqueeze(1).expand(batch_size, num_heads, num_noise_latents, 1)
    loss = loss_fn(object_token_attn_prob, object_segmaps)
    return loss.mean()


def get_object_localization_loss(cross_attention_scores, object_segmaps, loss_fn, fuse_index):
    if len(cross_attention_scores) == 0:
        return torch.tensor(0.0, device=object_segmaps.device, dtype=object_segmaps.dtype)

    total = 0.0
    for _, layer_scores in cross_attention_scores.items():
        total = total + get_object_localization_loss_for_one_layer(
            layer_scores,
            object_segmaps,
            loss_fn,
            fuse_index,
        )
    return total / len(cross_attention_scores)


class ObjectClearNewTrainer(ObjectClearTrainer):
    """ObjectClear 4-step distillation trainer with train_objectclear-style losses."""

    def init_models(self):
        super().init_models()
        self.cross_attention_scores = {}
        self.object_localization_loss_fn = None

        enable_attn_hook = getattr(self.config, "object_localization", False) or getattr(
            self.config, "apply_attention_guided_fusion", False
        )
        loguru.logger.info(f"Use localization: {self.config.object_localization}")
        loguru.logger.info(f"Use attention fusion: {self.config.apply_attention_guided_fusion}")

        if enable_attn_hook:
            unwrapped_unet = self.unwrap_model(self.G)
            unet_store_cross_attention_scores(
                unwrapped_unet,
                self.cross_attention_scores,
                layers=getattr(self.config, "attn_loss_layers", 5),
            )
        if getattr(self.config, "object_localization", False):
            self.object_localization_loss_fn = BalancedL1Loss(
                threshold=getattr(self.config, "object_localization_threshold", 0.1),
                background_loss_weight=getattr(self.config, "background_loss_weight", 1.0),
            )

    def forward_generator(self):
        z = self.batch_inputs.z_lq
        image_latents = self.batch_inputs.masked_image_latents
        noise = torch.randn_like(image_latents)
        mask = self.batch_inputs.mask_latent.to(dtype=z.dtype)
        use_attention_guided_fusion = getattr(self.config, "apply_attention_guided_fusion", False)
        fuse_index = getattr(self.config, "fuse_index", 5)
        attn_map = None

        if use_attention_guided_fusion:
            clear_cross_attention_scores(self.cross_attention_scores)

        timesteps = self._set_inference_timesteps(getattr(self.config, "distill_steps", 4))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((z.shape[0],), int(t), dtype=torch.long, device=self.device)
            z_model = self.scheduler.scale_model_input(z, t)
            # if self.G.config.in_channels == 9:
            z_model = torch.cat([z_model, self.batch_inputs.mask_latent, self.batch_inputs.masked_image_latents], dim=1)

            eps = self.G(
                z_model,
                t_batch,
                encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                added_cond_kwargs={
                    "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                    "time_ids": self.batch_inputs.add_time_ids,
                },
            ).sample
            z = self.scheduler.step(eps, t, z).prev_sample
            if use_attention_guided_fusion:
                if i == 0 and len(timesteps) > 1:
                    init_latents_proper = image_latents
                    noise_timestep = timesteps[i + 1]
                    if not torch.is_tensor(noise_timestep):
                        noise_timestep = torch.tensor(noise_timestep, dtype=torch.long, device=self.device)
                    noise_timestep = noise_timestep.reshape(1).to(self.device)
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper,
                        noise,
                        noise_timestep,
                    )
                    # z = (1 - mask) * init_latents_proper + mask * z

                if i == len(timesteps) - 1 and len(self.cross_attention_scores) > 0:
                    # _, attn_map = next(iter(self.cross_attention_scores.items()))
                    # attn_map = resize_attn_map_divide2(attn_map, mask, fuse_index)
                    # attn_map = attn_map.mean(dim=1, keepdim=True)
                    attn_map_list = []
                    for layer_name, layer_attn in self.cross_attention_scores.items():
                        # 对每一个层提取并缩放到 mask 分辨率
                        curr_attn = resize_attn_map_divide2(layer_attn, mask, fuse_index)
                        curr_attn = curr_attn.mean(dim=1, keepdim=True)
                        attn_map_list.append(curr_attn)
                    attn_map = torch.stack(attn_map_list, dim=0).mean(dim=0)
                    
        if use_attention_guided_fusion:
            clear_cross_attention_scores(self.cross_attention_scores)

        if use_attention_guided_fusion and attn_map is not None:
            z = (1 - attn_map) * image_latents + attn_map * z
        x_pred = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        return x_pred, z

    def _compute_train_objectclear_style_loss(self):
        z_gt = self.batch_inputs.z_gt.to(dtype=self.weight_dtype)
        mask = self.batch_inputs.mask_latent.to(dtype=self.weight_dtype)
        bs = z_gt.shape[0]

        noise = torch.randn_like(z_gt)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
            dtype=torch.long,
        )

        noisy_latents = self.scheduler.add_noise(z_gt, noise, timesteps)
        noisy_model_input = self.scheduler.scale_model_input(noisy_latents, timesteps)

        # if self.G.config.in_channels == 9:
        noisy_model_input = torch.cat([noisy_model_input, mask, self.batch_inputs.masked_image_latents], dim=1)

        model_pred = self.G(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
            added_cond_kwargs={
                "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                "time_ids": self.batch_inputs.add_time_ids,
            },
        ).sample

        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(z_gt, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        snr_gamma = getattr(self.config, "snr_gamma", None)
        mask_weight = getattr(self.config, "mask_weight", None)

        if snr_gamma is None:
            mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            if mask_weight is not None:
                latent_mask = mask.expand_as(mse)
                weight_matrix = torch.where(latent_mask > 0.5, torch.full_like(mse, float(mask_weight)), torch.ones_like(mse))
                mse = mse * weight_matrix
            diffusion_loss = mse.mean()
        else:
            snr = compute_snr(self.scheduler, timesteps)
            base_weight = torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
            if prediction_type == "v_prediction":
                mse_loss_weights = base_weight + 1
            else:
                mse_loss_weights = base_weight

            mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            mse = mse.mean(dim=list(range(1, len(mse.shape)))) * mse_loss_weights
            diffusion_loss = mse.mean()

        localization_loss = torch.tensor(0.0, device=self.device, dtype=diffusion_loss.dtype)
        if getattr(self.config, "object_localization", False) and self.object_localization_loss_fn is not None:
            fuse_index = getattr(self.config, "fuse_index", 5)
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                self.batch_inputs.object_effect_mask,
                self.object_localization_loss_fn,
                fuse_index,
            )
            clear_cross_attention_scores(self.cross_attention_scores)

        return diffusion_loss, localization_loss

    def optimize_generator(self):
        with self.accelerator.accumulate(self.G):
            if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None:
                self.unwrap_model(self.D).eval().requires_grad_(False)

            x, z = self.forward_generator()
            self.G_pred = x

            if getattr(self.config, "use_blended_image", False):
                effect_mask = self.batch_inputs.effect_mask
                # blended_x = (1 - effect_mask) * self.batch_inputs.gt + effect_mask * x
            loss_mse_img = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * getattr(self.config, "lambda_l2", 1.0)
            loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * getattr(self.config, "lambda_lpips", 0.0)

            if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None: 
                if getattr(self.config, "use_D_sdxl", False):
                    loss_disc = self.D(
                        latents=z, 
                        mask=self.batch_inputs.mask_latent, 
                        reference_latents=self.batch_inputs.masked_image_latents,
                        encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                        added_cond_kwargs={
                            "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                            "time_ids": self.batch_inputs.add_time_ids,},
                        for_G=True,
                        ).mean() * getattr(self.config, "lambda_gan", 0.0)
                else:
                    loss_disc = self.D(x, for_G=True).mean() * getattr(self.config, "lambda_gan", 0.0)
            else:
                loss_disc = torch.tensor(0.0, device=self.device, dtype=loss_mse_img.dtype)

            diffusion_loss, localization_loss = self._compute_train_objectclear_style_loss()

            loss = (
                loss_mse_img
                + loss_lpips
                + loss_disc
                + getattr(self.config, "lambda_diffusion", 1.0) * diffusion_loss
                + getattr(self.config, "object_localization_weight", 1.0) * localization_loss
            )

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()

        return {
            "G_total": loss,
            "G_mse": loss_mse_img,
            "G_lpips": loss_lpips,
            "G_disc": loss_disc,
            "G_diffusion": diffusion_loss,
            "G_localization": localization_loss,
        }

    def optimize_discriminator(self):
        gt = self.batch_inputs.gt
        z_gt = self.batch_inputs.z_gt
        with torch.no_grad():
            x, z = self.forward_generator()
        self.G_pred = x
        with self.accelerator.accumulate(self.D):
            self.unwrap_model(self.D).train().requires_grad_(True)
            if getattr(self.config, "use_D_sdxl", False):
                loss_D_real, real_logits = self.D(
                    z_gt, 
                    mask=self.batch_inputs.mask_latent, 
                    reference_latents=self.batch_inputs.masked_image_latents,
                    encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                    added_cond_kwargs={
                        "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                        "time_ids": self.batch_inputs.add_time_ids,},
                    for_real=True, return_logits=True)
                
                loss_D_fake, fake_logits = self.D(
                    z, 
                    mask=self.batch_inputs.mask_latent, 
                    reference_latents=self.batch_inputs.masked_image_latents,
                    encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                    added_cond_kwargs={
                        "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                        "time_ids": self.batch_inputs.add_time_ids,},
                    for_real=False, return_logits=True)
            else:
                loss_D_real, real_logits = self.D(gt, for_real=True, return_logits=True)
                loss_D_fake, fake_logits = self.D(x, for_real=True, return_logits=True)

            loss_D = loss_D_real.mean() + loss_D_fake.mean()
            self.accelerator.backward(loss_D)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.D_params, self.config.max_grad_norm)
            self.D_opt.step()
            self.D_opt.zero_grad()
        loss_dict = dict(D=loss_D)
        # logits = D(x) w/o sigmoid = log(p_real(x) / p_fake(x))
        with torch.no_grad():
            real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
            fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
        loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))
        return loss_dict
