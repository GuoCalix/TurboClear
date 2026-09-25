import csv
import gc
import os
import random
import shutil
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from PIL import Image
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from HYPIR.dataset.sd_inpaint_dataset import SDInpaintImageDataset
from HYPIR.model.fusion_module import LearnableAttentionFusion
from HYPIR.trainer.objectclear import ObjectClearTrainer
from HYPIR.dataset.ober import OBERDirectoryDataset
from HYPIR.utils.objectclear_helpers import (
    BalancedL1Loss,
    clear_cross_attention_scores,
    get_object_localization_loss,
    resize_attn_map_divide2,
    unet_store_cross_attention_scores,
)
from HYPIR.utils.common import print_vram_state

logger = get_logger(__name__, log_level="INFO")


def _rank_aware_worker_init_fn(worker_id):
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    worker_seed = (torch.initial_seed() + rank * 1000003) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _resolve_state_file(weight_path: Optional[str]) -> Optional[str]:
    if not weight_path:
        return None
    weight_path = os.path.abspath(os.path.expanduser(str(weight_path)))
    if os.path.isdir(weight_path):
        candidates = [
            os.path.join(weight_path, "state_dict.pth"),
            os.path.join(weight_path, "pytorch_model.bin"),
            os.path.join(weight_path, "model.pth"),
            os.path.join(weight_path, "model.pt"),
            os.path.join(weight_path, "model.safetensors"),
            os.path.join(weight_path, "unet", "diffusion_pytorch_model.bin"),
            os.path.join(weight_path, "unet", "diffusion_pytorch_model.safetensors"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None
    return weight_path if os.path.isfile(weight_path) else None


def _resolve_unet_dir(weight_path: Optional[str]) -> Optional[str]:
    if not weight_path:
        return None
    weight_path = os.path.abspath(os.path.expanduser(str(weight_path)))
    candidates = []
    if os.path.isdir(weight_path):
        candidates.extend([os.path.join(weight_path, "unet"), weight_path])
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


def _safe_load_state_file(path: str) -> Any:
    if path.endswith(".safetensors"):
        return load_file(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, Mapping):
        for key in ["state_dict", "model", "unet", "module", "G", "generator", "model_state_dict"]:
            value = ckpt_obj.get(key)
            if isinstance(value, Mapping):
                return value
    return ckpt_obj


def _strip_state_key_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "_orig_mod.",
        "_fsdp_wrapped_module.",
        "model.",
        "unet.",
        "G.",
        "generator.",
        "student.",
        "student_G.",
        "base_model.model.",
    )
    current = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix) :]
                changed = True
                break
    return current


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            normalized[_strip_state_key_prefixes(str(key))] = value
    return normalized


class ObjectClearFusionTrainer(ObjectClearTrainer):
    """Fusion-only trainer for a frozen one-step ObjectClear generator."""

    def init_models(self):
        print("Use fusion-only training: freeze generator, train LearnableAttentionFusion.")
        self.init_scheduler()
        self.init_text_models()
        self.init_objectclear_modules()
        self.init_vae()
        self.init_generator()
        self.init_fusion_module()
        self.init_lpips()

        self.cross_attention_scores = {}
        self.object_localization_loss_fn = BalancedL1Loss(
            threshold=getattr(self.config, "object_localization_threshold", 0.1),
            background_loss_weight=getattr(self.config, "background_loss_weight", 1.0),
        )
        unet_store_cross_attention_scores(
            self.G,
            self.cross_attention_scores,
            layers=getattr(self.config, "attn_loss_layers", 5),
        )
        logger.info("Installed frozen-generator cross-attention hook for fusion alpha prior.")

    def init_generator(self):
        generator_checkpoint_path = getattr(self.config, "generator_checkpoint_path", None)
        unet_dir = _resolve_unet_dir(generator_checkpoint_path)
        if unet_dir is not None:
            self.G = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=self.weight_dtype,
            ).to(self.device)
            self.G.eval().requires_grad_(False)
            logger.info(f"Loaded frozen generator directly from Diffusers UNet directory: {unet_dir}")
            return

        self.G = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.G.eval().requires_grad_(False)

        state_file = _resolve_state_file(generator_checkpoint_path)
        if state_file is None:
            logger.warning("No generator_checkpoint_path resolved; fusion will train against base UNet output.")
            return

        ckpt_obj = _safe_load_state_file(state_file)
        state_dict = _extract_state_dict(ckpt_obj)
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"Unsupported generator checkpoint format: {state_file}")

        keys = [str(key) for key, value in state_dict.items() if isinstance(value, torch.Tensor)]
        is_lora = any("lora_" in key for key in keys)
        if is_lora:
            lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=self.config.lora_modules,
            )
            self.G.add_adapter(lora_cfg)
            logger.info("Detected LoRA generator checkpoint; added matching adapter before loading.")

        state_dict = _normalize_state_dict_keys(state_dict)
        model_state = self.G.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        missing, unexpected = self.G.load_state_dict(filtered, strict=False)
        del ckpt_obj, state_dict
        self.G.eval().requires_grad_(False)
        logger.info(
            f"Loaded frozen generator from {state_file}: "
            f"filtered={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}, is_lora={is_lora}"
        )
        if not filtered:
            raise ValueError(f"No tensors from generator checkpoint matched the UNet: {state_file}")

    def init_fusion_module(self):
        self.fusion_module = LearnableAttentionFusion(
            latent_channels=int(getattr(self.config, "fusion_latent_channels", 4)),
            hidden_channels=int(getattr(self.config, "fusion_hidden_channels", 32)),
            num_layers=int(getattr(self.config, "fusion_num_layers", 3)),
            logit_eps=float(getattr(self.config, "fusion_logit_eps", 1e-4)),
        ).to(self.device, dtype=torch.float32)

        fusion_path = getattr(self.config, "fusion_checkpoint_path", None)
        if fusion_path:
            fusion_path = os.path.abspath(os.path.expanduser(str(fusion_path)))
            state_path = os.path.join(fusion_path, "fusion_module.pth") if os.path.isdir(fusion_path) else fusion_path
            if os.path.exists(state_path):
                state_dict = torch.load(state_path, map_location="cpu")
                missing, unexpected = self.fusion_module.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"Loaded fusion module from {state_path}: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
            else:
                logger.warning(f"fusion_checkpoint_path not found, training from init: {state_path}")
        self.fusion_module.train().requires_grad_(True)

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizer for fusion module")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Unsupported optimizer_type={self.config.optimizer_type}")

        self.fusion_params = [param for param in self.fusion_module.parameters() if param.requires_grad]
        if not self.fusion_params:
            raise ValueError("No trainable parameters found in fusion_module.")
        self.fusion_opt = optimizer_cls(
            self.fusion_params,
            lr=float(getattr(self.config, "lr_fusion", 1e-4)),
            **self.config.opt_kwargs,
        )

    def init_dataset(self):
        train_input_size = getattr(self.config, "input_size", 512)
        dataset = SDInpaintImageDataset(
            dataset_path=self.config.dataset_path,
            is_sdxl=True,
            input_size=train_input_size,
            use_blank_mask=getattr(self.config, "use_blank_mask", False),
            blank_mask_prob=getattr(self.config, "blank_mask_prob", 0.0),
            color_augmentation=getattr(self.config, "color_augmentation", False),
            flip_augmentation=getattr(self.config, "flip_augmentation", True),
            rotation_augmentation=getattr(self.config, "rotation_augmentation", False),
            rotation_prob=getattr(self.config, "rotation_prob", 0.5),
            random_mask_dilation=getattr(self.config, "random_mask_dilation", True),
            random_mask_erosion=getattr(self.config, "random_mask_erosion", True),
            return_minimal=True,
        )
        self.dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_num_workers,
            worker_init_fn=_rank_aware_worker_init_fn,
            pin_memory=getattr(self.config, "pin_memory", True),
            persistent_workers=bool(getattr(self.config, "persistent_workers", False))
            and self.config.dataloader_num_workers > 0,
            prefetch_factor=getattr(self.config, "prefetch_factor", 4) if self.config.dataloader_num_workers > 0 else None,
        )

        val_path = getattr(self.config, "validation_dataset_path", None)
        self.val_dataloader = None
        if val_path:
            val_input_size = getattr(self.config, "validation_input_size", train_input_size)
            val_max_samples = getattr(self.config, "validation_max_samples", -1)
            is_decoded_ober_dir = (
                os.path.isdir(val_path)
                and os.path.isdir(os.path.join(val_path, "image"))
                and os.path.isdir(os.path.join(val_path, "mask"))
                and os.path.isdir(os.path.join(val_path, "GT"))
            )
            if is_decoded_ober_dir:
                val_dataset = OBERDirectoryDataset(
                    root=val_path,
                    input_size=val_input_size,
                    max_samples=val_max_samples,
                    prompt=getattr(self.config, "validation_prompt", "remove the instance of object"),
                )
            else:
                val_dataset = SDInpaintImageDataset(
                    dataset_path=val_path,
                    is_sdxl=True,
                    input_size=val_input_size,
                    use_blank_mask=False,
                    blank_mask_prob=0.0,
                    color_augmentation=False,
                    flip_augmentation=False,
                    rotation_augmentation=False,
                    random_mask_dilation=False,
                    random_mask_erosion=False,
                    return_minimal=True,
                )
                if val_max_samples is not None and int(val_max_samples) > 0:
                    val_dataset = torch.utils.data.Subset(
                        val_dataset,
                        range(min(int(val_max_samples), len(val_dataset))),
                    )
            self.val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=getattr(self.config, "validation_batch_size", 1),
                num_workers=getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers),
                worker_init_fn=_rank_aware_worker_init_fn,
                pin_memory=getattr(self.config, "pin_memory", True),
                persistent_workers=bool(getattr(self.config, "persistent_workers", False))
                and getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers) > 0,
                prefetch_factor=getattr(self.config, "prefetch_factor", 4)
                if getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers) > 0
                else None,
            )
            logger.info(f"Fusion validation dataset initialized with {len(val_dataset)} samples from {val_path}.")

    def prepare_all(self):
        logger.info("Wrapping fusion module, optimizer and train dataloader")
        self.fusion_module, self.fusion_opt, self.dataloader = self.accelerator.prepare(
            self.fusion_module,
            self.fusion_opt,
            self.dataloader,
        )
        self.fusion_params = [param for param in self.fusion_module.parameters() if param.requires_grad]
        print_vram_state("After accelerator.prepare", logger=logger)

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if weights:
                weights.pop(0)
            while models:
                models.pop()
            if not self.accelerator.is_main_process:
                return
            state_dict = self.accelerator.unwrap_model(self.fusion_module).state_dict()
            torch.save(state_dict, os.path.join(output_dir, "fusion_module.pth"))
            logger.info(f"Saved fusion module checkpoint: tensors={len(state_dict)}")

        def load_model_hook(models, input_dir):
            while models:
                models.pop()
            path = os.path.join(input_dir, "fusion_module.pth")
            if not os.path.exists(path):
                raise FileNotFoundError(f"fusion_module.pth missing in checkpoint: {input_dir}")
            state_dict = torch.load(path, map_location="cpu")
            missing, unexpected = self.accelerator.unwrap_model(self.fusion_module).load_state_dict(
                state_dict,
                strict=False,
            )
            del state_dict
            logger.info(
                f"Loaded fusion module from {path}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def on_training_start(self):
        global_step = 0
        if self.config.resume_from_checkpoint:
            path = str(self.config.resume_from_checkpoint)
            ckpt_name = os.path.basename(path.rstrip("/"))
            logger.info(f"Resuming fusion trainer from checkpoint {path}")
            self.load_accelerator_state(path)
            try:
                global_step = int(ckpt_name.split("-")[1])
            except Exception:
                global_step = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.max_train_steps),
            initial=global_step,
            desc="Fusion Steps",
            disable=not self.accelerator.is_main_process,
        )

    def _select_generator_timesteps(self, batch_size):
        noise_timesteps = self._set_inference_timesteps(int(getattr(self.config, "student_steps", 1)))
        noise_timesteps = torch.full((batch_size,), int(noise_timesteps[0]), dtype=torch.long, device=self.device)
        if str(getattr(self.config, "scheduler_timestep_spacing", "trailing")).lower() != "fixed":
            return noise_timesteps, noise_timesteps
        configured_model_timestep = getattr(self.config, "generator_timestep", None)
        if configured_model_timestep is None:
            return noise_timesteps, noise_timesteps
        if int(getattr(self.config, "student_steps", 1)) != 1:
            raise ValueError("generator_timestep is only supported for 1-step generator inference.")
        model_timesteps = torch.full(
            (batch_size,),
            int(configured_model_timestep),
            dtype=torch.long,
            device=self.device,
        )
        return noise_timesteps, model_timesteps

    def _repeat_condition_to_batch(self, tensor, batch_size, name):
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] <= 0 or batch_size % tensor.shape[0] != 0:
            raise ValueError(f"{name} batch size {tensor.shape[0]} cannot repeat to {batch_size}.")
        repeats = [batch_size // tensor.shape[0]] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeats)

    def _unet_pred(self, latents, timesteps):
        model_dtype = next(self.G.parameters()).dtype
        batch_size = latents.shape[0]
        if timesteps.ndim > 0:
            timesteps = self._repeat_condition_to_batch(timesteps, batch_size, "timesteps")
        model_input = self.scheduler.scale_model_input(latents, timesteps).to(dtype=model_dtype)
        model_input = torch.cat(
            [
                model_input,
                self.batch_inputs.mask_latent.to(dtype=model_dtype),
                self.batch_inputs.masked_image_latents.to(dtype=model_dtype),
            ],
            dim=1,
        )
        return self.G(
            model_input,
            timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"].to(dtype=model_dtype),
            added_cond_kwargs={
                "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"].to(dtype=model_dtype),
                "time_ids": self.batch_inputs.add_time_ids.to(dtype=model_dtype),
            },
        ).sample

    def _pred_x0(self, model_output, sample, timesteps):
        prediction_type = self.scheduler.config.prediction_type
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t

        if prediction_type == "epsilon":
            return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if prediction_type == "v_prediction":
            return alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    def _decode_latents(self, latents):
        z = latents.to(self.weight_dtype) / self.vae.config.scaling_factor
        if torch.is_grad_enabled() and z.requires_grad and getattr(self.config, "vae_decode_checkpointing", True):
            return checkpoint(lambda x: self.vae.decode(x).sample, z, use_reentrant=False).float()
        return self.vae.decode(z).sample.float()

    @torch.no_grad()
    def _forward_frozen_generator(self):
        clear_cross_attention_scores(self.cross_attention_scores)
        z = self.batch_inputs.z_lq
        noise_timesteps, model_timesteps = self._select_generator_timesteps(z.shape[0])
        eps = self._unet_pred(z, model_timesteps)
        z_pred = self._pred_x0(eps.float(), z.float(), noise_timesteps).to(dtype=z.dtype)

        if len(self.cross_attention_scores) == 0:
            logger.warning("No cross-attention scores captured; falling back to object mask latent as attention prior.")
            attn_map = self.batch_inputs.mask_latent.to(dtype=z.dtype)
            localization_loss = torch.tensor(0.0, device=self.device)
        else:
            _, layer_attn = next(iter(self.cross_attention_scores.items()))
            attn_map = resize_attn_map_divide2(
                layer_attn,
                self.batch_inputs.mask_latent,
                getattr(self.config, "fuse_index", 5),
            ).mean(dim=1, keepdim=True)
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                self.batch_inputs.object_effect_mask,
                self.object_localization_loss_fn,
                getattr(self.config, "fuse_index", 5),
            )
        clear_cross_attention_scores(self.cross_attention_scores)
        return z_pred.detach(), attn_map.detach().clamp(0.0, 1.0), localization_loss.detach()

    def _pixel_alpha(self, alpha_pixel, size):
        return F.interpolate(alpha_pixel.float(), size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    def _forward_fusion(self, decode_unfused: bool = False):
        z_pred, attn_map, localization_monitor = self._forward_frozen_generator()
        # The fusion head must not see effect/shadow masks at inference time. Use only
        # the deployable object mask as an input feature; effect masks remain supervision.
        fusion_input_mask = str(getattr(self.config, "fusion_input_mask", "object")).lower()
        if fusion_input_mask != "object":
            raise ValueError(
                f"Unsupported fusion_input_mask={fusion_input_mask!r}. "
                "Learnable fusion must use the deployable object mask; effect/shadow masks are supervision only."
            )
        fusion_mask = self.batch_inputs.mask_latent
        fusion_out = self.fusion_module(
            z_pred.float(),
            self.batch_inputs.masked_image_latents.float(),
            fusion_mask.float(),
            attn_map.float(),
        )
        x_latent_fused = self._decode_latents(fusion_out["z_fused"])
        alpha_pixel_img = self._pixel_alpha(fusion_out["alpha_pixel"], x_latent_fused.shape[-2:])
        x_fused = alpha_pixel_img * x_latent_fused + (1.0 - alpha_pixel_img) * self.batch_inputs.input_img.float()
        x_fused = x_fused.clamp(-1.0, 1.0)

        x_unfused = None
        if decode_unfused:
            with torch.no_grad():
                x_unfused = self._decode_latents(z_pred)
        return {
            "x_fused": x_fused,
            "x_latent_fused": x_latent_fused,
            "x_unfused": x_unfused,
            "z_pred": z_pred,
            "z_fused": fusion_out["z_fused"],
            "attn_map": attn_map,
            "alpha_latent": fusion_out["alpha_latent"],
            "alpha_pixel": fusion_out["alpha_pixel"],
            "alpha_pixel_img": alpha_pixel_img,
            "delta_latent": fusion_out["delta_latent"],
            "delta_pixel": fusion_out["delta_pixel"],
            "localization_monitor": localization_monitor,
        }

    @staticmethod
    def _dilate_mask(mask, radius):
        radius = int(radius)
        if radius <= 0:
            return mask
        kernel_size = 2 * radius + 1
        return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=radius).clamp(0.0, 1.0)

    @staticmethod
    def _erode_mask(mask, radius):
        radius = int(radius)
        if radius <= 0:
            return mask
        return 1.0 - ObjectClearFusionTrainer._dilate_mask(1.0 - mask.float(), radius)

    @staticmethod
    def _tv_loss(alpha):
        return (
            (alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).abs().mean()
            + (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs().mean()
        )

    @staticmethod
    def _masked_l1(pred, target, mask):
        loss = (pred.float() - target.float()).abs()
        mask = mask.to(device=loss.device, dtype=loss.dtype)
        if mask.shape[1] == 1 and loss.shape[1] != 1:
            mask = mask.repeat(1, loss.shape[1], 1, 1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    @staticmethod
    def _masked_psnr(pred, gt, mask):
        pred_01 = ((pred.float() + 1.0) / 2.0).clamp(0, 1)
        gt_01 = ((gt.float() + 1.0) / 2.0).clamp(0, 1)
        mask = (mask.float() > 0.5).float().to(device=pred_01.device)
        if mask.shape[1] == 1 and pred_01.shape[1] != 1:
            mask = mask.repeat(1, pred_01.shape[1], 1, 1)
        mse = ((pred_01 - gt_01) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        mse_val = float(mse.detach().cpu())
        return -10.0 * np.log10(max(mse_val, 1e-12))

    @staticmethod
    def _full_psnr(pred, gt):
        pred_01 = ((pred.float() + 1.0) / 2.0).clamp(0, 1)
        gt_01 = ((gt.float() + 1.0) / 2.0).clamp(0, 1)
        mse = ((pred_01 - gt_01) ** 2).mean()
        mse_val = float(mse.detach().cpu())
        return -10.0 * np.log10(max(mse_val, 1e-12))

    def _alpha_regularization_losses(self, alpha_pixel_img):
        effect_mask = self.batch_inputs.object_effect_mask.float().to(device=alpha_pixel_img.device)
        if effect_mask.shape[-2:] != alpha_pixel_img.shape[-2:]:
            effect_mask = F.interpolate(effect_mask, size=alpha_pixel_img.shape[-2:], mode="nearest")

        fg_core = self._erode_mask(effect_mask, getattr(self.config, "alpha_fg_erode", 8))
        bg_safe = 1.0 - self._dilate_mask(effect_mask, getattr(self.config, "alpha_bg_dilate", 16))

        loss_alpha_fg = ((1.0 - alpha_pixel_img) * fg_core).sum() / (fg_core.sum() + 1e-8)
        loss_alpha_bg = (alpha_pixel_img * bg_safe).sum() / (bg_safe.sum() + 1e-8)
        loss_alpha_tv = self._tv_loss(alpha_pixel_img)
        return loss_alpha_bg, loss_alpha_fg, loss_alpha_tv, fg_core, bg_safe

    def _localization_loss(self):
        if len(self.cross_attention_scores) == 0:
            return torch.tensor(0.0, device=self.device)
        return get_object_localization_loss(
            self.cross_attention_scores,
            self.batch_inputs.object_effect_mask,
            self.object_localization_loss_fn,
            getattr(self.config, "fuse_index", 5),
        )

    def optimize_fusion(self):
        with self.accelerator.accumulate(self.fusion_module):
            outputs = self._forward_fusion(decode_unfused=True)
            x_fused = outputs["x_fused"]
            effect_mask = self.batch_inputs.object_effect_mask.float()
            bg_mask = 1.0 - effect_mask
            bg_target_name = str(getattr(self.config, "fusion_bg_target", "gt")).lower()
            bg_target = self.batch_inputs.input_img if bg_target_name == "input" else self.batch_inputs.gt

            loss_mask_l1 = self._masked_l1(x_fused, self.batch_inputs.gt, effect_mask) * float(
                getattr(self.config, "lambda_fusion_mask_l1", 1.0)
            )
            loss_bg_l1 = self._masked_l1(x_fused, bg_target, bg_mask) * float(
                getattr(self.config, "lambda_fusion_bg_l1", 0.5)
            )
            loss_lpips = self.net_lpips(x_fused.float(), self.batch_inputs.gt.float()).mean() * float(
                getattr(self.config, "lambda_fusion_lpips", 0.2)
            )
            loss_alpha_bg, loss_alpha_fg, loss_alpha_tv, fg_core, bg_safe = self._alpha_regularization_losses(
                outputs["alpha_pixel_img"]
            )
            loss_alpha_bg = loss_alpha_bg * float(getattr(self.config, "lambda_alpha_bg", 0.05))
            loss_alpha_fg = loss_alpha_fg * float(getattr(self.config, "lambda_alpha_fg", 0.05))
            loss_alpha_tv = loss_alpha_tv * float(getattr(self.config, "lambda_alpha_tv", 0.01))

            loss_prior = torch.tensor(0.0, device=self.device, dtype=x_fused.dtype)
            lambda_prior = float(getattr(self.config, "lambda_alpha_prior", 0.0))
            if lambda_prior > 0.0:
                attn_img = self._pixel_alpha(outputs["attn_map"], outputs["alpha_pixel_img"].shape[-2:])
                loss_prior = F.l1_loss(outputs["alpha_pixel_img"].float(), attn_img.float()) * lambda_prior

            loss = (
                loss_mask_l1
                + loss_bg_l1
                + loss_lpips
                + loss_alpha_bg
                + loss_alpha_fg
                + loss_alpha_tv
                + loss_prior
            )
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.fusion_params, self.config.max_grad_norm)
            self.fusion_opt.step()
            self.fusion_opt.zero_grad()

        self.G_pred = x_fused.detach()
        self.last_fusion_outputs = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }
        self.last_fusion_outputs["fg_core"] = fg_core.detach()
        self.last_fusion_outputs["bg_safe"] = bg_safe.detach()

        return {
            "fusion_total": loss.detach(),
            "fusion_mask_l1": loss_mask_l1.detach(),
            "fusion_bg_l1": loss_bg_l1.detach(),
            "fusion_lpips": loss_lpips.detach(),
            "alpha_bg": loss_alpha_bg.detach(),
            "alpha_fg": loss_alpha_fg.detach(),
            "alpha_tv": loss_alpha_tv.detach(),
            "alpha_prior": loss_prior.detach(),
            "localization_monitor": outputs["localization_monitor"].detach(),
            "alpha_pixel_mean": outputs["alpha_pixel_img"].detach().mean(),
            "alpha_pixel_fg_mean": (outputs["alpha_pixel_img"].detach() * fg_core).sum() / (fg_core.sum() + 1e-8),
            "alpha_pixel_bg_mean": (outputs["alpha_pixel_img"].detach() * bg_safe).sum() / (bg_safe.sum() + 1e-8),
        }

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        validation_steps = int(getattr(self.config, "validation_steps", 0))

        while self.global_step < self.config.max_train_steps:
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                loss_dict = self.optimize_fusion()

                log_dict = {}
                for key, value in loss_dict.items():
                    value = value.detach().float().mean().reshape(1).to(self.device)
                    log_dict[f"loss/{key}"] = self.accelerator.gather(value).mean().item()

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.pbar.update(1)
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"Fusion Step, VRAM peak: {peak:.2f} GB")

                    self.accelerator.log(log_dict, step=self.global_step)
                    self.log_metrics_csv(log_dict, self.global_step)

                    if self.global_step % self.config.log_image_steps == 0 or self.global_step == 1:
                        self.log_images()
                    if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1:
                        self.save_checkpoint()
                    if validation_steps > 0 and (
                        self.global_step % validation_steps == 0 or self.global_step == 1
                    ):
                        self.validate()

                if self.global_step >= self.config.max_train_steps:
                    break
        self.accelerator.end_training()

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.config.checkpoints_total_limit is not None:
                checkpoints = [
                    d
                    for d in os.listdir(self.config.output_dir)
                    if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
                ]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                    for removing_checkpoint in checkpoints[:num_to_remove]:
                        shutil.rmtree(os.path.join(self.config.output_dir, removing_checkpoint))
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            logger.info(f"Saved fusion trainer state to {save_path}")

    @torch.no_grad()
    def validate(self):
        if self.val_dataloader is None:
            return
        self.accelerator.wait_for_everyone()
        self.fusion_module.eval()
        self.G.eval()

        pred_names = ["unfused", "latent_fused", "learned_fused"]
        metric_names = []
        for pred_name in pred_names:
            metric_names.extend([f"{pred_name}_masked_psnr", f"{pred_name}_full_psnr", f"{pred_name}_lpips"])
        metrics = {name: 0.0 for name in metric_names}
        metric_counts = {name: 0 for name in metric_names}
        num_batches = 0
        num_samples = 0

        val_image_root = getattr(
            self.config,
            "validation_image_dir",
            os.path.join(self.config.output_dir, "validation", "images"),
        )
        val_save_dir = os.path.join(val_image_root, f"{self.global_step:07}")
        if self.accelerator.is_main_process:
            os.makedirs(val_save_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        save_batches = int(getattr(self.config, "validation_save_batches", 4))
        for batch_idx, batch in enumerate(self.val_dataloader):
            max_batches = int(getattr(self.config, "validation_max_batches", -1))
            if max_batches > 0 and batch_idx >= max_batches:
                break
            self.prepare_batch_inputs(batch)
            outputs = self._forward_fusion(decode_unfused=True)

            preds = {
                "unfused": outputs["x_unfused"],
                "latent_fused": outputs["x_latent_fused"],
                "learned_fused": outputs["x_fused"],
            }
            effect_mask = self.batch_inputs.object_effect_mask.clamp(0, 1)
            bs = self.batch_inputs.gt.shape[0]
            for pred_name, pred_x in preds.items():
                masked_psnr_sum = 0.0
                full_psnr_sum = 0.0
                for sample_idx in range(bs):
                    masked_psnr_sum += self._masked_psnr(
                        pred_x[sample_idx : sample_idx + 1],
                        self.batch_inputs.gt[sample_idx : sample_idx + 1],
                        effect_mask[sample_idx : sample_idx + 1],
                    )
                    full_psnr_sum += self._full_psnr(
                        pred_x[sample_idx : sample_idx + 1],
                        self.batch_inputs.gt[sample_idx : sample_idx + 1],
                    )
                metrics[f"{pred_name}_masked_psnr"] += masked_psnr_sum
                metrics[f"{pred_name}_full_psnr"] += full_psnr_sum
                metrics[f"{pred_name}_lpips"] += (
                    self.net_lpips(pred_x.float(), self.batch_inputs.gt.float()).mean().item() * bs
                )
                metric_counts[f"{pred_name}_masked_psnr"] += bs
                metric_counts[f"{pred_name}_full_psnr"] += bs
                metric_counts[f"{pred_name}_lpips"] += bs
            num_batches += 1
            num_samples += bs

            if self.accelerator.is_main_process and batch_idx < save_batches:
                self._save_validation_images(val_save_dir, batch_idx, outputs)

        log_metrics = {}
        if num_samples > 0:
            log_metrics = {
                f"validation/{key}": metrics[key] / max(metric_counts[key], 1)
                for key in metric_names
            }
            if self.accelerator.is_main_process:
                self.accelerator.log(log_metrics, step=self.global_step)
                logger.info(f"Fusion validation at step {self.global_step}: {log_metrics}")
        if self.accelerator.is_main_process:
            self._append_validation_csv(log_metrics, num_batches, num_samples, val_save_dir)
        self.fusion_module.train()
        self.accelerator.wait_for_everyone()

    def _to_uint8_np(self, image):
        image = ((image.float() + 1.0) / 2.0).clamp(0, 1)
        return (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def _save_validation_images(self, val_save_dir, batch_idx, outputs):
        input_img = ((self.batch_inputs.input_img + 1.0) / 2.0).clamp(0, 1)
        gt = ((self.batch_inputs.gt + 1.0) / 2.0).clamp(0, 1)
        mask = self.batch_inputs.object_mask.clamp(0, 1)
        effect_mask = self.batch_inputs.object_effect_mask.clamp(0, 1)
        alpha = outputs["alpha_pixel_img"].clamp(0, 1)
        attn = self._pixel_alpha(outputs["attn_map"], alpha.shape[-2:]).clamp(0, 1)
        preds = {
            "unfused": outputs["x_unfused"],
            "latent_fused": outputs["x_latent_fused"],
            "learned_fused": outputs["x_fused"],
        }

        bs = input_img.shape[0]
        for sample_idx in range(bs):
            for pred_name, pred in preds.items():
                pred_dir = os.path.join(val_save_dir, pred_name)
                os.makedirs(pred_dir, exist_ok=True)
                Image.fromarray(self._to_uint8_np(pred[sample_idx])).save(
                    os.path.join(pred_dir, f"batch{batch_idx}_sample{sample_idx}_{pred_name}.png")
                )

            grid = make_grid(
                [
                    input_img[sample_idx].cpu(),
                    gt[sample_idx].cpu(),
                    ((preds["unfused"][sample_idx].cpu() + 1.0) / 2.0).clamp(0, 1),
                    ((preds["latent_fused"][sample_idx].cpu() + 1.0) / 2.0).clamp(0, 1),
                    ((preds["learned_fused"][sample_idx].cpu() + 1.0) / 2.0).clamp(0, 1),
                    alpha[sample_idx].repeat(3, 1, 1).cpu(),
                    attn[sample_idx].repeat(3, 1, 1).cpu(),
                    effect_mask[sample_idx].repeat(3, 1, 1).cpu(),
                    mask[sample_idx].repeat(3, 1, 1).cpu(),
                ],
                nrow=9,
            )
            image_arr = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(image_arr).save(
                os.path.join(
                    val_save_dir,
                    f"batch{batch_idx}_sample{sample_idx}_input_gt_unfused_latent_learned_alpha_attn_effect_mask.png",
                )
            )

    def _append_validation_csv(self, log_metrics, num_batches, num_samples, val_save_dir):
        csv_path = getattr(
            self.config,
            "validation_metrics_csv",
            os.path.join(self.config.output_dir, "validation", "metrics.csv"),
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        row = {
            "step": int(self.global_step),
            "num_batches": int(num_batches),
            "num_samples": int(num_samples),
            "image_dir": val_save_dir,
        }
        row.update({key.replace("validation/", ""): float(value) for key, value in log_metrics.items()})
        base_fields = ["step", "num_batches", "num_samples"]
        tail_fields = ["image_dir"]
        metric_fields = [key for key in row.keys() if key not in base_fields and key not in tail_fields]
        desired_fields = base_fields + metric_fields + tail_fields

        existing_rows = []
        existing_fields = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = [
                    {key: value for key, value in existing_row.items() if key is not None}
                    for existing_row in reader
                ]

        extra_existing_fields = [
            field
            for field in existing_fields
            if field not in desired_fields and field not in tail_fields
        ]
        fieldnames = base_fields + metric_fields + extra_existing_fields + tail_fields
        should_rewrite = bool(existing_rows or existing_fields) and existing_fields != fieldnames
        mode = "w" if should_rewrite or not os.path.exists(csv_path) else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
            writer.writerow(row)

    def log_images(self):
        if not hasattr(self, "last_fusion_outputs"):
            return
        outputs = self.last_fusion_outputs
        n = min(4, self.batch_inputs.gt.shape[0])
        image_logs = {
            "input": (self.batch_inputs.input_img[:n] + 1.0) / 2.0,
            "gt": (self.batch_inputs.gt[:n] + 1.0) / 2.0,
            "unfused": (outputs["x_unfused"][:n] + 1.0) / 2.0,
            "latent_fused": (outputs["x_latent_fused"][:n] + 1.0) / 2.0,
            "learned_fused": (outputs["x_fused"][:n] + 1.0) / 2.0,
            "alpha_pixel": outputs["alpha_pixel_img"][:n].repeat(1, 3, 1, 1),
            "attn_prior": self._pixel_alpha(outputs["attn_map"][:n], outputs["alpha_pixel_img"].shape[-2:]).repeat(1, 3, 1, 1),
        }

        if not self.accelerator.is_main_process:
            return
        for key, images in image_logs.items():
            images = images.clamp(0, 1)
            for tracker in self.accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_image(
                        f"image/{key}",
                        make_grid(images.float().cpu(), nrow=min(4, n)),
                        self.global_step,
                    )
            save_dir = os.path.join(self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
            os.makedirs(save_dir, exist_ok=True)
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for idx, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{idx}.png"))
