import copy
import os
import csv
import gc
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from HYPIR.dataset.sd_inpaint_dataset import SDInpaintImageDataset
from HYPIR.trainer.objectclear import ObjectClearTrainer
from HYPIR.utils.common import print_vram_state
from HYPIR.trainer.objectclear_new import (
    ObjectClearNewTrainer,
    BalancedL1Loss,
    get_object_localization_loss,
    unet_store_cross_attention_scores,
)

logger = get_logger(__name__, log_level="INFO")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_image_files(folder):
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    )


def _pil_rgb_to_tensor(path, out_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((out_size, out_size), Image.BICUBIC)
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _pil_mask_to_tensor(path, out_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize((out_size, out_size), Image.NEAREST)
    arr = (np.array(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


class OBERDirectoryDataset(Dataset):
    def __init__(self, root, input_size=512, max_samples=-1, prompt="remove the instance of object"):
        self.root = root
        self.input_size = input_size
        self.prompt = prompt

        image_dir = os.path.join(root, "image")
        mask_dir = os.path.join(root, "mask")
        gt_dir = os.path.join(root, "GT")
        effect_mask_dir = os.path.join(root, "effect_mask")
        for required_dir in [image_dir, mask_dir, gt_dir]:
            if not os.path.isdir(required_dir):
                raise ValueError(f"Missing OBER-Test validation directory: {required_dir}")

        mask_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_image_files(mask_dir)}
        gt_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_image_files(gt_dir)}
        effect_mask_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_image_files(effect_mask_dir)} if os.path.isdir(effect_mask_dir) else {}

        samples = []
        for image_path in _list_image_files(image_dir):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = mask_files.get(stem)
            gt_path = gt_files.get(stem)
            if mask_path is None or gt_path is None:
                continue
            effect_mask_path = effect_mask_files.get(stem, mask_path)
            samples.append((stem, image_path, mask_path, effect_mask_path, gt_path))

        if max_samples is not None and int(max_samples) > 0:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError(f"No OBER-Test validation samples found under: {root}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, image_path, mask_path, effect_mask_path, gt_path = self.samples[idx]
        return {
            "name": name,
            "image": _pil_rgb_to_tensor(image_path, self.input_size),
            "GT": _pil_rgb_to_tensor(gt_path, self.input_size),
            "object_mask": _pil_mask_to_tensor(mask_path, self.input_size),
            "object_effect_mask": _pil_mask_to_tensor(effect_mask_path, self.input_size),
            "txt": self.prompt,
        }


def resolve_state_file(weight_path):
    if not weight_path:
        return None
    if os.path.isdir(weight_path):
        candidate = os.path.join(weight_path, "state_dict.pth")
        return candidate if os.path.exists(candidate) else None
    if os.path.isfile(weight_path):
        return weight_path
    return None


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model", "unet"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["module.", "unet."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


class ObjectClearLCMTrainer(ObjectClearNewTrainer):
    """LCM-style one-step consistency distillation for ObjectClear inpainting."""

    def _free_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _profile_enabled_for_step(self):
        if not bool(getattr(self.config, "step_profiler", False)):
            return False
        warmup = int(getattr(self.config, "step_profiler_warmup", 0))
        max_profile_steps = int(getattr(self.config, "step_profiler_steps", 20))
        every = max(1, int(getattr(self.config, "step_profiler_every", 1)))
        local_step = int(getattr(self, "_step_profile_local_step", 0))
        if local_step < warmup:
            return False
        if max_profile_steps > 0 and local_step >= warmup + max_profile_steps:
            return False
        return ((local_step - warmup) % every) == 0

    def _profile_sync(self):
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_begin(self):
        self._step_profile_active = self._profile_enabled_for_step()
        self._step_profile_times = {}
        if not self._step_profile_active:
            return
        self._profile_sync()
        now = time.perf_counter()
        wait_start = getattr(self, "_step_profile_wait_start", None)
        if wait_start is not None:
            self._step_profile_times["dataloader_wait"] = now - wait_start
        self._step_profile_last_time = now

    def _profile_mark(self, name):
        if not getattr(self, "_step_profile_active", False):
            return
        self._profile_sync()
        now = time.perf_counter()
        self._step_profile_times[name] = self._step_profile_times.get(name, 0.0) + (
            now - self._step_profile_last_time
        )
        self._step_profile_last_time = now

    def _profile_collect_log(self):
        if not getattr(self, "_step_profile_active", False) or not self._step_profile_times:
            return {}
        keys = sorted(self._step_profile_times)
        local_times = torch.tensor(
            [self._step_profile_times[key] for key in keys],
            device=self.device,
            dtype=torch.float32,
        )
        gathered = self.accelerator.gather(local_times).reshape(-1, len(keys))
        mean_times = gathered.mean(dim=0)
        max_times = gathered.max(dim=0).values
        profile_log = {}
        for idx, key in enumerate(keys):
            profile_log[f"profile/mean_sec/{key}"] = mean_times[idx].item()
            profile_log[f"profile/max_sec/{key}"] = max_times[idx].item()

        if self.accelerator.is_main_process:
            parts = [
                f"{key}: mean={mean_times[idx].item():.4f}s max={max_times[idx].item():.4f}s"
                for idx, key in enumerate(keys)
            ]
            local_step = int(getattr(self, "_step_profile_local_step", 0))
            logger.info(f"Step profile at global_step {self.global_step} local_step {local_step}: " + " | ".join(parts))
        return profile_log

    def init_models(self):
        ObjectClearTrainer.init_models(self)
        self.cross_attention_scores = {}
        self.object_localization_loss_fn = None
        logger.info(f"Use localization: {self.config.object_localization}")
        logger.info(f"Use attention fusion: {self.config.apply_attention_guided_fusion}")
        self.teacher_cross_attention_scores = {}
        self.init_teacher_generator()
        self.load_initial_student_weights()
        self.init_target_generator()

        enable_attn_hook = getattr(self.config, "object_localization", False) or getattr(
            self.config, "apply_attention_guided_fusion", False
        )
        if enable_attn_hook:
            unet_store_cross_attention_scores(
                self.G,
                self.cross_attention_scores,
                layers=getattr(self.config, "attn_loss_layers", 5),
            )
        if getattr(self.config, "object_localization", False):
            self.object_localization_loss_fn = BalancedL1Loss(
                threshold=getattr(self.config, "object_localization_threshold", 0.1),
                background_loss_weight=getattr(self.config, "background_loss_weight", 1.0),
            )

    def init_teacher_generator(self):
        self.teacher_G = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.teacher_G.eval().requires_grad_(False)

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=self.config.lora_modules,
        )
        self.teacher_G.add_adapter(lora_cfg)
        self.teacher_G.eval().requires_grad_(False)

        teacher_path = getattr(self.config, "teacher_checkpoint_path", None)
        if teacher_path is None:
            logger.info("No teacher_checkpoint_path set; using base_model_path UNet as frozen LCM teacher.")
        else:
            self._load_unet_weights(self.teacher_G, teacher_path, "teacher")

        if getattr(self.config, "teacher_apply_attention_guided_fusion", False):
            unet_store_cross_attention_scores(
                self.teacher_G,
                self.teacher_cross_attention_scores,
                layers=getattr(self.config, "attn_loss_layers", 5),
            )

    def init_target_generator(self):
        self.target_G = copy.deepcopy(self.G).to(self.device)
        self.target_G.eval().requires_grad_(False)
        self._refresh_target_ema_params()
        self.update_target_generator(decay=0.0)

    def load_initial_student_weights(self):
        student_path = getattr(self.config, "student_init_checkpoint_path", None)
        if student_path is None and getattr(self.config, "init_student_from_teacher", True):
            student_path = getattr(self.config, "teacher_checkpoint_path", None)
        if student_path:
            self._load_unet_weights(self.G, student_path, "student")

    def _load_unet_weights(self, model, weight_path, role):
        state_file = resolve_state_file(weight_path)
        if state_file is None:
            raise FileNotFoundError(f"Could not resolve {role} checkpoint from: {weight_path}")

        ckpt_obj = torch.load(state_file, map_location="cpu")
        state_dict = extract_state_dict(ckpt_obj)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported {role} checkpoint format: {state_file}")
        state_dict = normalize_state_dict_keys(state_dict)

        model_state = model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        logger.info(
            f"Loaded {role} weights from {state_file}: "
            f"filtered={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.config.optimizer_type}")

        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        self.G_opt = optimizer_cls(self.G_params, lr=self.config.lr_G, **self.config.opt_kwargs)

        if getattr(self.config, "use_D", False) and hasattr(self, "D"):
            self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
            self.D_opt = optimizer_cls(self.D_params, lr=self.config.lr_D, **self.config.opt_kwargs)

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
            pin_memory=getattr(self.config, "pin_memory", True),
            persistent_workers=bool(getattr(self.config, "persistent_workers", False))
            and self.config.dataloader_num_workers > 0,
            prefetch_factor=getattr(self.config, "prefetch_factor", 4) if self.config.dataloader_num_workers > 0 else None,
        )

        val_path = getattr(self.config, "validation_dataset_path", None)
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
                )
                if val_max_samples is not None and int(val_max_samples) > 0:
                    val_dataset = torch.utils.data.Subset(val_dataset, range(min(int(val_max_samples), len(val_dataset))))
            self.val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=getattr(self.config, "validation_batch_size", 1),
                num_workers=getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers),
                pin_memory=getattr(self.config, "pin_memory", True),
                persistent_workers=bool(getattr(self.config, "persistent_workers", False))
                and getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers) > 0,
                prefetch_factor=getattr(self.config, "prefetch_factor", 4)
                if getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers) > 0
                else None,
            )
            logger.info(f"OBER validation dataset initialized with {len(val_dataset)} samples from {val_path}.")
        else:
            self.val_dataloader = None

    def prepare_all(self):
        logger.info("Wrapping LCM student models, optimizers and dataloaders")
        attrs = ["G", "G_opt", "dataloader"]
        if getattr(self.config, "use_D", False) and hasattr(self, "D") and hasattr(self, "D_opt"):
            attrs.extend(["D", "D_opt"])
        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        self.teacher_G.to(self.device).eval().requires_grad_(False)
        self.target_G.to(self.device).eval().requires_grad_(False)
        self._refresh_target_ema_params()

    def on_training_start(self):
        super().on_training_start()
        self.update_target_generator(decay=0.0)

    def update_target_generator(self, decay=None):
        decay = float(getattr(self.config, "target_ema_decay", 0.95) if decay is None else decay)
        if not hasattr(self, "_target_ema_param_pairs"):
            self._refresh_target_ema_params()
        with torch.no_grad():
            for target_param, student_param in self._target_ema_param_pairs:
                source = student_param.detach()
                if source.dtype != target_param.dtype:
                    source = source.to(dtype=target_param.dtype)
                if decay == 0.0:
                    target_param.copy_(source)
                else:
                    target_param.mul_(decay).add_(source, alpha=1.0 - decay)

    def _refresh_target_ema_params(self):
        student = self.unwrap_model(self.G) if hasattr(self, "accelerator") else self.G
        student_trainable_params = {
            name: param for name, param in student.named_parameters() if param.requires_grad
        }
        target_params = dict(self.target_G.named_parameters())
        missing = [name for name in student_trainable_params if name not in target_params]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Target generator is missing {len(missing)} trainable student params, e.g. {preview}")

        self._target_ema_param_pairs = [
            (target_params[name], student_param)
            for name, student_param in student_trainable_params.items()
        ]
        self._target_ema_numel = sum(param.numel() for param in student_trainable_params.values())
        logger.info(
            "Target EMA tracks %d trainable parameters (%.2f M), frozen base UNet parameters are skipped.",
            len(self._target_ema_param_pairs),
            self._target_ema_numel / 1_000_000,
        )

    def _unet_pred(self, model, latents, timesteps, attention_scores=None):
        model_input = self.scheduler.scale_model_input(latents, timesteps)
        model_input = torch.cat(
            [model_input, self.batch_inputs.mask_latent, self.batch_inputs.masked_image_latents],
            dim=1,
        )
        return model(
            model_input,
            timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
            added_cond_kwargs={
                "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                "time_ids": self.batch_inputs.add_time_ids,
            },
        ).sample

    def _pred_x0(self, model_output, sample, timesteps):
        prediction_type = self.scheduler.config.prediction_type
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t

        if prediction_type == "epsilon":
            pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif prediction_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        return pred_x0

    def _pred_noise(self, model_output, sample, timesteps):
        prediction_type = self.scheduler.config.prediction_type
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t

        if prediction_type == "epsilon":
            pred_noise = model_output
        elif prediction_type == "v_prediction":
            pred_noise = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        return pred_noise

    def _add_noise_at(self, clean_latents, noise, timesteps):
        return self.scheduler.add_noise(clean_latents, noise, timesteps)

    @torch.no_grad()
    def _teacher_prev_sample(self, noisy_latents, timesteps, prev_timesteps):
        teacher_pred = self._unet_pred(self.teacher_G, noisy_latents, timesteps)
        teacher_x0 = self._pred_x0(teacher_pred.float(), noisy_latents.float(), timesteps)
        teacher_noise = self._pred_noise(teacher_pred.float(), noisy_latents.float(), timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        alpha_prev = alphas_cumprod[prev_timesteps].view(-1, 1, 1, 1)
        beta_prev = 1.0 - alpha_prev
        return alpha_prev.sqrt() * teacher_x0.to(noisy_latents.dtype) + beta_prev.sqrt() * teacher_noise.to(noisy_latents.dtype)

    def _masked_l1(self, pred, target, mask):
        loss = (pred.float() - target.float()).abs()
        mask = mask.to(device=loss.device, dtype=loss.dtype)
        if mask.shape[1] == 1 and loss.shape[1] != 1:
            mask = mask.repeat(1, loss.shape[1], 1, 1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    def _decode_latents(self, latents):
        z = latents.to(self.weight_dtype) / self.vae.config.scaling_factor
        if torch.is_grad_enabled() and z.requires_grad and getattr(self.config, "vae_decode_checkpointing", True):
            return checkpoint(lambda x: self.vae.decode(x).sample, z, use_reentrant=False).float()
        return self.vae.decode(z).sample.float()

    def _compute_lcm_loss(self):
        z_gt = self.batch_inputs.z_gt.to(dtype=self.weight_dtype)
        bs = z_gt.shape[0]
        noise = torch.randn_like(z_gt)

        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        min_timestep = int(getattr(self.config, "lcm_min_timestep", 20))
        max_timestep = int(getattr(self.config, "lcm_max_timestep", num_train_timesteps - 1))
        skip = int(getattr(self.config, "lcm_skip_steps", 20))
        max_timestep = min(max_timestep, num_train_timesteps - 1)
        min_timestep = max(min_timestep, skip)

        timestep = torch.randint(
            min_timestep,
            max_timestep + 1,
            (1,),
            device=self.device,
            dtype=torch.long,
        )
        timestep = ((timestep // skip) * skip).clamp(min=skip, max=num_train_timesteps - 1)
        timesteps = timestep.repeat(bs)
        prev_timesteps = (timesteps - skip).clamp(min=0)
        self.last_lcm_timestep = timestep.detach().float().mean()

        z_t = self._add_noise_at(z_gt, noise, timesteps)
        self.last_noisy_gt_latents = z_t.detach()
        student_model_pred = self._unet_pred(self.G, z_t, timesteps)
        student_x0 = self._pred_x0(student_model_pred.float(), z_t.float(), timesteps)

        with torch.no_grad():
            teacher_z_prev = self._teacher_prev_sample(z_t, timesteps, prev_timesteps)
            target_model_pred = self._unet_pred(self.target_G, teacher_z_prev, prev_timesteps)
            target_x0 = self._pred_x0(target_model_pred.float(), teacher_z_prev.float(), prev_timesteps)

        lcm_loss = F.mse_loss(student_x0.float(), target_x0.float()) * getattr(self.config, "lambda_lcm", 1.0)
        gt_latent_loss = F.mse_loss(student_x0.float(), z_gt.float()) * getattr(self.config, "lambda_gt_latent", 0.0)
        return lcm_loss, gt_latent_loss, student_x0, target_x0

    def forward_generator(self):
        z = self.batch_inputs.z_lq
        steps = int(getattr(self.config, "student_steps", 1))
        timesteps = self._set_inference_timesteps(steps)
        for t in timesteps:
            t_batch = torch.full((z.shape[0],), int(t), dtype=torch.long, device=self.device)
            eps = self._unet_pred(self.G, z, t_batch)
            z = self.scheduler.step(eps, t, z).prev_sample
        x = self._decode_latents(z)
        return x, z

    def optimize_generator(self):
        with self.accelerator.accumulate(self.G):
            if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None:
                self.unwrap_model(self.D).eval().requires_grad_(False)

            loss_lcm, loss_gt_latent, student_x0, target_x0 = self._compute_lcm_loss()
            self._profile_mark("lcm_forward")
            x_pred = self._decode_latents(student_x0)
            self._profile_mark("vae_decode_student")
            self.G_pred = x_pred.detach()

            effect_mask = self.batch_inputs.object_effect_mask
            bg_mask = 1.0 - effect_mask
            loss_mask_l1 = self._masked_l1(x_pred, self.batch_inputs.gt, effect_mask) * getattr(self.config, "lambda_mask_l1", 0.0)
            loss_mask_background = self._masked_l1(x_pred, self.batch_inputs.gt, bg_mask) * getattr(self.config, "lambda_mask_background", 0.0)
            lambda_teacher_image_l1 = float(getattr(self.config, "lambda_teacher_image_l1", 0.0))
            if lambda_teacher_image_l1 > 0.0:
                with torch.no_grad():
                    self.G_target = self._decode_latents(target_x0).detach()
                self._profile_mark("vae_decode_target")
                loss_teacher_l1 = F.l1_loss(x_pred.float(), self.G_target.float(), reduction="mean") * lambda_teacher_image_l1
            else:
                self.G_target = None
                loss_teacher_l1 = torch.zeros((), device=self.device, dtype=loss_lcm.dtype)
            loss_lpips = self.net_lpips(x_pred, self.batch_inputs.gt).mean() * getattr(self.config, "lambda_lpips_gt", 0.0)
            self._profile_mark("lpips")

            zero = torch.tensor(0.0, device=self.device, dtype=loss_lcm.dtype)
            lambda_diffusion = float(getattr(self.config, "lambda_diffusion", 0.0))
            lambda_localization = float(getattr(self.config, "object_localization_weight", 0.0))
            if lambda_diffusion > 0.0 or (
                getattr(self.config, "object_localization", False) and lambda_localization > 0.0
            ):
                diffusion_loss, localization_loss = self._compute_train_objectclear_style_loss()
            else:
                diffusion_loss, localization_loss = zero, zero
            self._profile_mark("diffusion_localization")
            loss_diffusion = diffusion_loss * lambda_diffusion
            loss_localization = localization_loss * lambda_localization

            loss = (
                loss_lcm
                + loss_gt_latent
                + loss_mask_l1
                + loss_mask_background
                + loss_teacher_l1
                + loss_lpips
                + loss_diffusion
                + loss_localization
            )

            self.accelerator.backward(loss)
            self._profile_mark("backward")
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
                self._profile_mark("clip_grad")
            self.G_opt.step()
            self.G_opt.zero_grad()
            self._profile_mark("optimizer")

            if self.accelerator.sync_gradients:
                self.update_target_generator()
                self._profile_mark("target_ema")

        return {
            "G_total": loss,
            "G_lcm": loss_lcm,
            "G_gt_latent": loss_gt_latent,
            "G_mask_l1": loss_mask_l1,
            "G_mask_background": loss_mask_background,
            "G_teacher_l1": loss_teacher_l1,
            "G_lpips": loss_lpips,
            "G_diffusion": loss_diffusion,
            "G_localization": loss_localization,
            "lcm_timestep": getattr(self, "last_lcm_timestep", torch.tensor(0.0, device=self.device)),
        }

    def optimize_discriminator(self):
        return {}

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        self._step_profile_local_step = 0
        validation_steps = int(getattr(self.config, "validation_steps", 0))

        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            self._step_profile_wait_start = time.perf_counter()
            for batch in self.dataloader:
                self._profile_begin()
                self.prepare_batch_inputs(batch)
                self._profile_mark("prepare_batch")
                loss_dict = self.optimize_generator()

                for key, value in loss_dict.items():
                    avg_loss = self.accelerator.gather(value.detach().float().reshape(1)).mean()
                    train_loss[key] = train_loss.get(key, 0.0) + avg_loss.item() / self.config.gradient_accumulation_steps
                self._profile_mark("loss_gather")

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    self.ema_handler.update()
                    self._profile_mark("ema_handler")
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"LCM Generator Step, VRAM peak: {peak:.2f} GB")
                    self.global_step += 1
                    self.pbar.update(1)

                    log_dict = {f"loss/{key}": value for key, value in train_loss.items()}
                    log_dict.update(self._profile_collect_log())
                    train_loss = {}
                    self.accelerator.log(log_dict, step=self.global_step)
                    self.log_metrics_csv(log_dict, self.global_step)

                    if self.config.use_vae and (self.global_step % self.config.log_image_steps == 0 or self.global_step == 1):
                        self.log_images()
                    if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1:
                        self.save_checkpoint()
                    if validation_steps > 0 and (self.global_step % validation_steps == 0 or self.global_step == 1):
                        self.validate()

                if self.global_step >= self.config.max_train_steps:
                    break
                self._step_profile_local_step += 1
                self._step_profile_wait_start = time.perf_counter()

        self.accelerator.end_training()

    @torch.no_grad()
    def validate(self):
        if self.val_dataloader is None:
            return
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        saved_G = self.G
        self.G = self.unwrap_model(self.G)
        was_training = self.G.training
        self.G.eval()

        max_batches = int(getattr(self.config, "validation_max_batches", -1))
        save_batches = int(getattr(self.config, "validation_save_batches", 4))
        logger.info(f"Running OBER validation at step {self.global_step}")

        total_mse = 0.0
        total_psnr = 0.0
        total_masked_psnr = 0.0
        total_lpips = 0.0
        num_batches = 0
        num_samples = 0

        val_image_root = getattr(self.config, "validation_image_dir", None)
        if val_image_root is None:
            val_image_root = os.path.join(self.config.output_dir, "validation", "images")
        val_save_dir = os.path.join(val_image_root, f"{self.global_step:07}")
        os.makedirs(val_save_dir, exist_ok=True)

        for batch_idx, batch in enumerate(self.val_dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            self.prepare_batch_inputs(batch)
            # Validation uses the actual deployment path: one-step inference from random latents.
            student_x, _ = self.forward_generator()
            pred = ((student_x + 1.0) / 2.0).clamp(0, 1)
            gt = ((self.batch_inputs.gt + 1.0) / 2.0).clamp(0, 1)
            mask = self.batch_inputs.object_effect_mask.float()
            mask_3 = mask.repeat(1, 3, 1, 1)
            bs = pred.shape[0]

            sample_mse = (pred - gt).pow(2).flatten(1).mean(dim=1)
            sample_psnr = -10.0 * torch.log10(sample_mse.clamp_min(1e-12))
            sample_masked_mse = ((pred - gt).pow(2) * mask_3).flatten(1).sum(dim=1) / (
                mask_3.flatten(1).sum(dim=1) + 1e-8
            )
            sample_masked_psnr = -10.0 * torch.log10(sample_masked_mse.clamp_min(1e-12))
            lpips_val = self.net_lpips(student_x, self.batch_inputs.gt).view(bs)

            total_mse += float(sample_mse.detach().sum())
            total_psnr += float(sample_psnr.detach().sum())
            total_masked_psnr += float(sample_masked_psnr.detach().sum())
            total_lpips += float(lpips_val.detach().sum())
            num_batches += 1
            num_samples += bs

            if batch_idx < save_batches:
                input_img = ((self.batch_inputs.input_img + 1.0) / 2.0).clamp(0, 1)
                vis = torch.cat([input_img, gt, pred, mask_3], dim=3)
                image_arrs = (vis * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for sample_idx, image_arr in enumerate(image_arrs):
                    Image.fromarray(image_arr).save(os.path.join(val_save_dir, f"batch{batch_idx}_sample{sample_idx}_input_gt_infer1step_mask.png"))

        if num_samples == 0:
            if was_training:
                self.G.train()
            self.G = saved_G
            self.accelerator.wait_for_everyone()
            return

        log_metrics = {
            "val/mse": total_mse / num_samples,
            "val/psnr": total_psnr / num_samples,
            "val/masked_psnr": total_masked_psnr / num_samples,
            "val/lpips": total_lpips / num_samples,
        }
        logger.info(
            f"OBER validation step {self.global_step}: "
            f"PSNR={log_metrics['val/psnr']:.4f}, "
            f"masked_PSNR={log_metrics['val/masked_psnr']:.4f}, "
            f"LPIPS={log_metrics['val/lpips']:.4f}"
        )
        self.accelerator.log(log_metrics, step=self.global_step)
        self._append_validation_csv(log_metrics, num_batches, num_samples, val_save_dir)

        if was_training:
            self.G.train()
        self.G = saved_G
        self.accelerator.wait_for_everyone()

    def _append_validation_csv(self, log_metrics, num_batches, num_samples, val_save_dir):
        if not self.accelerator.is_main_process:
            return
        csv_path = getattr(self.config, "validation_metrics_csv", None)
        if csv_path is None:
            csv_path = os.path.join(self.config.output_dir, "validation", "metrics.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = ["step", "num_batches", "num_samples", "mse", "psnr", "masked_psnr", "lpips", "image_dir"]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "step": self.global_step,
                    "num_batches": num_batches,
                    "num_samples": num_samples,
                    "mse": log_metrics["val/mse"],
                    "psnr": log_metrics["val/psnr"],
                    "masked_psnr": log_metrics["val/masked_psnr"],
                    "lpips": log_metrics["val/lpips"],
                    "image_dir": val_save_dir,
                }
            )

    def log_images(self):
        if not hasattr(self, "G_pred"):
            return
        with torch.no_grad():
            infer_x, _ = self.forward_generator()
            self.G_infer_1step = infer_x
        n = min(4, self.G_pred.shape[0])
        image_logs = {
            "input": (self.batch_inputs.input_img[:n] + 1) / 2,
            "gt": (self.batch_inputs.gt[:n] + 1) / 2,
            "student_lcm_x0": (self.G_pred[:n] + 1) / 2,
        }
        if hasattr(self, "last_noisy_gt_latents"):
            noisy_gt = self._decode_latents(self.last_noisy_gt_latents[:n])
            image_logs["noisy_gt_lcm"] = (noisy_gt + 1) / 2
        if hasattr(self, "G_infer_1step"):
            image_logs["student_infer_1step"] = (self.G_infer_1step[:n] + 1) / 2
        if getattr(self, "G_target", None) is not None:
            image_logs["target_lcm"] = (self.G_target[:n] + 1) / 2

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(f"image/{tag}", make_grid(images.float(), nrow=4), self.global_step)

        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))
        if hasattr(self, "G_infer_1step"):
            del self.G_infer_1step
