from typing import Dict, List, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from HYPIR.model.clip_encoder import CLIPImageEncoder
from HYPIR.model.postfuse_module import PostfuseModule
from HYPIR.trainer.base_sdxl import BaseTrainer, BatchInput
from HYPIR.utils.common import print_vram_state

logger = get_logger(__name__, log_level="INFO")


class ObjectClearTrainer(BaseTrainer):

    def init_models(self):
        print(f"Use VAE: {self.config.use_vae}, Use D: {self.config.use_D}, Use EMA: {self.config.use_ema}")
        self.init_scheduler()
        self.init_text_models()
        self.init_objectclear_modules()
        if self.config.use_vae:
            self.init_vae()
        self.init_generator()
        if self.config.use_D:
            self.init_discriminator()
        self.init_lpips()

    def _sync_module_from_rank0(self, module, name):
        if not (
            bool(getattr(self.config, "sync_conditioning_modules_from_rank0", True))
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return
        for tensor in list(module.parameters()) + list(module.buffers()):
            torch.distributed.broadcast(tensor.data, src=0)
        logger.info(f"Synchronized {name} parameters from rank0.")

    def init_scheduler(self):
        scheduler_type = str(getattr(self.config, "scheduler_type", "ddpm")).lower()
        timestep_spacing = str(getattr(self.config, "scheduler_timestep_spacing", "trailing")).lower()
        scheduler_cls = {
            "ddpm": DDPMScheduler,
            "ddim": DDIMScheduler,
        }.get(scheduler_type)
        if scheduler_cls is None:
            raise ValueError(f"Unsupported scheduler_type={scheduler_type!r}. Expected 'ddpm' or 'ddim'.")
        if timestep_spacing not in {"leading", "trailing", "linspace", "fixed"}:
            raise ValueError(
                f"Unsupported scheduler_timestep_spacing={timestep_spacing!r}. "
                "Expected 'trailing', 'leading', 'linspace', or 'fixed'."
            )
        scheduler_timestep_spacing = "trailing" if timestep_spacing == "fixed" else timestep_spacing
        self.scheduler = scheduler_cls.from_pretrained(
            self.config.base_model_path,
            subfolder="scheduler",
            timestep_spacing=scheduler_timestep_spacing,
        )
        if hasattr(self.scheduler, "register_to_config"):
            self.scheduler.register_to_config(timestep_spacing=scheduler_timestep_spacing)
        logger.info(
            f"Using {scheduler_cls.__name__} with timestep_spacing={timestep_spacing} "
            f"(scheduler={scheduler_timestep_spacing})."
        )

    def _get_configured_noise_timestep(self) -> int:
        configured = getattr(self.config, "noise_timestep", None)
        if configured is None:
            raise ValueError(
                "scheduler_timestep_spacing='fixed' requires noise_timestep to be set for 1-step inference."
            )
        timestep = int(configured)
        max_timestep = int(getattr(self.scheduler.config, "num_train_timesteps", 1000)) - 1
        if timestep < 0 or timestep > max_timestep:
            raise ValueError(
                f"Invalid noise_timestep={timestep}. Expected an integer in [0, {max_timestep}]."
            )
        return timestep

    def _set_inference_timesteps(self, num_inference_steps: int):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        if str(getattr(self.config, "scheduler_timestep_spacing", "trailing")).lower() == "fixed":
            if int(num_inference_steps) != 1:
                raise ValueError("scheduler_timestep_spacing='fixed' is only supported for 1-step inference.")
            timesteps = torch.tensor(
                [self._get_configured_noise_timestep()],
                dtype=torch.long,
                device=self.device,
            )
            self.scheduler.timesteps = timesteps
        return timesteps

    def init_text_models(self):
        self._prompt_embed_cache = {}
        self._text_encoders_offloaded = False
        text_encoder_dtype_name = str(getattr(self.config, "text_encoder_dtype", self.weight_dtype)).lower()
        self.text_encoder_dtype = torch.float32
        if text_encoder_dtype_name in ("fp16", "float16"):
            self.text_encoder_dtype = torch.float16
        elif text_encoder_dtype_name in ("bf16", "bfloat16"):
            self.text_encoder_dtype = torch.bfloat16
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer_2")

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.text_encoder_dtype,
        ).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.text_encoder_dtype,
        ).to(self.device)

        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder_2.eval().requires_grad_(False)
        self._sync_module_from_rank0(self.text_encoder, "text_encoder")
        self._sync_module_from_rank0(self.text_encoder_2, "text_encoder_2")

    def _resolve_postfuse_weights_path(self) -> Optional[str]:
        local_path = os.path.join(self.config.base_model_path, "postfuse_module", "model.safetensors")
        if os.path.exists(local_path):
            return local_path

        if self.config.base_model_path == "jixin0101/ObjectClear":
            try:
                from huggingface_hub import hf_hub_download

                return hf_hub_download(
                    repo_id="jixin0101/ObjectClear",
                    filename="model.safetensors",
                    subfolder="postfuse_module",
                    cache_dir=getattr(self.config, "cache_dir", None),
                )
            except Exception as exc:
                logger.warning(f"Failed to download postfuse weights from hub: {exc}")
                return None

        return None

    def init_objectclear_modules(self):
        object_encoder_dtype_name = str(getattr(self.config, "object_encoder_dtype", self.weight_dtype)).lower()
        self.object_encoder_dtype = torch.float32
        if object_encoder_dtype_name in ("fp16", "float16"):
            self.object_encoder_dtype = torch.float16
        elif object_encoder_dtype_name in ("bf16", "bfloat16"):
            self.object_encoder_dtype = torch.bfloat16

        postfuse_dtype_name = str(getattr(self.config, "postfuse_dtype", self.object_encoder_dtype)).lower()
        self.postfuse_dtype = torch.float32
        if postfuse_dtype_name in ("fp16", "float16"):
            self.postfuse_dtype = torch.float16
        elif postfuse_dtype_name in ("bf16", "bfloat16"):
            self.postfuse_dtype = torch.bfloat16

        self.image_prompt_encoder = CLIPImageEncoder.from_pretrained(
            self.config.base_model_path,
            cache_dir=getattr(self.config, "cache_dir", None),
        ).to(self.device, dtype=self.object_encoder_dtype)
        self.image_prompt_encoder.eval().requires_grad_(False)

        self.postfuse_module = PostfuseModule(embed_dim=2048, embed_dim_img=768).to(
            self.device, dtype=self.postfuse_dtype
        )

        postfuse_path = self._resolve_postfuse_weights_path()
        if postfuse_path is not None and os.path.exists(postfuse_path):
            from safetensors.torch import load_file

            state_dict = load_file(postfuse_path)
            try:
                self.postfuse_module.load_state_dict(state_dict)
                logger.info(f"Loaded postfuse weights from {postfuse_path}")
            except RuntimeError as exc:
                logger.warning(f"Postfuse strict load failed, trying non-strict load: {exc}")
                self.postfuse_module.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("No postfuse safetensors found. Using randomly initialized postfuse module.")

        self.postfuse_module.eval().requires_grad_(False)
        self._sync_module_from_rank0(self.image_prompt_encoder, "image_prompt_encoder")
        self._sync_module_from_rank0(self.postfuse_module, "postfuse_module")

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype
        ).to(self.device)
        self.vae.eval().requires_grad_(False)
        logger.info("SDXL VAE loaded.")
        print_vram_state("After VAE to(device)", logger=logger)

    def init_generator(self):
        self.G = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.G.eval().requires_grad_(False)

        if self.config.gradient_checkpointing:
            self.G.enable_gradient_checkpointing()

        target_modules = self.config.lora_modules
        logger.info(f"Add lora parameters to {target_modules}")
        G_lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.G.add_adapter(G_lora_cfg)
        lora_params = [p for p in self.G.parameters() if p.requires_grad]
        assert lora_params, "Failed to find lora parameters"
        for p in lora_params:
            p.data = p.to(torch.float32)

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)
                assert isinstance(model, UNet2DConditionModel)
                state_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.detach().clone().data
                torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))

        def load_model_hook(models, input_dir):
            model = models.pop(0)
            assert isinstance(model, UNet2DConditionModel)
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"), map_location="cpu")
            _, unexpected = model.load_state_dict(state_dict, strict=False)
            del state_dict
            logger.info(f"Loading lora parameters, unexpected keys: {unexpected}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def _pick_tensor_key(self, batch: Dict[str, torch.Tensor], keys: List[str], required: bool = True):
        for key in keys:
            if key in batch:
                return batch[key]
        if required:
            raise KeyError(f"Missing required key. Tried keys: {keys}")
        return None

    def _to_image_range(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 2.0 - 1.0
        return image

    def _to_binary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = (mask > 0.5).float()
        return mask

    def resolve_prompt(self, batch: Dict[str, torch.Tensor], batch_size: int) -> List[str]:
        config_prompt = getattr(self.config, "prompt", None)
        if config_prompt is not None:
            if isinstance(config_prompt, str):
                return [config_prompt] * batch_size
            if isinstance(config_prompt, (list, tuple)):
                if len(config_prompt) == 1:
                    return [str(config_prompt[0])] * batch_size
                if len(config_prompt) != batch_size:
                    raise ValueError(
                        f"config.prompt has length {len(config_prompt)}, but batch size is {batch_size}."
                    )
                return [str(item) for item in config_prompt]

        prompt = batch.get("txt", None)
        if prompt is None:
            prompt = batch.get("prompt", None)
        if prompt is None:
            return [""] * batch_size
        if isinstance(prompt, str):
            return [prompt] * batch_size
        return list(prompt)

    def encode_prompt(
        self,
        prompt: List[str],
        use_config_prompt: bool = True,
        allow_offload: bool = True,
    ) -> Dict[str, torch.Tensor]:
        config_prompt = getattr(self.config, "prompt", None) if use_config_prompt else None
        static_prompt = None
        if isinstance(config_prompt, str):
            static_prompt = config_prompt
        elif isinstance(config_prompt, (list, tuple)) and len(config_prompt) == 1:
            static_prompt = str(config_prompt[0])

        if static_prompt is not None:
            cache_key = ("static_prompt", static_prompt)
            encode_prompt = [static_prompt]
        elif config_prompt is not None:
            cache_key = tuple(prompt)
            encode_prompt = prompt
        elif not use_config_prompt:
            if len(prompt) > 0 and all(item == prompt[0] for item in prompt):
                cache_key = ("raw_prompt", prompt[0])
                encode_prompt = [prompt[0]]
                static_prompt = prompt[0]
            else:
                cache_key = ("raw_prompt", tuple(prompt))
                encode_prompt = prompt
        else:
            cache_key = None
            encode_prompt = prompt

        if cache_key is not None and cache_key in self._prompt_embed_cache:
            cached = self._prompt_embed_cache[cache_key]
            embeds = {key: value.to(self.device) for key, value in cached.items()}
            if static_prompt is not None and len(prompt) != embeds["prompt_embeds"].shape[0]:
                embeds = {
                    "prompt_embeds": embeds["prompt_embeds"].repeat(len(prompt), 1, 1),
                    "pooled_prompt_embeds": embeds["pooled_prompt_embeds"].repeat(len(prompt), 1),
                }
            return embeds

        text_input_ids = self.tokenizer(
            encode_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids_2 = self.tokenizer_2(
            encode_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        with torch.no_grad():
            for text_input_ids_i, text_encoder in zip(
                [text_input_ids, text_input_ids_2], [self.text_encoder, self.text_encoder_2]
            ):
                text_outputs = text_encoder(
                    text_input_ids_i.to(self.accelerator.device),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = text_outputs[0]
                text_hidden = text_outputs.hidden_states[-2]
                bs_embed, seq_len, _ = text_hidden.shape
                text_hidden = text_hidden.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(text_hidden)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(len(encode_prompt), -1)
        embeds = {
            "prompt_embeds": prompt_embeds.to(dtype=self.weight_dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=self.weight_dtype),
        }
        for key, value in embeds.items():
            if not torch.isfinite(value.float()).all():
                nonfinite_ratio = 1.0 - torch.isfinite(value.float()).float().mean().item()
                raise FloatingPointError(
                    f"Non-finite {key} from text encoder for prompt={encode_prompt!r}; "
                    f"ratio={nonfinite_ratio:.6f}, text_encoder_dtype={self.text_encoder_dtype}."
                )
        if cache_key is not None:
            cache_on_gpu = bool(getattr(self.config, "cache_prompt_embeds_on_gpu", False))
            self._prompt_embed_cache[cache_key] = {
                key: value.detach() if cache_on_gpu else value.detach().cpu()
                for key, value in embeds.items()
            }
        if static_prompt is not None and len(prompt) != embeds["prompt_embeds"].shape[0]:
            embeds = {
                "prompt_embeds": embeds["prompt_embeds"].repeat(len(prompt), 1, 1),
                "pooled_prompt_embeds": embeds["pooled_prompt_embeds"].repeat(len(prompt), 1),
            }
        return embeds

    def _get_add_time_ids(self, batch_size: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype, device=self.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def _encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.weight_dtype)
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def prepare_batch_inputs(self, batch):
        input_img = self._pick_tensor_key(batch, ["input", "INPUT", "LQ", "lq", "image", "Image"])
        gt = self._pick_tensor_key(batch, ["GT", "gt", "target", "inpaint_gt", "output"])
        object_mask = self._pick_tensor_key(batch, ["object_mask", "mask", "mask_image", "MASK"])
        effect_mask = self._pick_tensor_key(
            batch,
            ["object_effect_mask", "effect_mask", "loss_mask", "shadow_mask"],
            required=False,
        )
        if effect_mask is None:
            effect_mask = object_mask

        input_img = self._to_image_range(input_img).to(self.device)
        gt = self._to_image_range(gt).to(self.device)
        object_mask = self._to_binary_mask(object_mask).to(self.device)
        effect_mask = self._to_binary_mask(effect_mask).to(self.device)

        bs, _, h, w = input_img.shape
        prompt = self.resolve_prompt(batch, bs)
        c_txt = self.encode_prompt(prompt)
        uncond_txt = self.encode_prompt([""] * bs, use_config_prompt=False)
        add_time_ids = self._get_add_time_ids(bs, h, w, dtype=c_txt["pooled_prompt_embeds"].dtype)

        obj_only = input_img * (object_mask > 0.5)
        with torch.no_grad():
            object_embeds = self.image_prompt_encoder(obj_only.to(dtype=self.object_encoder_dtype))
            if not torch.isfinite(object_embeds.float()).all():
                nonfinite_ratio = 1.0 - torch.isfinite(object_embeds.float()).float().mean().item()
                raise FloatingPointError(
                    f"Non-finite object_embeds from image_prompt_encoder; ratio={nonfinite_ratio:.6f}, "
                    f"object_encoder_dtype={self.object_encoder_dtype}."
                )
            fused_prompt_embeds = self.postfuse_module(
                c_txt["prompt_embeds"].to(dtype=self.postfuse_dtype),
                object_embeds.to(dtype=self.postfuse_dtype),
                getattr(self.config, "fuse_index", 5),
            ).to(dtype=self.weight_dtype)
            if not torch.isfinite(fused_prompt_embeds.float()).all():
                nonfinite_ratio = 1.0 - torch.isfinite(fused_prompt_embeds.float()).float().mean().item()
                raise FloatingPointError(
                    f"Non-finite fused_prompt_embeds after postfuse; ratio={nonfinite_ratio:.6f}, "
                    f"object_encoder_dtype={self.object_encoder_dtype}, postfuse_dtype={self.postfuse_dtype}."
                )

        z_gt = self._encode_to_latents(gt)
        # For one-step distillation, start from pure Gaussian noise latents.
        z_in = torch.randn_like(z_gt)

        latent_h, latent_w = z_in.shape[-2:]
        mask_latent = F.interpolate(object_mask, size=(latent_h, latent_w), mode="nearest").to(dtype=self.weight_dtype)
        effect_mask_latent = F.interpolate(effect_mask, size=(latent_h, latent_w), mode="nearest").to(dtype=self.weight_dtype)
        # Keep masked image behavior aligned with ObjectClear pipeline (encode init image directly).
        masked_image_latents = self._encode_to_latents(input_img)

        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)

        self.batch_inputs = BatchInput(
            input_img=input_img,
            gt=gt,
            object_mask=object_mask,
            object_effect_mask=effect_mask,
            z_lq=z_in,
            z_gt=z_gt,
            mask_latent=mask_latent,
            effect_mask_latent=effect_mask_latent,
            masked_image_latents=masked_image_latents,
            c_txt={
                "prompt_embeds": fused_prompt_embeds,
                "pooled_prompt_embeds": c_txt["pooled_prompt_embeds"],
            },
            uncond_txt=uncond_txt,
            add_time_ids=add_time_ids,
            timesteps=timesteps,
            prompt=prompt,
        )

    def forward_generator(self):
        z = self.batch_inputs.z_lq
        timesteps = self._set_inference_timesteps(4)

        for t in timesteps:
            t_batch = torch.full((z.shape[0],), int(t), dtype=torch.long, device=self.device)
            z_model = self.scheduler.scale_model_input(z, t)
            if self.G.config.in_channels == 9:
                z_model = torch.cat(
                    [z_model, self.batch_inputs.mask_latent, self.batch_inputs.masked_image_latents], dim=1
                )

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

        x_pred = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()

        return x_pred, z
