from typing import Dict, List, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
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

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.config.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer_2")

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype,
        ).to(self.device)

        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder_2.eval().requires_grad_(False)

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
        self.image_prompt_encoder = CLIPImageEncoder.from_pretrained(
            self.config.base_model_path,
            cache_dir=getattr(self.config, "cache_dir", None),
        ).to(self.device, dtype=self.weight_dtype)
        self.image_prompt_encoder.eval().requires_grad_(False)

        self.postfuse_module = PostfuseModule(embed_dim=2048, embed_dim_img=768).to(
            self.device, dtype=self.weight_dtype
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
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"))
            _, unexpected = model.load_state_dict(state_dict, strict=False)
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

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds_list = []
        pooled_prompt_embeds = None
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
        pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompt), -1)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

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
            ["object_effect_mask", "effect_mask", "loss_mask"],
            required=False,
        )
        if effect_mask is None:
            effect_mask = object_mask

        input_img = self._to_image_range(input_img).to(self.device)
        gt = self._to_image_range(gt).to(self.device)
        object_mask = self._to_binary_mask(object_mask).to(self.device)
        effect_mask = self._to_binary_mask(effect_mask).to(self.device)

        prompt = batch.get("txt", None)
        if prompt is None:
            prompt = [""] * input_img.shape[0]

        c_txt = self.encode_prompt(prompt)
        bs, _, h, w = input_img.shape
        add_time_ids = self._get_add_time_ids(bs, h, w, dtype=c_txt["pooled_prompt_embeds"].dtype)

        obj_only = input_img * (object_mask > 0.5)
        with torch.no_grad():
            object_embeds = self.image_prompt_encoder(obj_only.to(dtype=self.weight_dtype))
            fused_prompt_embeds = self.postfuse_module(
                c_txt["prompt_embeds"], object_embeds, getattr(self.config, "fuse_index", 5)
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
            add_time_ids=add_time_ids,
            timesteps=timesteps,
            prompt=prompt,
        )

    def forward_generator(self):
        z = self.batch_inputs.z_lq
        self.scheduler.set_timesteps(4, device=self.device)

        for t in self.scheduler.timesteps:
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