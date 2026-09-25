from typing import List, Dict
import os
import torch
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
try:
    from peft import mark_only_lora_as_trainable
except ImportError:  # Older PEFT versions do not expose this helper
    def mark_only_lora_as_trainable(model):  # noqa: D401 - simple compatibility shim
        """Fallback that freezes base weights and keeps LoRA weights trainable."""
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
from transformers import CLIPTextModel, CLIPTokenizer

from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.utils.common import (
    instantiate_from_config,
    log_txt_as_img,
    print_vram_state,
    SuppressLogging,
    module_param_memory,
    human_bytes,
)
from HYPIR.dataset.latent_dataset import LatentWithGTDataset
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
from hyimage.models.hunyuan.modules.hunyuanimage_dit import load_hunyuan_dit_state_dict, HYImageDiffusionTransformer
from hyimage.models.text_encoder import PROMPT_TEMPLATE
from hyimage.models.text_encoder.byT5 import load_glyph_byT5_v2
from hyimage.common.format_prompt import MultilingualPromptFormat
from hyimage.common.config import instantiate
from hyimage.models.model_zoo import (
    HUNYUANIMAGE_V2_1_DIT,
    HUNYUANIMAGE_V2_1_DIT_CFG_DISTILL,
    HUNYUANIMAGE_V2_1_VAE_32x,
    HUNYUANIMAGE_V2_1_TEXT_ENCODER,
)

logger = get_logger(__name__, log_level="INFO")

class HunyuanImage21Trainer(BaseTrainer):
    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()    

    def init_dataset(self):
        data_cfg = self.config.data_config
        # 1) 读取 latent 数据集（来自保存好的 shards）
        latent_ds = instantiate_from_config(data_cfg.train.latent_data)

        # 2) 基于 filename 直接读取 GT（无需依赖 metadata 顺序）
        ds_params = data_cfg.train.dataset
        file_meta = ds_params.file_meta
        out_size = int(ds_params.get("out_size", 2048))
        crop_type = str(ds_params.get("crop_type", "center"))
        apply_usm = bool(ds_params.get("apply_usm", False))

        # 合并：同索引从 latent 里拿 filename，按 filename 读取 GT，返回 {'gt_latent','lq_latent','gt','filename'}
        self.dataset = LatentWithGTDataset(
            latent_ds=latent_ds,
            image_path_prefix=file_meta.get("image_path_prefix", ""),
            out_size=out_size,
            crop_type=crop_type,
            apply_usm=apply_usm,
        )

        # 随机批读取 + 高效 IO
        num_workers = int(data_cfg.train.dataloader_num_workers)
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0
        prefetch_factor = 4 if num_workers > 0 else None
        dl_kwargs = {
            "dataset": self.dataset,
            "shuffle": True,
            "batch_size": int(data_cfg.train.batch_size),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "drop_last": False,
        }
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor
        self.dataloader = torch.utils.data.DataLoader(**dl_kwargs)

        # load from saved debug inputs
        pos_text_emb = torch.load(f"debug_inputs/pos_text_emb.pt").to(self.device)
        pos_text_mask = torch.load(f"debug_inputs/pos_text_mask.pt").to(self.device)
        pos_byt5_emb = torch.load(f"debug_inputs/pos_byt5_emb.pt").to(self.device)
        pos_byt5_mask = torch.load(f"debug_inputs/pos_byt5_mask.pt").to(self.device)

        self.c_txt = {"text_emb": pos_text_emb, "text_mask": pos_text_mask, 
                 "byt5_emb": pos_byt5_emb, "byt5_mask": pos_byt5_mask}

    def prepare_batch_inputs(self, batch):
        # gt: CHW float32 [0,1]
        # 兼容两种 batch 形式：
        # 1) dict of tensors: {'gt_latent': (B,C,H,W), 'lq_latent': (B,C,H,W)}
        # 2) list of dicts: [{'gt_latent':(C,H,W), 'lq_latent':(C,H,W)}, ...]
        if isinstance(batch, dict):
            gt_latent = batch['gt_latent'].to(self.device, dtype=torch.bfloat16, non_blocking=True)
            lq_latent = batch['lq_latent'].to(self.device, dtype=torch.bfloat16, non_blocking=True)
            gt = batch['gt'] # CHW float32 [0,1]
        else:
            gt_latent = torch.stack([b['gt_latent'] for b in batch], dim=0).to(self.device, dtype=torch.bfloat16, non_blocking=True)
            lq_latent = torch.stack([b['lq_latent'] for b in batch], dim=0).to(self.device, dtype=torch.bfloat16, non_blocking=True)
            gt = torch.stack([b['gt'] for b in batch], dim=0)
        bs = lq_latent.shape[0]

        # 基础校验
        assert gt_latent.ndim == 4 and lq_latent.ndim == 4, f"Expect 4D latents, got {gt_latent.shape} and {lq_latent.shape}"
        assert gt_latent.shape == lq_latent.shape, f"lr/hr shape mismatch: {lq_latent.shape} vs {gt_latent.shape}"
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
        
        self.batch_inputs = BatchInput(
            z_gt=gt_latent, z_lq=lq_latent, gt=gt,
            timesteps=timesteps
        )
        print_vram_state("After prepare_batch_inputs", logger=logger)

    def init_models(self):
        print(f"Use VAE: {self.config.use_vae}, Use D: {self.config.use_D}, Use EMA: {self.config.use_ema}")
        if getattr(self.config, "use_vae", True):
            self.init_vae()
        self.init_generator()
        if getattr(self.config, "use_D", True):
            self.init_discriminator()
        if getattr(self.config, "use_vae", True):
            self.init_lpips()
        try:
            vae_bytes = module_param_memory(self.vae)
            G_bytes_trainable = module_param_memory(self.G, only_trainable=True)
            G_bytes_all = module_param_memory(self.G, only_trainable=False)
            D_bytes = module_param_memory(self.D)
            lpips_bytes = module_param_memory(self.net_lpips)
            logger.info(
                "[Param memory] VAE=%s, G(trainable)=%s, G(all)=%s, D=%s, LPIPS=%s",
                human_bytes(vae_bytes),
                human_bytes(G_bytes_trainable),
                human_bytes(G_bytes_all),
                human_bytes(D_bytes),
                human_bytes(lpips_bytes),
            )
        except Exception as exc:
            logger.warning(f"Param memory report failed: {exc}")

    def init_vae(self):
        vae_config = HUNYUANIMAGE_V2_1_VAE_32x()
        self.vae = instantiate(
            vae_config.model,
            vae_path=vae_config.load_from,
        )
        if hasattr(self.vae, 'encoder'):
            del self.vae.encoder  # 删除Encoder
            logger.info("✓ Encoder removed to save memory")
        # Use float32 for VAE to avoid potential bf16 conv2d kernel issues
        self.vae = self.vae.to(self.device, dtype=torch.float32)
        logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)
        print_vram_state("After VAE to(device)", logger=logger)

    def init_generator(self):
        dit_config = HUNYUANIMAGE_V2_1_DIT()
        self.G = instantiate(dit_config.model, dtype=self.weight_dtype, device=self.device)
        load_hunyuan_dit_state_dict(self.G, dit_config.load_from, strict=True)
        self.G = self.G.to(self.device, dtype=self.weight_dtype)
        self.G.eval()
        if getattr(dit_config, "use_compile", False):
            self.G = torch.compile(self.G)
        logger.info("✓ DiT model loaded")
        self.G.eval().requires_grad_(False)
        print_vram_state("After DiT to(device)", logger=logger)

        if self.config.gradient_checkpointing:
            self.G.enable_gradient_checkpointing()
        if getattr(self.config, "gradient_checkpointing_mlp_only", False):
            # Only checkpoint MLP paths in DiT blocks to reduce activation memory safely
            try:
                base_model = self.G
                if hasattr(base_model, "get_base_model"):
                    base_model = base_model.get_base_model()
                if hasattr(base_model, "module"):
                    base_model = base_model.module
                base_model.enable_gradient_checkpointing_mlp_only()
                logger.info("✓ Enabled MLP-only activation checkpointing")
            except Exception as e:
                logger.warning(f"Failed to enable MLP-only checkpointing: {e}")

        # Handle LoRA configuration
        target_patterns = list(getattr(self.config, "lora_modules", []) or [])
        if target_patterns:
            resolved_targets = self._resolve_lora_targets(self.G, target_patterns)
            if not resolved_targets:
                raise ValueError(
                    f"Failed to match any LoRA target modules. Requested patterns: {target_patterns}"
                )
            logger.info(f"Add LoRA parameters to {resolved_targets}")
            G_lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_patterns,
            )
            self.G = get_peft_model(self.G, G_lora_cfg)
            mark_only_lora_as_trainable(self.G)
            lora_params = [p for p in self.G.parameters() if p.requires_grad]
            assert lora_params, "Failed to find LoRA parameters"
            for p in lora_params:
                p.data = p.data.to(device=self.device, dtype=torch.float32)
            self.G.to(self.device)
            self.lora_target_modules = resolved_targets
            print_vram_state("After enabling LoRA", logger=logger)
        else:
            logger.warning("LoRA modules list is empty; generator will remain frozen.")

        self._set_byt5_precision(self.weight_dtype)
        # Ensure module is in training mode so gradient checkpointing can take effect,
        # while keeping only LoRA params trainable
        self.G.train()

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)
                assert isinstance(model, HYImageDiffusionTransformer)
                state_dict = {
                    name: param.detach().cpu() # param.detach().clone().data
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                # state_dict = {}
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         state_dict[name] = param.detach().clone().data
                torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))

        def load_model_hook(models, input_dir):
            model = models.pop(0)
            model = self.unwrap_model(model) # This line is added by GPT
            assert isinstance(model, HYImageDiffusionTransformer)
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"), map_location="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing:
                logger.info(f"LoRA missing keys: {missing}")
            if unexpected:
                logger.info(f"LoRA unexpected keys: {unexpected}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def _resolve_lora_targets(self, model, target_patterns):
        candidate_types = (
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
        )
        matched = []
        for module_name, module in model.named_modules():
            if not isinstance(module, candidate_types):
                continue
            if any(module_name.endswith(pattern) for pattern in target_patterns):
                matched.append(module_name)
        return sorted(set(matched))

    def _set_byt5_precision(self, dtype: torch.dtype):
        base_model = self.G
        # unwrap Peft or DDP wrappers to reach underlying module
        if hasattr(base_model, "get_base_model"):
            base_model = base_model.get_base_model()
        if hasattr(base_model, "module"):
            base_model = base_model.module
        byt5_module = getattr(base_model, "byt5_in", None)
        if byt5_module is None:
            logger.warning("ByT5 module not found; skip precision adjustment")
            return
        byt5_module.to(device=self.device, dtype=dtype)
        # ensure LayerNorm params stay in desired dtype
        if hasattr(byt5_module, "layernorm"):
            byt5_module.layernorm.to(dtype=dtype)
        for param in byt5_module.parameters():
            param.requires_grad = False

    def _denoise_step(self, latents, timesteps, text_emb, text_mask, byt5_emb, byt5_mask, timesteps_r=None):
        """
        Perform one denoising step.

        Args:
            latents: Latent tensor
            timesteps: Timesteps tensor
            text_emb: Text embedding
            text_mask: Text mask
            byt5_emb: byT5 embedding
            byt5_mask: byT5 mask
            guidance_scale: Guidance scale
            timesteps_r: Optional next timestep

        Returns:
            Noise prediction tensor
        """
        latents = latents.to(device=self.device, dtype=self.weight_dtype)
        timesteps = timesteps.to(device=self.device)
        text_emb = text_emb.to(device=self.device, dtype=self.weight_dtype)
        text_mask = text_mask.to(device=self.device)

        if byt5_emb is not None and byt5_mask is not None:
            target_dtype = getattr(self, "weight_dtype", latents.dtype)
            byt5_emb = byt5_emb.to(device=self.device, dtype=target_dtype)
            byt5_mask = byt5_mask.to(device=self.device)
            extra_kwargs = {
                "byt5_text_states": byt5_emb,
                "byt5_text_mask": byt5_mask,
            }
        else:
            if self.use_byt5:
                raise ValueError("Must provide byt5_emb and byt5_mask for HunyuanImage 2.1")
            extra_kwargs = {}

        if getattr(self, "accelerator", None) is None or self.accelerator.is_local_main_process:
            base_model = self.G.module if hasattr(self.G, "module") else self.G
            byt5_ln = None
            try:
                byt5_ln = base_model.byt5_in.layernorm
            except AttributeError:
                byt5_ln = None
            logger.info(
                "[dtype debug] latents=%s@%s text_emb=%s byt5_states=%s weight_dtype=%s",
                latents.dtype,
                latents.device,
                text_emb.dtype,
                None if byt5_emb is None else byt5_emb.dtype,
                self.weight_dtype,
            )
            if byt5_ln is not None:
                logger.info(
                    "[dtype debug] byt5 layernorm weight=%s bias=%s device=%s",
                    byt5_ln.weight.dtype,
                    None if byt5_ln.bias is None else byt5_ln.bias.dtype,
                    byt5_ln.weight.device,
                )
            # [dtype debug] latents=torch.bfloat16@cuda:0 text_emb=torch.bfloat16 byt5_states=torch.float32 weight_dtype=torch.bfloat16
            # [dtype debug] byt5 layernorm weight=torch.bfloat16 bias=torch.bfloat16 device=cuda:0

        guidance_expand = None

        noise_pred = self.G(
            latents,
            timesteps,
            text_states=text_emb,
            encoder_attention_mask=text_mask,
            guidance=guidance_expand,
            return_dict=False,
            extra_kwargs=extra_kwargs,
            timesteps_r=timesteps_r,
        )[0]

        return noise_pred

    def forward_generator(self):
        # z_in = self.batch_inputs.z_lq * self.vae.config.scaling_factor
        # eps = self.G(
        #     z_in,
        #     self.batch_inputs.timesteps,
        #     encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
        # ).sample
        # z = self.scheduler.step(eps, self.config.coeff_t, z_in).pred_original_sample
        # x = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        
        t_expand = self.batch_inputs.timesteps
        sigmas = torch.tensor([self.config.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        latent_model_input = self.batch_inputs.z_lq
        print(f"{latent_model_input.shape=}, {latent_model_input.max()=}, {latent_model_input.min()=}") 

        noise_pred = self._denoise_step(
            latent_model_input, t_expand, 
            self.c_txt["text_emb"], self.c_txt["text_mask"], 
            self.c_txt["byt5_emb"], self.c_txt["byt5_mask"], 
            timesteps_r=None
        )
        print_vram_state("After _denoise_step (G forward)", logger=logger)
        print(f"{t_expand=}, {sigmas=}")
        latents = self.step(latent_model_input, noise_pred, sigmas, 0)
        print(f"{latents.shape=}, {latents.max()=}, {latents.min()=}")

        # If we are training in latent space, return latents directly
        if not getattr(self.config, "use_vae", True):
            return latents

        def _decode_latents(latents):
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            latents = latents.unsqueeze(2)

            # Decode in float32 for maximum compatibility and disable autocast to avoid bf16 kernels
            latents = latents.to(dtype=torch.float32)
            latents = latents.contiguous()

            # Decode device selection
            mode = getattr(self.config, "vae_decode_mode", "auto")  # choices: auto | cuda | cpu | auto_highres_cpu
            thr_hw = getattr(self.config, "vae_highres_min_hw", (768, 768))
            try:
                thr_hw = tuple(thr_hw)
            except Exception:
                thr_hw = (int(thr_hw), int(thr_hw))
            thr_area = int(getattr(self.config, "vae_highres_min_area", 1024 * 1024))

            H, W = latents.shape[-2], latents.shape[-1]
            highres = (H >= thr_hw[0] and W >= thr_hw[1]) or (H * W >= thr_area)

            # Optional cached CPU fallback persists across steps
            if getattr(self, "_vae_decode_on_cpu", False):
                image = self.vae.to("cpu", dtype=torch.float32).decode(latents.cpu(), return_dict=False)[0]
                image = image.to(self.device, non_blocking=True)
            else:
                use_cpu = False
                if mode == "cpu":
                    use_cpu = True
                elif mode == "auto_highres_cpu" and highres:
                    use_cpu = True
                elif mode == "cuda" or mode == "auto":
                    use_cpu = False

                if use_cpu:
                    image = self.vae.to("cpu", dtype=torch.float32).decode(latents.cpu(), return_dict=False)[0]
                    image = image.to(self.device, non_blocking=True)
                else:
                    with torch.autocast(device_type="cuda", enabled=False):
                        try:
                            image = self.vae.decode(latents, return_dict=False)[0]
                        except Exception as e:
                            logger.warning(f"VAE CUDA decode failed ({e}). Falling back to CPU float32 decode; this will be slower.")
                            self._vae_decode_on_cpu = True
                            image = self.vae.to("cpu", dtype=torch.float32).decode(latents.cpu(), return_dict=False)[0]
                            image = image.to(self.device, non_blocking=True)
            # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            #     image = self.vae.decode(latents, return_dict=False)[0]
            
            print_vram_state("Inside _decode_latents: after VAE.decode", logger=logger)
            
            print(f"Decoding: {image.shape=}, {image.max()=}, {image.min()=}")
            image = image[:, :, 0]  # Remove frame dimension for images
            print(f"Decoding: {image.shape=}, {image.max()=}, {image.min()=}")
            
            return image
        
        x = _decode_latents(latents.to(self.weight_dtype)).float()
        print_vram_state("After decode -> x", logger=logger)
        print(f"{x.shape=}, {x.max()=}, {x.min()=}")
        # x = x[..., :h1, :w1]
        # x = (x + 1) / 2
        # x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        # x = wavelet_reconstruction(x, ref.to(device=self.device))
        return x
