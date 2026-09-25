from typing import List, Dict
import os
import torch
import random
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
try:
    from peft import mark_only_lora_as_trainable
except ImportError:
    def mark_only_lora_as_trainable(model):
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
from HYPIR.utils.common import (
    instantiate_from_config,
    log_txt_as_img,
    print_vram_state,
    SuppressLogging,
    module_param_memory,
    human_bytes,
)
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from HYPIR.model.backbone import CNNRefiner
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, T5Tokenizer, T5EncoderModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.utils.others import NoOpContext, EdgeDetectionModel, total_variation_loss

logger = get_logger(__name__, log_level="INFO")

# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.base_model_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.base_model_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.base_model_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

class SD35Trainer(BaseTrainer):
    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()    
    
    def init_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.base_model_path, subfolder="scheduler"
        )

    def init_dataset(self):
        super().init_dataset()

        
    def prepare_batch_inputs(self, batch, transform=None):
        if transform == None:
            transform = self.batch_transform
        batch = transform(batch)
        gt = (batch["GT"] * 2 - 1).float().to(self.device)
        lq = (batch["LQ"] * 2 - 1).float().to(self.device)
        origin_lq = batch["low_LQ"].float().to(self.device)
        bs = lq.shape[0]
        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
        z_gt = self.vae.encode(gt.to(self.weight_dtype)).latent_dist.sample()
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
        prompt = batch["txt"]
        self.c_txt = {}
        # Build text conditioning
        if getattr(self.config, "use_qwen", False):
            # Use unnormalized LQ (in [0,1]) before scaling; fall back to lq if absent
            raw_lq = batch.get("LQ", None)
            if raw_lq is None:
                raw_lq = (lq + 1) / 2
            else:
                raw_lq = raw_lq.clamp(0, 1)
            # Default prompt if not provided
            prompts = batch.get("txt", None)
            if prompts is None:
                prompts = [getattr(self.config, "qwen_prompt", "Describe this image in detail")] * bs
            qwen_text_embeds, qwen_pooled = self.extract_qwen_feature(raw_lq, prompts)
            self.c_txt = {"text_embeds": qwen_text_embeds, "pooled_embeds": qwen_pooled}
        else:
            # use cached debug embeddings
            pos_promt_emb = torch.load(f"debug_inputs/pos_prompt_embeds.pt").to(self.device)
            pos_pooled_prompt_emb = torch.load(f"debug_inputs/pos_pooled_prompt_embeds.pt").to(self.device)
            self.c_txt = {"text_embeds": pos_promt_emb, "pooled_embeds": pos_pooled_prompt_emb}

        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq, z_gt=z_gt,
            timesteps=timesteps,
        )

    def init_models(self):
        print(f"Use VAE: {self.config.use_vae}, Use D: {self.config.use_D}, Use EMA: {self.config.use_ema}")
        self.init_scheduler()
        if getattr(self.config, "use_vae", True):
            self.init_vae()
        self.init_generator()
        if getattr(self.config, "use_D", True):
            self.init_discriminator()
        if getattr(self.config, "use_vae", True):
            self.init_lpips()
        if getattr(self.config, "use_txt", False):
            self.init_text_models()
        if getattr(self.config, "use_qwen", False):
            self.init_qwen()
            self.init_qwen_projector()
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

    def init_qwen_projector(self):
        """Project Qwen hidden states to SD3 text-embed dims and pooled dims."""
        base_model = self.G.module if hasattr(self.G, "module") else self.G
        # SD3 text embedding dim
        text_dim = base_model.text_embed_dim if hasattr(base_model, "text_embed_dim") else base_model.config.joint_attention_dim
        pooled_dim = getattr(base_model.config, "pooled_projection_dim", 1536)
        qwen_dim = getattr(self, "qwen_hidden_size", 3584)
        logger.info(f"Initializing qwen_projector_text: {qwen_dim} -> {text_dim}, qwen_projector_pooled: {qwen_dim} -> {pooled_dim}")
        self.qwen_projector_text = torch.nn.Linear(qwen_dim, text_dim).to(self.device, dtype=self.weight_dtype)
        self.qwen_projector_pooled = torch.nn.Linear(qwen_dim, pooled_dim).to(self.device, dtype=self.weight_dtype)
        self.qwen_projector_text.train().requires_grad_(True)
        self.qwen_projector_pooled.train().requires_grad_(True)

    def init_qwen(self):
        logger.info("Loading Qwen3-VL for VLM text embeddings...")
        model_path = self.config.qwen_model_path
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
        ).eval()
        self.qwen_model.requires_grad_(False)
        self.qwen_processor = AutoProcessor.from_pretrained(model_path)
        self.image_token_id = self.qwen_model.config.image_token_id
        self.qwen_hidden_size = self.qwen_model.config.text_config.hidden_size
        logger.info(f"✓ Qwen3-VL loaded. Hidden size: {self.qwen_hidden_size}")

    def extract_qwen_feature(self, lq, prompts):
        """Extract SD3-compatible text_embeds and pooled_embeds from images via Qwen."""
        batch_size = len(prompts)
        lq_images_denorm = (lq * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        from PIL import Image
        messages_batch = []
        for i in range(batch_size):
            img_pil = Image.fromarray(lq_images_denorm[i])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": prompts[i]},
                    ],
                }
            ]
            messages_batch.append(messages)

        texts = [
            self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.qwen_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.qwen_model.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [B, Seq, H_qwen]
        text_embeds_list = []
        pooled_list = []
        for i in range(batch_size):
            input_ids = inputs.input_ids[i]
            attn_mask = inputs.attention_mask[i]
            text_mask = (attn_mask == 1) & (input_ids != self.image_token_id)
            text_tokens = last_hidden_state[i][text_mask]  # [N_txt, H_qwen]
            text_tokens = text_tokens.to(dtype=self.weight_dtype)
            proj_tokens = self.qwen_projector_text(text_tokens)  # [N_txt, H_text]
            text_embeds_list.append(proj_tokens)
            pooled_src = text_tokens.mean(dim=0, keepdim=True)  # [1, H_qwen]
            pooled = self.qwen_projector_pooled(pooled_src)      # [1, H_pooled]
            pooled_list.append(pooled)

        text_embeds = torch.nn.utils.rnn.pad_sequence(
            text_embeds_list, batch_first=True
        )  # stays on same device/dtype
        return text_embeds, torch.cat(pooled_list, dim=0)

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        
        # if hasattr(self.vae, 'encoder'):
        #     del self.vae.encoder  # 删除Encoder
        #     logger.info("✓ Encoder removed to save memory")
            
        # Use float32 for VAE to avoid potential bf16 conv2d kernel issues
        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)
        print_vram_state("After VAE to(device)", logger=logger)

    def init_generator(self):
        self.G = SD3Transformer2DModel.from_pretrained(
            self.config.base_model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        ).to(self.device)

        logger.info("✓ DiT model loaded")
        print_vram_state("After DiT to(device)", logger=logger)

        if getattr(self.config, "gradient_checkpointing", False):
            if hasattr(self.G, "enable_gradient_checkpointing"):
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

        # Apply LoRA if configured
        target_patterns = list(getattr(self.config, "lora_modules", []) or [])
        if target_patterns:
            matched = self._resolve_lora_targets(self.G, target_patterns)
            logger.info(f"Matched LoRA target modules: {matched}")
            lora_cfg = LoraConfig(
                r=getattr(self.config, "lora_rank", 16),
                lora_alpha=getattr(self.config, "lora_alpha", 16),
                lora_dropout=getattr(self.config, "lora_dropout", 0.0),
                target_modules=matched,
                bias="none",
            )
            self.G = get_peft_model(self.G, lora_cfg)
            mark_only_lora_as_trainable(self.G)
            logger.info("✓ Applied LoRA to DiT (trainable params are LoRA only)")
        else:
            logger.info("No LoRA target patterns provided; training full model.")

        self._set_byt5_precision(self.weight_dtype)
        # Ensure module is in training mode so gradient checkpointing can take effect,
        # while keeping only LoRA params trainable
        self.G.train()

    def _resolve_lora_targets(self, model, target_patterns):
        """Match module names against patterns for LoRA injection."""
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
            for pat in target_patterns:
                if pat in module_name:
                    matched.append(module_name)
                    break
        return sorted(set(matched))

    def init_text_models(self):
        # Tokenizer: prefer tokenizer_3 (T5) for SD3.5
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            self.config.base_model_path,
            subfolder="tokenizer",
            revision=self.config.revision,
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            self.config.base_model_path,
            subfolder="tokenizer_2",
            revision=self.config.revision,
        )
        try:
            self.tokenizer_three = T5TokenizerFast.from_pretrained(
                self.config.base_model_path,
                subfolder="tokenizer_3",
                revision=self.config.revision,
            )
        except Exception as e:
            logger.warning(f"Could not load T5TokenizerFast, falling back to T5Tokenizer. Reason: {e}")
            self.tokenizer_three = T5Tokenizer.from_pretrained(
                self.config.base_model_path,
                subfolder="tokenizer_3",
                revision=self.config.revision,
            )
        # Text encoder: SD3.5 medium commonly uses T5 in text_encoder_3
        self.text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.config.base_model_path, self.config.revision
        )
        self.text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.config.base_model_path, self.config.revision, subfolder="text_encoder_2"
        )
        self.text_encoder_cls_three = import_model_class_from_model_name_or_path(
            self.config.base_model_path, self.config.revision, subfolder="text_encoder_3"
        )

        self.text_encoder_one, self.text_encoder_two, self.text_encoder_three = load_text_encoders(
            self.text_encoder_cls_one, self.text_encoder_cls_two, self.text_encoder_cls_three, self.config
        )

        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.text_encoder_three.requires_grad_(False)

        self.text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_three.to(self.accelerator.device, dtype=self.weight_dtype)

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(self.accelerator.device)
        attention_mask = tok.attention_mask.to(self.accelerator.device)

        enc_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        prompt_embeds = enc_out.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        summed = (prompt_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_prompt_embeds = summed / denom

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    
    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)
                assert isinstance(model, SD3Transformer2DModel) or hasattr(model, "base_model")
                state_dict = {
                    name: param.detach().cpu()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))
                # Save qwen_projector if present
                # Save Qwen projectors if present
                if getattr(self.config, "use_qwen", False):
                    if hasattr(self, "qwen_projector_text"):
                        torch.save(
                            self.unwrap_model(self.qwen_projector_text).state_dict(),
                            os.path.join(output_dir, "qwen_projector_text.pth"),
                        )
                    if hasattr(self, "qwen_projector_pooled"):
                        torch.save(
                            self.unwrap_model(self.qwen_projector_pooled).state_dict(),
                            os.path.join(output_dir, "qwen_projector_pooled.pth"),
                        )
                if getattr(self.config, "use_refiner", False):
                    for i in range(len(models) - 1, -1, -1):
                        model = models[i]
                        unwrapped_model = self.accelerator.unwrap_model(model)
                        if unwrapped_model is self.accelerator.unwrap_model(self.refiner):
                            models.pop(i)
                            weights.pop(i)
                            logger.info("Excluded refiner from accelerator checkpoint (will be saved separately).")
                            break

        def load_model_hook(models, input_dir):
            model = models.pop(0)
            model = self.unwrap_model(model)
            assert isinstance(model, SD3Transformer2DModel) or hasattr(model, "base_model")
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"), map_location="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing:
                logger.info(f"LoRA missing keys: {missing}")
            if unexpected:
                logger.info(f"LoRA unexpected keys: {unexpected}")
            # Load qwen_projector if present
            # Load Qwen projectors if present
            if getattr(self.config, "use_qwen", False):
                if hasattr(self, "qwen_projector_text"):
                    p = os.path.join(input_dir, "qwen_projector_text.pth")
                    if os.path.exists(p):
                        logger.info(f"Loading qwen_projector_text from {p}")
                        self.unwrap_model(self.qwen_projector_text).load_state_dict(
                            torch.load(p, map_location="cpu")
                        )
                if hasattr(self, "qwen_projector_pooled"):
                    p = os.path.join(input_dir, "qwen_projector_pooled.pth")
                    if os.path.exists(p):
                        logger.info(f"Loading qwen_projector_pooled from {p}")
                        self.unwrap_model(self.qwen_projector_pooled).load_state_dict(
                            torch.load(p, map_location="cpu")
                        )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

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

    def _denoise_step(self, latents, timesteps, text_emb, pooled_emb, timesteps_r=None):
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
        pooled_emb = pooled_emb.to(device=self.device, dtype=self.weight_dtype)

        if getattr(self, "accelerator", None) is None or self.accelerator.is_local_main_process:
            base_model = self.G.module if hasattr(self.G, "module") else self.G

        guidance_expand = None
        
        noise_pred = self.G(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=text_emb,
            pooled_projections=pooled_emb,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        return noise_pred
    
    def forward_generator(self):
        t_expand = self.batch_inputs.timesteps
        sigmas = torch.tensor([self.config.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        latent_model_input = self.batch_inputs.z_lq
        noise_pred = self._denoise_step(
            latent_model_input, t_expand, 
            self.c_txt["text_embeds"], 
            self.c_txt["pooled_embeds"],  
            timesteps_r=None
        )
        latents = self.step(latent_model_input, noise_pred, sigmas, 0)
        # If we are training in latent space, return latents directly
        if not getattr(self.config, "use_vae", True):
            return latents
        
        x = self._decode_latents(latents.to(self.weight_dtype)).float()
        return x, latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor

        latents = latents.to(dtype=self.weight_dtype).contiguous()
        image = self.vae.decode(latents, return_dict=False)[0]
        
        if getattr(self.config, "use_refiner", False):
            image = self.refiner(image)
        return image