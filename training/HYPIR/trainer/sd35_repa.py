from typing import List, Dict
import os
import torch
from transformers import CLIPModel    
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
try:
    from peft import mark_only_lora_as_trainable
except ImportError:
    def mark_only_lora_as_trainable(model):
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
import torch.nn.functional as F
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from vision_aided_loss.cv_losses import multilevel_loss
from HYPIR.utils.common import (
    instantiate_from_config,
    log_txt_as_img,
    print_vram_state,
    SuppressLogging,
    module_param_memory,
    human_bytes,
)
from HYPIR.model.backbone import CNNRefiner
from REPA.utils import load_encoders
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
)
from HYPIR.model.autoencoder_kl import AutoencoderKL
from ..model.transformer_sd3 import SD3Transformer2DModel
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, T5Tokenizer, T5EncoderModel
from HYPIR.trainer.base import BaseTrainer, BatchInput

logger = get_logger(__name__, log_level="INFO")

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    target_resolution = 896
    if 'clip' in enc_type:
        x = x / 255.
        # --- FIX: Always resize to 224 for CLIP ---
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic', antialias=True)
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = torch.nn.functional.interpolate(x, target_resolution, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x

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
        args.base_model_path, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        args.base_model_path, subfolder="text_encoder_2", revision=None, variant=None
    )
    text_encoder_three = class_three.from_pretrained(
        args.base_model_path, subfolder="text_encoder_3", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

class SD35REPATrainer(BaseTrainer):
    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()    
    
    def init_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.base_model_path, subfolder="scheduler"
        )

    def init_dataset(self):
        super().init_dataset()
        # data_cfg = self.config.data_config
        # dataset = instantiate_from_config(data_cfg.train.latent_data)
        # # 随机批读取 + 高效 IO
        # num_workers = int(data_cfg.train.dataloader_num_workers)
        # pin_memory = torch.cuda.is_available()
        # persistent_workers = num_workers > 0
        # prefetch_factor = 4 if num_workers > 0 else None
        # dl_kwargs = {
        #     "dataset": dataset,
        #     "shuffle": True,
        #     "batch_size": int(data_cfg.train.batch_size),
        #     "num_workers": num_workers,
        #     "pin_memory": pin_memory,
        #     "persistent_workers": persistent_workers,
        #     "drop_last": False,
        # }
        # if prefetch_factor is not None:
        #     dl_kwargs["prefetch_factor"] = prefetch_factor
        # self.dataloader = torch.utils.data.DataLoader(**dl_kwargs)

        # load from saved debug inputs
        if not self.config.use_clip and not self.config.use_txt:
            pos_prompt_emb = torch.load(f"debug_inputs/pos_prompt_embeds.pt").to(self.device)
            pos_pooled_prompt_emb = torch.load(f"debug_inputs/pos_pooled_prompt_embeds.pt").to(self.device)
            print(f"Shape of pos_emb and pooled_emb: {pos_prompt_emb.shape}, {pos_pooled_prompt_emb.shape}")
            print(f"min/max pos_emb: {pos_prompt_emb.min()} / {pos_prompt_emb.max()}")
            print(f"min/max pooled_pos_emb: {pos_pooled_prompt_emb.min()} / {pos_pooled_prompt_emb.max()}")
            self.c_txt = {"prompt_embeds": pos_prompt_emb,
                        "pooled_prompt_embeds": pos_pooled_prompt_emb}
    
    def init_repa(self):
        target_resolution = 896
        if self.config.enc_type != None:
            self.repa_encoders, self.repa_encoder_types, self.repa_architectures = load_encoders(
                self.config.enc_type, self.device, target_resolution
            )
            for i in range(len(self.repa_encoders)):
                self.repa_encoders[i] = self.repa_encoders[i].to(dtype=self.weight_dtype)
        else:
            raise NotImplementedError()
    
    def init_refiner(self):
        self.refiner = CNNRefiner().to(self.accelerator.device)
        self.refiner_opt = torch.optim.AdamW(self.refiner.parameters(), lr=1e-4, weight_decay=1e-2)

    def prepare_batch_inputs(self, batch):
        batch = self.batch_transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        prompt = batch["txt"]
        # print(f"{gt.shape=}, {gt.max()=}, {gt.min()=}")
        # print(f"{lq.shape=}, {lq.max()=}, {lq.min()=}")
        bs = lq.shape[0]
        if getattr(self.config, "train_encoder", False):
            self.vae.encoder.eval()

        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
        z_gt = self.vae.encode(gt.to(self.weight_dtype)).latent_dist.sample()

        if getattr(self.config, "train_encoder", False):
            self.vae.encoder.train()

        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self.config.use_clip:
                lq_for_clip = (lq + 1) / 2.0  # 反归一化到 [0, 1]
                lq_for_clip = preprocess_raw_image(lq_for_clip * 255., 'clip')

                # 2. 使用CLIP图像编码器提取特征
                vision_output_one = self.clip_model_one.vision_model(lq_for_clip)
                vision_output_two = self.clip_model_two.vision_model(lq_for_clip)

                image_embeds_one = vision_output_one.last_hidden_state[:, :-1, :]
                pooled_embeds_one = vision_output_one.pooler_output

                image_embeds_two = vision_output_two.last_hidden_state[:, :-1, :]
                pooled_embeds_two = vision_output_two.pooler_output
                print(f"Shape of pooled_embeds before: {pooled_embeds_one.shape}, {pooled_embeds_two.shape}")
                pooled_embeds_one = self.clip_model_one.visual_projection(pooled_embeds_one) # shape: [1, 768]
                pooled_embeds_two = self.clip_model_two.visual_projection(pooled_embeds_two) # shape: [1, 1024]
                print(f"Shape of pooled_embeds after: {pooled_embeds_one.shape}, {pooled_embeds_two.shape}")
                self.image_embeds = torch.cat([image_embeds_one, image_embeds_two], dim=-1)
                self.image_embeds = torch.nn.functional.pad(self.image_embeds, (0, 4096-self.image_embeds.shape[-1]))
                
                self.pooled_embeds = torch.cat([pooled_embeds_one, pooled_embeds_two], dim=-1)
                self.pooled_embeds = torch.nn.functional.pad(self.pooled_embeds, (0, 2048-self.pooled_embeds.shape[-1]))
                print(f"----------Shape of image_embeds: {self.image_embeds.shape}------------")
                print(f"----------Shape of pooled_embeds: {self.pooled_embeds.shape}-----------")

            if getattr(self.config, "use_txt", True):
                text_encoders = [self.text_encoder_one, self.text_encoder_two, self.text_encoder_three]
                text_tokenizers = [self.tokenizer_one, self.tokenizer_two, self.tokenizer_three]
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                    text_encoders=text_encoders,
                    tokenizers=text_tokenizers,
                    prompt=prompt,
                    max_sequence_length=256
                )
                self.c_txt={
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                }
                logger.info(f"Use text embeds, prompt shape: {prompt_embeds.shape}, pooled prompt shape: {pooled_prompt_embeds.shape}")
            if self.config.use_repa:
                zs = []
                with self.accelerator.autocast():
                    for encoder, encoder_type, arch in zip(self.repa_encoders, self.repa_encoder_types, self.repa_architectures):
                        raw_image_ = preprocess_raw_image(gt, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)
            else:
                zs = []
        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq, z_gt=z_gt, z_s=zs,
            timesteps=timesteps,
        )

    def init_models(self):
        print(f"Use VAE: {self.config.use_vae}, Use D: {self.config.use_D}, Use EMA: {self.config.use_ema}")
        if getattr(self.config, "use_vae", True):
            self.init_vae()
        self.init_generator()
        if getattr(self.config, "use_D", True):
            self.init_discriminator()
        if getattr(self.config, "use_vae", True):
            self.init_lpips()
        if getattr(self.config, "use_refiner", False):
            self.init_refiner()
        if getattr(self.config, "use_clip", False):
            self.init_clip_model()
        if getattr(self.config, "use_txt", True):
            self.init_text_models()
        try:
            vae_bytes = module_param_memory(self.vae)
            vae_bytes_trainable = module_param_memory(self.vae, only_trainable=True)
            G_bytes_trainable = module_param_memory(self.G, only_trainable=True)
            G_bytes_all = module_param_memory(self.G, only_trainable=False)
            D_bytes = module_param_memory(self.D)
            lpips_bytes = module_param_memory(self.net_lpips)
            logger.info(
                "[Param memory] VAE=%s, VAE(trainable)=%s, G(trainable)=%s, G(all)=%s, D=%s, LPIPS=%s",
                human_bytes(vae_bytes),
                human_bytes(vae_bytes_trainable),
                human_bytes(G_bytes_trainable),
                human_bytes(G_bytes_all),
                human_bytes(D_bytes),
                human_bytes(lpips_bytes),
            )
        except Exception as exc:
            logger.warning(f"Param memory report failed: {exc}")

    def init_clip_model(self):
        """Initializes the CLIP Vision model for image encoding."""
        model_one = CLIPModel.from_pretrained('./models')
        model_two = CLIPModel.from_pretrained('./models')

        self.clip_model_one = model_one.to(self.device)
        self.clip_model_two = model_two.to(self.device)
        self.clip_model_one.requires_grad_(False)
        self.clip_model_two.requires_grad_(False)
        self.clip_model_one.eval()
        self.clip_model_two.eval()
        logger.info("✓ CLIP Vision model Large/Giant (ViT-L-14) loaded.")

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        
        # if hasattr(self.vae, 'encoder'):
        #     del self.vae.encoder  # 删除Encoder
        #     logger.info("✓ Encoder removed to save memory")
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        if getattr(self.config, "train_encoder", True):
            for param in self.vae.encoder.parameters():
                param.requires_grad = True
            self.vae.encoder.train()

            # 4. （可选但推荐）明确保持 Decoder 在评估模式
            if hasattr(self.vae, 'decoder'):
                self.vae.decoder.eval()
            
            logger.info("✓ VAE loaded. Decoder is frozen, Encoder is trainable.")
        elif self.config.train_decoder:
            for param in self.vae.decoder.parameters():
                param.requires_grad = True
            
            # 3. 将 Decoder 切换到训练模式
            #    这对于 Dropout 或 BatchNorm 等层是必要的（如果存在）
            self.vae.decoder.train()

            # 4. （可选但推荐）明确保持 Encoder 在评估模式
            if hasattr(self.vae, 'encoder'):
                self.vae.encoder.eval()
            
            logger.info("✓ VAE loaded. Encoder is frozen, Decoder is trainable.")
        else:
            logger.info("✓ VAE loaded.")
        # Use float32 for VAE to avoid potential bf16 conv2d kernel issues
        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        # logger.info("✓ VAE loaded")
        # self.vae.eval().requires_grad_(False)

        print_vram_state("After VAE to(device)", logger=logger)

    def init_generator(self):
        self.G = SD3Transformer2DModel.from_pretrained_local(
            self.config.base_model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        ).to(self.device)

        logger.info(f"✓ DiT model loaded from {self.config.base_model_path}")
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

        self._set_byt5_precision(self.weight_dtype)
        # Ensure module is in training mode so gradient checkpointing can take effect,
        # while keeping only LoRA params trainable
        self.G.train()

    def init_text_models(self):
        # Tokenizer: prefer tokenizer_3 (T5) for SD3.5
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            self.config.base_model_path,
            subfolder="tokenizer",
            revision=None,
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            self.config.base_model_path,
            subfolder="tokenizer_2",
            revision=None,
        )
        try:
            self.tokenizer_three = T5TokenizerFast.from_pretrained(
                self.config.base_model_path,
                subfolder="tokenizer_3",
                revision=None,
            )
        except Exception as e:
            logger.warning(f"Could not load T5TokenizerFast, falling back to T5Tokenizer. Reason: {e}")
            self.tokenizer_three = T5Tokenizer.from_pretrained(
                self.config.base_model_path,
                subfolder="tokenizer_3",
                revision=None,
            )
        # Text encoder: SD3.5 medium commonly uses T5 in text_encoder_3
        self.text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.config.base_model_path, None
        )
        self.text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.config.base_model_path, None, subfolder="text_encoder_2"
        )
        self.text_encoder_cls_three = import_model_class_from_model_name_or_path(
            self.config.base_model_path, None, subfolder="text_encoder_3"
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
        
        logger.info("✓ T5/ClipG/ClipL model loaded")
        
    # Copied from dreambooth sd3 example
    def _encode_prompt_with_t5(
        self,
        text_encoder,
        tokenizer,
        max_sequence_length,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds


    # Copied from dreambooth sd3 example
    def _encode_prompt_with_clip(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, pooled_prompt_embeds


    # Copied from dreambooth sd3 example
    def encode_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = tokenizers[:2]
        clip_text_encoders = text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device if device is not None else self.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = self._encode_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else self.device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        # prompt_embeds = clip_prompt_embeds

        return prompt_embeds, pooled_prompt_embeds
    
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
            if getattr(self.config, "use_refiner", False):
                for i in range(len(models) - 1, -1, -1):
                    model = models[i]
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    if unwrapped_model is self.accelerator.unwrap_model(self.refiner):
                        models.pop(i)
                        weights.pop(i-1)
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

    def _denoise_step(self, latents, timesteps, prompt_embeds, pooled_prompt_embeds, timesteps_r=None):
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
        
        # image_emb = image_emb.to(device=self.device, dtype=self.weight_dtype)
        # pooled_image_emb = pooled_image_emb.to(device=self.device, dtype=self.weight_dtype)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.weight_dtype)
        # if byt5_emb is not None and byt5_mask is not None:
        #     target_dtype = getattr(self, "weight_dtype", latents.dtype)
        #     byt5_emb = byt5_emb.to(device=self.device, dtype=target_dtype)
        #     byt5_mask = byt5_mask.to(device=self.device)
        #     extra_kwargs = {
        #         "byt5_text_states": byt5_emb,
        #         "byt5_text_mask": byt5_mask,
        #     }
        # else:
        #     if self.use_byt5:
        #         raise ValueError("Must provide byt5_emb and byt5_mask for HunyuanImage 2.1")
        #     extra_kwargs = {}

        if getattr(self, "accelerator", None) is None or self.accelerator.is_local_main_process:
            base_model = self.G.module if hasattr(self.G, "module") else self.G
            # byt5_ln = None
            # try:
            #     byt5_ln = base_model.byt5_in.layernorm
            # except AttributeError:
            #     byt5_ln = None
            # logger.info(
            #     "[dtype debug] latents=%s@%s text_emb=%s weight_dtype=%s",
            #     latents.dtype,
            #     latents.device,
            #     text_emb.dtype,
            #     self.weight_dtype,
            # )
            # if byt5_ln is not None:
            #     logger.info(
            #         "[dtype debug] byt5 layernorm weight=%s bias=%s device=%s",
            #         byt5_ln.weight.dtype,
            #         None if byt5_ln.bias is None else byt5_ln.bias.dtype,
            #         byt5_ln.weight.device,
            #     )
            # [dtype debug] latents=torch.bfloat16@cuda:0 text_emb=torch.bfloat16 byt5_states=torch.float32 weight_dtype=torch.bfloat16
            # [dtype debug] byt5 layernorm weight=torch.bfloat16 bias=torch.bfloat16 device=cuda:0

        guidance_expand = None

        noise_pred, zs_pred = self.G(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        
        return noise_pred, zs_pred
    
    def forward_generator(self):
        t_expand = self.batch_inputs.timesteps
        sigmas = torch.tensor([self.config.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        latent_model_input = self.batch_inputs.z_lq
        # print(f"{latent_model_input.shape=}, {latent_model_input.max()=}, {latent_model_input.min()=}") 

        if self.config.use_clip:
            noise_pred, zs_pred = self._denoise_step(
                latent_model_input, t_expand, 
                self.image_embeds, 
                self.pooled_embeds,  
                timesteps_r=None
            )
        else:
            noise_pred, zs_pred = self._denoise_step(
                latent_model_input, t_expand, 
                self.c_txt["prompt_embeds"], 
                self.c_txt["pooled_prompt_embeds"],  
                timesteps_r=None
            )    
        # print_vram_state("After _denoise_step (G forward)", logger=logger)
        # print(f"{t_expand=}, {sigmas=}")
        latents = self.step(latent_model_input, noise_pred, sigmas, 0)
        print(f"{latents.shape=}, {latents.mean()=}, {latents.std()=}")

        # If we are training in latent space, return latents directly
        if not getattr(self.config, "use_vae", True):
            return latents, zs_pred
        
        x = self._decode_latents(latents.to(self.weight_dtype)).float()
        # print_vram_state("After decode -> x", logger=logger)
        # print(f"{x.shape=}, {x.max()=}, {x.min()=}")
        # x = x[..., :h1, :w1]
        # x = (x + 1) / 2
        # x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        # x = wavelet_reconstruction(x, ref.to(device=self.device))
        return x, latents, zs_pred


    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Align to hy.py decode ergonomics and stability
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor

        # latents = latents.to(dtype=torch.float32).contiguous()
        latents = latents.to(dtype=self.weight_dtype).contiguous()
        # print(f"{latents.shape=}, {latents.max()=}, {latents.min()=}")

        # with torch.autocast(device_type="cuda", enabled=False):
        image = self.vae.decode(latents, return_dict=False)[0]
            
        if getattr(self.config, "use_refiner", False):
            image = self.refiner(image)
        return image

        mode = getattr(self.config, "vae_decode_mode", "auto")  # auto | cuda | cpu | auto_highres_cpu
        thr_hw = getattr(self.config, "vae_highres_min_hw", (768, 768))
        try:
            thr_hw = tuple(thr_hw)
        except Exception:
            thr_hw = (int(thr_hw), int(thr_hw))
        thr_area = int(getattr(self.config, "vae_highres_min_area", 1024 * 1024))

        H, W = latents.shape[-2], latents.shape[-1]
        highres = (H >= thr_hw[0] and W >= thr_hw[1]) or (H * W >= thr_area)

        if getattr(self, "_vae_decode_on_cpu", False):
            image = self.vae.to("cpu", dtype=torch.float32).decode(latents.cpu(), return_dict=False)[0]
            return image.to(self.device, non_blocking=True)

        use_cpu = False
        if mode == "cpu":
            use_cpu = True
        elif mode == "auto_highres_cpu" and highres:
            use_cpu = True

        if use_cpu:
            image = self.vae.to("cpu", dtype=torch.float32).decode(latents.cpu(), return_dict=False)[0]
            return image.to(self.device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=False):
            try:
                image = self.vae.decode(latents, return_dict=False)[0]
            except Exception as e:
                logger.warning(f"VAE CUDA decode failed ({e}). Falling back to CPU float32 decode.")
                self._vae_decode_on_cpu = True
                image = self.vae.to("cpu", dtype=torch.float32).decode(latents.cpu(), return_dict=False)[0]
                image = image.to(self.device, non_blocking=True)
        return image

    # def optimize_discriminator(self):
    #     gt = self.batch_inputs.z_gt
    #     with torch.no_grad():
    #         if self.config.use_repa:
    #             x, latents, _ = self.forward_generator()
    #         else:
    #             x = self.forward_generator()
    #     self.G_pred = x
    #     ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
    #     # Avoid accelerate.accumulate (which uses no_sync) when ZeRO stage >= 2
    #     use_null_ctx = bool(ds_plugin and getattr(ds_plugin, "zero_stage", 0) >= 2)
    #     ctx = nullcontext() if use_null_ctx else self.accelerator.accumulate(self.D)
    #     with ctx:  
    #         self.unwrap_model(self.D).train().requires_grad_(True)
    #         loss_fn = multilevel_loss(alpha=0.8)
    #         if self.config.use_clip:
    #             real_logits = self.D(
    #                 hidden_states=gt, 
    #                 encoder_hidden_states=self.image_embeds,
    #                 pooled_projections=self.pooled_embeds,
    #                 timestep=self.batch_inputs.timesteps,
    #                 for_real=True, 
    #                 return_logits=True
    #             )
    #             fake_logits = self.D(
    #                 hidden_states=latents, 
    #                 encoder_hidden_states=self.image_embeds,
    #                 pooled_projections=self.pooled_embeds,
    #                 timestep=self.batch_inputs.timesteps,
    #                 for_real=False, 
    #                 return_logits=True
    #             )
    #         else:
    #             real_logits = self.D(
    #                 hidden_states=gt, 
    #                 encoder_hidden_states=self.c_txt["prompt_embeds"],
    #                 pooled_projections=self.c_txt["pooled_prompt_embeds"], 
    #                 timestep=self.batch_inputs.timesteps,
    #                 for_real=True, 
    #                 return_logits=True
    #             )
    #             fake_logits = self.D(
    #                 hidden_states=latents, 
    #                 encoder_hidden_states=self.c_txt["prompt_embeds"],
    #                 pooled_projections=self.c_txt["pooled_prompt_embeds"], 
    #                 timestep=self.batch_inputs.timesteps,
    #                 for_real=False, 
    #                 return_logits=True
    #             )
    #         loss_D_real = loss_fn(real_logits, for_real=True)
    #         loss_D_fake = loss_fn(fake_logits, for_real=False)
    #         loss_D = loss_D_real.mean() + loss_D_fake.mean()
    #         self.accelerator.backward(loss_D)
    #         if self.accelerator.sync_gradients:
    #             self.accelerator.clip_grad_norm_(self.D_params, self.config.max_grad_norm)
    #         self.D_opt.step()
    #         self.D_opt.zero_grad()
    #     loss_dict = dict(D=loss_D)
    #     # logits = D(x) w/o sigmoid = log(p_real(x) / p_fake(x))
    #     with torch.no_grad():
    #         real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
    #         fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
    #     loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))
    #     return loss_dict

    # def optimize_generator_image(self):
    #     ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
    #     # Avoid accelerate.accumulate (which uses no_sync) when ZeRO stage >= 2
    #     use_null_ctx = bool(ds_plugin and getattr(ds_plugin, "zero_stage", 0) >= 2)
    #     ctx = nullcontext() if use_null_ctx else self.accelerator.accumulate(self.G)
    #     with ctx:
    #         if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None:
    #             self.unwrap_model(self.D).eval().requires_grad_(False)

    #         if self.config.use_repa:
    #             x, latents, zs_pred = self.forward_generator()
    #             # print(f"-------------Shape of x: {x.shape}")
    #             # print(f"-------------Shape of latent: {latent.shape}------------")
    #             zs = self.batch_inputs.z_s
    #             # print(f"------------Shape of zs: {zs[0].shape}")
    #             # print(f"------------Shape of zs pred: {zs_pred[0].shape}------------")
    #         else:
    #             x = self.forward_generator()

    #         self.G_pred = x
    #         # print(f"G_pred: {x}")
    #         # print(f"GT: {self.batch_inputs.gt}")
    #         loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
    #         # loss_l2 = F.mse_loss(latents.float(), self.batch_inputs.z_gt.float(), reduction="mean") * self.config.lambda_l2
    #         loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips

    #         if getattr(self.config, "lambda_edge_detect", False):
    #             edge_x = self.edge_detection_model(x)
    #             edge_gt = self.edge_detection_model(self.batch_inputs.gt)
    #             loss_edge = self.net_lpips(edge_x, edge_gt).mean() * self.config.lambda_edge_detect
    #             # print(f"------------loss edge: {loss_edge}------------")
    #         else:
    #             loss_edge = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

    #         if getattr(self.config, "lambda_tv", False):
    #             tv_x = total_variation_loss(x)
    #             tv_gt = total_variation_loss(self.batch_inputs.gt)
    #             loss_tv = self.net_lpips(tv_x, tv_gt).mean() * self.config.lambda_tv
    #             # print(f"------------loss tv: {loss_tv}------------")
    #         else:
    #             loss_tv = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

    #         if self.config.use_repa:
    #             proj_loss = (1 - self.repa_loss(zs, zs_pred)) * self.config.proj_coef
    #             # print(f"------------loss repa: {proj_loss}------------")
    #         else:
    #             proj_loss = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

    #         if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None:
    #             loss_fn = multilevel_loss(alpha=0.8)
    #             logits = self.D(
    #                 hidden_states=latents, 
    #                 encoder_hidden_states=self.c_txt["prompt_embeds"],
    #                 pooled_projections=self.c_txt["pooled_prompt_embeds"], 
    #                 timestep=self.batch_inputs.timesteps,
    #                 for_G=True, 
    #             )
    #             loss_disc = loss_fn(logits, for_G=True).mean() * self.config.lambda_gan
    #         else:
    #             loss_disc = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

    #         loss_G = loss_l2 + loss_lpips + loss_disc + proj_loss + loss_edge + loss_tv
    #         self.accelerator.backward(loss_G)
    #         if self.accelerator.sync_gradients:
    #             self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
    #         self.G_opt.step()
    #         self.G_opt.zero_grad()
    #     # Log something
    #     loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips, G_disc=loss_disc, G_repa=proj_loss, G_edge=loss_edge, G_tv=loss_tv)
    #     return loss_dict