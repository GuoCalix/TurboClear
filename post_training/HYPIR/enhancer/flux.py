import os
import torch
import loguru
from transformers import CLIPModel  
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model
from transformers import T5TokenizerFast, T5Tokenizer, T5EncoderModel
from safetensors.torch import load_file
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from HYPIR.model.transformer_sd3 import SD3Transformer2DModel
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, T5Tokenizer, T5EncoderModel
from HYPIR.trainer.base import BaseTrainer, BatchInput
from ..model.models.upsampling import Upsample2D
from HYPIR.enhancer.base import BaseEnhancer

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

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, args):
    text_encoder = class_one.from_pretrained(
        args["base_model_path"], subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_2 = class_two.from_pretrained(
        args["base_model_path"], subfolder="text_encoder_2", revision=None, variant=None
    )
    return text_encoder, text_encoder_2

class FluxEnhancer(BaseEnhancer):
    def init_models(self):
        self.init_vae()
        self.init_generator()
        # self.init_clip_model()
        self.init_text_models()
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.dit.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.dit.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype
        )
        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        loguru.logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)

        # Standard SD VAE
        # decoder_path = "./models"
        # self.vae = AutoencoderKL.from_pretrained(
        #     self.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype
        # )
        # state_dict = torch.load(decoder_path)

        # self.vae.decoder.load_state_dict(state_dict, strict=False)
        # loguru.logger.info("✓ Fine-tuned Decoder loaded")
        # self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        # loguru.logger.info("✓ VAE loaded")
        # self.vae.eval().requires_grad_(False)
        # config = AutoencoderKL.load_config(self.base_model_path, subfolder="vae")
        # self.vae = AutoencoderKL.from_config(config, torch_dtype=self.weight_dtype)
        # loguru.logger.info("Initialized VAE from config (without pre-trained weights).")

        # # 2. 手动替换 Decoder 中的上采样模块
        # loguru.logger.info("Replacing original upsamplers with PixelShuffle upsamplers.")
        # for up_block in self.vae.decoder.up_blocks:
        #     if hasattr(up_block, 'upsamplers') and up_block.upsamplers is not None:
        #         # 假设每个 up_block 只有一个 upsampler
        #         original_upsampler = up_block.upsamplers[0]
                
        #         # 创建一个新的使用 PixelShuffle 的 Upsampler
        #         # 我们需要从原始 upsampler 获取 in_channels 和 out_channels
        #         new_upsampler = Upsample2D(
        #             channels=original_upsampler.channels,
        #             out_channels=original_upsampler.out_channels,
        #             use_pixel_shuffle=True,
        #             name=original_upsampler.name
        #         )
        #         up_block.upsamplers[0] = new_upsampler
        # self.vae.to(self.device, dtype=self.weight_dtype)
        # # 3. 加载预训练模型的完整 state_dict 到内存
        # loguru.logger.info(f"Loading pretrained VAE weights from {self.base_model_path}")
        # pretrained_state_dict = AutoencoderKL.from_pretrained(
        #     self.base_model_path, subfolder="vae"
        # ).state_dict()

        # # 4. 筛选出 Encoder 的权重并加载
        # # 我们只加载键以 "encoder." 开头的权重
        # encoder_state_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder.")}
        
        # # 使用 load_state_dict 加载筛选后的权重。strict=False 是必要的，
        # # 因为我们故意忽略了 decoder 和其他部分的权重。
        # incompatible_keys = self.vae.load_state_dict(encoder_state_dict, strict=False)

        # decoder_path = "./models"
        # decoder_state_dict = torch.load(decoder_path)

        # self.vae.decoder.load_state_dict(decoder_state_dict, strict=False)
        # loguru.logger.info("✓ Fine-tuned Decoder loaded")
        
    def init_text_models(self):
        # Tokenizer: prefer tokenizer_3 (T5) for SD3.5
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer",
        )
        self.tokenizer_2 = T5Tokenizer.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer_2",
        )

        self.text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.base_model_path, revision=None
        )
        self.text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.base_model_path, revision=None, subfolder="text_encoder_2"
        )

        args = dict(base_model_path=self.base_model_path)
        self.text_encoder, self.text_encoder_2 = load_text_encoders(
            self.text_encoder_cls_one, self.text_encoder_cls_two, args=args
        )
        loguru.logger.info("✓ T5/Clip tokenizer + encoder loaded")

        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.device, dtype=self.weight_dtype)


    def init_generator(self):
        # SD3.5 transformer
        # self.dit = SD3Transformer2DModel.from_pretrained(
        #     self.base_model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        # )
        # local_transformer_path = self.weight_path + '/state_dict.pth'
        # print(f"Load Model from {local_transformer_path}")
        # 使用 from_single_file 方法加载
        # 它需要一个预训练模型的路径或名称来获取正确的 config.json
        # 这里我们继续使用 self.base_model_path 来提供配置信息
        self.dit = FluxTransformer2DModel.from_pretrained(
            self.base_model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=False,  # Set to False to instantiate the model fully in memory
        )

        # Step 2: Load the state dictionary from your local .safetensors file.
        # state_dict = torch.load(local_transformer_path)

        # Step 3: Load the weights into the model structure.
        # The `strict=True` (default) will ensure all keys match. If you have missing/extra keys (e.g., from LoRA),
        # you might need to adjust this or filter the state_dict.
        # self.dit.load_state_dict(state_dict, strict=False)
        # self.dit = SD3Transformer2DModel.from_pretrained(
        #     self.base_model_path,
        #     subfolder="transformer",
        #     torch_dtype=self.weight_dtype,
        #     low_cpu_mem_usage=False,  # Set to False to instantiate the model fully in memory
        # )
        
        self.dit = self.dit.to(self.device, dtype=self.weight_dtype)
        loguru.logger.info("✓ Flux-schnell DiT model loaded")

        # # Optionally attach LoRA and load minimal checkpoint if provided (same workflow as hy.py)
        self.dit.eval().requires_grad_(False)

    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.weight_dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            loguru.logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            loguru.logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self.device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(prompt)

        # Save for debugging (optional)
        os.makedirs("debug_inputs", exist_ok=True)
        torch.save(prompt_embeds.cpu(), "debug_inputs/flux_pos_prompt_embeds.pt")
        torch.save(pooled_prompt_embeds.cpu(), "debug_inputs/flux_pos_pooled_prompt_embeds.pt")
        # torch.save(attn_mask.cpu(), "debug_inputs/pos_attn_mask.pt")
        loguru.logger.info("Saved debug inputs to debug_inputs/")

        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt={"text_embeds": prompt_embeds, "pooled_embeds": pooled_prompt_embeds, 
            "text_ids": text_ids},
            timesteps=timesteps,
        )
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _denoise_step(self, latents, timesteps, timesteps_r=None):
        """
        One DiT forward to predict noise.
        """
        text_embeds = self.inputs["c_txt"]["text_embeds"]
        pooled_embeds = self.inputs["c_txt"]["pooled_embeds"]

        latents = latents.to(device=self.device, dtype=self.weight_dtype)
        timesteps = timesteps.to(device=self.device)
        txt_ids = self.inputs["c_txt"]["text_ids"].to(self.device)
        img_ids = self.inputs["c_txt"]["img_ids"].to(self.device)

        print(f"Shape of latents: {latents.shape}")
        print(f"Shape of text embeds: {text_embeds.shape}")
        print(f"Shape of pooled embeds: {pooled_embeds.shape}")
        print(f"Shape of txt ids: {txt_ids.shape}")
        print(f"Shape of img ids: {img_ids.shape}")

        if self.dit.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale=3.5, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.weight_dtype, enabled=True):
            noise_pred= self.dit(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=self.inputs["c_txt"]["text_embeds"],
                pooled_projections=self.inputs["c_txt"]["pooled_embeds"],
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=True,
                guidance=guidance,
            )[0]

        return noise_pred

    def forward_generator(self, z_lq: torch.Tensor):
        # 1. 获取当前潜空间图块的高度和宽度
        latent_h, latent_w = z_lq.shape[2], z_lq.shape[3]
        print(f"Shape of z_lq: {z_lq.shape}")
        # 2. 获取 Transformer 的 patch_size
        patch_size = self.dit.config.patch_size
        
        # 3. 计算 patch 的数量
        num_patches_h = latent_h // 2
        num_patches_w = latent_w // 2
        
        latent_image_ids = torch.zeros(num_patches_h, num_patches_w, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(num_patches_h)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(num_patches_w)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        
        # 6. 将其放入 self.inputs，以便 _denoise_step 可以访问
        self.inputs["c_txt"]["img_ids"] = latent_image_ids

        sigmas=torch.tensor([self.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device) 
        
        batch_size, num_channels_latents, height, width = z_lq.shape
        latents = self._pack_latents(z_lq, batch_size, num_channels_latents, height, width)

        print(f"{latents.shape=}, {z_lq.max()=}, {z_lq.min()=}") 
        t_expand = self.inputs["timesteps"] # [200, 0]

        noise_pred = self._denoise_step(
            latents, t_expand, timesteps_r=None
        )

        print(f"{t_expand=}, {sigmas=}")
        latents = self.step(latents, noise_pred, sigmas, 0)
        print(f"{latents.shape=}")
        print(f"{latents.max()=}, {latents.min()=}")
        latents = self._unpack_latents(latents, height, width)

        
        return latents
