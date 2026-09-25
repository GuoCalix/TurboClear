import os
import torch
import loguru
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model
from transformers import T5TokenizerFast, T5Tokenizer, T5EncoderModel
from safetensors.torch import load_file
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, T5Tokenizer, T5EncoderModel
from HYPIR.trainer.base import BaseTrainer, BatchInput

from HYPIR.enhancer.base import BaseEnhancer


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
        args["base_model_path"], subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        args["base_model_path"], subfolder="text_encoder_2", revision=None, variant=None
    )
    text_encoder_three = class_three.from_pretrained(
        args["base_model_path"], subfolder="text_encoder_3", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

class SD35Enhancer(BaseEnhancer):
    def init_models(self):
        self.init_vae()
        self.init_generator()
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
        # Standard SD VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype
        )
        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        loguru.logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)

    def init_text_models(self):
        # Tokenizer: prefer tokenizer_3 (T5) for SD3.5
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer",
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer_2",
        )
        try:
            self.tokenizer_3 = T5TokenizerFast.from_pretrained(
                self.base_model_path,
                subfolder="tokenizer_3",
            )
        except Exception as e:
            loguru.logger.warning(f"Could not load T5TokenizerFast, falling back to T5Tokenizer. Reason: {e}")
            self.tokenizer_3 = T5Tokenizer.from_pretrained(
                self.base_model_path,
                subfolder="tokenizer_3",

            )
        # Text encoder: SD3.5 medium commonly uses T5 in text_encoder_3
        self.text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.base_model_path, revision=None
        )
        self.text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.base_model_path, revision=None, subfolder="text_encoder_2"
        )
        self.text_encoder_cls_three = import_model_class_from_model_name_or_path(
            self.base_model_path, revision=None, subfolder="text_encoder_3"
        )
        args = dict(base_model_path=self.base_model_path)
        self.text_encoder, self.text_encoder_2, self.text_encoder_3 = load_text_encoders(
            self.text_encoder_cls_one, self.text_encoder_cls_two, self.text_encoder_cls_three, args=args
        )
        loguru.logger.info("✓ T5/ClipG/ClipL tokenizer + encoder loaded")

        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)

        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_3.to(self.device, dtype=self.weight_dtype)


    def init_generator(self):
        # SD3.5 transformer
        # self.dit = SD3Transformer2DModel.from_pretrained(
        #     self.base_model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        # )
        local_transformer_path = './models'

        # 使用 from_single_file 方法加载
        # 它需要一个预训练模型的路径或名称来获取正确的 config.json
        # 这里我们继续使用 self.base_model_path 来提供配置信息
        self.dit = SD3Transformer2DModel.from_pretrained(
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

        self.dit = self.dit.to(self.device, dtype=self.weight_dtype)
        loguru.logger.info("✓ SD3.5 DiT model loaded")

        # Optionally attach LoRA and load minimal checkpoint if provided (same workflow as hy.py)
        self.dit.eval().requires_grad_(False)

    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.dit.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            loguru.logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self.device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            loguru.logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]]=None,
        prompt_3: Union[str, List[str]]=None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
    ):
        device = device or self.device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access i

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        return prompt_embeds, pooled_prompt_embeds


    def prepare_inputs(self, batch_size: int, prompt: str):
        bs = batch_size
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # Save for debugging (optional)
        os.makedirs("debug_inputs", exist_ok=True)
        torch.save(prompt_embeds.cpu(), "debug_inputs/pos_prompt_embeds.pt")
        torch.save(pooled_prompt_embeds.cpu(), "debug_inputs/pos_pooled_prompt_embeds.pt")
        # torch.save(attn_mask.cpu(), "debug_inputs/pos_attn_mask.pt")
        loguru.logger.info("Saved debug inputs to debug_inputs/")

        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt={"text_embeds": prompt_embeds, "pooled_embeds": pooled_prompt_embeds},
            timesteps=timesteps,
        )

    def _denoise_step(self, latents, timesteps, prompt_embeds, pooled_prompt_embeds, timesteps_r=None):
        """
        One DiT forward to predict noise.
        """
        latents = latents.to(device=self.device, dtype=self.weight_dtype)
        timesteps = timesteps.to(device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.weight_dtype, enabled=True):
            noise_pred= self.dit(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            )[0]

        return noise_pred

    def forward_generator(self, z_lq: torch.Tensor):
        print(f"{z_lq.shape=}")
        sampling_steps = 1
        shift = 1
        timesteps=torch.tensor([self.coeff_t]).to(dtype=torch.float32, device=self.device)
        sigmas=torch.tensor([self.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        noise = torch.rand_like(z_lq, device=z_lq.device, dtype=z_lq.dtype)
        latent_model_input = (1- self.coeff_t / 1000.0) * z_lq + self.coeff_t / 1000.0 * noise 
        print(f"{latent_model_input.shape=}, {z_lq.max()=}, {z_lq.min()=}") 
        t_expand = self.inputs["timesteps"] # [200, 0]

        text_emb = self.inputs["c_txt"]["text_embeds"]
        pooled_emb = self.inputs["c_txt"]["pooled_embeds"]
        # text_mask = self.inputs["c_txt"]["text_mask"]
        # byt5_emb = self.inputs["c_txt"]["byt5_emb"]
        # byt5_mask = self.inputs["c_txt"]["byt5_mask"]

        noise_pred = self._denoise_step(
            latent_model_input, t_expand, text_emb, pooled_emb, timesteps_r=None
        )
        print(f"{t_expand=}, {sigmas=}")
        latents = self.step(z_lq, noise_pred, sigmas, 0)

        # z_in = z_lq * self.vae.config.scaling_factor
        # eps = self.G(
        #     z_in, self.inputs["timesteps"],
        #     encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
        # ).sample
        # z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        # z_out = z / self.vae.config.scaling_factor
        print(f"{latents.shape=}")
        print(f"{latents.max()=}, {latents.min()=}")
        return latents
