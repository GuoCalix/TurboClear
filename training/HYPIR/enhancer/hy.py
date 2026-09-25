import os
import re
import torch
import loguru
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from HYPIR.enhancer.base import BaseEnhancer
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
from hyimage.models.hunyuan.modules.hunyuanimage_dit import load_hunyuan_dit_state_dict
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

class HunyuanEnhancer(BaseEnhancer):
    def init_models(self):
        self.init_vae()
        self.init_generator()
        self.init_text_models()
        self.init_byt5()

    def init_vae(self):
        vae_config = HUNYUANIMAGE_V2_1_VAE_32x()
        self.vae = instantiate(
            vae_config.model,
            vae_path=vae_config.load_from,
        )
        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        loguru.logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)
    
    def init_text_models(self):
        text_encoder_config = HUNYUANIMAGE_V2_1_TEXT_ENCODER()
        if not text_encoder_config.load_from:
            raise ValueError("Must provide checkpoint path for text encoder")

        if text_encoder_config.prompt_template is not None:
            prompt_template = PROMPT_TEMPLATE[text_encoder_config.prompt_template]
            crop_start = prompt_template.get("crop_start", 0)
        else:
            crop_start = 0
            prompt_template = None
        max_length = text_encoder_config.text_len + crop_start

        self.text_encoder = instantiate(
            text_encoder_config.model,
            max_length=max_length,
            text_encoder_path=os.path.join(text_encoder_config.load_from, "llm"),
            prompt_template=prompt_template,
            logger=None,
            device=self.device,
        )
        self.text_encoder.eval().requires_grad_(False)

        loguru.logger.info("✓ HunyuanImage text encoder loaded")

    def init_byt5(self):
        assert self.dit is not None, "DiT model must be loaded before byT5"
        text_encoder_config = HUNYUANIMAGE_V2_1_TEXT_ENCODER()

        glyph_root = os.path.join(text_encoder_config.load_from, "Glyph-SDXL-v2")
        if not os.path.exists(glyph_root):
            raise RuntimeError(
                f"Glyph checkpoint not found from '{glyph_root}'. \n"
                "Please download from https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files.\n\n"
                "- Required files:\n"
                "    Glyph-SDXL-v2\n"
                "    ├── assets\n"
                "    │   ├── color_idx.json\n"
                "    │   └── multilingual_10-lang_idx.json\n"
                "    └── checkpoints\n"
                "        └── byt5_model.pt\n"
            )
                

        byT5_google_path = os.path.join(text_encoder_config.load_from, "byt5-small")
        if not os.path.exists(byT5_google_path):
            loguru.logger.warning(f"ByT5 google path not found from: {byT5_google_path}. Try downloading from https://huggingface.co/google/byt5-small.")
            byT5_google_path = "google/byt5-small"


        multilingual_prompt_format_color_path = os.path.join(glyph_root, "assets/color_idx.json")
        multilingual_prompt_format_font_path = os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json")

        byt5_args = dict(
            byT5_google_path=byT5_google_path,
            byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
            multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
            multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
            byt5_max_length=128
        )

        self.byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=self.device)
        self.prompt_format = MultilingualPromptFormat(
            font_path=multilingual_prompt_format_font_path,
            color_path=multilingual_prompt_format_color_path
        )
        loguru.logger.info("✓ byT5 glyph processor loaded")

    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()    

    def init_generator(self):
        dit_config = HUNYUANIMAGE_V2_1_DIT()

        self.dit = instantiate(dit_config.model, dtype=self.weight_dtype, device=self.device)
        load_hunyuan_dit_state_dict(self.dit, dit_config.load_from, strict=True)
        self.dit = self.dit.to(self.device, dtype=self.weight_dtype)
        self.dit.eval()
        if getattr(dit_config, "use_compile", False):
            self.dit = torch.compile(self.dit)
        loguru.logger.info("✓ DiT model loaded")

        # Optionally attach LoRA and load minimal checkpoint if provided
        self._maybe_apply_lora()

        # self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        #     self.base_model_path, subfolder="unet", weight_dtype=self.weight_dtype).to(self.device)
        # target_modules = self.lora_modules
        # G_lora_cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_rank,
        #     init_lora_weights="gaussian", target_modules=target_modules)
        # self.G.add_adapter(G_lora_cfg)

        # print(f"Load model weights from {self.weight_path}")
        # state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        # self.G.load_state_dict(state_dict, strict=False)
        # input_keys = set(state_dict.keys())
        # required_keys = set([k for k in self.G.state_dict().keys() if "lora" in k])
        # missing = required_keys - input_keys
        # unexpected = input_keys - required_keys
        # assert required_keys == input_keys, f"Missing: {missing}, Unexpected: {unexpected}"

        self.dit.eval().requires_grad_(False)

    def _infer_lora_targets_and_rank(self, state_dict):
        """Infer LoRA target module suffixes and rank from a minimal checkpoint state_dict.

        Heuristic:
        - For keys like ...mlp.fc1.lora_A..., infer target 'mlp.fc1'
        - For keys like ...linear2.fc.lora_A..., infer 'linear2.fc'
        - For single-segment modules like 'attn_q', infer that segment
        - Rank comes from the first non-empty lora_A weight's first dim
        """
        targets = set()
        inferred_rank = None
        for k, v in state_dict.items():
            if not isinstance(k, str):
                continue
            parts = k.split(".")
            if "lora_A" in parts or "lora_B" in parts:
                idx = parts.index("lora_A") if "lora_A" in parts else parts.index("lora_B")
                # Compose a two-level suffix if available
                if idx >= 2:
                    module_name = parts[idx - 2] + "." + parts[idx - 1]
                elif idx >= 1:
                    module_name = parts[idx - 1]
                else:
                    continue
                targets.add(module_name)
                if inferred_rank is None and isinstance(v, torch.Tensor) and v.numel() > 0:
                    # lora_A: (r, in_features); lora_B: (out_features, r)
                    inferred_rank = int(v.shape[0]) if parts[idx] == "lora_A" else int(v.shape[1])
        # Fallback to common defaults if nothing inferred
        default_targets = [
            "attn_q", "attn_k", "attn_v", "attn_proj",
            "linear1_q", "linear1_k", "linear1_v", "linear1_mlp",
            "linear2.fc", "mlp.fc1", "mlp.fc2", "final_layer.linear",
        ]
        if not targets:
            targets = set(default_targets)
        if inferred_rank is None or inferred_rank <= 0:
            inferred_rank = max(8, getattr(self, "lora_rank", 8) or 8)
        return sorted(targets), int(inferred_rank)

    def _maybe_apply_lora(self):
        """Attach PEFT LoRA modules and load weights from minimal checkpoint if available."""
        weight_path = getattr(self, "weight_path", None)
        if not weight_path:
            loguru.logger.info("No weight_path provided; skip LoRA loading.")
            return
        # Resolve state dict file
        state_file = None
        if os.path.isdir(weight_path):
            cand = os.path.join(weight_path, "state_dict.pth")
            if os.path.exists(cand):
                state_file = cand
        elif os.path.isfile(weight_path):
            state_file = weight_path
        if state_file is None or not os.path.exists(state_file):
            loguru.logger.warning(f"LoRA state file not found under {weight_path}; skip LoRA loading.")
            return

        loguru.logger.info(f"Loading LoRA minimal checkpoint from: {state_file}")
        state_dict = torch.load(state_file, map_location="cpu")

        # Determine targets and rank: infer from checkpoint when CLI gives None/0
        ckpt_targets, ckpt_rank = self._infer_lora_targets_and_rank(state_dict)
        if self.lora_modules and self.lora_modules != ["None"] and self.lora_modules != [None]:
            target_modules = self.lora_modules
        else:
            target_modules = ckpt_targets
        rank = int(self.lora_rank) if getattr(self, "lora_rank", 0) else ckpt_rank
        rank = max(1, rank)

        loguru.logger.info(f"Attaching LoRA with rank={rank}, targets={target_modules}")
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.dit = get_peft_model(self.dit, lora_cfg)
        self.dit.to(self.device)
        # Filter checkpoint to matching keys and shapes to avoid size mismatch
        model_state = self.dit.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state:
                try:
                    if isinstance(v, torch.Tensor) and v.numel() > 0 and model_state[k].shape == v.shape:
                        filtered[k] = v
                except Exception:
                    continue
        missing_keys = [k for k in state_dict.keys() if k not in filtered]
        if missing_keys:
            loguru.logger.info(f"Filtered out {len(missing_keys)} keys due to shape/name mismatch; loading {len(filtered)} keys.")
        # Load LoRA weights (trainable params) into wrapped model
        load_result = self.dit.load_state_dict(filtered, strict=False)
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        if missing:
            loguru.logger.info(f"LoRA missing keys: {missing}")
        if unexpected:
            loguru.logger.info(f"LoRA unexpected keys: {unexpected}")
        loguru.logger.info("✓ LoRA weights loaded")

    def _encode_text(self, prompt: str, data_type: str = "image"):
        """
        Encode text prompt to embeddings.

        Args:
            prompt: The text prompt
            data_type: The type of data ("image" by default)

        Returns:
            Tuple of (text_emb, text_mask)
        """
        self.text_encoder.to(self.device)
        text_inputs = self.text_encoder.text2tokens(prompt)
        with torch.no_grad():
            text_outputs = self.text_encoder.encode(
                text_inputs,
                data_type=data_type,
            )
            text_emb = text_outputs.hidden_state
            text_mask = text_outputs.attention_mask
        return text_emb, text_mask
    
    def _encode_glyph(self, prompt: str):
        """
        Encode glyph information using byT5.

        Args:
            prompt: The text prompt

        Returns:
            Tuple of (byt5_emb, byt5_mask)
        """
        if not prompt:
            return (
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"], 1472), device=self.device),
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"]), device=self.device, dtype=torch.int64)
            )

        text_prompt_texts = []
        pattern_quote_double = r'\"(.*?)\"'
        pattern_quote_chinese_single = r'‘(.*?)’'
        pattern_quote_chinese_double = r'“(.*?)”'

        matches_quote_double = re.findall(pattern_quote_double, prompt)
        matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
        matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

        text_prompt_texts.extend(matches_quote_double)
        text_prompt_texts.extend(matches_quote_chinese_single)
        text_prompt_texts.extend(matches_quote_chinese_double)

        if not text_prompt_texts:
            self.ocr_mask = [False]
            return (
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"], 1472), device=self.device),
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"]), device=self.device, dtype=torch.int64)
            )
        self.ocr_mask = [True]

        text_prompt_style_list = [{'color': None, 'font-family': None} for _ in range(len(text_prompt_texts))]
        glyph_text_formatted = self.prompt_format.format_prompt(text_prompt_texts, text_prompt_style_list)

        byt5_text_ids, byt5_text_mask = self._get_byt5_text_tokens(
            self.byt5_kwargs["byt5_tokenizer"],
            self.byt5_kwargs["byt5_max_length"],
            glyph_text_formatted
        )

        byt5_text_ids = byt5_text_ids.to(device=self.device)
        byt5_text_mask = byt5_text_mask.to(device=self.device)

        byt5_prompt_embeds = self.byt5_kwargs["byt5_model"](
            byt5_text_ids, attention_mask=byt5_text_mask.float()
        )
        byt5_emb = byt5_prompt_embeds[0]

        return byt5_emb, byt5_text_mask

    def _get_byt5_text_tokens(self, tokenizer, max_length, text_list):
        """
        Get byT5 text tokens.

        Args:
            tokenizer: The tokenizer object
            max_length: Maximum token length
            text_list: List or string of text

        Returns:
            Tuple of (byt5_text_ids, byt5_text_mask)
        """
        if isinstance(text_list, list):
            text_prompt = " ".join(text_list)
        else:
            text_prompt = text_list

        byt5_text_inputs = tokenizer(
            text_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        byt5_text_ids = byt5_text_inputs.input_ids
        byt5_text_mask = byt5_text_inputs.attention_mask

        return byt5_text_ids, byt5_text_mask

    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        pos_text_emb, pos_text_mask = self._encode_text(prompt)
        pos_byt5_emb, pos_byt5_mask = self._encode_glyph(prompt)
        # save pos_text_emb, pos_text_mask, pos_byt5_emb, pos_byt5_mask for use in forward
        os.makedirs("debug_inputs", exist_ok=True)
        torch.save(pos_text_emb.cpu(), f"debug_inputs/pos_text_emb.pt")
        torch.save(pos_text_mask.cpu(), f"debug_inputs/pos_text_mask.pt")
        torch.save(pos_byt5_emb.cpu(), f"debug_inputs/pos_byt5_emb.pt")
        torch.save(pos_byt5_mask.cpu(), f"debug_inputs/pos_byt5_mask.pt")
        print(f"Saved debug inputs to debug_inputs/")
        print(f"{pos_text_emb.shape=}, {pos_text_mask.shape=}, {pos_byt5_emb.shape=}, {pos_byt5_mask.shape=}")
        print(f"{pos_text_emb=}")
        print(f"{pos_text_mask=}")
        print(f"{pos_byt5_emb=}")
        print(f"{pos_byt5_mask=}")

        c_txt = {"text_emb": pos_text_emb, "text_mask": pos_text_mask, 
                 "byt5_emb": pos_byt5_emb, "byt5_mask": pos_byt5_mask}
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt=c_txt,
            timesteps=timesteps,
        )

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
        if byt5_emb is not None and byt5_mask is not None:
            extra_kwargs = {
                "byt5_text_states": byt5_emb,
                "byt5_text_mask": byt5_mask,
            }
        else:
            if self.use_byt5:
                raise ValueError("Must provide byt5_emb and byt5_mask for HunyuanImage 2.1")
            extra_kwargs = {}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            guidance_expand = None

            noise_pred = self.dit(
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

    def forward_generator(self, z_lq):
        print(f"{z_lq.shape=}")
        sampling_steps = 1
        shift = 1
        timesteps=torch.tensor([self.coeff_t]).to(dtype=torch.float32, device=self.device)
        sigmas=torch.tensor([self.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        latent_model_input = z_lq 
        print(f"{latent_model_input.shape=}, {z_lq.max()=}, {z_lq.min()=}") 
        t_expand = self.inputs["timesteps"] # [200, 0]

        text_emb = self.inputs["c_txt"]["text_emb"]
        text_mask = self.inputs["c_txt"]["text_mask"]
        byt5_emb = self.inputs["c_txt"]["byt5_emb"]
        byt5_mask = self.inputs["c_txt"]["byt5_mask"]

        noise_pred = self._denoise_step(
            latent_model_input, t_expand, text_emb, text_mask, byt5_emb, byt5_mask, timesteps_r=None
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
