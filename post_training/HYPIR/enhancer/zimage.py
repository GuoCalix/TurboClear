import os
import re
import torch
import loguru
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from HYPIR.enhancer.base import BaseEnhancer
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers import ZImageTransformer2DModel
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.utils.others import NoOpContext, EdgeDetectionModel, total_variation_loss

class ZImageEnhancer(BaseEnhancer):
    def init_models(self):
        self.init_vae()
        self.init_generator()

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, 
            subfolder="vae", 
            torch_dtype=self.weight_dtype
        ).to(self.device)

        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        loguru.logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)

    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()    

    def init_generator(self):
        self.dit = ZImageTransformer2DModel.from_pretrained(
            self.base_model_path, 
            subfolder="transformer", 
            torch_dtype=self.weight_dtype
        ).to(self.device)

        self.dit.eval()
        loguru.logger.info("✓ DiT model loaded")
        self._maybe_apply_lora()

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
            'to_q', 'to_k', 'to_v',
            'feedforward.w1', 'feedforward.w2', 'feedforward.w3'
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
        trainable_keys = set(n for n, p in self.dit.named_parameters() if p.requires_grad)
        real_missing = [k for k in missing if k in trainable_keys]
        if real_missing:
            loguru.logger.info(f"LoRA missing keys (trainable parameters that failed to load): {real_missing}")
        elif missing:
            loguru.logger.info(f"LoRA missing keys: {missing}")
        if unexpected:
            loguru.logger.info(f"LoRA unexpected keys: {unexpected}")
        loguru.logger.info("✓ LoRA weights loaded")


    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        prompt_embeds = torch.load("debug_inputs/prompt_embeds.pt")
        c_txt = {"prompt_embeds": [prompt_embeds]}
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt=c_txt,
            timesteps=timesteps,
        )

    def _denoise_step(self, latents, timesteps, prompt_embeds, timesteps_r=None):
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
        latent_model_input = latents.to(device=self.device, dtype=self.weight_dtype)
        timestep_model_input = timesteps.to(device=self.device)
        prompt_embeds_model_input = [embeds.to(self.device) for embeds in prompt_embeds]

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            noise_pred = self.dit(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )[0]

        return noise_pred

    def forward_generator(self, z_lq):
        print(f"{z_lq.shape=}")
        timesteps=torch.tensor([self.coeff_t]).to(dtype=torch.float32, device=self.device)
        sigmas=torch.tensor([self.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        latent_model_input = z_lq 
        t_expand = self.inputs["timesteps"] # [200, 0]
        t_expand = (1000 - t_expand) / 1000

        prompt_embeds = self.inputs["c_txt"]["prompt_embeds"]
        noise_pred = self._denoise_step(
            latent_model_input, t_expand, prompt_embeds, timesteps_r=None
        )
        noise_pred = torch.stack([t.float() for t in noise_pred], dim=0)
        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        latents = self.step(z_lq, noise_pred, sigmas, 0)
        return latents