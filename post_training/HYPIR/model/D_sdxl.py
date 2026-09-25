from contextlib import nullcontext
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from vision_aided_loss.cv_losses import multilevel_loss

try:
    from HYPIR.model.sd_unet_forward import classify_forward
except ImportError:
    classify_forward = None


class SDXLPredBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1),
            # nn.GroupNorm(num_groups=32, num_channels=1280),
            # nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0),
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.net(rep).squeeze(dim=[2, 3])


class SDXLInpaintDiscriminator(nn.Module):
    def __init__(
        self,
        model_id: str,
        precision: str = "bf16",
        renoise_max_timestep: int = 200,
        default_seq_len: int = 77,
        default_cross_attn_dim: int = 2048,
        default_text_embeds_dim: int = 1280,
        default_time_ids_dim: int = 6,
    ):
        super().__init__()
        if classify_forward is None:
            raise ImportError(
                "Cannot import classify_forward. Please ensure main.sd_unet_forward is on PYTHONPATH."
            )

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.unet.forward = types.MethodType(classify_forward, self.unet)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.unet.eval().requires_grad_(False) 

        self.pred_branch = SDXLPredBranch()
        self.loss_fn = multilevel_loss(alpha=0.8)

        self.renoise_max_timestep = renoise_max_timestep
        self.default_seq_len = default_seq_len
        self.default_cross_attn_dim = default_cross_attn_dim
        self.default_text_embeds_dim = default_text_embeds_dim
        self.default_time_ids_dim = default_time_ids_dim

        if precision == "fp16":
            self.compute_dtype = torch.float16
        elif precision == "bf16":
            self.compute_dtype = torch.bfloat16
        else:
            self.compute_dtype = torch.float32

    def train(self, mode=True):
        self.unet.eval()
        self.pred_branch.train(mode)
        return self

    def eval(self):
        self.train(False)
        return self

    def requires_grad_(self, requires_grad=True):
        self.unet.requires_grad_(False)
        self.pred_branch.requires_grad_(requires_grad)
        return self

    def _build_default_condition(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        encoder_hidden_states = torch.zeros(
            (batch_size, self.default_seq_len, self.default_cross_attn_dim), dtype=dtype, device=device
        )
        added_cond_kwargs = {
            "text_embeds": torch.zeros((batch_size, self.default_text_embeds_dim), dtype=dtype, device=device),
            "time_ids": torch.zeros((batch_size, self.default_time_ids_dim), dtype=dtype, device=device),
        }
        return encoder_hidden_states, added_cond_kwargs

    def forward(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        reference_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        added_cond_kwargs: dict = None,
        timesteps: torch.Tensor = None,
        for_real: bool = True,
        for_G: bool = False,
        verbose: bool = False,
        return_logits: bool = False,
    ):
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Expected mask shape [B, 1, H, W], got {tuple(mask.shape)}")
        if latents.ndim != 4:
            raise ValueError(f"Expected latents shape [B, C, H, W], got {tuple(latents.shape)}")
        if reference_latents.shape != latents.shape:
            raise ValueError(
                f"reference_latents must match latents shape, got {tuple(reference_latents.shape)} vs {tuple(latents.shape)}"
            )

        latents = latents.float()
        reference_latents = reference_latents.float()
        mask = mask.float()
        if mask.min() < 0:
            mask = (mask + 1.0) / 2.0
        mask = mask.clamp(0.0, 1.0)

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.renoise_max_timestep,
                (latents.shape[0],),
                device=latents.device,
                dtype=torch.long,
            )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        latent_mask = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
        masked_reference_latents = reference_latents
        model_input = torch.cat([noisy_latents, latent_mask, masked_reference_latents], dim=1)

        if encoder_hidden_states is None or added_cond_kwargs is None:
            default_encoder_hidden_states, default_added_cond_kwargs = self._build_default_condition(
                batch_size=latents.shape[0], dtype=model_input.dtype, device=model_input.device
            )
            if encoder_hidden_states is None:
                encoder_hidden_states = default_encoder_hidden_states
            if added_cond_kwargs is None:
                added_cond_kwargs = default_added_cond_kwargs

        device_type = "cuda" if model_input.is_cuda else "cpu"
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=self.compute_dtype)
            if self.compute_dtype != torch.float32
            else nullcontext()
        )
        with autocast_ctx:
            reps = self.unet.forward(
                model_input.to(self.compute_dtype),
                timesteps,
                encoder_hidden_states.to(self.compute_dtype),
                added_cond_kwargs={k: v.to(self.compute_dtype) for k, v in added_cond_kwargs.items()},
                classify_mode=True,
            )

        bottleneck_rep = reps[-1].float()
        # print(f"shape of bottleneck rep: {bottleneck_rep.shape}")
        logits = self.pred_branch(bottleneck_rep)
        logits_list = [logits]

        if verbose:
            for idx, rep in enumerate(reps):
                print(f"{idx}-th feature: {tuple(rep.shape)}")
            print(f"logits shape: {tuple(logits.shape)}")

        loss = self.loss_fn(logits_list, for_real=for_real, for_G=for_G)
        if not return_logits:
            return loss
        return loss, logits_list