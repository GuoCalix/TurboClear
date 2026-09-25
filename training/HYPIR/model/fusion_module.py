import torch
import torch.nn as nn


class LearnableAttentionFusion(nn.Module):
    """Small alpha-calibration head for ObjectClear attention-guided fusion.

    The module treats the normalized object-token attention map as a prior and
    learns residual logit corrections for latent and pixel alpha maps. With the
    final convolution initialized to zero, the initial alpha equals the input
    attention map up to numerical clamping.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_channels: int = 32,
        num_layers: int = 3,
        logit_eps: float = 1e-4,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.logit_eps = float(logit_eps)
        in_channels = latent_channels * 3 + 2
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                ]
            )
        out = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

        self.latent_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.latent_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.pixel_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.pixel_logit_bias = nn.Parameter(torch.tensor(0.0))

    def _base_logit(self, attn_map: torch.Tensor) -> torch.Tensor:
        attn_map = attn_map.clamp(self.logit_eps, 1.0 - self.logit_eps)
        return torch.log(attn_map) - torch.log1p(-attn_map)

    def forward(
        self,
        z_pred: torch.Tensor,
        image_latents: torch.Tensor,
        mask_latent: torch.Tensor,
        attn_map: torch.Tensor,
    ):
        dtype = z_pred.dtype
        image_latents = image_latents.to(device=z_pred.device, dtype=dtype)
        mask_latent = mask_latent.to(device=z_pred.device, dtype=dtype)
        attn_map = attn_map.to(device=z_pred.device, dtype=dtype).clamp(0.0, 1.0)

        diff = (z_pred - image_latents).abs()
        features = torch.cat([z_pred, image_latents, diff, mask_latent, attn_map], dim=1)
        delta_latent, delta_pixel = self.net(features).chunk(2, dim=1)
        base_logit = self._base_logit(attn_map)

        alpha_latent = torch.sigmoid(
            self.latent_logit_scale.to(dtype=dtype) * base_logit
            + self.latent_logit_bias.to(dtype=dtype)
            + delta_latent
        )
        alpha_pixel = torch.sigmoid(
            self.pixel_logit_scale.to(dtype=dtype) * base_logit
            + self.pixel_logit_bias.to(dtype=dtype)
            + delta_pixel
        )
        z_fused = (1.0 - alpha_latent) * image_latents + alpha_latent * z_pred

        return {
            "z_fused": z_fused,
            "alpha_latent": alpha_latent,
            "alpha_pixel": alpha_pixel,
            "delta_latent": delta_latent,
            "delta_pixel": delta_pixel,
        }
