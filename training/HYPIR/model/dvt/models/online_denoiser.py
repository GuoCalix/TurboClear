from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from torch import Tensor

from .vit_wrapper import PretrainedViTWrapper

ADALN_EMBED_DIM = 256

class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        out = self.adaLN_modulation(c)
        scale = 1.0 + out
        x = self.norm_final(x.to(out.dtype)) * scale.unsqueeze(1)
        x = self.linear(x)
        return x
    
class Denoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 64,
        noise_map_width: int = 64,
        feat_dim: int = 3840,
        vit: PretrainedViTWrapper = None,
        enable_pe: bool = True,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.vit = vit
        self.target_dim = 64
        self.denoiser = nn.Sequential(
            *[
                Block(
                    dim=feat_dim,
                    num_heads=feat_dim // 64,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_norm=False,
                    init_values=None,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    mlp_layer=Mlp,
                )
                for _ in range(num_blocks)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(feat_dim, ADALN_EMBED_DIM), mid_size=1024)

        self.final_layer = FinalLayer(feat_dim, self.target_dim)

        self.pos_embed = None
        if enable_pe:
            seq_len = noise_map_height * noise_map_width
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, feat_dim) * 0.02)

    def forward(
        self,
        x,
        t,
        return_dict=False,
        return_channel_first=False,
        return_class_token=False,
        norm=True,
    ):
        class_tokens = None

        t = t * 1000.0
        t = self.t_embedder(t)
        adaln_input = t

        b, hw, c = x.shape
        h = w = 64
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed(self.pos_embed, (h, w), num_prefix_tokens=0)
        
        x = self.denoiser(x)
        x = self.final_layer(x, adaln_input)

        x = x.reshape(b, h, w, -1)
        if return_channel_first:
            x = x.permute(0, 3, 1, 2)

        if return_class_token:
            assert class_tokens is not None
            return x, class_tokens
        return x
