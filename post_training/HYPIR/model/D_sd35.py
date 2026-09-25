import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from vision_aided_loss.cv_losses import multilevel_loss
from HYPIR.model.sd3_backbone import SD3Transformer2DModel

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.scale).to(dtype)

class DiscriminatorHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        # 1. Single Learnable Token (Query)
        # 初始化为一个可学习的向量
        self.query_token = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Input Projections
        self.q_proj = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, input_dim, bias=False)

        self.norm_visual = RMSNorm(input_dim)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # 3. MLP (Figure 1: "MLP" block)
        # self.mlp_norm = RMSNorm(input_dim)
        self.mlp_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, visual_tokens):
        """
        visual_tokens: Image Features [B, N, C]
        """
        batch_size = visual_tokens.shape[0]

        query = self.query_token.expand(batch_size, -1, -1)

        visual_tokens = self.norm_visual(visual_tokens)

        q = self.q_proj(query)
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 4. QK RMSNorm
        # Apply Norm on the head_dim
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 5. Scaled Dot Product Attention
        # Transpose for torch functional attention: [B, Num_Heads, Len, Head_Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.inner_dim)

        attn_logit = self.o_proj(attn_out) + query

        logit = self.mlp(self.mlp_norm(attn_logit)) + attn_logit

        return logit.squeeze(1)

class SD3Discriminator(nn.Module):
    def __init__(
        self,
        sd3_model_path: str,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.backbone = SD3Transformer2DModel.from_pretrained_local(
            sd3_model_path, 
            subfolder="transformer",
            torch_dtype=dtype
        )

        self.loss_fn = multilevel_loss(alpha=0.8)
        # self.backbone.requires_grad_(False)
        # self.backbone.eval()

        self.backbone.enable_gradient_checkpointing()
        self.hidden_size = 1536 
        self.text_encoder_dim = 4096 
        
        self.target_layers = [11, 17, 23] 
        
        # 初始化多层 Heads
        self.heads = nn.ModuleList([
            DiscriminatorHead(input_dim=self.hidden_size)
            for _ in range(len(self.target_layers))
        ])
        
        concat_dim = self.hidden_size * len(self.target_layers)
        self.final_head = nn.Sequential(
            nn.Linear(concat_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size), # 图中写的是 LayerNorm
            nn.LeakyReLU(0.2, inplace=True), # 通常加个激活
            nn.Linear(self.hidden_size, 1)
        )

        self.register_buffer("image_mean", torch.tensor([0.5], dtype=torch.float32)) # Latent 均值通常接近0
        self.register_buffer("image_std", torch.tensor([0.5], dtype=torch.float32))

        self.feature_storage = {} # 用于临时存储 Hook 捕获的特征
        self._register_hooks()
    
    def _register_hooks(self):
        """
        为目标层注册 Forward Hook。
        SD3 的 transformer_blocks 是一个 ModuleList。
        """
        for idx in self.target_layers:
            # 确保索引在范围内
            if idx < len(self.backbone.transformer_blocks):
                layer = self.backbone.transformer_blocks[idx]
                layer.register_forward_hook(self._get_hook_fn(idx))
            else:
                print(f"Warning: Layer index {idx} out of bounds for backbone.")

    def _get_hook_fn(self, layer_idx):
        """
        创建 Hook 函数。
        JointTransformerBlock 的输出通常是 (encoder_hidden_states, hidden_states)
        我们需要的是 hidden_states (图像特征)。
        """
        def hook(module, input, output):
            # output 是一个 tuple: (text_feats, image_feats)
            if isinstance(output, tuple):
                # 取第二个元素作为图像特征
                self.feature_storage[layer_idx] = output[1]
            else:
                #以此防备万一某些层只返回一个 tensor
                self.feature_storage[layer_idx] = output
        return hook
       
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        for_real: bool = True,
        for_G: bool = False,
        return_logits: bool = True
    ):
        self.feature_storage.clear()

        # 1. 运行 Backbone 并获取中间层输出
        # diffusers 的 SD3 模型支持 output_hidden_states=True
        outputs = self.backbone(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=True,
        )
        if any(idx not in self.feature_storage for idx in self.target_layers):
            print("Missing features:", [idx for idx in self.target_layers if idx not in self.feature_storage])
        # outputs.hidden_states 包含了 (input_emb, layer_0, ... layer_23)
        # 索引映射: layer_i 的输出通常在 index i+1 (因为 index 0 是 embedding)
        # 具体需根据 diffusers 版本确认，通常 tuple 长度为 num_layers + 1

        head_outpus = []

        for i, layer_idx in enumerate(self.target_layers):
            
            if layer_idx not in self.feature_storage:
                raise ValueError(f"Layer {layer_idx} feature not captured. Check hooks.")
            print(f"Layer {layer_idx} feature found!")
            feature = self.feature_storage[layer_idx]
                    
            score = self.heads[i](visual_tokens=feature)
            head_outpus.append(score)

        concat_features = torch.cat(head_outpus, dim=-1)

        logit = self.final_head(concat_features)

        if not return_logits:
            return self.loss_fn(logit, for_real=for_real, for_G=for_G)
        else:
            return self.loss_fn(logit, for_real=for_real, for_G=for_G), logit
