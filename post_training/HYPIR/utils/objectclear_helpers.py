import gc
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

    unet_layer_names = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(unet_layer_names) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = unet_layer_names[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(query, key, attention_mask)
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.new_get_attention_scores = types.MethodType(make_new_get_attention_scores_fn(name), module)
            module.get_attention_scores = module.new_get_attention_scores

    return unet


def clear_cross_attention_scores(cross_attention_scores):
    keys = list(cross_attention_scores.keys())
    for key in keys:
        del cross_attention_scores[key]
    gc.collect()

def resize_attn_map_divide2(attn_map, mask, fuse_index):
    """Extract one token attention map and resize it to latent mask resolution."""
    bxh, num_noise_latents, _ = attn_map.shape
    batch_size = mask.shape[0]

    if bxh % batch_size != 0:
        raise ValueError(f"Unexpected attention shape {attn_map.shape} for batch size {batch_size}.")

    num_heads = bxh // batch_size
    size = int(num_noise_latents**0.5)
    if size * size != num_noise_latents:
        raise ValueError(f"num_noise_latents={num_noise_latents} is not a square number.")

    attn_map = attn_map.view(batch_size, num_heads, num_noise_latents, -1)
    index_tensor = torch.full(
        (batch_size, num_heads, num_noise_latents, 1),
        fuse_index,
        dtype=torch.long,
        device=attn_map.device,
    )
    attn_map = torch.gather(attn_map, dim=3, index=index_tensor).squeeze(-1)
    attn_map = attn_map.view(batch_size, num_heads, size, size)
    attn_map = F.interpolate(attn_map, size=mask.shape[-2:], mode="bilinear", antialias=True)

    attn_min = attn_map.amin(dim=(-2, -1), keepdim=True)
    attn_max = attn_map.amax(dim=(-2, -1), keepdim=True)
    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    return attn_map

class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=0.1, normalize=False, background_loss_weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize
        self.background_loss_weight = background_loss_weight

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )

        object_segmaps = (object_segmaps > self.threshold).to(object_segmaps.dtype)
        background_segmaps = 1 - object_segmaps

        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(dim=2) / background_segmaps_sum
        object_loss = (object_token_attn_prob * object_segmaps).sum(dim=2) / object_segmaps_sum

        return self.background_loss_weight * background_loss - object_loss + 1


def get_object_localization_loss_for_one_layer(cross_attention_scores, object_segmaps, loss_fn, fuse_index):
    bxh, num_noise_latents, _ = cross_attention_scores.shape
    batch_size, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    object_segmaps = F.interpolate(object_segmaps, size=(size, size), mode="bilinear", antialias=True)
    object_segmaps = object_segmaps.view(batch_size, max_num_objects, -1)

    num_heads = bxh // batch_size
    cross_attention_scores = cross_attention_scores.view(batch_size, num_heads, num_noise_latents, -1)

    index_tensor = torch.full(
        (batch_size, num_heads, num_noise_latents, 1),
        fuse_index,
        dtype=torch.long,
        device=cross_attention_scores.device,
    )
    object_token_attn_prob = torch.gather(cross_attention_scores, dim=3, index=index_tensor)

    object_segmaps = object_segmaps.permute(0, 2, 1).unsqueeze(1).expand(batch_size, num_heads, num_noise_latents, 1)
    loss = loss_fn(object_token_attn_prob, object_segmaps)
    return loss.mean()


def get_object_localization_loss(cross_attention_scores, object_segmaps, loss_fn, fuse_index):
    if len(cross_attention_scores) == 0:
        return torch.tensor(0.0, device=object_segmaps.device, dtype=object_segmaps.dtype)

    total = 0.0
    for _, layer_scores in cross_attention_scores.items():
        total = total + get_object_localization_loss_for_one_layer(
            layer_scores,
            object_segmaps,
            loss_fn,
            fuse_index,
        )
    return total / len(cross_attention_scores)

