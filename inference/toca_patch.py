import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

# ToCa parameters container
class ToCaConfig:
    def __init__(self):
        # Step-wise cache: lists of 1-indexed step numbers where each component is cached/skipped
        self.cache_sa_steps = []   # Steps to skip Self-Attention (use cached output)
        self.cache_ca_steps = []   # Steps to skip Cross-Attention (use cached output)
        self.cache_mlp_steps = []  # Steps to use selective MLP token refresh
        self.cache_cnn_steps = []  # Steps to skip CNN/ResNet (use cached output)

        # MLP selective token refresh ratio (used with cache_mlp_steps)
        self.cache_ratio = 0.5    # Ratio of tokens to cache (0.0 = no caching, 1.0 = cache all)

        # Score evaluation weights for toca_cutfresh (MLP selective refresh)
        self.s1_weight = 1.0     # Self-attention score weight
        self.s2_weight = 1.0     # Cross-attention score weight
        self.s3_weight = 0.5     # Cache frequency weight
        self.s4_bonus = 0.6      # Spatial uniform bonus
        self.grid_size = 2       # Grid size for spatial uniform bonus


# Global Cache Dictionary
toca_cache_dic = {
    'cache': {},          # For storing hidden states
    'attn_score1': {},    # For storing s1 score (self-attention sums)
    'attn_score2': {},    # For storing s2 score (cross-attention sums)
    'cache_counter': {},  # For tracking how long tokens have been cached (s3)
    'step': 0,
}

def toca_score_evaluate(hidden_states, block_name, config: ToCaConfig):
    B, N, C = hidden_states.shape

    # Extract s1 (Self-Attn Score)
    s1 = toca_cache_dic['attn_score1'].get(block_name, torch.ones(B, N, device=hidden_states.device))
    if s1.shape[1] != N:
        s1 = torch.ones(B, N, device=hidden_states.device)
    s1 = F.normalize(s1, dim=-1, p=2)

    # Extract s2 (Cross-Attn Score)
    s2 = toca_cache_dic['attn_score2'].get(block_name, torch.ones(B, N, device=hidden_states.device))
    if s2.shape[1] != N:
        s2 = torch.ones(B, N, device=hidden_states.device)
    s2 = F.normalize(s2, dim=-1, p=2)

    # Combine Attention Scores
    score = config.s1_weight * s1 + config.s2_weight * s2

    # s3 (Cache Frequency Score)
    # Penalize tokens that have been cached for a long time
    counter = toca_cache_dic['cache_counter'].get(block_name, torch.zeros(B, N, device=hidden_states.device))
    soft_step_score = counter.float() / max(1, len(config.cache_mlp_steps))
    score = score + config.s3_weight * soft_step_score

    # s4 (Uniform Spatial Distribution Bonus)
    grid_size = config.grid_size
    H = int(N**0.5)
    if H * H == N and H % grid_size == 0:
        # local_selection_with_bonus
        block_size = grid_size * grid_size
        score_reshaped = score.view(B, H // grid_size, grid_size, H // grid_size, grid_size)
        score_reshaped = score_reshaped.permute(0, 1, 3, 2, 4).contiguous()
        score_reshaped = score_reshaped.view(B, -1, block_size)

        max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)
        mask = torch.zeros_like(score_reshaped)
        mask.scatter_(-1, max_indices, 1)

        score_reshaped = score_reshaped + (mask * max_scores * config.s4_bonus)

        score_modified = score_reshaped.view(B, H // grid_size, H // grid_size, grid_size, grid_size)
        score_modified = score_modified.permute(0, 1, 3, 2, 4).contiguous()
        score = score_modified.view(B, N)

    return score


def toca_cutfresh(hidden_states, block_name, config: ToCaConfig):
    B, N, C = hidden_states.shape

    fresh_ratio = 1.0 - config.cache_ratio
    topk = max(1, int(fresh_ratio * N))

    # [FVCORE Hot-fix] Bypasses un-traceable dynamic ranking logic when checking tracing flags
    if toca_cache_dic.get('is_tracing', False):
        fresh_indices = torch.arange(topk, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        fresh_tokens = hidden_states[:, :topk, :]
        return fresh_indices, None, fresh_tokens

    score = toca_score_evaluate(hidden_states, block_name, config)
    indices = score.argsort(dim=-1, descending=True)

    fresh_indices = indices[:, :topk]
    stale_indices = indices[:, topk:]

    # Update cache counter
    counter = toca_cache_dic['cache_counter'].get(block_name, torch.zeros(B, N, dtype=torch.int32, device=hidden_states.device))
    counter += 1
    # Reset fresh tokens counter to 0
    counter.scatter_(dim=1, index=fresh_indices, src=torch.zeros_like(fresh_indices, dtype=torch.int32, device=fresh_indices.device))
    toca_cache_dic['cache_counter'][block_name] = counter

    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, C)
    fresh_tokens = torch.gather(input=hidden_states, dim=1, index=fresh_indices_expand)

    return fresh_indices, stale_indices, fresh_tokens


class ToCaAttnProcessor:
    """Attention processor that records attention scores for toca_cutfresh.

    This processor is only called when the corresponding attention module is NOT
    cached (i.e. on full-computation steps).  It always uses the slow
    head-to-batch-dim path so that:
      1. Attention scores are recorded for ``toca_cutfresh`` (MLP selective refresh).
      2. AGF hooks (``get_attention_scores``) work correctly.
    """

    def __init__(self, block_name, is_cross_attn):
        self.block_name = block_name
        self.is_cross_attn = is_cross_attn

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Always use full computation with score recording
        query_3d = attn.head_to_batch_dim(query)
        key_3d = attn.head_to_batch_dim(key)
        value_3d = attn.head_to_batch_dim(value)

        if hasattr(attn, "get_attention_scores"):
            attention_probs = attn.get_attention_scores(query_3d, key_3d, attention_mask)
        else:
            attention_scores = torch.bmm(query_3d, key_3d.transpose(-1, -2)) * attn.scale
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = attention_scores.softmax(dim=-1)

        # Record attention scores for toca_cutfresh
        B_H, N_q, N_k = attention_probs.shape
        B = B_H // attn.heads
        att_probs_4d = attention_probs.view(B, attn.heads, N_q, N_k)

        if self.is_cross_attn:
            score2 = att_probs_4d.sum(dim=(1, 3))
            toca_cache_dic['attn_score2'][self.block_name] = score2
        else:
            score1 = att_probs_4d.sum(dim=(1, 2))
            toca_cache_dic['attn_score1'][self.block_name] = score1

        hidden_states = torch.bmm(attention_probs, value_3d)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def patch_module(module, name, config: ToCaConfig):
    from diffusers.models.attention import BasicTransformerBlock
    from diffusers.models.attention_processor import Attention
    from diffusers.models.resnet import ResnetBlock2D

    for n, sub_module in module.named_modules():
        full_name = f"{name}.{n}"
        if isinstance(sub_module, BasicTransformerBlock):

            # Wrap standard forward
            old_forward = sub_module.forward

            def make_forward(block_name, old_fw, block_mod):
                def cached_forward(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=None,
                    cross_attention_kwargs=None,
                    class_labels=None,
                    **kwargs,
                ):
                    current_step = toca_cache_dic['step']
                    # DYNAMICALLY fetch the latest config to support instant parameter swapping in persistent sessions!
                    active_config = toca_cache_dic.get('config', config)

                    cache_sa_steps = getattr(active_config, 'cache_sa_steps', [])
                    cache_ca_steps = getattr(active_config, 'cache_ca_steps', [])
                    cache_mlp_steps = getattr(active_config, 'cache_mlp_steps', [])
                    cache_ratio = getattr(active_config, 'cache_ratio', 0.0)

                    # 1-indexed step number for comparison with user-specified step lists
                    step_1indexed = current_step + 1

                    skip_sa = step_1indexed in cache_sa_steps
                    skip_ca = step_1indexed in cache_ca_steps
                    skip_mlp = step_1indexed in cache_mlp_steps and cache_ratio > 0.0

                    # Force full round for the specific Attention Guided Fusion (AGF)
                    # block to preserve cross-attention pipeline hooks
                    is_target_agf_block = 'down_blocks.1.attentions.0.transformer_blocks.0' in block_name
                    if is_target_agf_block:
                        skip_sa = False
                        skip_ca = False

                    def get_hook(key):
                        def hook(module, args, output):
                            toca_cache_dic['cache'][block_name + key] = output
                        return hook

                    hooks = []
                    old_attn1_fw = block_mod.attn1.forward
                    old_attn2_fw = None
                    if getattr(block_mod, 'attn2', None) is not None:
                        old_attn2_fw = block_mod.attn2.forward
                    old_ff_fw = block_mod.ff.forward

                    try:
                        # --- Self-Attention ---
                        if skip_sa:
                            def cached_attn1_fw(*args, **kw):
                                x = args[0]
                                c = toca_cache_dic['cache'][block_name + '_attn1']
                                return c + (x.flatten()[0] * 0.0).to(x.dtype)
                            block_mod.attn1.forward = cached_attn1_fw
                        else:
                            hooks.append(block_mod.attn1.register_forward_hook(get_hook('_attn1')))

                        # --- Cross-Attention ---
                        if old_attn2_fw is not None:
                            if skip_ca:
                                def cached_attn2_fw(*args, **kw):
                                    x = args[0]
                                    c = toca_cache_dic['cache'][block_name + '_attn2']
                                    return c + (x.flatten()[0] * 0.0).to(x.dtype)
                                block_mod.attn2.forward = cached_attn2_fw
                            else:
                                hooks.append(block_mod.attn2.register_forward_hook(get_hook('_attn2')))

                        # --- MLP / Feed-Forward ---
                        if skip_mlp:
                            def selective_ff_fw(ff_hidden_states, *args, **kw):
                                fresh_indices, _, fresh_tokens = toca_cutfresh(ff_hidden_states, block_name, active_config)
                                fresh_ff_out = old_ff_fw(fresh_tokens, *args, **kw)
                                stale_ff_out = toca_cache_dic['cache'][block_name + '_ff']
                                final_ff_out = stale_ff_out.clone()
                                fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, stale_ff_out.shape[-1])
                                final_ff_out.scatter_(dim=1, index=fresh_indices_expand, src=fresh_ff_out)
                                return final_ff_out
                            block_mod.ff.forward = selective_ff_fw
                        else:
                            hooks.append(block_mod.ff.register_forward_hook(get_hook('_ff')))

                        # Execute native `.forward`! It will correctly route into
                        # our overrides OR hooks!
                        res = old_fw(
                            hidden_states,
                            attention_mask=attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            timestep=timestep,
                            cross_attention_kwargs=cross_attention_kwargs,
                            class_labels=class_labels,
                            **kwargs
                        )
                    finally:
                        for h in hooks:
                            h.remove()
                        block_mod.attn1.forward = old_attn1_fw
                        if old_attn2_fw is not None:
                            block_mod.attn2.forward = old_attn2_fw
                        block_mod.ff.forward = old_ff_fw

                    return res

                return cached_forward

            sub_module.forward = make_forward(full_name, old_forward, sub_module)

            # Add custom processor to capture Attention Scores during FULL rounds
            if hasattr(sub_module, 'attn1'):
                sub_module.attn1.set_processor(ToCaAttnProcessor(full_name, is_cross_attn=False))
            if hasattr(sub_module, 'attn2'):
                sub_module.attn2.set_processor(ToCaAttnProcessor(full_name, is_cross_attn=True))

        elif isinstance(sub_module, ResnetBlock2D) and len(getattr(config, 'cache_cnn_steps', [])) > 0:
            old_forward = sub_module.forward

            def make_resnet_forward(block_name, old_fw, block_mod):
                def cached_forward(*args, **kwargs):
                    current_step = toca_cache_dic['step']
                    # Dynamically fetch the latest config
                    active_config = toca_cache_dic.get('config', config)
                    cache_cnn_steps = getattr(active_config, 'cache_cnn_steps', [])

                    step_1indexed = current_step + 1
                    skip_cnn = step_1indexed in cache_cnn_steps

                    if not skip_cnn:
                        # Full round: run normally and save conv outputs for future cached steps
                        def get_hook(key):
                            def hook(module, a, output):
                                toca_cache_dic['cache'][block_name + key] = output
                            return hook

                        hooks = []
                        if getattr(block_mod, 'conv1', None) is not None:
                            hooks.append(block_mod.conv1.register_forward_hook(get_hook('_conv1')))
                        if getattr(block_mod, 'conv2', None) is not None:
                            hooks.append(block_mod.conv2.register_forward_hook(get_hook('_conv2')))
                        if getattr(block_mod, 'conv_shortcut', None) is not None:
                            hooks.append(block_mod.conv_shortcut.register_forward_hook(get_hook('_conv_shortcut')))

                        res = old_fw(*args, **kwargs)

                        for h in hooks:
                            h.remove()
                        return res
                    else:
                        # Cache round: skip convolutions, inject cached outputs
                        old_c1 = getattr(block_mod, 'conv1', None)
                        old_c1_fw = old_c1.forward if old_c1 else None

                        old_c2 = getattr(block_mod, 'conv2', None)
                        old_c2_fw = old_c2.forward if old_c2 else None

                        old_cs = getattr(block_mod, 'conv_shortcut', None)
                        old_cs_fw = old_cs.forward if old_cs else None

                        try:
                            if old_c1_fw is not None:
                                def cached_c1_fw(*a, **kw):
                                    x = a[0]
                                    c = toca_cache_dic['cache'][block_name + '_conv1']
                                    return c + (x.flatten()[0] * 0.0).to(x.dtype)
                                block_mod.conv1.forward = cached_c1_fw
                            if old_c2_fw is not None:
                                def cached_c2_fw(*a, **kw):
                                    x = a[0]
                                    c = toca_cache_dic['cache'][block_name + '_conv2']
                                    return c + (x.flatten()[0] * 0.0).to(x.dtype)
                                block_mod.conv2.forward = cached_c2_fw
                            if old_cs_fw is not None:
                                def cached_cs_fw(*a, **kw):
                                    x = a[0]
                                    c = toca_cache_dic['cache'][block_name + '_conv_shortcut']
                                    return c + (x.flatten()[0] * 0.0).to(x.dtype)
                                block_mod.conv_shortcut.forward = cached_cs_fw

                            res = old_fw(*args, **kwargs)
                        finally:
                            if old_c1_fw is not None: block_mod.conv1.forward = old_c1_fw
                            if old_c2_fw is not None: block_mod.conv2.forward = old_c2_fw
                            if old_cs_fw is not None: block_mod.conv_shortcut.forward = old_cs_fw

                        return res
                return cached_forward

            sub_module.forward = make_resnet_forward(full_name, old_forward, sub_module)


def apply_toca_patch(pipe, cache_sa_steps=None, cache_ca_steps=None,
                     cache_mlp_steps=None, cache_cnn_steps=None, cache_ratio=0.5):
    """Apply step-wise caching patch to a diffusers pipeline.

    Args:
        pipe: A diffusers pipeline with a ``.unet`` attribute.
        cache_sa_steps: List of 1-indexed step numbers to cache Self-Attention.
        cache_ca_steps: List of 1-indexed step numbers to cache Cross-Attention.
        cache_mlp_steps: List of 1-indexed step numbers to use selective MLP refresh.
        cache_cnn_steps: List of 1-indexed step numbers to cache CNN.
        cache_ratio: Ratio of tokens to cache in MLP selective refresh.
    """
    config = ToCaConfig()
    config.cache_sa_steps = cache_sa_steps or []
    config.cache_ca_steps = cache_ca_steps or []
    config.cache_mlp_steps = cache_mlp_steps or []
    config.cache_cnn_steps = cache_cnn_steps or []
    config.cache_ratio = cache_ratio
    toca_cache_dic['config'] = config

    # Store old callback to wrap around
    def toca_callback(pipe, step_index, timestep, callback_kwargs):
        toca_cache_dic['step'] = step_index
        return callback_kwargs

    # Patch the UNet
    patch_module(pipe.unet, "unet", config)

    # Initialize dictionary
    toca_cache_dic['cache'].clear()
    toca_cache_dic['attn_score1'].clear()
    toca_cache_dic['attn_score2'].clear()
    toca_cache_dic['cache_counter'].clear()
    toca_cache_dic['step'] = 0

    return toca_callback
