"""
Fusion-guided token pruning for ObjectClear SDXL UNet.

Phase 2: Prunes background tokens across the entire UNet.
- Step 0: full computation, cache all block outputs
- Steps 1+: compute normally up to DB1's first cross-attention,
  generate pruning mask, downsample to lower resolutions,
  then apply asymmetric attention + fg-only FF everywhere.

Patched blocks:
  DB1  (640ch,  H/2):  attentions[0].tb[0] = mask gen, rest pruned
  DB2  (1280ch, H/4):  all pruned (mask downsampled 2x)
  Mid  (1280ch, H/4):  all pruned
  UB0  (1280ch, H/4):  all pruned
  UB1  (640ch,  H/2):  all pruned (same mask as DB1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionPruneConfig:
    def __init__(self):
        self.threshold = 0.5       # Binary threshold for attention map
        self.dilate_kernel = 3     # Dilation kernel size at token resolution
        self.fuse_index = 5        # Text token index for attention map extraction
        self.blocks = 'all'        # Which blocks to prune: 'all', 'db1', 'db1+ub1', 'db2+mid+ub0', etc.
        self.prune_last_n = 1      # How many final steps to prune (default=1 = only last step)


# Global cache — multi-resolution
fusion_prune_cache = {
    'step': 0,
    'total_steps': 4,
    'cache_step': 2,           # step index to cache at (= total_steps - prune_last_n - 1)
    'block_cache': {},
    'masks': {},
    'fg_indices': {},
    'N_fg': {},
    'config': None,
    'flops_profiler': None,
}

# Convenience helpers
def _get_mask(res_key):
    return fusion_prune_cache['masks'].get(res_key)

def _get_fg(res_key):
    return (fusion_prune_cache['fg_indices'].get(res_key),
            fusion_prune_cache['N_fg'].get(res_key, 0))


def _infer_hw_from_tokens(num_tokens, ref_hw=None):
    """Infer a 2D token grid from token count, preferring the reference aspect."""
    if ref_hw is not None:
        ref_h, ref_w = ref_hw
        if ref_h > 0 and ref_w > 0 and ref_h * ref_w == num_tokens:
            return int(ref_h), int(ref_w)
        target_aspect = float(ref_h) / max(float(ref_w), 1.0)
    else:
        target_aspect = 1.0

    best_hw = None
    best_score = float("inf")
    for h in range(1, int(num_tokens ** 0.5) + 1):
        if num_tokens % h != 0:
            continue
        for cand_h, cand_w in ((h, num_tokens // h), (num_tokens // h, h)):
            aspect = float(cand_h) / max(float(cand_w), 1.0)
            score = abs(torch.log(torch.tensor(aspect / target_aspect)).item())
            if ref_hw is not None:
                score += 0.001 * (abs(cand_h - ref_hw[0]) + abs(cand_w - ref_hw[1]))
            if score < best_score:
                best_score = score
                best_hw = (cand_h, cand_w)
    return best_hw


def generate_pruning_mask(attn_probs, fuse_index=5, threshold=0.5, dilate_kernel=3, batch_size=None):
    """
    Generate (B, N) foreground mask from cross-attention probabilities.
    Returns fg_mask at DB1 token resolution.
    Supports non-square latent layouts via latent_hw stored in fusion_prune_cache.
    """
    B_H, N_q, N_kv = attn_probs.shape

    # Infer spatial dims from the real query-token count. For non-standard sizes
    # latent_hw is only an aspect hint; padding/resize paths can make latent//2
    # disagree with the attention map (e.g. 43x32 tokens).
    latent_hw = fusion_prune_cache.get('latent_hw')
    ref_hw = None
    if latent_hw is not None:
        # DB1 attention operates at roughly latent_h/2 x latent_w/2.
        ref_hw = (max(1, latent_hw[0] // 2), max(1, latent_hw[1] // 2))

    inferred_hw = _infer_hw_from_tokens(N_q, ref_hw=ref_hw)
    if inferred_hw is None:
        fusion_prune_cache.pop('db1_hw', None)
        B = int(batch_size or 1)
        return torch.ones(B, N_q, dtype=torch.bool, device=attn_probs.device)
    size_h, size_w = inferred_hw

    # Store DB1 spatial dims for downstream use (e.g., _downsample_mask)
    fusion_prune_cache['db1_hw'] = (size_h, size_w)

    B = int(batch_size or 1)
    if B <= 0 or B_H % B != 0:
        B = 1
    H = B_H // B

    attn = attn_probs[:, :, min(fuse_index, N_kv - 1)]
    attn = attn.view(B, H, size_h, size_w).mean(dim=1)

    flat = attn.reshape(B, -1)
    a_min = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
    a_max = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
    attn = (attn - a_min) / (a_max - a_min + 1e-6)

    fg_2d = (attn > threshold)

    if dilate_kernel > 1:
        fg_f = fg_2d.unsqueeze(1).float()
        fg_f = F.max_pool2d(fg_f, dilate_kernel, stride=1, padding=dilate_kernel // 2)
        fg_2d = fg_f.squeeze(1) > 0.5

    return fg_2d.reshape(B, -1)


def _downsample_mask(fg_mask_flat, src_h, src_w):
    """Downsample a 1D fg mask by 2x using conservative max-pool."""
    B = fg_mask_flat.shape[0]
    fg_2d = fg_mask_flat.reshape(B, 1, src_h, src_w).float()
    pad_h = src_h % 2
    pad_w = src_w % 2
    if pad_h or pad_w:
        fg_2d = F.pad(fg_2d, (0, pad_w, 0, pad_h), value=0.0)
    fg_down = F.max_pool2d(fg_2d, kernel_size=2, stride=2)
    return (fg_down > 0.5).reshape(B, -1)


def _compute_fg_indices(fg_mask):
    """Compute padded foreground indices for gather/scatter."""
    B, N = fg_mask.shape
    max_fg = max(1, int(fg_mask.sum(dim=1).max().item()))
    idx_list = []
    for b in range(B):
        fg = torch.where(fg_mask[b])[0]
        if len(fg) == 0:
            fg = torch.zeros(1, dtype=torch.long, device=fg_mask.device)
        if len(fg) < max_fg:
            fg = torch.cat([fg, fg[-1:].expand(max_fg - len(fg))])
        idx_list.append(fg[:max_fg])
    return torch.stack(idx_list), max_fg


def _store_mask(res_key, fg_mask):
    """Store a mask + precomputed indices for a resolution level."""
    fusion_prune_cache['masks'][res_key] = fg_mask
    idx, n = _compute_fg_indices(fg_mask)
    fusion_prune_cache['fg_indices'][res_key] = idx
    fusion_prune_cache['N_fg'][res_key] = n


def _report_attention_flops(q_len, kv_len, heads, head_dim, batch_size):
    profiler = fusion_prune_cache.get('flops_profiler')
    if profiler is None:
        return
    profiler.add_attention_flops(
        q_len=q_len,
        kv_len=kv_len,
        heads=heads,
        head_dim=head_dim,
        batch_size=batch_size,
        bucket_name='transformer',
    )


# ────────────────────────────────────────────────────────────
# Asymmetric attention helpers
# ────────────────────────────────────────────────────────────

def _asymmetric_self_attn(block, hidden_states, fg_indices, N_fg, B, N, C):
    """Self-attention with Q only for fg tokens, K/V for all tokens."""
    norm_hs = block.norm1(hidden_states)
    fg_expand = fg_indices.unsqueeze(-1).expand(-1, -1, C)
    fg_norm = torch.gather(norm_hs, 1, fg_expand)

    attn = block.attn1
    query = attn.to_q(fg_norm)
    key = attn.to_k(norm_hs)
    value = attn.to_v(norm_hs)

    heads = attn.heads
    hd = attn.inner_dim // heads

    q4 = query.view(B, N_fg, heads, hd).transpose(1, 2)
    k4 = key.view(B, N, heads, hd).transpose(1, 2)
    v4 = value.view(B, N, heads, hd).transpose(1, 2)

    _report_attention_flops(q_len=N_fg, kv_len=N, heads=heads, head_dim=hd, batch_size=B)
    out = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(B, N_fg, attn.inner_dim)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)

    residual = torch.zeros(B, N, C, device=hidden_states.device, dtype=hidden_states.dtype)
    residual.scatter_(1, fg_expand, out)
    return hidden_states + residual


def _fg_only_cross_attn(block, hidden_states, encoder_hidden_states,
                        fg_indices, N_fg, B, N, C):
    """Cross-attention with Q only for fg tokens."""
    if block.attn2 is None or encoder_hidden_states is None:
        return hidden_states

    norm_hs = block.norm2(hidden_states)
    fg_expand = fg_indices.unsqueeze(-1).expand(-1, -1, C)
    fg_norm = torch.gather(norm_hs, 1, fg_expand)

    attn = block.attn2
    ca_input = encoder_hidden_states
    if attn.norm_cross:
        ca_input = attn.norm_encoder_hidden_states(ca_input)

    query = attn.to_q(fg_norm)
    key = attn.to_k(ca_input)
    value = attn.to_v(ca_input)

    heads = attn.heads
    hd = attn.inner_dim // heads
    N_kv = ca_input.shape[1]

    q4 = query.view(B, N_fg, heads, hd).transpose(1, 2)
    k4 = key.view(B, N_kv, heads, hd).transpose(1, 2)
    v4 = value.view(B, N_kv, heads, hd).transpose(1, 2)

    _report_attention_flops(q_len=N_fg, kv_len=N_kv, heads=heads, head_dim=hd, batch_size=B)
    out = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(B, N_fg, attn.inner_dim)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)

    residual = torch.zeros(B, N, C, device=hidden_states.device, dtype=hidden_states.dtype)
    residual.scatter_(1, fg_expand, out)
    return hidden_states + residual


def _fg_only_ff(block, hidden_states, fg_indices, N_fg, B, N, C):
    """Feed-forward only for fg tokens."""
    norm_hs = block.norm3(hidden_states)
    fg_expand = fg_indices.unsqueeze(-1).expand(-1, -1, C)
    fg_norm = torch.gather(norm_hs, 1, fg_expand)

    ff_out = block.ff(fg_norm)

    residual = torch.zeros(B, N, C, device=hidden_states.device, dtype=hidden_states.dtype)
    residual.scatter_(1, fg_expand, ff_out)
    return hidden_states + residual


# ────────────────────────────────────────────────────────────
# Patching
# ────────────────────────────────────────────────────────────

def _make_prunable_forward(block, block_name, old_fw, res_key):
    """Create a pruned forward for a BasicTransformerBlock."""

    def pruned_forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        **kwargs,
    ):
        step = fusion_prune_cache['step']
        cache_step = fusion_prune_cache['cache_step']

        if step <= cache_step:
            output = old_fw(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                **kwargs,
            )
            if step == cache_step:
                fusion_prune_cache['block_cache'][block_name] = output.detach().clone()
            return output

        # ── Steps after cache_step: pruned ──
        fg_mask = _get_mask(res_key)
        fg_indices, N_fg = _get_fg(res_key)
        cached = fusion_prune_cache['block_cache'].get(block_name)

        if fg_mask is None or cached is None or fg_indices is None or N_fg == 0:
            return old_fw(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                **kwargs,
            )

        B, N, C = hidden_states.shape

        if cached.shape != hidden_states.shape or fg_mask.shape[-1] != N:
            return old_fw(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                **kwargs,
            )
        if fg_mask.shape[0] != B:
            if fg_mask.shape[0] == 1:
                fg_mask = fg_mask.expand(B, -1)
                fg_indices, N_fg = _compute_fg_indices(fg_mask)
            else:
                return old_fw(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    **kwargs,
                )

        # Inject cached output for background tokens
        hs = hidden_states.clone()
        bg_exp = (~fg_mask).unsqueeze(-1).expand_as(hs)
        hs[bg_exp] = cached[bg_exp]

        # Asymmetric SA → fg-only CA → fg-only FF
        hs = _asymmetric_self_attn(block, hs, fg_indices, N_fg, B, N, C)
        hs = _fg_only_cross_attn(block, hs, encoder_hidden_states,
                                 fg_indices, N_fg, B, N, C)
        hs = _fg_only_ff(block, hs, fg_indices, N_fg, B, N, C)

        return hs

    return pruned_forward


def _make_mask_gen_forward(block, block_name, old_fw, cross_attention_scores):
    """Wrap DB1's transformer_blocks[0] to generate pruning masks after forward."""

    def mask_gen_forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        **kwargs,
    ):
        step = fusion_prune_cache['step']
        cache_step = fusion_prune_cache['cache_step']

        output = old_fw(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
            **kwargs,
        )

        if step == cache_step:
            fusion_prune_cache['block_cache'][block_name] = output.detach().clone()

        # Generate pruning masks at multiple resolutions
        cfg = fusion_prune_cache.get('config') or FusionPruneConfig()
        found_key = None
        for k in cross_attention_scores:
            if 'down_blocks.1.attentions.0.transformer_blocks.0' in k:
                found_key = k
                break

        if found_key and found_key in cross_attention_scores:
            attn_probs = cross_attention_scores[found_key]

            # DB1 resolution mask
            fg_db1 = generate_pruning_mask(
                attn_probs,
                fuse_index=cfg.fuse_index,
                threshold=cfg.threshold,
                dilate_kernel=cfg.dilate_kernel,
                batch_size=hidden_states.shape[0],
            )
            _store_mask('db1', fg_db1)

            # DB2 / Mid / UB0 resolution mask (2x downsampled)
            db1_hw = fusion_prune_cache.get('db1_hw')
            if db1_hw is not None:
                db1_h, db1_w = db1_hw
                if db1_h > 1 and db1_w > 1 and db1_h * db1_w == fg_db1.shape[-1]:
                    fg_db2 = _downsample_mask(fg_db1, db1_h, db1_w)
                    _store_mask('db2', fg_db2)

        return output

    return mask_gen_forward


def patch_fusion_prune(unet, config, cross_attention_scores):
    """
    Patch the UNet for fusion-guided token pruning.

    config.blocks controls which sections to prune:
      'all'          → DB1 + DB2 + Mid + UB0 + UB1
      'db1'          → only Down Block 1
      'db1+ub1'      → DB1 + UB1 (same resolution, skip deep blocks)
      'db2+mid+ub0'  → only deep blocks
      'db1+db2+mid'  → DB1 + DB2 + Mid (no up blocks)
    """
    fusion_prune_cache['config'] = config
    fusion_prune_cache['block_cache'].clear()
    fusion_prune_cache['masks'].clear()
    fusion_prune_cache['fg_indices'].clear()
    fusion_prune_cache['N_fg'].clear()
    fusion_prune_cache.pop('db1_hw', None)
    fusion_prune_cache.pop('latent_hw', None)
    fusion_prune_cache['step'] = 0
    fusion_prune_cache['_sample_idx'] = 0

    # cache_step = last fully-computed step; pruning starts at cache_step+1
    total = fusion_prune_cache.get('total_steps', 4)
    prune_last_n = getattr(config, 'prune_last_n', 1)
    cache_step = max(0, total - prune_last_n - 1)
    fusion_prune_cache['cache_step'] = cache_step
    print(f"[FusionPrune] total_steps={total}, prune_last_n={prune_last_n}, cache_step={cache_step} (prune steps {cache_step+1}..{total-1})")

    blocks_str = getattr(config, 'blocks', 'all').lower()
    enabled = {
        'db1': 'db1' in blocks_str or blocks_str == 'all',
        'db2': 'db2' in blocks_str or blocks_str == 'all',
        'mid': 'mid' in blocks_str or blocks_str == 'all',
        'ub0': 'ub0' in blocks_str or blocks_str == 'all',
        'ub1': 'ub1' in blocks_str or blocks_str == 'all',
    }

    total_pruned = 0

    # ── Mask generator (always needed) ──
    db1 = unet.down_blocks[1]
    tb0 = db1.attentions[0].transformer_blocks[0]
    tb0.forward = _make_mask_gen_forward(
        tb0, 'db1.a0.tb0', tb0.forward, cross_attention_scores
    )

    # ── DB1 ──
    db1_count = 0
    if enabled['db1']:
        for i in range(1, len(db1.attentions[0].transformer_blocks)):
            tb = db1.attentions[0].transformer_blocks[i]
            tb.forward = _make_prunable_forward(tb, f'db1.a0.tb{i}', tb.forward, 'db1')
            db1_count += 1
        for j, attn_mod in enumerate(db1.attentions[1:], start=1):
            for i, tb in enumerate(attn_mod.transformer_blocks):
                tb.forward = _make_prunable_forward(tb, f'db1.a{j}.tb{i}', tb.forward, 'db1')
                db1_count += 1
    total_pruned += db1_count
    print(f"[FusionPrune]   DB1: {db1_count} pruned blocks")

    # ── DB2 ──
    db2_count = 0
    if enabled['db2'] and len(unet.down_blocks) > 2 and hasattr(unet.down_blocks[2], 'attentions'):
        for j, attn_mod in enumerate(unet.down_blocks[2].attentions):
            for i, tb in enumerate(attn_mod.transformer_blocks):
                tb.forward = _make_prunable_forward(tb, f'db2.a{j}.tb{i}', tb.forward, 'db2')
                db2_count += 1
    total_pruned += db2_count
    print(f"[FusionPrune]   DB2: {db2_count} pruned blocks")

    # ── Mid ──
    mid_count = 0
    if enabled['mid'] and hasattr(unet.mid_block, 'attentions'):
        for j, attn_mod in enumerate(unet.mid_block.attentions):
            for i, tb in enumerate(attn_mod.transformer_blocks):
                tb.forward = _make_prunable_forward(tb, f'mid.a{j}.tb{i}', tb.forward, 'db2')
                mid_count += 1
    total_pruned += mid_count
    print(f"[FusionPrune]   Mid: {mid_count} pruned blocks")

    # ── UB0 ──
    ub0_count = 0
    if enabled['ub0'] and len(unet.up_blocks) > 0 and hasattr(unet.up_blocks[0], 'attentions'):
        for j, attn_mod in enumerate(unet.up_blocks[0].attentions):
            for i, tb in enumerate(attn_mod.transformer_blocks):
                tb.forward = _make_prunable_forward(tb, f'ub0.a{j}.tb{i}', tb.forward, 'db2')
                ub0_count += 1
    total_pruned += ub0_count
    print(f"[FusionPrune]   UB0: {ub0_count} pruned blocks")

    # ── UB1 ──
    ub1_count = 0
    if enabled['ub1'] and len(unet.up_blocks) > 1 and hasattr(unet.up_blocks[1], 'attentions'):
        for j, attn_mod in enumerate(unet.up_blocks[1].attentions):
            for i, tb in enumerate(attn_mod.transformer_blocks):
                tb.forward = _make_prunable_forward(tb, f'ub1.a{j}.tb{i}', tb.forward, 'db1')
                ub1_count += 1
    total_pruned += ub1_count
    print(f"[FusionPrune]   UB1: {ub1_count} pruned blocks")

    print(f"[FusionPrune] Total: {total_pruned} pruned blocks (blocks={blocks_str})")
