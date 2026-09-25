import numpy as np
import cv2
from scipy.ndimage import convolve, zoom
from .image_utils import pad_to_multiple, crop_to_original


def wavelet_blur_np(image: np.ndarray, radius: int):
    kernel = np.array([
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ], dtype=np.float32)

    blurred = np.empty_like(image)
    for c in range(image.shape[0]):
        blurred_c = convolve(image[c], kernel, mode="nearest")
        if radius > 1:
            blurred_c = zoom(zoom(blurred_c, 1 / radius, order=1), radius, order=1)
        blurred[c] = blurred_c
    return blurred


def wavelet_decomposition_np(image: np.ndarray, levels=5):
    high_freq = np.zeros_like(image)
    for i in range(levels):
        radius = 2**i
        low_freq = wavelet_blur_np(image, radius)
        high_freq += image - low_freq
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction_np(content_feat: np.ndarray, style_feat: np.ndarray):
    content_high, _ = wavelet_decomposition_np(content_feat)
    _, style_low = wavelet_decomposition_np(style_feat)
    return content_high + style_low


def wavelet_color_fix_np(fused: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fused_np = fused.astype(np.float32) / 255.0
    mask_np = mask.astype(np.float32) / 255.0

    fused_np = fused_np.transpose(2, 0, 1)
    mask_np = mask_np.transpose(2, 0, 1)

    result_np = wavelet_reconstruction_np(fused_np, mask_np)

    result_np = result_np.transpose(1, 2, 0)
    result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

    return result_np


def attention_guided_fusion(
    ori: np.ndarray,
    removed: np.ndarray,
    attn_map: np.ndarray,
    multiple: int = 8,
    threshold: float = 0.5,
    blur_ksize: int = 21,
    blur_sigma: float = 4.0,
):
    h, w = ori.shape[:2]
    am = attn_map.astype(np.float32)
    if am.ndim == 3:
        am = am.mean(axis=-1)
    if am.max() > 1.0:
        am = am / 255.0
    am = cv2.resize(am, (w, h), interpolation=cv2.INTER_LINEAR)

    hard = (am >= threshold).astype(np.float32)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    alpha = cv2.GaussianBlur(hard, (blur_ksize, blur_ksize), sigmaX=blur_sigma)
    alpha = np.maximum(hard, np.clip(alpha, 0.0, 1.0))
    alpha_3c = np.stack([alpha, alpha, alpha], axis=-1)

    fused = ori.astype(np.float32) * (1.0 - alpha_3c) + removed.astype(np.float32) * alpha_3c
    return np.clip(fused, 0, 255).astype(np.uint8)


def _normalize_attn_map(attn_map: np.ndarray, h: int, w: int) -> np.ndarray:
    am = attn_map.astype(np.float32)
    if am.ndim == 3:
        am = am.mean(axis=0)
    if am.ndim != 2:
        raise ValueError(f"Expected 2D/3D attn_map, got shape={attn_map.shape}")

    am = cv2.resize(am, (w, h), interpolation=cv2.INTER_LINEAR)
    lo = float(np.percentile(am, 2.0))
    hi = float(np.percentile(am, 98.0))
    am = (am - lo) / (hi - lo + 1e-6)
    return np.clip(am, 0.0, 1.0)


def _build_soft_alpha(attn_map: np.ndarray, h: int, w: int, blur_ksize: int = 25, gamma: float = 0.9) -> np.ndarray:
    alpha = _normalize_attn_map(attn_map, h, w)
    alpha = np.power(alpha, gamma)

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), sigmaX=0)
    return np.clip(alpha, 0.0, 1.0)


def _local_color_match_on_ring(
    ori: np.ndarray,
    removed: np.ndarray,
    alpha: np.ndarray,
    ring_width: int = 15,
) -> np.ndarray:
    hard = (alpha > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    dil = cv2.dilate(hard, kernel, iterations=1)
    ero = cv2.erode(hard, kernel, iterations=1)
    ring = ((dil - ero) > 0).astype(np.uint8)

    if ring.sum() < 64:
        return removed

    out = removed.astype(np.float32).copy()
    ori_f = ori.astype(np.float32)
    rem_f = removed.astype(np.float32)

    ring_mask = ring.astype(bool)
    for c in range(3):
        ori_vals = ori_f[:, :, c][ring_mask]
        rem_vals = rem_f[:, :, c][ring_mask]
        if ori_vals.size == 0 or rem_vals.size == 0:
            continue

        mu_o = float(ori_vals.mean())
        sd_o = float(ori_vals.std() + 1e-6)
        mu_r = float(rem_vals.mean())
        sd_r = float(rem_vals.std() + 1e-6)

        out[:, :, c] = (out[:, :, c] - mu_r) * (sd_o / sd_r) + mu_o

    return np.clip(out, 0, 255).astype(np.uint8)


def attention_guided_fusion_v2(
    ori: np.ndarray,
    removed: np.ndarray,
    attn_map: np.ndarray,
    multiple: int = 8,
    blur_ksize: int = 25,
    gamma: float = 0.9,
    ring_width: int = 15,
):
    """Modified fusion with soft attention and boundary color alignment.

    This version avoids hard-threshold masks and aligns color statistics on
    a ring around the transition area to reduce visible seams.
    """
    h, w = ori.shape[:2]

    alpha = _build_soft_alpha(attn_map, h, w, blur_ksize=blur_ksize, gamma=gamma)
    alpha_3c = np.stack([alpha, alpha, alpha], axis=-1)

    removed_corr = _local_color_match_on_ring(ori, removed, alpha, ring_width=ring_width)

    # Keep a low-frequency color prior similar to the original wavelet branch.
    ori_bg = (ori.astype(np.float32) * (1.0 - alpha_3c)).astype(np.uint8)
    rem_bg = (removed_corr.astype(np.float32) * (1.0 - alpha_3c)).astype(np.uint8)
    ori_pad, h0, w0 = pad_to_multiple(ori_bg, multiple)
    rem_pad, _, _ = pad_to_multiple(rem_bg, multiple)
    wave_rgb = wavelet_color_fix_np(ori_pad, rem_pad)
    wave = crop_to_original(wave_rgb, h0, w0)

    fused = wave.astype(np.float32) * (1.0 - alpha_3c) + removed_corr.astype(np.float32) * alpha_3c
    return np.clip(fused, 0, 255).astype(np.uint8)
