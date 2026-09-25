import argparse
import contextlib
import csv
import os
import random
import types
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from omegaconf import OmegaConf
from peft import LoraConfig
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from HYPIR.dataset.sd_inpaint_dataset import SDInpaintImageDataset
from HYPIR.model.clip_encoder import CLIPImageEncoder
from HYPIR.model.fusion_module import LearnableAttentionFusion
from HYPIR.model.postfuse_module import PostfuseModule
from HYPIR.utils.attention_guided_fusion import attention_guided_fusion

import time

FLOP_BUCKET_KEYS = (
    "condition_encode",
    "removal_vae_encode",
    "transformer",
    "vae_decode",
)


class TheoreticalFLOPsProfiler:
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.current_bucket = None
        self.totals = {key: 0.0 for key in FLOP_BUCKET_KEYS}
        self._handles = []

    def register_model(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self._handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._linear_hook))

            if self._is_attention_module(module):
                self._handles.append(module.register_forward_hook(self._attention_hook, with_kwargs=True))

    def reset_totals(self):
        for key in self.totals:
            self.totals[key] = 0.0

    def get_totals(self):
        return dict(self.totals)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    @contextlib.contextmanager
    def bucket(self, bucket_name: str):
        previous = self.current_bucket
        self.current_bucket = bucket_name
        try:
            yield
        finally:
            self.current_bucket = previous

    @contextlib.contextmanager
    def suspend(self):
        previous_enabled = self.enabled
        previous_bucket = self.current_bucket
        self.enabled = False
        self.current_bucket = None
        try:
            yield
        finally:
            self.enabled = previous_enabled
            self.current_bucket = previous_bucket

    def add_attention_flops(self, q_len, kv_len, heads, head_dim, batch_size=1, bucket_name=None):
        bucket = bucket_name or self.current_bucket
        if not self.enabled or bucket is None:
            return
        flops = float(4.0 * batch_size * heads * q_len * kv_len * head_dim)
        self.totals[bucket] += flops

    def _add_flops(self, bucket_name, flops):
        if not self.enabled or bucket_name is None:
            return
        self.totals[bucket_name] += float(flops)

    def _linear_hook(self, module, inputs, output):
        if not self.enabled or self.current_bucket is None or not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x) or x.numel() == 0:
            return
        out_features = module.out_features
        ops_per_output = 2 * module.in_features + (1 if module.bias is not None else 0)
        flops = x.numel() / x.shape[-1] * out_features * ops_per_output
        self._add_flops(self.current_bucket, flops)

    def _conv_hook(self, module, inputs, output):
        if not self.enabled or self.current_bucket is None or not inputs:
            return
        x = inputs[0]
        y = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(x) or not torch.is_tensor(y) or y.numel() == 0:
            return
        batch_size = y.shape[0]
        out_channels = y.shape[1]
        out_spatial = int(np.prod(y.shape[2:]))
        kernel_mul = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
        ops_per_output = 2 * kernel_mul + (1 if module.bias is not None else 0)
        flops = batch_size * out_channels * out_spatial * ops_per_output
        self._add_flops(self.current_bucket, flops)

    def _attention_hook(self, module, args, kwargs, output):
        if not self.enabled or self.current_bucket is None:
            return

        hidden_states = self._get_attention_hidden_states(module, args, kwargs)
        if hidden_states is None or hidden_states.ndim < 3:
            return

        batch_size = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        kv_len = encoder_hidden_states.shape[1] if torch.is_tensor(encoder_hidden_states) else q_len
        heads, head_dim = self._get_attention_heads(module, hidden_states)
        if heads is None or head_dim is None:
            return
        self.add_attention_flops(q_len=q_len, kv_len=kv_len, heads=heads, head_dim=head_dim, batch_size=batch_size)

    @staticmethod
    def _is_attention_module(module):
        return (
            (hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"))
            or (hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"))
            or isinstance(module, nn.MultiheadAttention)
        )

    @staticmethod
    def _get_attention_hidden_states(module, args, kwargs):
        if args and torch.is_tensor(args[0]):
            return args[0]
        if torch.is_tensor(kwargs.get("hidden_states")):
            return kwargs["hidden_states"]
        if torch.is_tensor(kwargs.get("query")):
            return kwargs["query"]
        return None

    @staticmethod
    def _get_attention_heads(module, hidden_states):
        if hasattr(module, "heads") and hasattr(module, "inner_dim"):
            heads = int(module.heads)
            return heads, int(module.inner_dim // max(heads, 1))
        if hasattr(module, "num_heads"):
            heads = int(module.num_heads)
            if hasattr(module, "head_dim"):
                return heads, int(module.head_dim)
            embed_dim = getattr(module, "embed_dim", hidden_states.shape[-1])
            return heads, int(embed_dim // max(heads, 1))
        return None, None


DEFAULT_ATTENTION_CAPTURE_LAYER = "down_blocks.1.attentions.0.transformer_blocks.0.attn2"


def unet_store_cross_attention_scores(
    unet,
    attention_scores,
    layers=5,
    capture_mode="all",
    target_layer=DEFAULT_ATTENTION_CAPTURE_LAYER,
):
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

    candidates = [
        (name, module)
        for name, module in unet.named_modules()
        if isinstance(module, Attention)
        and "attn2" in name
        and any(layer in name for layer in applicable_layers)
    ]
    capture_mode = str(capture_mode).lower()
    if capture_mode == "single" and candidates:
        selected = next(
            ((name, module) for name, module in candidates if name == target_layer),
            candidates[0],
        )
        candidates = [selected]
    elif capture_mode != "all":
        raise ValueError(f"Unsupported attention_capture_mode={capture_mode!r}. Expected 'single' or 'all'.")

    for name, module in candidates:
        if isinstance(module.processor, AttnProcessor2_0):
            module.set_processor(AttnProcessor())
        module.old_get_attention_scores = module.get_attention_scores
        module.new_get_attention_scores = types.MethodType(make_new_get_attention_scores_fn(name), module)
        module.get_attention_scores = module.new_get_attention_scores

    selected_names = [name for name, _ in candidates]
    if selected_names:
        print(f"[Info] attention_capture_mode={capture_mode}, layers={selected_names}")
    else:
        print("[Warn] No cross-attention layer matched the requested capture configuration.")

    return unet


def clear_cross_attention_scores(cross_attention_scores):
    cross_attention_scores.clear()


def _infer_spatial_hw_from_tokens(num_tokens: int, ref_hw: Tuple[int, int]) -> Tuple[int, int]:
    ref_h, ref_w = ref_hw
    target_aspect = float(ref_h) / max(float(ref_w), 1.0)
    best_hw = None
    best_score = float("inf")
    for h in range(1, int(num_tokens ** 0.5) + 1):
        if num_tokens % h != 0:
            continue
        for cand_h, cand_w in ((h, num_tokens // h), (num_tokens // h, h)):
            aspect = float(cand_h) / max(float(cand_w), 1.0)
            score = abs(np.log(max(aspect, 1e-8) / max(target_aspect, 1e-8)))
            score += 0.001 * (abs(cand_h - ref_h) + abs(cand_w - ref_w))
            if score < best_score:
                best_score = score
                best_hw = (cand_h, cand_w)
    if best_hw is None:
        raise ValueError(f"Cannot infer spatial dims from num_tokens={num_tokens}, ref_hw={ref_hw}")
    return best_hw


def resize_attn_map_divide2(attn_map, mask, fuse_index):
    bxh, num_noise_latents, _ = attn_map.shape
    batch_size = mask.shape[0]

    if bxh % batch_size != 0:
        raise ValueError(f"Unexpected attention shape {attn_map.shape} for batch size {batch_size}.")

    num_heads = bxh // batch_size

    # Infer spatial dimensions from token count and mask aspect ratio.
    mask_h, mask_w = mask.shape[-2:]
    latent_h, latent_w = _infer_spatial_hw_from_tokens(num_noise_latents, (mask_h, mask_w))

    attn_map = attn_map.view(batch_size, num_heads, num_noise_latents, -1)
    attn_map = attn_map[..., fuse_index]
    attn_map = attn_map.view(batch_size, num_heads, latent_h, latent_w)
    attn_map = F.interpolate(attn_map, size=mask.shape[-2:], mode="bilinear", antialias=True)

    attn_min = attn_map.amin(dim=(-2, -1), keepdim=True)
    attn_max = attn_map.amax(dim=(-2, -1), keepdim=True)
    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    return attn_map


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_weight_dtype(precision: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[precision]


def parse_torch_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    name = str(value).lower().replace("torch.", "")
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={value!r}. Expected fp32, bf16, or fp16.")


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = ((x + 1.0) / 2.0).clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().cpu().numpy()
    return x

def uint8_image_to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    t = t * 2.0 - 1.0
    return t.to(device=device, dtype=dtype)

def save_tensor_image(x: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8_image(x)).save(path)


def _to_01_image(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def _to_3ch_01(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    x = x.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[-2:] != size:
        x = _resize_chw_tensor(x, size, mode="bilinear")
    x = x.clamp(0.0, 1.0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x[:3]


def save_fusion_debug_strip(
    input_img: torch.Tensor,
    gt: torch.Tensor,
    unfused: torch.Tensor,
    latent_fused: torch.Tensor,
    learned_fused: torch.Tensor,
    alpha: torch.Tensor,
    attn: torch.Tensor,
    mask: torch.Tensor,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = input_img.shape[-2:]
    strip = torch.cat(
        [
            _to_01_image(input_img),
            _to_01_image(gt),
            _to_01_image(unfused),
            _to_01_image(latent_fused),
            _to_01_image(learned_fused),
            _to_3ch_01(alpha, size),
            _to_3ch_01(attn, size),
            _to_3ch_01(mask, size),
        ],
        dim=2,
    )
    arr = (strip.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_grayscale_tensor(x: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.detach().float().cpu()
    x = x.squeeze()
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    arr = (x.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    m = mask.squeeze(0).clamp(0, 1)
    m = (m * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(m, mode="L")


def _normalize_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def _list_image_files(folder: str) -> List[str]:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in valid_exts:
            files.append(path)
    return files


def _load_rgb_tensor(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_rgb_tensor_square(path: str, out_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((out_size, out_size), Image.BICUBIC)
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_mask_tensor(path: str, out_hw: Optional[tuple] = None) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if out_hw is not None:
        h, w = out_hw
        img = img.resize((w, h), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _load_mask_tensor_square(path: str, out_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((out_size, out_size), Image.NEAREST)
    arr = np.array(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _resize_chw_tensor(x: torch.Tensor, out_hw: Tuple[int, int], mode: str) -> torch.Tensor:
    x4 = x.unsqueeze(0)
    if mode in ("bilinear", "bicubic"):
        y4 = F.interpolate(x4, size=out_hw, mode=mode, align_corners=False)
    else:
        y4 = F.interpolate(x4, size=out_hw, mode=mode)
    return y4.squeeze(0).contiguous()


def _resize_by_short_side_tensor(x: torch.Tensor, target_short: int, mode: str = "bilinear", align_to: int = 8) -> torch.Tensor:
    """Resize CHW tensor so shorter side equals target_short, preserving aspect ratio.
    Output dimensions are rounded down to the nearest multiple of align_to."""
    _, h, w = x.shape
    if h <= w:
        new_h = target_short
        new_w = int(round(w * target_short / h))
    else:
        new_w = target_short
        new_h = int(round(h * target_short / w))
    new_h = max(align_to, (new_h // align_to) * align_to)
    new_w = max(align_to, (new_w // align_to) * align_to)
    return _resize_chw_tensor(x, (new_h, new_w), mode=mode)


def _pad_chw_to_multiple(
    x: torch.Tensor,
    multiple: int = 8,
    mode: str = "constant",
    value: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x4 = x.unsqueeze(0)
    if mode == "constant":
        padded = F.pad(x4, (pad_left, pad_right, pad_top, pad_bottom), mode=mode, value=value)
    else:
        padded = F.pad(x4, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return padded.squeeze(0).contiguous(), (pad_top, pad_bottom, pad_left, pad_right)


def _crop_chw_tensor(x: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return x[:, top : top + height, left : left + width].contiguous()


class DirectoryInpaintDataset(Dataset):
    """Read input/mask/(optional gt) images from folders matched by filename stem."""

    def __init__(
        self,
        input_dir: str,
        mask_dir: str,
        gt_dir: Optional[str] = None,
        prompt: str = "",
        infer_size: int = 512,
        resize_mode: str = "short_side",
    ):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.prompt = prompt
        self.infer_size = infer_size
        self.resize_mode = str(resize_mode).lower()
        if self.resize_mode not in {"short_side", "square", "pad_to_multiple"}:
            raise ValueError(
                f"Unsupported resize_mode={resize_mode!r}. Expected 'short_side', 'square', or 'pad_to_multiple'."
            )

        input_files = _list_image_files(input_dir)
        mask_files = _list_image_files(mask_dir)
        gt_files = _list_image_files(gt_dir) if gt_dir and os.path.isdir(gt_dir) else []

        mask_map = {_normalize_name(p): p for p in mask_files}
        gt_map = {_normalize_name(p): p for p in gt_files}

        self.samples = []
        for input_path in input_files:
            stem = _normalize_name(input_path)
            mask_path = mask_map.get(stem)
            if mask_path is None:
                continue
            gt_path = gt_map.get(stem)
            self.samples.append(
                {
                    "name": stem,
                    "input_path": input_path,
                    "mask_path": mask_path,
                    "gt_path": gt_path,
                    "has_gt": gt_path is not None,
                }
            )

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found. input_dir={input_dir}, mask_dir={mask_dir}. "
                "Files must share the same filename stem."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[index]
        with Image.open(item["input_path"]) as image:
            w, h = image.size

        if self.resize_mode == "square":
            input_img = _load_rgb_tensor_square(item["input_path"], self.infer_size)
            object_mask = _load_mask_tensor_square(item["mask_path"], self.infer_size)
            gt = _load_rgb_tensor_square(item["gt_path"], self.infer_size) if item["has_gt"] else input_img.clone()
            pad_top = pad_bottom = pad_left = pad_right = 0
        elif self.resize_mode == "pad_to_multiple":
            input_img = _load_rgb_tensor(item["input_path"])
            object_mask = _load_mask_tensor(item["mask_path"], out_hw=(h, w))

            if item["has_gt"]:
                gt = _load_rgb_tensor(item["gt_path"])
                if gt.shape[-2:] != (h, w):
                    gt = _resize_chw_tensor(gt, (h, w), mode="bilinear")
            else:
                gt = input_img.clone()

            input_img, pad_info = _pad_chw_to_multiple(input_img, multiple=8, mode="reflect")
            gt, _ = _pad_chw_to_multiple(gt, multiple=8, mode="reflect")
            object_mask, _ = _pad_chw_to_multiple(object_mask, multiple=8, mode="constant", value=0.0)
            pad_top, pad_bottom, pad_left, pad_right = pad_info
        else:
            input_img = _load_rgb_tensor(item["input_path"])
            object_mask = _load_mask_tensor(item["mask_path"], out_hw=(h, w))

            if item["has_gt"]:
                gt = _load_rgb_tensor(item["gt_path"])
                if gt.shape[-2:] != (h, w):
                    gt = _resize_chw_tensor(gt, (h, w), mode="bilinear")
            else:
                gt = input_img.clone()

            input_img = _resize_by_short_side_tensor(input_img, self.infer_size, mode="bilinear")
            infer_h, infer_w = input_img.shape[-2:]
            gt = _resize_chw_tensor(gt, (infer_h, infer_w), mode="bilinear")
            object_mask = _resize_chw_tensor(object_mask, (infer_h, infer_w), mode="nearest")
            pad_top = pad_bottom = pad_left = pad_right = 0

        return {
            "input": input_img,
            "GT": gt,
            "object_mask": object_mask,
            "object_effect_mask": object_mask.clone(),
            "txt": self.prompt,
            "sample_name": item["name"],
            "orig_h": torch.tensor(h, dtype=torch.int64),
            "orig_w": torch.tensor(w, dtype=torch.int64),
            "pad_top": torch.tensor(pad_top, dtype=torch.int64),
            "pad_bottom": torch.tensor(pad_bottom, dtype=torch.int64),
            "pad_left": torch.tensor(pad_left, dtype=torch.int64),
            "pad_right": torch.tensor(pad_right, dtype=torch.int64),
            "has_gt": torch.tensor(item["has_gt"], dtype=torch.bool),
        }


def resolve_state_file(weight_path: Optional[str]) -> Optional[str]:
    if not weight_path:
        return None
    if os.path.isdir(weight_path):
        candidates = [
            os.path.join(weight_path, "state_dict.pth"),
            os.path.join(weight_path, "pytorch_model.bin"),
            os.path.join(weight_path, "model.pth"),
            os.path.join(weight_path, "model.pt"),
            os.path.join(weight_path, "model.safetensors"),
            os.path.join(weight_path, "unet", "diffusion_pytorch_model.bin"),
            os.path.join(weight_path, "unet", "diffusion_pytorch_model.safetensors"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None
    if os.path.isfile(weight_path):
        return weight_path
    return None


def safe_load_state_file(path: str) -> Any:
    if path.endswith(".safetensors"):
        return load_file(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_fusion_module_state_file(weight_path: Optional[str]) -> Optional[str]:
    if not weight_path:
        return None
    weight_path = os.path.abspath(os.path.expanduser(str(weight_path)))
    if os.path.isdir(weight_path):
        candidates = [
            os.path.join(weight_path, "fusion_module.pth"),
            os.path.join(weight_path, "state_dict.pth"),
            os.path.join(weight_path, "model.pth"),
            os.path.join(weight_path, "model.pt"),
            os.path.join(weight_path, "model.safetensors"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None
    if os.path.isfile(weight_path):
        return weight_path
    return None


def strip_module_prefixes_from_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    normalized = {}
    prefixes = ("module.", "_orig_mod.", "model.", "fusion_module.")
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
                    break
        normalized[new_key] = value
    return normalized


class ObjectClearValInfer:
    """Validation inferencer that mirrors HYPIR/trainer/objectclear.py forward flow."""

    def __init__(self, config, weight_path: Optional[str], device: torch.device, weight_dtype: torch.dtype):
        self.config = config
        self.weight_path = weight_path
        self.device = device
        self.weight_dtype = weight_dtype
        self.batch_inputs = None
        self.cross_attention_scores = {}
        self.use_attention_guided_fusion = bool(getattr(self.config, "apply_attention_guided_fusion", False))
        self.request_learnable_fusion = bool(getattr(self.config, "learnable_fusion", False))
        self.fusion_module_path = getattr(self.config, "fusion_module_path", None)
        self.use_learnable_fusion = self.request_learnable_fusion and bool(self.fusion_module_path)
        if self.request_learnable_fusion and not self.fusion_module_path:
            self.use_attention_guided_fusion = True
            print("[Warn] --learnable_fusion was set without --fusion_module_path; falling back to default fusion.")
        self.alpha_fusion_threshold = float(getattr(self.config, "alpha_fusion_threshold", 0.5))
        self.debug_dump_intermediates = bool(getattr(self.config, "debug_dump_intermediates", False))
        self.debug_dump_dir = getattr(self.config, "debug_dump_dir", None)
        self.inference_only = bool(getattr(self.config, "inference_only", False))
        self.save_fusion_debug = bool(getattr(self.config, "save_fusion_debug", False))
        self.attention_capture_mode = str(getattr(self.config, "attention_capture_mode", "all")).lower()
        self.attention_capture_layer = str(
            getattr(self.config, "attention_capture_layer", DEFAULT_ATTENTION_CAPTURE_LAYER)
        )
        self._prompt_embed_cache = {}
        self._debug_sample_counter = 0
        self.flops_profiler = None
        self.state_file = resolve_state_file(self.weight_path)
        self.ckpt_mode = self._infer_ckpt_mode(self.state_file)

        self.init_scheduler()
        self.init_text_models()
        self.init_objectclear_modules()
        self.init_vae()
        self.init_generator()
        self.load_generator_weights()
        self.init_fusion_module()

    def _extract_state_dict(self, ckpt_obj):
        if isinstance(ckpt_obj, Mapping):
            for key in ["state_dict", "model", "unet", "module", "G", "generator", "model_state_dict"]:
                value = ckpt_obj.get(key)
                if isinstance(value, Mapping):
                    return value
        return ckpt_obj

    def _strip_state_key_prefixes(self, key: str) -> str:
        prefixes = (
            "module.",
            "_orig_mod.",
            "_fsdp_wrapped_module.",
            "model.",
            "unet.",
            "G.",
            "generator.",
            "student.",
            "student_G.",
            "base_model.model.",
        )
        current = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if current.startswith(prefix):
                    current = current[len(prefix) :]
                    changed = True
                    break
        return current

    def _state_key_variants(self, key: str) -> List[str]:
        prefixes = (
            "module.",
            "_orig_mod.",
            "_fsdp_wrapped_module.",
            "model.",
            "unet.",
            "G.",
            "generator.",
            "student.",
            "student_G.",
            "base_model.model.",
        )
        variants = []
        current = str(key)
        while current not in variants:
            variants.append(current)
            next_key = current
            for prefix in prefixes:
                if next_key.startswith(prefix):
                    next_key = next_key[len(prefix) :]
                    break
            if next_key == current:
                break
            current = next_key
        stripped = self._strip_state_key_prefixes(key)
        for candidate in (stripped, f"base_model.model.{stripped}", f"module.{stripped}", f"unet.{stripped}"):
            if candidate not in variants:
                variants.append(candidate)
        return variants

    def _normalize_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = {}
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            nk = self._strip_state_key_prefixes(k)
            normalized[nk] = v
        return normalized

    def _filter_compatible_state_dict(
        self,
        state_dict: Mapping[str, Any],
        model_state: Mapping[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        filtered = {}
        shape_mismatch = 0
        unmatched = 0
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            matched_key = None
            for candidate in self._state_key_variants(str(key)):
                if candidate in model_state:
                    matched_key = candidate
                    break
            if matched_key is None:
                unmatched += 1
                continue
            target = model_state[matched_key]
            if tuple(target.shape) != tuple(value.shape):
                shape_mismatch += 1
                continue
            tensor = value.detach()
            if tensor.is_floating_point() and tensor.dtype != target.dtype:
                tensor = tensor.to(dtype=target.dtype)
            filtered[matched_key] = tensor
        return filtered, unmatched, shape_mismatch

    def _state_dict_dtype_summary(self, state_dict: Mapping[str, Any]) -> str:
        counts = {}
        for value in state_dict.values():
            if isinstance(value, torch.Tensor):
                counts[str(value.dtype)] = counts.get(str(value.dtype), 0) + 1
        return ", ".join(f"{dtype}:{count}" for dtype, count in sorted(counts.items())) or "no tensors"

    def _infer_ckpt_mode(self, state_file: Optional[str]) -> str:
        if state_file is None:
            return "none"
        try:
            ckpt_obj = safe_load_state_file(state_file)
            state_dict = self._extract_state_dict(ckpt_obj)
            if not isinstance(state_dict, Mapping):
                return "unknown"
            keys = [k for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
            if any("lora_" in k for k in keys):
                return "lora"
            return "full"
        except Exception as exc:
            print(f"[Warn] Failed to inspect checkpoint type from {state_file}: {exc}")
            return "unknown"

    def init_scheduler(self):
        timestep_spacing = self._requested_timestep_spacing()
        scheduler_spacing = "trailing" if timestep_spacing == "fixed" else timestep_spacing
        scheduler_type = str(getattr(self.config, "scheduler_type", "ddpm")).lower()
        scheduler_cls = {
            "ddpm": DDPMScheduler,
            "ddim": DDIMScheduler,
        }.get(scheduler_type)
        if scheduler_cls is None:
            raise ValueError(f"Unsupported scheduler_type={scheduler_type!r}. Expected 'ddpm' or 'ddim'.")

        scheduler_kwargs = {}
        if scheduler_spacing is not None:
            scheduler_kwargs["timestep_spacing"] = scheduler_spacing
        self.scheduler = scheduler_cls.from_pretrained(
            self.config.base_model_path,
            subfolder="scheduler",
            **scheduler_kwargs,
        )
        if timestep_spacing is not None:
            self.scheduler.register_to_config(timestep_spacing=scheduler_spacing)
            print(
                f"[Info] scheduler={scheduler_cls.__name__}, timestep_spacing={timestep_spacing} "
                f"(diffusers={self.scheduler.config.timestep_spacing})"
            )
        else:
            print(f"[Info] scheduler={scheduler_cls.__name__}")

    def _requested_timestep_spacing(self) -> Optional[str]:
        timestep_spacing = getattr(
            self.config,
            "timestep_spacing",
            getattr(self.config, "scheduler_timestep_spacing", None),
        )
        if timestep_spacing is None:
            return None
        timestep_spacing = str(timestep_spacing).lower()
        if timestep_spacing == "fix":
            timestep_spacing = "fixed"
        return timestep_spacing

    def _get_fixed_timestep(self) -> int:
        configured = getattr(self.config, "fixed_timestep", None)
        if configured is None:
            configured = getattr(self.config, "noise_timestep", None)
        if configured is None:
            configured = getattr(self.config, "generator_timestep", None)
        if configured is None:
            configured = getattr(self.config, "dmd_fixed_timestep", None)
        if configured is None:
            raise ValueError("timestep_spacing='fixed' requires --fixed_timestep (or config fixed_timestep/noise_timestep).")
        timestep = int(configured)
        max_timestep = int(getattr(self.scheduler.config, "num_train_timesteps", 1000)) - 1
        if timestep < 0 or timestep > max_timestep:
            raise ValueError(f"Invalid fixed_timestep={timestep}. Expected an integer in [0, {max_timestep}].")
        return timestep

    def _set_inference_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        num_inference_steps = int(num_inference_steps)
        timestep_spacing = self._requested_timestep_spacing()
        fixed_timestep = self._get_fixed_timestep() if timestep_spacing == "fixed" else None
        cache_key = (num_inference_steps, timestep_spacing, fixed_timestep)
        if timestep_spacing == "fixed" and getattr(self, "_inference_timesteps_cache_key", None) == cache_key:
            return self._inference_timesteps_cache

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        if timestep_spacing == "fixed":
            if num_inference_steps != 1:
                raise ValueError("timestep_spacing='fixed' is only supported for 1-step inference.")
            timesteps = torch.tensor([fixed_timestep], dtype=torch.long, device=self.device)
            self.scheduler.timesteps = timesteps
            self._inference_timesteps_cache_key = cache_key
            self._inference_timesteps_cache = timesteps
        if not getattr(self, "_logged_inference_timesteps", False):
            print(f"[Info] inference timesteps={timesteps.detach().cpu().tolist()}")
            self._logged_inference_timesteps = True
        return timesteps

    def _pred_x0(self, model_output: torch.Tensor, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        prediction_type = self.scheduler.config.prediction_type
        cache_key = (sample.device.type, sample.device.index, sample.dtype)
        if getattr(self, "_alphas_cumprod_cache_key", None) != cache_key:
            self._alphas_cumprod_cache_key = cache_key
            self._alphas_cumprod_cache = self.scheduler.alphas_cumprod.to(
                device=sample.device,
                dtype=sample.dtype,
            )
        alphas_cumprod = self._alphas_cumprod_cache
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t

        if prediction_type == "epsilon":
            return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if prediction_type == "v_prediction":
            return alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        if prediction_type == "sample":
            return model_output
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    def _one_step_output_type(self, num_inference_steps: int) -> str:
        mode = str(getattr(self.config, "one_step_output_type", "auto")).lower()
        if mode == "auto":
            if int(num_inference_steps) == 1 and self._requested_timestep_spacing() == "fixed":
                return "pred_x0"
            return "scheduler"
        if mode not in {"pred_x0", "scheduler"}:
            raise ValueError(f"Unsupported one_step_output_type={mode!r}. Expected 'auto', 'pred_x0', or 'scheduler'.")
        return mode

    def _fusion_precision_mode(self) -> str:
        mode = str(getattr(self.config, "fusion_precision", "fp32")).lower()
        aliases = {
            "float": "fp32",
            "float32": "fp32",
            "full": "fp32",
            "autocast": "amp",
            "auto": "amp",
            "weight_dtype": "weight",
            "model": "weight",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"fp32", "amp", "weight"}:
            raise ValueError(f"Unsupported fusion_precision={mode!r}. Expected 'fp32', 'amp', or 'weight'.")
        return mode

    def _fusion_autocast_context(self):
        if self.device.type == "cuda" and self._fusion_precision_mode() == "fp32":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    def _fusion_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._fusion_precision_mode() == "weight":
            return tensor.to(dtype=self.weight_dtype)
        return tensor.float()

    def _fusion_validation_mode(self) -> str:
        mode = str(getattr(self.config, "fusion_validation_mode", "training")).lower()
        aliases = {
            "train": "training",
            "trainer": "training",
            "strict": "training",
            "inference": "pipeline",
            "legacy": "pipeline",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"training", "pipeline"}:
            raise ValueError(f"Unsupported fusion_validation_mode={mode!r}. Expected 'training' or 'pipeline'.")
        return mode

    def _select_generator_timesteps(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        student_steps = int(getattr(self.config, "student_steps", getattr(self.config, "distill_steps", 1)))
        noise_timesteps = self._set_inference_timesteps(student_steps)
        noise_timesteps = torch.full(
            (batch_size,),
            int(noise_timesteps[0]),
            dtype=torch.long,
            device=self.device,
        )
        if self._requested_timestep_spacing() != "fixed":
            return noise_timesteps, noise_timesteps

        configured_model_timestep = getattr(self.config, "generator_timestep", None)
        if configured_model_timestep is None:
            configured_model_timestep = getattr(self.config, "fixed_timestep", None)
        if configured_model_timestep is None:
            configured_model_timestep = getattr(self.config, "noise_timestep", None)
        if configured_model_timestep is None:
            return noise_timesteps, noise_timesteps
        if student_steps != 1:
            raise ValueError("generator_timestep/fixed_timestep is only supported for 1-step generator inference.")

        model_timesteps = torch.full(
            (batch_size,),
            int(configured_model_timestep),
            dtype=torch.long,
            device=self.device,
        )
        return noise_timesteps, model_timesteps

    def _unet_pred_for_fusion_validation(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        model_dtype = next(self.G.parameters()).dtype
        batch_size = latents.shape[0]
        if timesteps.ndim > 0 and timesteps.shape[0] != batch_size:
            if batch_size % timesteps.shape[0] != 0:
                raise ValueError(f"timesteps batch size {timesteps.shape[0]} cannot repeat to {batch_size}.")
            timesteps = timesteps.repeat(batch_size // timesteps.shape[0])
        model_input = self.scheduler.scale_model_input(latents, timesteps).to(dtype=model_dtype)
        model_input = torch.cat(
            [
                model_input,
                self.batch_inputs.mask_latent.to(dtype=model_dtype),
                self.batch_inputs.masked_image_latents.to(dtype=model_dtype),
            ],
            dim=1,
        )
        transformer_ctx = self.flops_profiler.bucket("transformer") if self.flops_profiler is not None else contextlib.nullcontext()
        with transformer_ctx:
            return self.G(
                model_input,
                timesteps,
                encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"].to(dtype=model_dtype),
                added_cond_kwargs={
                    "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"].to(dtype=model_dtype),
                    "time_ids": self.batch_inputs.add_time_ids.to(dtype=model_dtype),
                },
            ).sample

    def _decode_latents_for_fusion_validation(self, latents: torch.Tensor) -> torch.Tensor:
        vae_decode_ctx = self.flops_profiler.bucket("vae_decode") if self.flops_profiler is not None else contextlib.nullcontext()
        with vae_decode_ctx:
            return self.vae.decode(latents.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()

    @staticmethod
    def _pixel_alpha(alpha_pixel: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(alpha_pixel.float(), size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    @torch.no_grad()
    def _forward_frozen_generator_for_fusion_validation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        clear_cross_attention_scores(self.cross_attention_scores)
        z = self.batch_inputs.z_lq
        noise_timesteps, model_timesteps = self._select_generator_timesteps(z.shape[0])
        eps = self._unet_pred_for_fusion_validation(z, model_timesteps)
        z_pred = self._pred_x0(eps.float(), z.float(), noise_timesteps).to(dtype=z.dtype)

        if len(self.cross_attention_scores) == 0:
            print("[Warn] No cross-attention scores captured; falling back to mask_latent as attention prior.")
            attn_map = self.batch_inputs.mask_latent.to(dtype=z.dtype)
        else:
            _, layer_attn = next(iter(self.cross_attention_scores.items()))
            attn_map = resize_attn_map_divide2(
                layer_attn,
                self.batch_inputs.mask_latent,
                getattr(self.config, "fuse_index", 5),
            ).mean(dim=1, keepdim=True)
        clear_cross_attention_scores(self.cross_attention_scores)
        return z_pred.detach(), attn_map.detach().clamp(0.0, 1.0)

    @torch.no_grad()
    def forward_learnable_fusion_validation(self):
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward_learnable_fusion_validation_impl()
        return self._forward_learnable_fusion_validation_impl()

    @torch.no_grad()
    def _forward_learnable_fusion_validation_impl(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        z_pred, attn_map = self._forward_frozen_generator_for_fusion_validation()
        fusion_mask = self.batch_inputs.mask_latent
        with self._fusion_autocast_context():
            fusion_out = self.fusion_module(
                self._fusion_tensor(z_pred),
                self._fusion_tensor(self.batch_inputs.masked_image_latents),
                self._fusion_tensor(fusion_mask),
                self._fusion_tensor(attn_map),
            )
            # CUDA Graph outputs from compiled modules are backed by reusable buffers.
            # Own the values before invoking another compiled module such as the VAE decoder.
            z_fused = fusion_out["z_fused"].clone()
            alpha_pixel = fusion_out["alpha_pixel"].clone()
            alpha_latent = fusion_out["alpha_latent"].clone() if self.save_fusion_debug else None
            x_latent_fused = self._decode_latents_for_fusion_validation(z_fused)
            x_unfused = None
            if self.save_fusion_debug:
                x_unfused = self._decode_latents_for_fusion_validation(z_pred)
            alpha_pixel_img = self._pixel_alpha(alpha_pixel, x_latent_fused.shape[-2:])
            x_fused = (
                alpha_pixel_img * x_latent_fused.float()
                + (1.0 - alpha_pixel_img) * self.batch_inputs.input_img.float()
            ).clamp(-1.0, 1.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_inference_time = time.time() - start_time
        if not self.save_fusion_debug:
            return x_fused, None
        return x_fused, {
            "x_fused": x_fused,
            "x_latent_fused": x_latent_fused,
            "x_unfused": x_unfused,
            "z_pred": z_pred,
            "z_fused": z_fused,
            "attn_map": attn_map,
            "alpha_pixel_img": alpha_pixel_img,
            "alpha_latent": alpha_latent,
            "alpha_pixel": alpha_pixel,
        }

    def init_text_models(self):
        self.text_encoder_dtype = parse_torch_dtype(
            getattr(self.config, "text_encoder_dtype", None),
            self.weight_dtype,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer_2")

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.text_encoder_dtype,
        ).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.text_encoder_dtype,
        ).to(self.device)

        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder_2.eval().requires_grad_(False)

    def _resolve_postfuse_weights_path(self) -> Optional[str]:
        local_path = os.path.join(self.config.base_model_path, "postfuse_module", "model.safetensors")
        if os.path.exists(local_path):
            return local_path

        if self.config.base_model_path == "jixin0101/ObjectClear":
            try:
                from huggingface_hub import hf_hub_download

                return hf_hub_download(
                    repo_id="jixin0101/ObjectClear",
                    filename="model.safetensors",
                    subfolder="postfuse_module",
                    cache_dir=getattr(self.config, "cache_dir", None),
                )
            except Exception:
                return None
        return None

    def init_objectclear_modules(self):
        self.object_encoder_dtype = parse_torch_dtype(
            getattr(self.config, "object_encoder_dtype", None),
            self.weight_dtype,
        )
        self.postfuse_dtype = parse_torch_dtype(
            getattr(self.config, "postfuse_dtype", None),
            self.object_encoder_dtype,
        )
        self.image_prompt_encoder = CLIPImageEncoder.from_pretrained(
            self.config.base_model_path,
            cache_dir=getattr(self.config, "cache_dir", None),
        ).to(self.device, dtype=self.object_encoder_dtype)
        self.image_prompt_encoder.eval().requires_grad_(False)

        self.postfuse_module = PostfuseModule(embed_dim=2048, embed_dim_img=768).to(
            self.device, dtype=self.postfuse_dtype
        )
        postfuse_path = self._resolve_postfuse_weights_path()
        if postfuse_path is not None and os.path.exists(postfuse_path):
            state_dict = load_file(postfuse_path)
            try:
                self.postfuse_module.load_state_dict(state_dict)
            except RuntimeError:
                self.postfuse_module.load_state_dict(state_dict, strict=False)
        self.postfuse_module.eval().requires_grad_(False)
        print(
            f"[Info] dtype setup: weight={self.weight_dtype}, text_encoder={self.text_encoder_dtype}, "
            f"object_encoder={self.object_encoder_dtype}, postfuse={self.postfuse_dtype}"
        )

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.vae.eval().requires_grad_(False)

    def init_generator(self):
        self.G = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.G.eval().requires_grad_(False)

        # Full-finetune checkpoints should be loaded into the plain UNet without LoRA adapters.
        if self.ckpt_mode != "full":
            target_modules = self.config.lora_modules
            lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            self.G.add_adapter(lora_cfg)

        if self.use_attention_guided_fusion or self.use_learnable_fusion or self.debug_dump_intermediates:
            unet_store_cross_attention_scores(
                self.G,
                self.cross_attention_scores,
                layers=7 if self.debug_dump_intermediates else getattr(self.config, "attn_loss_layers", 5),
                capture_mode=self.attention_capture_mode,
                target_layer=self.attention_capture_layer,
            )

    def init_fusion_module(self):
        self.fusion_module = None
        if not self.use_learnable_fusion:
            return

        state_file = resolve_fusion_module_state_file(self.fusion_module_path)
        if state_file is None:
            self.use_learnable_fusion = False
            self.use_attention_guided_fusion = True
            print(
                f"[Warn] Could not resolve fusion module checkpoint from {self.fusion_module_path}. "
                "Falling back to default fusion."
            )
            return

        fusion_precision = self._fusion_precision_mode()
        fusion_param_dtype = self.weight_dtype if fusion_precision == "weight" else torch.float32
        self.fusion_module = LearnableAttentionFusion(
            latent_channels=int(getattr(self.config, "fusion_latent_channels", 4)),
            hidden_channels=int(getattr(self.config, "fusion_hidden_channels", 32)),
            num_layers=int(getattr(self.config, "fusion_num_layers", 3)),
            logit_eps=float(getattr(self.config, "fusion_logit_eps", 1e-4)),
        ).to(self.device, dtype=fusion_param_dtype)

        ckpt_obj = safe_load_state_file(state_file)
        state_dict = self._extract_state_dict(ckpt_obj)
        if not isinstance(state_dict, Mapping):
            self.use_learnable_fusion = False
            self.use_attention_guided_fusion = True
            print(f"[Warn] Unsupported fusion module checkpoint format: {state_file}. Falling back to default fusion.")
            return

        state_dict = strip_module_prefixes_from_state_dict(state_dict)
        model_state = self.fusion_module.state_dict()
        compatible_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        if not compatible_state_dict:
            self.fusion_module = None
            self.use_learnable_fusion = False
            self.use_attention_guided_fusion = True
            print(
                f"[Warn] No tensors from fusion module checkpoint matched LearnableAttentionFusion: {state_file}. "
                "Falling back to default fusion."
            )
            return
        missing, unexpected = self.fusion_module.load_state_dict(compatible_state_dict, strict=False)
        del ckpt_obj, state_dict
        self.fusion_module.eval().requires_grad_(False)
        print(
            f"[Info] Loaded learnable fusion module from {state_file}: "
            f"filtered={len(compatible_state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"fusion_precision={fusion_precision}, param_dtype={fusion_param_dtype}"
        )

    def load_generator_weights(self):
        if self.state_file is None:
            if self.ckpt_mode == "none":
                print("[Info] No checkpoint provided. Running base model.")
            else:
                print("[Info] No valid state_dict.pth found. Running base model.")
            return

        ckpt_obj = safe_load_state_file(self.state_file)
        state_dict = self._extract_state_dict(ckpt_obj)
        if not isinstance(state_dict, Mapping):
            print(f"[Warn] Unsupported checkpoint format: {self.state_file}. Running base model.")
            return

        model_state = self.G.state_dict()
        filtered, unmatched, shape_mismatch = self._filter_compatible_state_dict(state_dict, model_state)
        dtype_summary = self._state_dict_dtype_summary(state_dict)
        del ckpt_obj, state_dict

        if self.ckpt_mode == "full":
            missing, unexpected = self.G.load_state_dict(filtered, strict=False)
            print(
                f"[Info] Loaded full-finetune UNet from {self.state_file}: "
                f"filtered={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}, "
                f"unmatched={unmatched}, shape_mismatch={shape_mismatch}, checkpoint_dtypes={dtype_summary}"
            )
            if not filtered:
                print("[Warn] No checkpoint tensors matched the UNet. Running base model weights.")
            return

        missing, unexpected = self.G.load_state_dict(filtered, strict=False)
        print(
            f"[Info] Loaded LoRA from {self.state_file}: "
            f"filtered={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"unmatched={unmatched}, shape_mismatch={shape_mismatch}, checkpoint_dtypes={dtype_summary}"
        )
        if not filtered:
            print("[Warn] No checkpoint tensors matched the LoRA UNet. Running base model weights.")

    def _pick_tensor_key(self, batch: Dict[str, torch.Tensor], keys: List[str], required: bool = True):
        for key in keys:
            if key in batch:
                return batch[key]
        if required:
            raise KeyError(f"Missing required key. Tried keys: {keys}")
        return None

    def _to_image_range(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 2.0 - 1.0
        return image

    def _to_binary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = (mask > 0.5).float()
        return mask

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        cache_key = tuple(prompt) if self.inference_only else None
        if cache_key is not None and cache_key in self._prompt_embed_cache:
            return self._prompt_embed_cache[cache_key]

        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for text_input_ids_i, text_encoder in zip(
            [text_input_ids, text_input_ids_2], [self.text_encoder, self.text_encoder_2]
        ):
            text_outputs = text_encoder(
                text_input_ids_i.to(self.device),
                output_hidden_states=True,
            )
            pooled_prompt_embeds = text_outputs[0]
            text_hidden = text_outputs.hidden_states[-2]
            bs_embed, seq_len, _ = text_hidden.shape
            text_hidden = text_hidden.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(text_hidden)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompt), -1)
        result = {
            "prompt_embeds": prompt_embeds.to(dtype=self.weight_dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=self.weight_dtype),
        }
        if cache_key is not None:
            self._prompt_embed_cache[cache_key] = result
        return result

    def _get_add_time_ids(self, batch_size: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype, device=self.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def _encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.weight_dtype)
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def prepare_batch_inputs(self, batch: Dict[str, torch.Tensor]):
        input_img = self._pick_tensor_key(batch, ["input", "INPUT", "LQ", "lq", "image", "Image"])
        gt = self._pick_tensor_key(batch, ["GT", "gt", "target", "inpaint_gt", "output"], required=False)
        object_mask = self._pick_tensor_key(batch, ["object_mask", "mask", "mask_image", "MASK"])
        effect_mask = self._pick_tensor_key(
            batch,
            ["object_effect_mask", "effect_mask", "loss_mask"],
            required=False,
        )
        has_gt = batch.get("has_gt", None)
        if effect_mask is None:
            effect_mask = object_mask
        if gt is None:
            gt = input_img
            if has_gt is None:
                has_gt = torch.zeros((input_img.shape[0],), dtype=torch.bool)
        elif has_gt is None:
            has_gt = torch.ones((input_img.shape[0],), dtype=torch.bool)

        input_img = self._to_image_range(input_img).to(self.device)
        gt = self._to_image_range(gt).to(self.device)
        object_mask = self._to_binary_mask(object_mask).to(self.device)
        effect_mask = self._to_binary_mask(effect_mask).to(self.device)
        has_gt = has_gt.to(self.device)

        prompt = batch.get("txt", None)
        if prompt is None:
            prompt = batch.get("prompt", None)
        if prompt is None:
            prompt = [""] * input_img.shape[0]
        sample_name = batch.get("sample_name", None)
        if sample_name is None:
            sample_name = [
                f"sample_{self._debug_sample_counter + i:06d}"
                for i in range(input_img.shape[0])
            ]
            self._debug_sample_counter += input_img.shape[0]
        elif isinstance(sample_name, str):
            sample_name = [sample_name]
        else:
            sample_name = list(sample_name)

        profiler = self.flops_profiler
        with (profiler.bucket("condition_encode") if profiler is not None else contextlib.nullcontext()):
            c_txt = self.encode_prompt(prompt)
        bs, _, h, w = input_img.shape
        add_time_ids = self._get_add_time_ids(bs, h, w, dtype=c_txt["pooled_prompt_embeds"].dtype)

        obj_only = input_img * (object_mask > 0.5)
        with torch.no_grad():
            with (profiler.bucket("condition_encode") if profiler is not None else contextlib.nullcontext()):
                object_embeds = self.image_prompt_encoder(obj_only.to(dtype=self.object_encoder_dtype))
                fused_prompt_embeds = self.postfuse_module(
                    c_txt["prompt_embeds"].to(dtype=self.postfuse_dtype),
                    object_embeds.to(dtype=self.postfuse_dtype),
                    getattr(self.config, "fuse_index", 5),
                ).to(dtype=self.weight_dtype)

        if self.inference_only:
            vae_encode_ctx = profiler.bucket("removal_vae_encode") if profiler is not None else contextlib.nullcontext()
            with vae_encode_ctx:
                masked_image_latents = self._encode_to_latents(input_img)
            z_gt = None
            z_in = torch.randn_like(masked_image_latents)
        else:
            if profiler is not None:
                with profiler.suspend():
                    z_gt = self._encode_to_latents(gt)
            else:
                z_gt = self._encode_to_latents(gt)
            z_in = torch.randn_like(z_gt)
            vae_encode_ctx = profiler.bucket("removal_vae_encode") if profiler is not None else contextlib.nullcontext()
            with vae_encode_ctx:
                masked_image_latents = self._encode_to_latents(input_img)

        latent_h, latent_w = z_in.shape[-2:]
        mask_latent = F.interpolate(object_mask, size=(latent_h, latent_w), mode="nearest").to(dtype=self.weight_dtype)
        effect_mask_latent = F.interpolate(effect_mask, size=(latent_h, latent_w), mode="nearest").to(dtype=self.weight_dtype)
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)

        self.batch_inputs = SimpleNamespace(
            input_img=input_img,
            gt=gt,
            object_mask=object_mask,
            object_effect_mask=effect_mask,
            z_lq=z_in,
            z_gt=z_gt,
            mask_latent=mask_latent,
            effect_mask_latent=effect_mask_latent,
            masked_image_latents=masked_image_latents,
            c_txt={
                "prompt_embeds": fused_prompt_embeds,
                "pooled_prompt_embeds": c_txt["pooled_prompt_embeds"],
            },
            add_time_ids=add_time_ids,
            timesteps=timesteps,
            prompt=prompt,
            sample_name=sample_name,
            has_gt=has_gt,
        )

    def _get_shallowest_cross_attention(self):
        if not self.cross_attention_scores:
            return None, None
        layer_order = [
            "down_blocks.0",
            "down_blocks.1",
            "down_blocks.2",
            "mid_block",
            "up_blocks.1",
            "up_blocks.2",
            "up_blocks.3",
        ]

        def sort_key(name):
            for idx, prefix in enumerate(layer_order):
                if prefix in name:
                    return idx
            return len(layer_order)

        key = min(self.cross_attention_scores.keys(), key=sort_key)
        return key, self.cross_attention_scores[key]

    @torch.no_grad()
    def _dump_debug_step(
        self,
        step_idx: int,
        timestep,
        z: torch.Tensor,
        mask: torch.Tensor,
        fuse_index: int,
        pred_original_sample: Optional[torch.Tensor] = None,
    ):
        if not self.debug_dump_intermediates or not self.debug_dump_dir:
            return

        sample_names = getattr(self.batch_inputs, "sample_name", None) or [
            f"sample_{i:06d}" for i in range(z.shape[0])
        ]
        t_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)

        preview = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        x0_preview = None
        if pred_original_sample is not None:
            x0_preview = self.vae.decode(
                pred_original_sample.to(self.weight_dtype) / self.vae.config.scaling_factor
            ).sample.float()
        attn_key, attn_probs = self._get_shallowest_cross_attention()
        attn_map = None
        if attn_probs is not None:
            try:
                attn_map = resize_attn_map_divide2(attn_probs, mask, fuse_index).mean(dim=1, keepdim=True)
            except Exception as exc:
                print(f"[DebugDump] Failed to resize cross-attention at step={step_idx}, layer={attn_key}: {exc}")

        for bi, sample_name in enumerate(sample_names):
            sample_dir = os.path.join(self.debug_dump_dir, _normalize_name(str(sample_name)))
            save_tensor_image(preview[bi], os.path.join(sample_dir, f"step_{step_idx:03d}_t{t_value}_prev_sample.png"))
            if x0_preview is not None:
                save_tensor_image(x0_preview[bi], os.path.join(sample_dir, f"step_{step_idx:03d}_t{t_value}_pred_x0.png"))
            if attn_map is not None:
                save_grayscale_tensor(attn_map[bi], os.path.join(sample_dir, f"step_{step_idx:03d}_t{t_value}_cross_attn.png"))

    @torch.no_grad()
    def forward_generator(self):
        if (
            self.use_learnable_fusion
            and self.fusion_module is not None
            and self._fusion_validation_mode() == "training"
        ):
            if not getattr(self, "_logged_fusion_validation_mode", False):
                print("[Info] fusion_validation_mode=training; using trainer-matched learnable fusion validation path.")
                self._logged_fusion_validation_mode = True
            return self.forward_learnable_fusion_validation()

        z = self.batch_inputs.z_lq
        image_latents = self.batch_inputs.masked_image_latents
        noise = torch.randn_like(image_latents)
        mask = self.batch_inputs.mask_latent.to(dtype=z.dtype)
        fuse_index = getattr(self.config, "fuse_index", 5)
        attn_map = None

        needs_attn_map = self.use_attention_guided_fusion or self.use_learnable_fusion

        if needs_attn_map:
            clear_cross_attention_scores(self.cross_attention_scores)

        timesteps = self._set_inference_timesteps(getattr(self.config, "distill_steps", 4))
        one_step_output_type = self._one_step_output_type(len(timesteps))
        if not getattr(self, "_logged_one_step_output_type", False):
            print(f"[Info] one_step_output_type={one_step_output_type}")
            self._logged_one_step_output_type = True
        
        import time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        loop_start_time = time.time()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((z.shape[0],), int(t), dtype=torch.long, device=self.device)
            z_model = self.scheduler.scale_model_input(z, t)
            if self.G.config.in_channels == 9:
                z_model = torch.cat([z_model, self.batch_inputs.mask_latent, self.batch_inputs.masked_image_latents], dim=1)

            transformer_ctx = self.flops_profiler.bucket("transformer") if self.flops_profiler is not None else contextlib.nullcontext()
            with transformer_ctx:
                eps = self.G(
                    z_model,
                    t_batch,
                    encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                    added_cond_kwargs={
                        "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                        "time_ids": self.batch_inputs.add_time_ids,
                    },
                ).sample

            if one_step_output_type == "pred_x0":
                pred_original_sample = self._pred_x0(eps.float(), z.float(), t_batch).to(dtype=z.dtype)
                scheduler_output = SimpleNamespace(
                    prev_sample=pred_original_sample,
                    pred_original_sample=pred_original_sample,
                )
            else:
                scheduler_output = self.scheduler.step(eps, t, z)
            z = scheduler_output.prev_sample

            if needs_attn_map:
                if i == 0 and len(timesteps) > 1:
                    init_latents_proper = image_latents
                    noise_timestep = timesteps[i + 1]
                    if not torch.is_tensor(noise_timestep):
                        noise_timestep = torch.tensor(noise_timestep, dtype=torch.long, device=self.device)
                    noise_timestep = noise_timestep.reshape(1).to(self.device)
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper,
                        noise,
                        noise_timestep,
                    )
                    #z = (1 - mask) * init_latents_proper + mask * z

                if i == len(timesteps) - 1 and len(self.cross_attention_scores) > 0:
                    _, attn_map = next(iter(self.cross_attention_scores.items()))
                    attn_map = resize_attn_map_divide2(attn_map, mask, fuse_index)
                    attn_map = attn_map.mean(dim=1, keepdim=True)

            self._dump_debug_step(
                i,
                t,
                z,
                mask,
                fuse_index,
                pred_original_sample=getattr(scheduler_output, "pred_original_sample", None),
            )

        if needs_attn_map or self.debug_dump_intermediates:
            clear_cross_attention_scores(self.cross_attention_scores)

        if self.use_learnable_fusion and attn_map is not None and self.fusion_module is not None:
            fusion_mask = getattr(self.batch_inputs, "effect_mask_latent", self.batch_inputs.mask_latent)
            with self._fusion_autocast_context():
                fusion_out = self.fusion_module(
                    self._fusion_tensor(z),
                    self._fusion_tensor(image_latents),
                    self._fusion_tensor(fusion_mask),
                    self._fusion_tensor(attn_map),
                )
            z = fusion_out["z_fused"].to(dtype=z.dtype)
            learned_alpha_pixel = fusion_out["alpha_pixel"].float().detach()
        elif self.use_attention_guided_fusion and attn_map is not None:
            z = (1 - attn_map) * image_latents + attn_map * z
            learned_alpha_pixel = None
        else:
            learned_alpha_pixel = None
            
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        loop_end_time = time.time()
        self.last_inference_time = loop_end_time - loop_start_time

        vae_decode_ctx = self.flops_profiler.bucket("vae_decode") if self.flops_profiler is not None else contextlib.nullcontext()
        with vae_decode_ctx:
            x_pred = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        
        
        # Keep validation output consistent with pipeline's final pixel-level attention fusion.
        if self.use_learnable_fusion and attn_map is not None and learned_alpha_pixel is not None:
            with self._fusion_autocast_context():
                alpha_pixel_img = F.interpolate(
                    learned_alpha_pixel.float(),
                    size=x_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                x_pred = (
                    alpha_pixel_img * x_pred.float()
                    + (1.0 - alpha_pixel_img) * self.batch_inputs.input_img.float()
                ).clamp(-1.0, 1.0)
        elif self.use_attention_guided_fusion and attn_map is not None:
            fused_list = []
            for bi in range(x_pred.shape[0]):
                ori_np = to_uint8_image(self.batch_inputs.input_img[bi])
                pred_np = to_uint8_image(x_pred[bi])
                attn_np = (attn_map[bi].mean(dim=0).float().detach().cpu().numpy() * 255.0).astype(np.uint8)
                fused_np = attention_guided_fusion(
                    ori_np,
                    pred_np,
                    attn_np,
                    threshold=self.alpha_fusion_threshold,
                )
                fused_list.append(uint8_image_to_tensor(fused_np, device=x_pred.device, dtype=x_pred.dtype))
            x_pred = torch.stack(fused_list, dim=0)

        return x_pred, z


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    pred_01 = ((pred + 1.0) / 2.0).clamp(0, 1)
    gt_01 = ((gt + 1.0) / 2.0).clamp(0, 1)
    mask = (mask > 0.5).float()
    if mask.shape[1] == 1:
        mask = mask.repeat(1, 3, 1, 1)
    mse = ((pred_01 - gt_01) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    mse_val = float(mse.detach().cpu())
    return -10.0 * np.log10(max(mse_val, 1e-12))


def full_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_01 = ((pred + 1.0) / 2.0).clamp(0, 1)
    gt_01 = ((gt + 1.0) / 2.0).clamp(0, 1)
    mse = ((pred_01 - gt_01) ** 2).mean()
    mse_val = float(mse.detach().cpu())
    return -10.0 * np.log10(max(mse_val, 1e-12))


def get_bounding_box(mask: torch.Tensor, min_size: int = 64):
    # mask is expected to be 4D tensor: [1, 1, H, W]
    if mask.sum() == 0:
        return None
    non_zero = torch.nonzero(mask > 0)
    h_min, h_max = non_zero[:, 2].min().item(), non_zero[:, 2].max().item()
    w_min, w_max = non_zero[:, 3].min().item(), non_zero[:, 3].max().item()
    
    H, W = mask.shape[2], mask.shape[3]
    h_len = h_max - h_min + 1
    w_len = w_max - w_min + 1
    
    if h_len < min_size:
        pad = min_size - h_len
        h_min = max(0, h_min - pad // 2)
        h_max = min(H - 1, h_min + min_size - 1)
        h_min = max(0, h_max - min_size + 1)
        
    if w_len < min_size:
        pad = min_size - w_len
        w_min = max(0, w_min - pad // 2)
        w_max = min(W - 1, w_min + min_size - 1)
        w_min = max(0, w_max - min_size + 1)
        
    return h_min, h_max, w_min, w_max


def build_runtime_config(config, args):
    cfg = OmegaConf.to_container(config, resolve=True)
    cfg = dict(cfg)
    if args.base_model_path is not None:
        cfg["base_model_path"] = args.base_model_path
    if args.text_encoder_dtype is not None:
        cfg["text_encoder_dtype"] = args.text_encoder_dtype
    if args.object_encoder_dtype is not None:
        cfg["object_encoder_dtype"] = args.object_encoder_dtype
    if args.postfuse_dtype is not None:
        cfg["postfuse_dtype"] = args.postfuse_dtype
    if args.model_t is not None:
        cfg["model_t"] = int(args.model_t)
    if args.fuse_index is not None:
        cfg["fuse_index"] = int(args.fuse_index)
    if args.distill_steps is not None:
        cfg["distill_steps"] = int(args.distill_steps)
    if args.scheduler_type is not None:
        cfg["scheduler_type"] = args.scheduler_type
    if args.timestep_spacing is not None:
        cfg["timestep_spacing"] = args.timestep_spacing
        cfg["scheduler_timestep_spacing"] = args.timestep_spacing
    if args.fixed_timestep is not None:
        cfg["fixed_timestep"] = int(args.fixed_timestep)
        cfg["noise_timestep"] = int(args.fixed_timestep)
        cfg["generator_timestep"] = int(args.fixed_timestep)
    if args.apply_attention_guided_fusion is not None:
        cfg["apply_attention_guided_fusion"] = bool(args.apply_attention_guided_fusion)
    cfg["learnable_fusion"] = bool(args.learnable_fusion)
    cfg["fusion_module_path"] = args.fusion_module_path
    cfg["fusion_precision"] = args.fusion_precision
    cfg["fusion_validation_mode"] = args.fusion_validation_mode
    cfg["one_step_output_type"] = args.one_step_output_type
    cfg["alpha_fusion_threshold"] = float(args.alpha_fusion_threshold)
    cfg["debug_dump_intermediates"] = bool(args.debug_dump_intermediates)
    cfg["debug_dump_dir"] = args.debug_dump_dir
    inference_only = bool(getattr(args, "inference_only", False))
    cfg["inference_only"] = inference_only
    cfg["save_fusion_debug"] = bool(getattr(args, "save_fusion_debug", False) or not inference_only)
    cfg["attention_capture_mode"] = str(getattr(args, "attention_capture_mode", "all"))
    cfg["attention_capture_layer"] = str(
        getattr(args, "attention_capture_layer", DEFAULT_ATTENTION_CAPTURE_LAYER)
    )
    return SimpleNamespace(**cfg)


def parse_args():
    parser = argparse.ArgumentParser("Validation inference script based on objectclear.py training flow")
    parser.add_argument("--config", type=str, required=True, help="Training config yaml path")
    parser.add_argument("--weight_path", type=str, default=None, help="Checkpoint dir or state_dict.pth path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images for direct inference")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory of mask images for direct inference")
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional directory of GT images for direct inference")
    parser.add_argument("--prompt", type=str, default="", help="Prompt used in directory inference mode")
    parser.add_argument("--infer_size", type=int, default=512, help="Target short-side size for resize, preserving aspect ratio (default: 512)")
    parser.add_argument(
        "--resize_mode",
        "--resize-mode",
        type=str,
        default="short_side",
        choices=["short_side", "square", "pad_to_multiple"],
        help="Direct-folder resize mode. 'pad_to_multiple' preserves native aspect ratio and pads to multiples of 8.",
    )
    parser.add_argument(
        "--save_resize",
        "--save-resize",
        type=str,
        default="original",
        choices=["original", "model"],
        help="Save direct-folder outputs at original size or model/input tensor size. 'model' matches fusion trainer validation.",
    )
    parser.add_argument("--base_model_path", type=str, default=None, help="Override base model path")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--object_encoder_dtype", type=str, default=None, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--postfuse_dtype", type=str, default=None, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=-1, help="Stop after N samples, -1 for full set")
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_t", type=int, default=None)
    parser.add_argument("--fuse_index", type=int, default=None)
    parser.add_argument("--distill_steps", type=int, default=None)
    parser.add_argument(
        "--scheduler_type",
        "--scheduler-type",
        type=str,
        default=None,
        choices=["ddpm", "ddim"],
        help="Override inference scheduler type. Fusion training uses ddim for the fixed 1-step setup.",
    )
    parser.add_argument(
        "--fixed_timestep",
        "--fixed-timestep",
        type=int,
        default=None,
        help="Use this training timestep when --timestep_spacing fixed is selected.",
    )
    parser.add_argument(
        "--timestep_spacing",
        "--timestep-spacing",
        type=str,
        default=None,
        choices=["leading", "trailing", "linspace", "fixed", "fix"],
        help="Override scheduler timestep spacing for inference. 'fixed'/'fix' uses --fixed_timestep for 1-step inference.",
    )
    agf_group = parser.add_mutually_exclusive_group()
    agf_group.add_argument(
        "--apply_attention_guided_fusion",
        dest="apply_attention_guided_fusion",
        action="store_true",
        help="Enable attention-guided fusion in forward_generator",
    )
    agf_group.add_argument(
        "--disable_attention_guided_fusion",
        dest="apply_attention_guided_fusion",
        action="store_false",
        help="Disable attention-guided fusion in forward_generator",
    )
    parser.set_defaults(apply_attention_guided_fusion=None)
    parser.add_argument(
        "--alpha_fusion_threshold",
        type=float,
        default=0.5,
        help="Hard threshold for final pixel-level attention fusion alpha map.",
    )
    parser.add_argument(
        "--one_step_output_type",
        "--one-step-output-type",
        type=str,
        default="auto",
        choices=["auto", "pred_x0", "scheduler"],
        help="For fixed 1-step inference, 'pred_x0' matches fusion-training validation; 'scheduler' keeps the old sampler step path.",
    )
    parser.add_argument(
        "--learnable_fusion",
        "--learnable-fusion",
        action="store_true",
        help="Use a trained LearnableAttentionFusion module instead of the rule-based latent/pixel fusion.",
    )
    parser.add_argument(
        "--fusion_module_path",
        "--fusion-module-path",
        "--fusion_module_ckpt",
        type=str,
        default=None,
        help="Directory or file containing fusion_module.pth. If omitted, validation falls back to default fusion.",
    )
    parser.add_argument(
        "--fusion_precision",
        "--fusion-precision",
        type=str,
        default="fp32",
        choices=["fp32", "amp", "weight"],
        help="Precision for learnable fusion only. 'fp32' disables autocast around fusion; UNet/VAE still follow --precision.",
    )
    parser.add_argument(
        "--fusion_validation_mode",
        "--fusion-validation-mode",
        type=str,
        default="training",
        choices=["training", "pipeline"],
        help="For learnable fusion, 'training' uses the exact fusion-trainer validation output path.",
    )
    parser.add_argument(
        "--attention_capture_mode",
        "--attention-capture-mode",
        choices=["single", "all"],
        default="all",
        help="Capture one deployment attention layer or all trainer validation layers.",
    )
    parser.add_argument(
        "--attention_capture_layer",
        "--attention-capture-layer",
        type=str,
        default=DEFAULT_ATTENTION_CAPTURE_LAYER,
        help="Cross-attention layer used when --attention_capture_mode single is selected.",
    )
    parser.add_argument(
        "--inference_only",
        "--inference-only",
        action="store_true",
        help="Save only final predictions and per-sample/average latency; skip all quality metrics.",
    )
    parser.add_argument(
        "--save_fusion_debug",
        "--save-fusion-debug",
        action="store_true",
        help="Decode and save unfused/latent-fused debug outputs. Disabled for latency inference.",
    )
    parser.add_argument(
        "--optimize_latency",
        "--optimize-latency",
        action="store_true",
        help="Enable CUDA inference backend tuning and channels-last model weights.",
    )
    parser.add_argument(
        "--torch_compile",
        "--torch-compile",
        action="store_true",
        help="Compile UNet and VAE decoder. Compilation happens during warmup.",
    )
    parser.add_argument(
        "--torch_compile_mode",
        "--torch-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--warmup_runs",
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup passes on the first batch, excluded from latency statistics.",
    )
    parser.add_argument("--save_aux", action="store_true", help="Also save input/gt/mask images")
    parser.add_argument(
        "--debug_dump_intermediates",
        "--debug-dump-intermediates",
        action="store_true",
        help="Save per-sample per-step latent previews and shallow cross-attention maps",
    )
    parser.add_argument(
        "--debug_dump_dir",
        "--debug-dump-dir",
        type=str,
        default=None,
        help="Directory for debug dumps. Defaults to output_dir/debug when debug dumping is enabled.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone for LPIPS metric",
    )
    
    parser.add_argument("--profile_flops", action="store_true", help="Profile theoretical FLOPs for the removal pipeline")
    return parser.parse_args()


def enable_latency_optimizations(model: ObjectClearValInfer, args, device: torch.device):
    if device.type != "cuda":
        print("[Warn] CUDA latency optimizations were requested on a non-CUDA device.")
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model.G.to(memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    if model.fusion_module is not None:
        model.fusion_module.to(memory_format=torch.channels_last)
    print("[Info] Enabled cuDNN benchmark, TF32, and channels-last inference weights.")

    if not args.torch_compile:
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("--torch_compile requires PyTorch 2.0 or newer.")

    model.G = torch.compile(model.G, mode=args.torch_compile_mode, fullgraph=False)
    model.vae.decoder = torch.compile(model.vae.decoder, mode=args.torch_compile_mode, fullgraph=False)
    print(
        f"[Info] torch.compile enabled for UNet/VAE decoder with mode={args.torch_compile_mode}; "
        "the small fusion head stays eager to avoid CUDA Graph output aliasing."
    )


def run_inference_only(model, dataloader, args, device, weight_dtype):
    pred_root = os.path.join(args.output_dir, "pred")
    os.makedirs(pred_root, exist_ok=True)
    latency_csv = os.path.join(args.output_dir, "latency.csv")
    latency_rows = []
    latency_values = []
    sample_idx = 0
    warmup_done = False
    print(
        "[Info] Latency scope: 1-step UNet + attention extraction + learnable fusion + "
        "one VAE decode + final pixel blend. Disk I/O and condition encoding are excluded."
    )

    amp_enabled = device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16)
    amp_dtype = torch.float16 if weight_dtype == torch.float16 else torch.bfloat16
    strict_fusion_validation = (
        model.use_learnable_fusion
        and model.fusion_module is not None
        and model._fusion_validation_mode() == "training"
    )

    def inference_context():
        if device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_enabled and not strict_fusion_validation,
        )

    pbar = tqdm(dataloader, desc="TurboClear Inference")
    for batch in pbar:
        if not warmup_done and args.warmup_runs > 0:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            with torch.inference_mode():
                for warmup_idx in range(args.warmup_runs):
                    with inference_context():
                        model.prepare_batch_inputs(batch)
                        model.forward_generator()
                    print(f"[Warmup] {warmup_idx + 1}/{args.warmup_runs} complete")
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
            warmup_done = True

        with torch.inference_mode():
            with inference_context():
                model.prepare_batch_inputs(batch)
                pred, _ = model.forward_generator()

        bs = pred.shape[0]
        batch_latency = model.last_inference_time / bs
        sample_names = batch.get("sample_name", None)
        orig_h = batch.get("orig_h", None)
        orig_w = batch.get("orig_w", None)
        pad_top_batch = batch.get("pad_top", None)
        pad_left_batch = batch.get("pad_left", None)

        for i in range(bs):
            if args.max_samples > 0 and sample_idx >= args.max_samples:
                break

            if sample_names is not None:
                base_name = _normalize_name(sample_names[i])
            else:
                base_name = f"{sample_idx:06d}"

            pred_save = pred[i]
            if args.save_resize == "original" and orig_h is not None and orig_w is not None:
                out_h = int(orig_h[i].item())
                out_w = int(orig_w[i].item())
                if args.resize_mode == "pad_to_multiple":
                    pad_top = int(pad_top_batch[i].item()) if pad_top_batch is not None else 0
                    pad_left = int(pad_left_batch[i].item()) if pad_left_batch is not None else 0
                    pred_save = _crop_chw_tensor(pred_save, pad_top, pad_left, out_h, out_w)
                else:
                    pred_save = _resize_chw_tensor(pred_save, (out_h, out_w), mode="bilinear")

            save_tensor_image(pred_save, os.path.join(pred_root, f"{base_name}_pred.png"))
            latency_rows.append(
                {
                    "index": sample_idx,
                    "name": base_name,
                    "latency_s": batch_latency,
                }
            )
            latency_values.append(batch_latency)
            sample_idx += 1
            pbar.set_postfix(latency=f"{batch_latency:.4f}s")
            print(f"[Latency] {base_name}: {batch_latency:.6f}s")

        if args.max_samples > 0 and sample_idx >= args.max_samples:
            break

    avg_latency = float(np.mean(latency_values)) if latency_values else float("nan")
    with open(latency_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "name", "latency_s"])
        writer.writeheader()
        writer.writerows(latency_rows)
        writer.writerow({"index": sample_idx, "name": "avg", "latency_s": avg_latency})

    print(f"[Done] Saved {sample_idx} final predictions to: {pred_root}")
    print(f"[Done] Per-sample latency csv: {latency_csv}")
    print(f"[Done] Average latency: {avg_latency:.6f}s")


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.debug_dump_intermediates and args.debug_dump_dir is None:
        args.debug_dump_dir = os.path.join(args.output_dir, "debug")

    config = OmegaConf.load(args.config)
    runtime_cfg = build_runtime_config(config, args)

    dataset_path = args.dataset_path if args.dataset_path is not None else getattr(runtime_cfg, "dataset_path", None)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug_dump_intermediates:
        os.makedirs(args.debug_dump_dir, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    weight_dtype = infer_weight_dtype(args.precision)
    if device.type == "cpu" and weight_dtype in (torch.float16, torch.bfloat16):
        print("[Warn] fp16/bf16 on CPU is unsupported for this workflow. Falling back to fp32.")
        weight_dtype = torch.float32

    if args.input_dir and args.mask_dir:
        dataset_desc = f"input_dir={args.input_dir}, mask_dir={args.mask_dir}, gt_dir={args.gt_dir}"
    else:
        dataset_desc = f"dataset={dataset_path}"
    print(f"[Info] device={device}, dtype={weight_dtype}, {dataset_desc}")

    model = ObjectClearValInfer(
        config=runtime_cfg,
        weight_path=args.weight_path,
        device=device,
        weight_dtype=weight_dtype,
    )

    if args.optimize_latency:
        enable_latency_optimizations(model, args, device)

    if args.profile_flops:
        profiler = TheoreticalFLOPsProfiler()
        profiler.register_model(model.text_encoder)
        profiler.register_model(model.text_encoder_2)
        profiler.register_model(model.image_prompt_encoder)
        profiler.register_model(model.postfuse_module)
        profiler.register_model(model.vae)
        profiler.register_model(model.G)
        model.flops_profiler = profiler
        print("[Profiler] Enabled theoretical FLOPs for removal pipeline: "
              "Condition Encode + Removal VAE Encode + Transformer + VAE Decode")
    else:
        model.flops_profiler = None

    if args.input_dir and args.mask_dir:
        dataset = DirectoryInpaintDataset(
            input_dir=args.input_dir,
            mask_dir=args.mask_dir,
            gt_dir=args.gt_dir,
            prompt=args.prompt,
            infer_size=args.infer_size,
            resize_mode=args.resize_mode,
        )
    else:
        if dataset_path is None:
            raise ValueError("Either --dataset_path (or config.dataset_path) OR --input_dir + --mask_dir must be provided.")
        dataset = SDInpaintImageDataset(
            dataset_path=dataset_path,
            is_sdxl=True,
            use_blank_mask=False,
            blank_mask_prob=0.0,
            color_augmentation=False,
            flip_augmentation=False,
            random_mask_dilation=False,
            random_mask_erosion=False,
        )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if args.inference_only:
        if args.profile_flops:
            raise ValueError("--inference_only cannot be combined with --profile_flops because hooks distort latency.")
        run_inference_only(model, dataloader, args, device, weight_dtype)
        return

    try:
        import lpips
    except Exception:
        lpips = None
    import pyiqa

    metrics_csv = os.path.join(args.output_dir, "metrics.csv")
    rows = []
    valid_psnr_values = []
    valid_full_psnr_values = []
    valid_lpips_values = []
    sample_idx = 0

    lpips_fn = None
    if lpips is not None:
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_fn.eval().requires_grad_(False)
    else:
        print("[Warn] lpips package is not available. LPIPS metric will be NaN.")

    print("[Info] Initializing pyiqa metrics...")
    musiq_fn = pyiqa.create_metric('musiq', device=device)
    clipiqa_fn = pyiqa.create_metric('clipiqa', device=device)

    cached_flops = None
    total_latency_values = []
    
    valid_lpips_local_values = []
    valid_musiq_values = []
    valid_clipiqa_values = []

    amp_enabled = device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16)
    amp_dtype = torch.float16 if weight_dtype == torch.float16 else torch.bfloat16
    strict_fusion_validation = (
        model.use_learnable_fusion
        and model.fusion_module is not None
        and model._fusion_validation_mode() == "training"
    )
    if strict_fusion_validation:
        print("[Info] Disabling validation-loop autocast to match fusion trainer validation.")

    pbar = tqdm(dataloader, desc="Validation Inference")
    for batch in pbar:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled and not strict_fusion_validation):
                should_measure_flops = args.profile_flops and cached_flops is None
                if args.profile_flops:
                    model.flops_profiler.reset_totals()
                    model.flops_profiler.set_enabled(should_measure_flops)
                    if should_measure_flops and cached_flops is None:
                        print("\n[Profiler] Measuring theoretical FLOPs on the first removal batch...")

                model.prepare_batch_inputs(batch)
                pred, pred_aux = model.forward_generator()

        if args.profile_flops:
            if should_measure_flops:
                current_flops = model.flops_profiler.get_totals()
                batch_flops = {k: float(v) / pred.shape[0] for k, v in current_flops.items()}
                if cached_flops is None:
                    cached_flops = dict(batch_flops)
            else:
                batch_flops = dict(cached_flops)
        else:
            batch_flops = None

        batch_latency = model.last_inference_time / pred.shape[0]
        if batch_flops is not None:
            cond_flops_t = batch_flops["condition_encode"] / 1e12
            removal_vae_encode_flops_t = batch_flops["removal_vae_encode"] / 1e12
            transformer_flops_t = batch_flops["transformer"] / 1e12
            vae_decode_flops_t = batch_flops["vae_decode"] / 1e12
            total_flops_t = (
                batch_flops["condition_encode"]
                + batch_flops["removal_vae_encode"]
                + batch_flops["transformer"]
                + batch_flops["vae_decode"]
            ) / 1e12
        else:
            cond_flops_t = float("nan")
            removal_vae_encode_flops_t = float("nan")
            transformer_flops_t = float("nan")
            vae_decode_flops_t = float("nan")
            total_flops_t = float("nan")


        bs = pred.shape[0]
        sample_names = batch.get("sample_name", None)
        orig_h = batch.get("orig_h", None)
        orig_w = batch.get("orig_w", None)
        pad_top_batch = batch.get("pad_top", None)
        pad_left_batch = batch.get("pad_left", None)
        for i in range(bs):
            if args.max_samples > 0 and sample_idx >= args.max_samples:
                break

            pred_i = pred[i]
            input_i = model.batch_inputs.input_img[i]
            gt_i = model.batch_inputs.gt[i]
            mask_i = model.batch_inputs.object_mask[i]
            effect_mask_i = model.batch_inputs.object_effect_mask[i]

            if args.save_resize == "original" and orig_h is not None and orig_w is not None:
                out_h = int(orig_h[i].item())
                out_w = int(orig_w[i].item())
                pad_top_i = int(pad_top_batch[i].item()) if pad_top_batch is not None else 0
                pad_left_i = int(pad_left_batch[i].item()) if pad_left_batch is not None else 0
                if args.resize_mode == "pad_to_multiple":
                    pred_save = _crop_chw_tensor(pred_i, pad_top_i, pad_left_i, out_h, out_w)
                    input_save = _crop_chw_tensor(input_i, pad_top_i, pad_left_i, out_h, out_w)
                    gt_save = _crop_chw_tensor(gt_i, pad_top_i, pad_left_i, out_h, out_w)
                    mask_save = _crop_chw_tensor(mask_i, pad_top_i, pad_left_i, out_h, out_w)
                    effect_mask_save = _crop_chw_tensor(effect_mask_i, pad_top_i, pad_left_i, out_h, out_w)
                else:
                    pred_save = _resize_chw_tensor(pred_i, (out_h, out_w), mode="bilinear")
                    input_save = _resize_chw_tensor(input_i, (out_h, out_w), mode="bilinear")
                    gt_save = _resize_chw_tensor(gt_i, (out_h, out_w), mode="bilinear")
                    mask_save = _resize_chw_tensor(mask_i, (out_h, out_w), mode="nearest")
                    effect_mask_save = _resize_chw_tensor(effect_mask_i, (out_h, out_w), mode="nearest")
            else:
                pred_save = pred_i
                input_save = input_i
                gt_save = gt_i
                mask_save = mask_i
                effect_mask_save = effect_mask_i

            if sample_names is not None:
                base_name = _normalize_name(sample_names[i])
            else:
                base_name = f"{sample_idx:06d}"
            pred_root = os.path.join(args.output_dir, "pred")
            os.makedirs(pred_root, exist_ok=True)

            has_fusion_debug = isinstance(pred_aux, Mapping) and all(
                key in pred_aux for key in ("x_unfused", "x_latent_fused", "x_fused")
            )
            if has_fusion_debug:
                unfused_i = pred_aux["x_unfused"][i]
                latent_fused_i = pred_aux["x_latent_fused"][i]
                learned_fused_i = pred_aux["x_fused"][i]
                alpha_batch = pred_aux.get("alpha_pixel_img")
                attn_batch = pred_aux.get("attn_map")
                alpha_i = alpha_batch[i] if isinstance(alpha_batch, torch.Tensor) else torch.zeros_like(mask_i[:1])
                attn_i = attn_batch[i] if isinstance(attn_batch, torch.Tensor) else torch.zeros_like(mask_i[:1])

                if args.save_resize == "original" and orig_h is not None and orig_w is not None:
                    if args.resize_mode == "pad_to_multiple":
                        unfused_save = _crop_chw_tensor(unfused_i, pad_top_i, pad_left_i, out_h, out_w)
                        latent_fused_save = _crop_chw_tensor(latent_fused_i, pad_top_i, pad_left_i, out_h, out_w)
                        learned_fused_save = _crop_chw_tensor(learned_fused_i, pad_top_i, pad_left_i, out_h, out_w)
                        alpha_save = _crop_chw_tensor(alpha_i, pad_top_i, pad_left_i, out_h, out_w)
                        attn_save = _crop_chw_tensor(attn_i, pad_top_i, pad_left_i, out_h, out_w)
                    else:
                        unfused_save = _resize_chw_tensor(unfused_i, (out_h, out_w), mode="bilinear")
                        latent_fused_save = _resize_chw_tensor(latent_fused_i, (out_h, out_w), mode="bilinear")
                        learned_fused_save = _resize_chw_tensor(learned_fused_i, (out_h, out_w), mode="bilinear")
                        alpha_save = _resize_chw_tensor(alpha_i, (out_h, out_w), mode="bilinear")
                        attn_save = _resize_chw_tensor(attn_i, (out_h, out_w), mode="bilinear")
                    strip_input = input_save
                    strip_gt = gt_save
                    strip_mask = mask_save
                else:
                    unfused_save = unfused_i
                    latent_fused_save = latent_fused_i
                    learned_fused_save = learned_fused_i
                    alpha_save = alpha_i
                    attn_save = attn_i
                    strip_input = input_i
                    strip_gt = gt_i
                    strip_mask = mask_i

                for pred_name, image_tensor in (
                    ("unfused", unfused_save),
                    ("latent_fused", latent_fused_save),
                    ("learned_fused", learned_fused_save),
                ):
                    pred_dir = os.path.join(pred_root, pred_name)
                    os.makedirs(pred_dir, exist_ok=True)
                    save_tensor_image(image_tensor, os.path.join(pred_dir, f"{base_name}_{pred_name}.png"))

                save_fusion_debug_strip(
                    strip_input,
                    strip_gt,
                    unfused_save,
                    latent_fused_save,
                    learned_fused_save,
                    alpha_save,
                    attn_save,
                    strip_mask,
                    os.path.join(pred_root, f"{base_name}_input_gt_unfused_latent_learned_alpha_attn_mask.png"),
                )
            else:
                save_tensor_image(pred_save, os.path.join(pred_root, f"{base_name}_pred.png"))

            if args.save_aux:
                os.makedirs(os.path.join(args.output_dir, "input"), exist_ok=True)
                os.makedirs(os.path.join(args.output_dir, "gt"), exist_ok=True)
                os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)
                save_tensor_image(input_save, os.path.join(args.output_dir, "input", f"{base_name}.png"))
                if bool(model.batch_inputs.has_gt[i].item()):
                    save_tensor_image(gt_save, os.path.join(args.output_dir, "gt", f"{base_name}.png"))
                tensor_mask_to_pil(mask_save).save(os.path.join(args.output_dir, "mask", f"{base_name}.png"))

            if bool(model.batch_inputs.has_gt[i].item()):
                psnr_masked = masked_psnr(pred_i.unsqueeze(0), gt_i.unsqueeze(0), effect_mask_i.unsqueeze(0))
                valid_psnr_values.append(psnr_masked)

                psnr_full = full_psnr(pred_i.unsqueeze(0), gt_i.unsqueeze(0))
                valid_full_psnr_values.append(psnr_full)

                if lpips_fn is not None:
                    with torch.no_grad():
                        lpips_val = float(
                            lpips_fn(
                                pred_i.unsqueeze(0).to(device=device, dtype=torch.float32),
                                gt_i.unsqueeze(0).to(device=device, dtype=torch.float32),
                            ).mean().detach().cpu()
                        )
                        
                        bb = get_bounding_box(effect_mask_i.unsqueeze(0), min_size=64)
                        if bb is not None:
                            h_min, h_max, w_min, w_max = bb
                            if h_max - h_min > 4 and w_max - w_min > 4:
                                pred_local = pred_i.unsqueeze(0)[:, :, h_min:h_max+1, w_min:w_max+1]
                                gt_local = gt_i.unsqueeze(0)[:, :, h_min:h_max+1, w_min:w_max+1]
                                lpips_local_val = float(lpips_fn(
                                    pred_local.to(device=device, dtype=torch.float32), 
                                    gt_local.to(device=device, dtype=torch.float32)
                                ).mean().detach().cpu())
                            else:
                                lpips_local_val = float("nan")
                        else:
                            lpips_local_val = float("nan")
                    valid_lpips_values.append(lpips_val)
                    if not torch.isnan(torch.tensor(lpips_local_val)):
                        valid_lpips_local_values.append(lpips_local_val)
                else:
                    lpips_val = float("nan")
                    lpips_local_val = float("nan")
            else:
                psnr_masked = float("nan")
                psnr_full = float("nan")
                lpips_val = float("nan")
                lpips_local_val = float("nan")
                
            # MUSIQ and CLIPIQA
            pred_01 = ((pred_i.unsqueeze(0) + 1.0) / 2.0).clamp(0, 1).to(device)
            musiq_val = float(musiq_fn(pred_01).item())
            clipiqa_val = float(clipiqa_fn(pred_01).item())
            valid_musiq_values.append(musiq_val)
            valid_clipiqa_values.append(clipiqa_val)
            total_latency_values.append(batch_latency)

            rows.append({
                "index": sample_idx,
                "name": base_name,
                "has_gt": int(bool(model.batch_inputs.has_gt[i].item())),
                "Latency_s": batch_latency,
                "Total_FLOPs_T": total_flops_t,
                "CondEncode_FLOPs_T": cond_flops_t,
                "Transformer_FLOPs_T": transformer_flops_t,
                "VAEDecode_FLOPs_T": vae_decode_flops_t,
                "RemovalVAEEncode_FLOPs_T": removal_vae_encode_flops_t,
                "masked_psnr": psnr_masked,
                "full_psnr": psnr_full,
                "lpips": lpips_val,
                "lpips_local": lpips_local_val,
                "musiq": musiq_val,
                "clipiqa": clipiqa_val
            })
            sample_idx += 1

        if args.max_samples > 0 and sample_idx >= args.max_samples:
            break
            
    avg_psnr = float(np.mean(valid_psnr_values)) if valid_psnr_values else float("nan")
    avg_full_psnr = float(np.mean(valid_full_psnr_values)) if valid_full_psnr_values else float("nan")
    avg_lpips = float(np.mean(valid_lpips_values)) if valid_lpips_values else float("nan")
    avg_lpips_local = float(np.mean(valid_lpips_local_values)) if valid_lpips_local_values else float("nan")
    avg_musiq = float(np.mean(valid_musiq_values)) if valid_musiq_values else float("nan")
    avg_clipiqa = float(np.mean(valid_clipiqa_values)) if valid_clipiqa_values else float("nan")
    avg_latency = float(np.mean(total_latency_values)) if total_latency_values else float("nan")
    valid_total_flops_values = [r["Total_FLOPs_T"] for r in rows if not torch.isnan(torch.tensor(r["Total_FLOPs_T"]))]
    valid_cond_flops_values = [r["CondEncode_FLOPs_T"] for r in rows if not torch.isnan(torch.tensor(r["CondEncode_FLOPs_T"]))]
    valid_transformer_flops_values = [r["Transformer_FLOPs_T"] for r in rows if not torch.isnan(torch.tensor(r["Transformer_FLOPs_T"]))]
    valid_vae_decode_flops_values = [r["VAEDecode_FLOPs_T"] for r in rows if not torch.isnan(torch.tensor(r["VAEDecode_FLOPs_T"]))]
    valid_removal_vae_encode_values = [r["RemovalVAEEncode_FLOPs_T"] for r in rows if not torch.isnan(torch.tensor(r["RemovalVAEEncode_FLOPs_T"]))]
    avg_flops = float(np.mean(valid_total_flops_values)) if valid_total_flops_values else float("nan")
    avg_cond_flops = float(np.mean(valid_cond_flops_values)) if valid_cond_flops_values else float("nan")
    avg_transformer_flops = float(np.mean(valid_transformer_flops_values)) if valid_transformer_flops_values else float("nan")
    avg_vae_decode_flops = float(np.mean(valid_vae_decode_flops_values)) if valid_vae_decode_flops_values else float("nan")
    avg_removal_vae_encode_flops = float(np.mean(valid_removal_vae_encode_values)) if valid_removal_vae_encode_values else float("nan")
    
    total_row = {
        "index": sample_idx,
        "name": "avg",
        "has_gt": "nan",
        "Latency_s": avg_latency,
        "Total_FLOPs_T": avg_flops,
        "CondEncode_FLOPs_T": avg_cond_flops,
        "Transformer_FLOPs_T": avg_transformer_flops,
        "VAEDecode_FLOPs_T": avg_vae_decode_flops,
        "RemovalVAEEncode_FLOPs_T": avg_removal_vae_encode_flops,
        "masked_psnr": avg_psnr,
        "full_psnr": avg_full_psnr,
        "lpips": avg_lpips,
        "lpips_local": avg_lpips_local,
        "musiq": avg_musiq,
        "clipiqa": avg_clipiqa
    }
    
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index", "name", "has_gt", "Latency_s",
            "Total_FLOPs_T", "CondEncode_FLOPs_T", "Transformer_FLOPs_T", "VAEDecode_FLOPs_T", "RemovalVAEEncode_FLOPs_T",
            "masked_psnr", "full_psnr", "lpips", "lpips_local", "musiq", "clipiqa"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    metrics_total_csv = os.path.join(args.output_dir, "metrics_total.csv")
    with open(metrics_total_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(total_row)

    
    print(f"[Done] Saved {len(rows)} samples to: {args.output_dir}")
    print(f"[Done] Metrics csv: {metrics_csv}")
    print(f"[Done] Average Theoretical Computation: {avg_flops:.4f} T")
    print(f"[Done] Average Condition Encode FLOPs: {avg_cond_flops:.4f} T")
    print(f"[Done] Average Transformer FLOPs: {avg_transformer_flops:.4f} T")
    print(f"[Done] Average VAE Decode FLOPs: {avg_vae_decode_flops:.4f} T")
    if args.profile_flops:
        print(f"[Done] Internal Removal VAE Encode FLOPs: {avg_removal_vae_encode_flops:.4f} T")
    print(f"[Done] Average masked PSNR (GT available only): {avg_psnr:.4f}")
    print(f"[Done] Average full-image PSNR (GT available only): {avg_full_psnr:.4f}")
    print(f"[Done] Average LPIPS (GT available only): {avg_lpips:.4f}")


if __name__ == "__main__":
    main()
