#!/usr/bin/env python
"""Merge ObjectClear/FlashClear PEFT LoRA weights into a base SDXL UNet.

The script is intentionally conservative:
- load the base UNet in fp32
- merge LoRA A/B pairs by hand for Linear and Conv2d modules
- write a new Diffusers `unet/` folder
- optionally link/copy the remaining ObjectClear pipeline components
- write `merge_report.json` for reproducibility
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_COMPONENTS = (
    "model_index.json",
    "README.md",
    "scheduler",
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "vae",
    "image_prompt_encoder",
    "postfuse_module",
)


@dataclass
class MergeItem:
    module_name: str
    module_type: str
    lora_a_key: str
    lora_b_key: str
    weight_shape: list[int]
    lora_a_shape: list[int]
    lora_b_shape: list[int]
    scale: float


@dataclass
class MergeReport:
    base_model_path: str
    lora_checkpoint: str
    output_dir: str
    unet_subfolder: str
    output_unet_subfolder: str
    dtype: str
    lora_rank_arg: int | None
    lora_alpha: float | None
    scale: float | None
    dry_run: bool
    save_dtype: str
    safe_serialization: bool
    link_base_components: bool
    copy_base_components: bool
    merged_linear_count: int = 0
    merged_conv2d_count: int = 0
    merged_items: list[MergeItem] = field(default_factory=list)
    missing_lora_pairs: list[str] = field(default_factory=list)
    missing_modules: list[str] = field(default_factory=list)
    unsupported_modules: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)
    unexpected_lora_keys: list[str] = field(default_factory=list)
    linked_components: list[str] = field(default_factory=list)
    copied_components: list[str] = field(default_factory=list)
    skipped_components: list[str] = field(default_factory=list)


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model", "unet", "module"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                return value
        return obj
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")


def normalize_key(key: str) -> str:
    for prefix in ("module.", "unet.", "model.", "base_model.model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            normalized[normalize_key(str(key))] = value
    return normalized


def resolve_lora_checkpoint(path: Path) -> Path:
    if path.is_dir():
        state_file = path / "state_dict.pth"
        if not state_file.exists():
            raise FileNotFoundError(f"No state_dict.pth found in checkpoint directory: {path}")
        return state_file
    if not path.exists():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    return path


def collect_lora_pairs(state_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, tuple[str | None, str | None]], list[str]]:
    pairs: dict[str, list[str | None]] = {}
    lora_keys: list[str] = []
    for key in sorted(state_dict):
        if ".lora_A." in key:
            module_name = key.split(".lora_A.", 1)[0]
            pairs.setdefault(module_name, [None, None])[0] = key
            lora_keys.append(key)
        elif ".lora_B." in key:
            module_name = key.split(".lora_B.", 1)[0]
            pairs.setdefault(module_name, [None, None])[1] = key
            lora_keys.append(key)
    return {name: (keys[0], keys[1]) for name, keys in pairs.items()}, lora_keys


def get_module(root: nn.Module, module_name: str) -> nn.Module | None:
    module_map = dict(root.named_modules())
    return module_map.get(module_name)


def merge_linear(module: nn.Linear, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Linear LoRA tensors must be 2D, got A={tuple(a.shape)}, B={tuple(b.shape)}")
    expected_a = (a.shape[0], module.in_features)
    expected_b = (module.out_features, a.shape[0])
    if tuple(a.shape) != expected_a or tuple(b.shape) != expected_b:
        raise ValueError(
            "Linear LoRA shape mismatch: "
            f"weight={tuple(module.weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}, "
            f"expected A={expected_a}, B={expected_b}"
        )
    return torch.matmul(b.float(), a.float()).reshape_as(module.weight).mul_(scale)


def merge_conv2d(module: nn.Conv2d, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    if a.ndim != 4 or b.ndim != 4:
        raise ValueError(f"Conv2d LoRA tensors must be 4D, got A={tuple(a.shape)}, B={tuple(b.shape)}")
    rank = a.shape[0]
    expected_a = (rank, module.in_channels // module.groups, *module.kernel_size)
    expected_b = (module.out_channels, rank, 1, 1)
    if tuple(a.shape) != expected_a or tuple(b.shape) != expected_b:
        raise ValueError(
            "Conv2d LoRA shape mismatch: "
            f"weight={tuple(module.weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}, "
            f"expected A={expected_a}, B={expected_b}"
        )
    if module.groups != 1:
        raise ValueError(f"Grouped Conv2d LoRA merge is not implemented for groups={module.groups}")
    a_flat = a.float().reshape(rank, -1)
    b_flat = b.float().reshape(module.out_channels, rank)
    return torch.matmul(b_flat, a_flat).reshape_as(module.weight).mul_(scale)


def infer_save_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported save dtype: {name}") from exc


def infer_lora_scale(args: argparse.Namespace, rank: int) -> float:
    if args.scale is not None:
        return float(args.scale)
    alpha = float(rank if args.lora_alpha is None else args.lora_alpha)
    denominator = float(rank if args.lora_rank is None else args.lora_rank)
    return alpha / denominator


def link_or_copy_components(
    base_model_path: Path,
    output_dir: Path,
    *,
    link_base_components: bool,
    copy_base_components: bool,
    overwrite_components: bool,
    report: MergeReport,
) -> None:
    if not link_base_components and not copy_base_components:
        return
    for name in BASE_COMPONENTS:
        src = base_model_path / name
        dst = output_dir / name
        if name == "unet":
            continue
        if not src.exists():
            report.skipped_components.append(name)
            continue
        if dst.exists() or dst.is_symlink():
            if overwrite_components:
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            else:
                report.skipped_components.append(name)
                continue
        if link_base_components:
            os.symlink(src, dst, target_is_directory=src.is_dir())
            report.linked_components.append(name)
        else:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            report.copied_components.append(name)


def write_report(report: MergeReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["merged_items"] = [asdict(item) for item in report.merged_items]
    with (output_dir / "merge_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def merge_lora(args: argparse.Namespace) -> MergeReport:
    from diffusers import UNet2DConditionModel

    base_model_path = Path(args.base_model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    lora_checkpoint = resolve_lora_checkpoint(Path(args.lora_checkpoint).expanduser())
    unet_subfolder = args.unet_subfolder
    output_unet_subfolder = getattr(args, "output_unet_subfolder", None) or unet_subfolder
    save_dtype = infer_save_dtype(args.save_dtype)

    if output_dir == base_model_path:
        raise ValueError("Refusing to write merged model into the base model directory.")
    if not args.dry_run and (output_dir / output_unet_subfolder).exists() and not args.overwrite:
        raise FileExistsError(f"Output UNet folder already exists: {output_dir / output_unet_subfolder}")

    report_scale: float | None
    if args.scale is not None:
        report_scale = float(args.scale)
    elif args.lora_alpha is not None and args.lora_rank is not None:
        report_scale = float(args.lora_alpha) / float(args.lora_rank)
    elif args.lora_alpha is None and args.lora_rank is None:
        report_scale = 1.0
    else:
        report_scale = None

    report = MergeReport(
        base_model_path=str(base_model_path),
        lora_checkpoint=str(lora_checkpoint),
        output_dir=str(output_dir),
        unet_subfolder=unet_subfolder,
        output_unet_subfolder=output_unet_subfolder,
        dtype="torch.float32",
        lora_rank_arg=args.lora_rank,
        lora_alpha=None if args.lora_alpha is None else float(args.lora_alpha),
        scale=report_scale,
        dry_run=bool(args.dry_run),
        save_dtype=str(save_dtype).replace("torch.", ""),
        safe_serialization=bool(args.safe_serialization),
        link_base_components=bool(args.link_base_components),
        copy_base_components=bool(args.copy_base_components),
    )

    print(f"[Info] Loading base UNet from {base_model_path}/{unet_subfolder} in fp32")
    unet = UNet2DConditionModel.from_pretrained(
        str(base_model_path),
        subfolder=unet_subfolder,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    )
    unet.eval()

    print(f"[Info] Loading LoRA checkpoint from {lora_checkpoint}")
    raw_state = safe_torch_load(lora_checkpoint)
    state_dict = normalize_state_dict(extract_state_dict(raw_state))
    lora_pairs, lora_keys = collect_lora_pairs(state_dict)
    if not lora_pairs:
        raise ValueError(f"No PEFT LoRA A/B keys found in {lora_checkpoint}")

    consumed_keys: set[str] = set()
    with torch.no_grad():
        for module_name, (a_key, b_key) in sorted(lora_pairs.items()):
            if a_key is None or b_key is None:
                report.missing_lora_pairs.append(module_name)
                continue
            module = get_module(unet, module_name)
            if module is None:
                report.missing_modules.append(module_name)
                continue
            a = state_dict[a_key]
            b = state_dict[b_key]
            rank = int(a.shape[0])
            scale = infer_lora_scale(args, rank)
            try:
                if isinstance(module, nn.Linear):
                    delta = merge_linear(module, a, b, scale)
                    module_type = "Linear"
                    report.merged_linear_count += 1
                elif isinstance(module, nn.Conv2d):
                    delta = merge_conv2d(module, a, b, scale)
                    module_type = "Conv2d"
                    report.merged_conv2d_count += 1
                else:
                    report.unsupported_modules.append(f"{module_name}: {type(module).__name__}")
                    continue
            except ValueError as exc:
                report.shape_mismatches.append(f"{module_name}: {exc}")
                continue

            if not args.dry_run:
                module.weight.data.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
            consumed_keys.update({a_key, b_key})
            report.merged_items.append(
                MergeItem(
                    module_name=module_name,
                    module_type=module_type,
                    lora_a_key=a_key,
                    lora_b_key=b_key,
                    weight_shape=[int(dim) for dim in module.weight.shape],
                    lora_a_shape=[int(dim) for dim in a.shape],
                    lora_b_shape=[int(dim) for dim in b.shape],
                    scale=scale,
                )
            )

    report.unexpected_lora_keys = sorted(set(lora_keys) - consumed_keys)

    print(
        "[Info] Merge summary: "
        f"linear={report.merged_linear_count}, conv2d={report.merged_conv2d_count}, "
        f"missing_pairs={len(report.missing_lora_pairs)}, missing_modules={len(report.missing_modules)}, "
        f"unsupported={len(report.unsupported_modules)}, mismatches={len(report.shape_mismatches)}"
    )

    if args.strict and (
        report.missing_lora_pairs
        or report.missing_modules
        or report.unsupported_modules
        or report.shape_mismatches
        or report.unexpected_lora_keys
    ):
        write_report(report, output_dir)
        raise RuntimeError(f"Strict merge failed. See report: {output_dir / 'merge_report.json'}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / output_unet_subfolder).exists() and args.overwrite:
            shutil.rmtree(output_dir / output_unet_subfolder)
        link_or_copy_components(
            base_model_path,
            output_dir,
            link_base_components=args.link_base_components,
            copy_base_components=args.copy_base_components,
            overwrite_components=args.overwrite_components,
            report=report,
        )
        if save_dtype != torch.float32:
            print(f"[Info] Casting merged UNet to {save_dtype} before save")
            unet.to(dtype=save_dtype)
        print(f"[Info] Saving merged UNet to {output_dir / output_unet_subfolder}")
        unet.save_pretrained(
            output_dir / output_unet_subfolder,
            safe_serialization=args.safe_serialization,
            max_shard_size=args.max_shard_size,
        )

    write_report(report, output_dir)
    print(f"[Info] Wrote merge report to {output_dir / 'merge_report.json'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge ObjectClear/FlashClear LoRA weights into a base UNet.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base ObjectClear/Diffusers model path.")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        required=True,
        help="Path to state_dict.pth or checkpoint directory containing state_dict.pth.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the merged model.")
    parser.add_argument("--unet_subfolder", type=str, default="unet", help="UNet subfolder name in the base model.")
    parser.add_argument(
        "--output_unet_subfolder",
        type=str,
        default=None,
        help="UNet subfolder name to write under output_dir. Defaults to --unet_subfolder.",
    )
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank. If set, default scale is alpha/rank.")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha. If omitted, alpha is inferred as each layer's rank, giving scale=1.",
    )
    parser.add_argument("--scale", type=float, default=None, help="Override LoRA scale directly.")
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"],
        help="Dtype used to save the merged UNet.",
    )
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="Diffusers save_pretrained shard size.")
    parser.add_argument(
        "--safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save UNet weights as safetensors.",
    )
    parser.add_argument("--link_base_components", action="store_true", help="Symlink non-UNet base components.")
    parser.add_argument("--copy_base_components", action="store_true", help="Copy non-UNet base components.")
    parser.add_argument(
        "--overwrite_components",
        action="store_true",
        help="Overwrite existing linked/copied base components in output_dir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output unet folder if it exists.")
    parser.add_argument("--dry_run", action="store_true", help="Inspect and report without saving merged weights.")
    parser.add_argument("--strict", action="store_true", help="Fail if any LoRA key cannot be merged cleanly.")
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass local_files_only to diffusers from_pretrained.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.link_base_components and args.copy_base_components:
        parser.error("--link_base_components and --copy_base_components are mutually exclusive.")
    merge_lora(args)


if __name__ == "__main__":
    main()
