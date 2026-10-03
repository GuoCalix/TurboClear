#!/usr/bin/env python3
"""Download inference weights into model_weights/; optionally download OBER."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "model_weights"
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"
DATASET_REPO = "sczhou/OBERDataset_ObjectClear"
DATASET_REVISION = "25e064b39bf675fefb9ac4d2e2cd962037b7206c"
BASE_REPO = "jixin0101/ObjectClear"
BASE_REVISION = "c73af80888dbd519819d0a521b5c8d0f3cda6859"
TURBO_REPO = "JGuo666/TurboClear"
TURBO_REVISION = "1fb323071034a80028b7881c7fac3ea227c79410"
BASE_FILES = [
    "model_index.json", "scheduler/scheduler_config.json",
    "tokenizer/merges.txt", "tokenizer/vocab.json",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
    "tokenizer_2/merges.txt", "tokenizer_2/vocab.json",
    "tokenizer_2/special_tokens_map.json", "tokenizer_2/tokenizer_config.json",
] + [
    f"{component}/{filename}"
    for component in ("text_encoder", "text_encoder_2", "image_prompt_encoder", "postfuse_module")
    for filename in ("config.json", "model.safetensors")
] + [
    f"{component}/{filename}"
    for component in ("unet", "vae")
    for filename in ("config.json", "diffusion_pytorch_model.safetensors")
]
TURBO_FILES = ["sdxl/state_dict.pth", "fusion/fusion_module.pth"]


def add_download_arguments(parser):
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--weights-dir", type=Path,
        help="Store ObjectClear/ and TurboClear/ here (default: repository model_weights/).",
    )
    destination.add_argument(
        "--cache-dir", type=Path,
        help="Use a Hugging Face cache directory instead of model_weights/.",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Use downloaded files without network access; fail if any are missing.",
    )


def download_weights(weights_dir=None, cache_dir=None, local_files_only=False):
    if weights_dir is not None and cache_dir is not None:
        raise ValueError("Choose either weights_dir or cache_dir")
    if weights_dir is None and cache_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    roots = []
    for repo, revision, files, folder in (
        (BASE_REPO, BASE_REVISION, BASE_FILES, "ObjectClear"),
        (TURBO_REPO, TURBO_REVISION, TURBO_FILES, "TurboClear"),
    ):
        local_dir = Path(weights_dir).expanduser().resolve() / folder if weights_dir else None
        if local_files_only and local_dir is not None:
            root = local_dir
        else:
            root = Path(snapshot_download(
                repo_id=repo,
                revision=revision,
                allow_patterns=files,
                local_dir=str(local_dir) if local_dir else None,
                cache_dir=str(Path(cache_dir).expanduser().resolve()) if cache_dir else None,
                local_files_only=local_files_only,
            ))
        # A cached snapshot can exist but lack some required files.
        missing = [name for name in files if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete weights for {repo} at {root}: {', '.join(missing)}")
        roots.append(root.resolve())
    base, turbo = roots
    return {
        "BASE_MODEL_PATH": str(base),
        "WEIGHT_PATH": str(turbo / "sdxl"),
        "FUSION_MODULE_PATH": str(turbo / "fusion/fusion_module.pth"),
    }


def download_dataset(datasets_dir=DEFAULT_DATASETS_DIR, local_files_only=False):
    root = Path(datasets_dir).expanduser().resolve() / "OBER"
    files = [f"data/train-{i:05d}-of-00053.parquet" for i in range(53)]
    files.append("data/test-00000-of-00001.parquet")
    if not local_files_only:
        snapshot_download(
            repo_id=DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
            local_dir=str(root), allow_patterns=files + ["README.md", "LICENSE*"],
        )
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete OBER dataset at {root}: {', '.join(missing)}")
    return str(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser)
    parser.add_argument("--with-dataset", action="store_true",
                        help="Also download OBER training/test parquet shards (off by default).")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR,
                        help="Dataset parent directory (default: repository datasets/).")
    args = parser.parse_args()
    paths = download_weights(args.weights_dir, args.cache_dir, args.local_files_only)
    if args.with_dataset:
        paths["OBER_DATASET_PATH"] = download_dataset(args.datasets_dir, args.local_files_only)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
