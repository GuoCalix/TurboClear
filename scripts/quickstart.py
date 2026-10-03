#!/usr/bin/env python3
"""Download weights and run the bundled example images with the active Python."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from download_weights import add_download_arguments, download_weights


REPO_ROOT = Path(__file__).resolve().parents[1]


def check_runtime():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Quickstart requires an NVIDIA CUDA GPU. Activate the turboclear Conda environment.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The released bf16 inference settings require a GPU with bfloat16 support.")
    print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "inputs/imgs")
    parser.add_argument("--mask-dir", type=Path, default=REPO_ROOT / "inputs/masks")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/quickstart")
    parser.add_argument("--max-samples", type=int, default=-1, help="Default: all 12 examples.")
    args = parser.parse_args()
    for directory in (args.input_dir, args.mask_dir):
        if not directory.expanduser().is_dir():
            parser.error(f"Directory does not exist: {directory}")
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 (all) or a positive integer")
    check_runtime()
    weights = download_weights(args.weights_dir, args.cache_dir, args.local_files_only)
    env = os.environ.copy()
    env.update(weights)
    env.update({
        "PYTHON_BIN": sys.executable,
        "INPUT_DIR": str(args.input_dir.expanduser().resolve()),
        "MASK_DIR": str(args.mask_dir.expanduser().resolve()),
        "OUTPUT_DIR": str(args.output_dir.expanduser().resolve()),
        "MAX_SAMPLES": str(args.max_samples),
    })
    if args.local_files_only:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(["bash", str(REPO_ROOT / "inference/inference_turboclear.sh")],
                   cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
