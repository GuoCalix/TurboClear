#!/usr/bin/env python3
"""Upload the first TurboClear checkpoint release to Hugging Face.

The large files are read directly from the local weights directory; they are
not copied into the Git repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo, e.g. GuoCalix/TurboClear")
    parser.add_argument(
        "--weights-root",
        default="/Users/calixguo/Research/weights/turboclear/turboclear",
        help="Directory containing checkpoint-25000-sdxl and checkpoint-7000-fusion",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.weights_root).expanduser().resolve()
    student = root / "checkpoint-25000-sdxl" / "state_dict.pth"
    fusion = root / "checkpoint-7000-fusion" / "fusion_module.pth"
    for path in (student, fusion):
        if not path.is_file():
            raise FileNotFoundError(path)

    api = HfApi(token=args.token)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
    model_card = Path(__file__).with_name("huggingface_model_card.md")
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=args.token,
    )
    api.upload_file(
        path_or_fileobj=str(student),
        path_in_repo="sdxl/state_dict.pth",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=args.token,
    )
    api.upload_file(
        path_or_fileobj=str(fusion),
        path_in_repo="fusion/fusion_module.pth",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=args.token,
    )
    print(f"Uploaded TurboClear weights to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
