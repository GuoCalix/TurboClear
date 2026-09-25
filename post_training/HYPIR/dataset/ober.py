import os
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_image_files(folder):
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    )


def _pil_rgb_to_tensor(path, out_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((out_size, out_size), Image.BICUBIC)
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _pil_mask_to_tensor(path, out_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize((out_size, out_size), Image.NEAREST)
    arr = (np.array(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


class OBERDirectoryDataset(Dataset):
    def __init__(self, root, input_size=512, max_samples=-1, prompt="remove the instance of object"):
        self.root = root
        self.input_size = input_size
        self.prompt = prompt

        image_dir = os.path.join(root, "image")
        mask_dir = os.path.join(root, "mask")
        gt_dir = os.path.join(root, "GT")
        effect_mask_dir = os.path.join(root, "effect_mask")
        for required_dir in [image_dir, mask_dir, gt_dir]:
            if not os.path.isdir(required_dir):
                raise ValueError(f"Missing OBER-Test validation directory: {required_dir}")

        mask_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_image_files(mask_dir)}
        gt_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_image_files(gt_dir)}
        effect_mask_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_image_files(effect_mask_dir)} if os.path.isdir(effect_mask_dir) else {}

        samples = []
        for image_path in _list_image_files(image_dir):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = mask_files.get(stem)
            gt_path = gt_files.get(stem)
            if mask_path is None or gt_path is None:
                continue
            effect_mask_path = effect_mask_files.get(stem, mask_path)
            samples.append((stem, image_path, mask_path, effect_mask_path, gt_path))

        if max_samples is not None and int(max_samples) > 0:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError(f"No OBER-Test validation samples found under: {root}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, image_path, mask_path, effect_mask_path, gt_path = self.samples[idx]
        return {
            "name": name,
            "image": _pil_rgb_to_tensor(image_path, self.input_size),
            "GT": _pil_rgb_to_tensor(gt_path, self.input_size),
            "object_mask": _pil_mask_to_tensor(mask_path, self.input_size),
            "object_effect_mask": _pil_mask_to_tensor(effect_mask_path, self.input_size),
            "txt": self.prompt,
        }

