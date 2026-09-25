import os
import glob
import torch
from typing import Iterator, Dict, List, Optional, Tuple
from torch.utils import data
from collections import OrderedDict
import bisect
from PIL import Image
import numpy as np
import time
from HYPIR.dataset.utils import augment, random_crop_arr, center_crop_arr, load_file_meta
from HYPIR.utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from HYPIR.utils.common import instantiate_from_config

class LatentGTDataset(data.Dataset):
    def __init__(
        self,
        shard_size: int = 1024,
        file_meta: Optional[Dict] = None,
        lr_name: str = "z_lr",
        hr_name: str = "z_hr",
        use_rank_subdir: bool = True,
        cache_size: int = 2,
        crop_type: str = "center",
        out_size = 2048,
        use_hflip = True,
        use_rot = False,
    ) -> None:
        super().__init__()
        self.file_meta = file_meta
        self.image_files = load_file_meta(file_meta)
        self.shard_size = int(shard_size)
        self.lr_name = lr_name
        self.hr_name = hr_name
        self.use_rank_subdir = bool(use_rank_subdir)
        self.cache_size = int(cache_size)
        self.crop_type = crop_type
        self.out_size = out_size
        self.use_hflip = use_hflip
        self.use_rot = use_rot

        if file_meta is None:
            raise ValueError("file_meta is required and must contain 'file_list' directory path")
        base_dir = file_meta.get("base_dir", None)
        print(f"Base Dir: {base_dir}")
        if base_dir is None:
            raise ValueError("file_meta must contain 'base_dir'")
        self.base_dir = base_dir

        self._shard_paths = self._discover_shards()
        if len(self._shard_paths) == 0:
            raise FileNotFoundError(f"No shard .pt files found under: {self.base_dir}")
        # Pre-scan each shard length to support accurate random indexing
        self._shard_sizes = self._scan_all_shard_sizes()
        self._prefix = self._build_prefix(self._shard_sizes)
        # shard cache: path -> {lr, hr}
        self._cache = OrderedDict()

    def _discover_shards(self) -> List[str]:
        base = self.base_dir
        candidates: List[str] = []

        rank = None
        if self.use_rank_subdir:
            for env_key in ("RANK", "LOCAL_RANK", "ACCELERATE_PROCESS_INDEX"):
                v = os.environ.get(env_key)
                if v is not None:
                    try:
                        rank = int(v)
                        break
                    except Exception:
                        pass
        # Prefer rank-specific directory if rank is known
        if rank is not None:
            rank_dir = os.path.join(base, f"rank{rank:02d}")
            if os.path.isdir(rank_dir):
                candidates.extend(glob.glob(os.path.join(rank_dir, "*.pt")))
        # If still empty, try common rank* pattern
        if not candidates and os.path.isdir(base):
            candidates.extend(glob.glob(os.path.join(base, "rank*", "*.pt")))
        # Final fallback: recursive scan under base (covers nested layouts)
        if not candidates:
            candidates.extend(glob.glob(os.path.join(base, "**", "*.pt"), recursive=True))

        candidates = sorted(set(candidates))
        return candidates

    def _scan_all_shard_sizes(self) -> List[int]:
        sizes: List[int] = []
        for p in self._shard_paths:
            blob = torch.load(p, map_location="cpu")
            if self.lr_name not in blob or self.hr_name not in blob:
                raise KeyError(
                    f"Shard {p} missing keys: expected '{self.lr_name}' and '{self.hr_name}'"
                )
            z_lr = blob[self.lr_name]
            z_hr = blob[self.hr_name]
            if not isinstance(z_lr, torch.Tensor) or not isinstance(z_hr, torch.Tensor):
                raise TypeError(f"Shard {p} values must be torch.Tensor")
            if z_lr.shape[0] != z_hr.shape[0]:
                raise ValueError(f"Shard {p} has mismatched batch: {z_lr.shape} vs {z_hr.shape}")
            sizes.append(int(z_lr.shape[0]))
            # free memory ASAP
            del blob, z_lr, z_hr
        return sizes

    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        try:
            # failed to decode image bytes
            image = Image.open(image_path).convert("RGB")
        except:
            return None

        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)

        return image
    
    @staticmethod
    def _build_prefix(sizes: List[int]) -> List[int]:
        prefix = [0]
        s = 0
        for n in sizes:
            s += n
            prefix.append(s)
        return prefix

    def __len__(self) -> int:
        return self._prefix[-1]

    def _locate(self, index: int) -> Tuple[str, int]:
        if index < 0:
            index = len(self) + index
        # prefix: [0, n0, n0+n1, ...]
        shard_id = bisect.bisect_right(self._prefix, index) - 1
        if shard_id < 0 or shard_id >= len(self._shard_paths):
            raise IndexError("index out of range")
        local = index - self._prefix[shard_id]
        if local < 0 or local >= self._shard_sizes[shard_id]:
            raise IndexError("index out of range")
        return self._shard_paths[shard_id], int(local)

    def _get_shard(self, path: str) -> Dict[str, torch.Tensor]:
        if path in self._cache:
            entry = self._cache.pop(path)
            self._cache[path] = entry
            return entry
        blob = torch.load(path, map_location="cpu")
        if self.lr_name not in blob or self.hr_name not in blob:
            raise KeyError(
                f"Shard {path} missing keys: expected '{self.lr_name}' and '{self.hr_name}'"
            )
        entry = {
            "lr": blob[self.lr_name],
            "hr": blob[self.hr_name],
        }
        self._cache[path] = entry
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, local = self._locate(index)
        shard = self._get_shard(path)
        z_lr = shard["lr"][local]
        z_hr = shard["hr"][local]
        img_gt = None
        while img_gt is None:
            # load meta file
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            prompt = image_file["prompt"]
            img_gt = self.load_gt_image(gt_path)
            if img_gt is None:
                print(f"failed to load {gt_path}, try another image")
                index = np.random.randint(0, len(self) - 1)
            # img_gt = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        img_hq = (img_gt[..., ::-1] / 255.0).astype(np.float32)

        img_hq = augment(img_hq, self.use_hflip, self.use_rot)

        img_hq = torch.from_numpy(img_hq[..., ::-1].transpose(2, 0, 1).copy()).float()
        return {
            'gt': img_hq,
            "lq_latent": z_lr,
            "gt_latent": z_hr,
        }