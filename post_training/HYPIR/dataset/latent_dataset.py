import os
import glob
import torch
import numpy as np
from typing import Iterator, Dict, List, Optional, Tuple
from torch.utils import data
from collections import OrderedDict
import bisect
from HYPIR.dataset.utils import USMSharp, center_crop_arr
from PIL import Image

class LatentDataset(data.Dataset):
    def __init__(
        self,
        shard_size: int = 1024,
        file_meta: Optional[Dict] = None,
        lr_name: str = "z_lr",
        hr_name: str = "z_hr",
        use_rank_subdir: bool = True,
        cache_size: int = 2,
    ) -> None:
        super().__init__()
        self.shard_size = int(shard_size)
        self.lr_name = lr_name
        self.hr_name = hr_name
        self.use_rank_subdir = bool(use_rank_subdir)
        self.cache_size = int(cache_size)

        if file_meta is None:
            raise ValueError("file_meta is required and must contain 'file_list' directory path")
        base_dir = file_meta.get("file_list", None)
        if base_dir is None:
            raise ValueError("file_meta must contain 'file_list'")
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
            # Optional: filenames length should match
            if "filename" in blob:
                try:
                    if len(blob["filename"]) != int(z_lr.shape[0]):
                        raise ValueError(
                            f"Shard {p} filename length mismatch: {len(blob['filename'])} vs {z_lr.shape[0]}"
                        )
                except TypeError:
                    # if filename is not list-like, ignore
                    pass
            sizes.append(int(z_lr.shape[0]))
            # free memory ASAP
            del blob, z_lr, z_hr
        return sizes

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
        if "filename" in blob:
            entry["filename"] = blob["filename"]
        self._cache[path] = entry
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, local = self._locate(index)
        shard = self._get_shard(path)
        z_lr = shard["lr"][local]
        z_hr = shard["hr"][local]
        item = {
            "lq_latent": z_lr,
            "gt_latent": z_hr,
        }
        if "filename" in shard:
            # shard["filename"] is expected to be a list-like of str
            try:
                item["filename"] = shard["filename"][local]
            except Exception:
                pass
        return item

class LatentWithGTDataset(torch.utils.data.Dataset):
    """将 latent 数据集与按 filename 加载的 GT 对齐并合并输出。"""
    def __init__(self, latent_ds, image_path_prefix: str, out_size: int, crop_type: str = "center", apply_usm: bool = False, skip_missing_filename: bool = True):
        super().__init__()
        self.latent_ds = latent_ds
        self._length = len(self.latent_ds)
        self.prefix = image_path_prefix
        self.out_size = int(out_size)
        self.crop_type = str(crop_type)
        self.apply_usm = bool(apply_usm)
        self.usm = USMSharp() if self.apply_usm else None
        self.skip_missing_filename = bool(skip_missing_filename)

        # Build valid index list for samples that have 'filename' in shards (auto-compatibility with old shards)
        self.valid_indices = None
        try:
            shard_paths = getattr(self.latent_ds, "_shard_paths", None)
            if shard_paths is not None:
                import torch as _torch
                valid = []
                offset = 0
                for sp in shard_paths:
                    blob = _torch.load(sp, map_location="cpu")
                    z = blob.get(getattr(self.latent_ds, "lr_name", "z_lr"))
                    n = int(z.shape[0]) if isinstance(z, torch.Tensor) else 0
                    if "filename" in blob and isinstance(blob["filename"], (list, tuple)):
                        # take all indices in this shard
                        valid.extend(range(offset, offset + n))
                    else:
                        # missing filename in this shard
                        if not self.skip_missing_filename:
                            valid.extend(range(offset, offset + n))
                        # else skip
                    offset += n
                if len(valid) > 0:
                    self.valid_indices = valid
                    self._length = len(self.valid_indices)
        except Exception:
            # fallback to direct indexing without filtering
            self.valid_indices = None

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self.valid_indices is not None:
            real_idx = self.valid_indices[idx]
        else:
            real_idx = idx
        lat = self.latent_ds[real_idx]  # {'lq_latent','gt_latent', maybe 'filename'}
        if 'filename' not in lat or lat['filename'] is None:
            if self.skip_missing_filename:
                raise KeyError("Latent shard missing 'filename'; please regenerate latents with filename saved or set skip_missing_filename=False.")
            else:
                # Try best-effort: cannot infer path without metadata; raise for visibility
                raise KeyError("Missing 'filename' and skip_missing_filename=False; cannot load GT deterministically.")
        fname = lat['filename']
        path = os.path.join(self.prefix, fname)
        img = Image.open(path).convert("RGB")
        # 参照 RealESRGANDataset 的中心裁剪逻辑（不使用先 Resize 再 CenterCrop 的 torchvision）
        if self.crop_type != "none":
            if img.height == self.out_size and img.width == self.out_size:
                arr = np.array(img)
            else:
                if self.crop_type == "center":
                    arr = center_crop_arr(img, self.out_size)
                elif self.crop_type == "random":
                    # 为保持确定性，测试阶段通常不应使用 random；这里仍保留分支，如需可后续接入相同的 random_crop_arr
                    arr = center_crop_arr(img, self.out_size)
                else:
                    arr = center_crop_arr(img, self.out_size)
        else:
            assert img.height == self.out_size and img.width == self.out_size
            arr = np.array(img)
        # HWC RGB -> CHW float32 [0,1]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float() / 255.0
        if self.apply_usm:
            tensor = self.usm(tensor.unsqueeze(0)).squeeze(0)
        return {
            'gt_latent': lat['gt_latent'],
            'lq_latent': lat['lq_latent'],
            'gt': tensor,
            'filename': fname,
        }