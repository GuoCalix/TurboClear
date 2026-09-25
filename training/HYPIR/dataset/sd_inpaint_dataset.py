import io
import os
import random

import cv2
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from pathlib import Path

class SDInpaintImageDataset(Dataset):
    """
    Parquet-backed inpaint dataset that keeps the public class name used by train_sd.py.
    """

    def __init__(
        self,
        dataset_path,
        is_sdxl=False,        
        input_size=512,
        use_blank_mask=False,
        blank_mask_prob=0.4,
        color_augmentation=False,
        flip_augmentation=True,
        rotation_augmentation=False,
        rotation_prob=0.5,
        random_mask_dilation=True,
        random_mask_erosion=True,
        return_minimal=False,
    ):
        self.dataset_path = dataset_path
        self.input_size = input_size
        self.is_sdxl = is_sdxl

        self.use_blank_mask = use_blank_mask
        self.blank_mask_prob = blank_mask_prob
        self.color_augmentation = color_augmentation
        self.flip_augmentation = flip_augmentation
        self.rotation_augmentation = rotation_augmentation
        self.rotation_prob = rotation_prob
        self.random_mask_dilation = random_mask_dilation
        self.random_mask_erosion = random_mask_erosion
        self.return_minimal = return_minimal
        self.parquet_paths = self._resolve_parquet_paths(dataset_path)
        self.base_dir = dataset_path if os.path.isdir(dataset_path) else os.path.dirname(self.parquet_paths[0])
        self.table = pl.read_parquet(self.parquet_paths)

        self.input_col = self._pick_column(
            ["input", "image", "input_image", "input_path", "image_path"],
            required=True,
            kind="input",
        )
        self.gt_col = self._pick_column(
            ["gt", "target", "gt_image", "target_image", "gt_path", "target_path"],
            required=False,
            kind="gt",
        )
        self.mask_col = self._pick_column(
            ["object_mask", "mask", "mask_image", "mask_path"],
            required=True,
            kind="mask",
        )
        self.shadow_mask_col = self._pick_column(
            ["object_effect_mask", "shadow_mask", "shadow_mask_image", "shadow_mask_path"],
            required=False,
            kind="shadow_mask",
        )
        self.prompt_col = self._pick_column(
            ["prompt", "caption", "text", "description"],
            required=False,
            kind="prompt",
        )
        self.length = self.table.height
        if self.length == 0:
            raise ValueError(f"Parquet dataset is empty: {self.parquet_paths}")

        if len(self.parquet_paths) == 1:
            print(f"Loaded parquet dataset: {self.parquet_paths[0]}")
        else:
            print(f"Loaded parquet dataset files: {len(self.parquet_paths)}")
        print(f"Dataset length: {self.length}")

    def _resolve_parquet_paths(self, dataset_path):
        excluded_name = "test-00000-of-00001.parquet"
        # excluded_name = ""

        if os.path.isfile(dataset_path):
            if dataset_path.endswith(".parquet"):
                if os.path.basename(dataset_path) == excluded_name:
                    raise ValueError(f"Excluded test parquet file cannot be used as training dataset: {dataset_path}")
                return [dataset_path]
            raise ValueError(f"Expected a parquet file, got: {dataset_path}")

        if os.path.isdir(dataset_path):
            parquet_files = sorted(
                str(path)
                for path in Path(dataset_path).rglob("*.parquet")
                if path.name != excluded_name
            )
            if len(parquet_files) == 0:
                raise ValueError(
                    f"No parquet file found in directory after excluding '{excluded_name}': {dataset_path}"
                )
            return parquet_files

        raise ValueError(f"dataset_path does not exist: {dataset_path}")

    def _pick_column(self, candidates, required, kind):
        existing = set(self.table.columns)
        for name in candidates:
            if name in existing:
                return name
        if required:
            raise ValueError(
                f"Missing required {kind} column. Tried {candidates}. "
                f"Available columns: {self.table.columns}"
            )
        return None

    def conservative_augment_image(self, image1, image2):
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)

        brightness_factor = random.uniform(0.8, 1.2)
        image1 = ImageEnhance.Brightness(image1).enhance(brightness_factor)
        image2 = ImageEnhance.Brightness(image2).enhance(brightness_factor)

        contrast_factor = random.uniform(0.8, 1.2)
        image1 = ImageEnhance.Contrast(image1).enhance(contrast_factor)
        image2 = ImageEnhance.Contrast(image2).enhance(contrast_factor)

        color_factor = random.uniform(0.8, 1.2)
        image1 = ImageEnhance.Color(image1).enhance(color_factor)
        image2 = ImageEnhance.Color(image2).enhance(color_factor)

        image1 = np.array(image1.convert("HSV"))
        image2 = np.array(image2.convert("HSV"))
        hue_shift = random.randint(-15, 15)
        image1[:, :, 0] = (image1[:, :, 0] + hue_shift) % 256
        image2[:, :, 0] = (image2[:, :, 0] + hue_shift) % 256
        image1 = np.array(Image.fromarray(image1, "HSV").convert("RGB"))
        image2 = np.array(Image.fromarray(image2, "HSV").convert("RGB"))
        return image1, image2

    def random_crop_with_object_center(self, mask_image_np, input_size):
        h, w = mask_image_np.shape

        rotate_flag = 0
        if h > w:
            mask_image_np = cv2.rotate(mask_image_np, cv2.ROTATE_90_CLOCKWISE)
            rotate_flag = 1
            h, w = mask_image_np.shape

        mask_indices = np.where(mask_image_np == 255)
        if len(mask_indices[0]) == 0 or len(mask_indices[1]) == 0:
            raise ValueError("Mask is empty, no object found.")

        center_row = int(np.mean(mask_indices[0]))
        center_col = int(np.mean(mask_indices[1]))

        max_crop_size = min(h, w)
        min_crop_size = input_size
        crop_size = random.randint(int(min_crop_size), int(max_crop_size))

        min_row = max(0, center_row - crop_size + 1)
        max_row = min(center_row, h - crop_size)
        min_col = max(0, center_col - crop_size + 1)
        max_col = min(center_col, w - crop_size)

        start_row = random.randint(min_row, max_row)
        start_col = random.randint(min_col, max_col)

        if rotate_flag == 1:
            temp = w - start_col - input_size
            start_col = start_row
            start_row = temp

        return start_row, start_col, crop_size

    def process_to_tensor(self, input_np, output_size, is_mask):
        output = input_np.astype(np.float32) / 255.0
        if is_mask:
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            output = torch.from_numpy(output).unsqueeze(0)
            output = F.interpolate(
                output.unsqueeze(0),
                size=(output_size, output_size),
                mode="nearest",
            ).squeeze(0)
        else:
            output = torch.from_numpy(output).permute(2, 0, 1).float()
            output = F.interpolate(
                output.unsqueeze(0),
                size=(output_size, output_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            output = torch.clamp(output, 0, 1)
        return output

    def random_rotate_sample(
        self,
        image_np,
        gt_image_np,
        mask_image_np,
        shadow_mask_image_np,
        ori_mask_image_np,
    ):
        if (not self.rotation_augmentation) or random.random() > self.rotation_prob:
            return image_np, gt_image_np, mask_image_np, shadow_mask_image_np, ori_mask_image_np

        # 改为从 90, 180, 270 中随机选择一个直角
        angle = random.choice([90.0, 180.0, 270.0])
        
        h, w = image_np.shape[:2]
        center = (w // 2, h // 2)
        rot_m = cv2.getRotationMatrix2D(center, angle, 1.0)

        image_np = cv2.warpAffine(
            image_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        gt_image_np = cv2.warpAffine(
            gt_image_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask_image_np = cv2.warpAffine(
            mask_image_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shadow_mask_image_np = cv2.warpAffine(
            shadow_mask_image_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        ori_mask_image_np = cv2.warpAffine(
            ori_mask_image_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        _, mask_image_np = cv2.threshold(mask_image_np, 127, 255, cv2.THRESH_BINARY)
        _, shadow_mask_image_np = cv2.threshold(shadow_mask_image_np, 127, 255, cv2.THRESH_BINARY)
        _, ori_mask_image_np = cv2.threshold(ori_mask_image_np, 127, 255, cv2.THRESH_BINARY)

        return image_np, gt_image_np, mask_image_np, shadow_mask_image_np, ori_mask_image_np
    def _decode_image_value(self, value, mode):
        if value is None:
            raise ValueError(f"Parquet cell for mode={mode} is None")

        if isinstance(value, dict):
            if "bytes" in value and value["bytes"] is not None:
                value = value["bytes"]
        
        if isinstance(value, str):
            candidate = value
            if not os.path.isabs(candidate):
                candidate = os.path.join(self.base_dir, candidate)
            return np.array(Image.open(candidate).convert(mode))

        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.array(Image.open(io.BytesIO(bytes(value))).convert(mode))

        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, np.ndarray):
            if mode == "L" and value.ndim == 3:
                value = value[..., 0]
            if value.dtype != np.uint8:
                if np.issubdtype(value.dtype, np.floating):
                    max_value = float(value.max()) if value.size > 0 else 1.0
                    if max_value <= 1.0:
                        value = np.clip(value * 255.0, 0, 255)
                value = value.astype(np.uint8)
            return value

        raise ValueError(f"Unsupported parquet cell type for mode={mode}: {type(value)}")

    def _tokenize_prompt(self, prompt):
        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        output = {"text_input_ids_one": text_input_ids_one}
        if self.is_sdxl:
            if self.tokenizer_two is None:
                raise ValueError("is_sdxl=True but tokenizer_two is None.")
            text_input_ids_two = self.tokenizer_two(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output["text_input_ids_two"] = text_input_ids_two
        return output

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.table.row(idx, named=True)

        image_np = self._decode_image_value(row[self.input_col], "RGB")
        gt_value = row[self.gt_col] if self.gt_col is not None else row[self.input_col]
        gt_image_np = self._decode_image_value(gt_value, "RGB")

        mask_image_np = self._decode_image_value(row[self.mask_col], "L")
        _, mask_image_np = cv2.threshold(mask_image_np, 127, 255, cv2.THRESH_BINARY)

        if self.shadow_mask_col is not None:
            shadow_mask_image_np = self._decode_image_value(row[self.shadow_mask_col], "L")
            _, shadow_mask_image_np = cv2.threshold(shadow_mask_image_np, 127, 255, cv2.THRESH_BINARY)
        else:
            shadow_mask_image_np = np.zeros_like(mask_image_np)

        ori_mask_image_np = mask_image_np.copy()

        if self.random_mask_dilation:
            if random.random() > 0.33:
                height, width = mask_image_np.shape[:2]
                area = np.count_nonzero(mask_image_np)
                area_ratio = area / (height * width)

                if area_ratio <= 0.005:
                    kernel_size = random.randint(1, 5)
                elif area_ratio <= 0.02:
                    kernel_size = random.randint(1, 8)
                elif area_ratio <= 0.05:
                    kernel_size = random.randint(1, 12)
                elif area_ratio <= 0.1:
                    kernel_size = random.randint(1, 17)
                elif area_ratio <= 0.25:
                    kernel_size = random.randint(1, 23)
                else:
                    kernel_size = random.randint(1, 30)

                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_image_np = cv2.dilate(mask_image_np, kernel, iterations=1)
            elif self.random_mask_erosion and random.random() > 0.5:
                height, width = mask_image_np.shape[:2]
                area = np.count_nonzero(mask_image_np)
                area_ratio = area / (height * width)

                if area_ratio <= 0.005:
                    max_k, min_k, max_attempts = 5, 1, 2
                elif area_ratio <= 0.02:
                    max_k, min_k, max_attempts = 8, 1, 2
                elif area_ratio <= 0.05:
                    max_k, min_k, max_attempts = 12, 1, 2
                elif area_ratio <= 0.1:
                    max_k, min_k, max_attempts = 17, 1, 2
                elif area_ratio <= 0.25:
                    max_k, min_k, max_attempts = 23, 1, 2
                else:
                    max_k, min_k, max_attempts = 30, 1, 2

                for attempt in range(max_attempts):
                    decay_ratio = attempt / (max_attempts - 1)
                    current_max = int(max_k - (max_k - min_k) * decay_ratio)
                    kernel_size = random.randint(1, current_max)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel_size = max(1, kernel_size)

                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    eroded_mask = cv2.erode(mask_image_np, kernel, iterations=1)
                    if np.count_nonzero(eroded_mask) > 0:
                        mask_image_np = eroded_mask
                        break

        start_row, start_col, crop_size = self.random_crop_with_object_center(
            mask_image_np,
            input_size=self.input_size,
        )

        if self.use_blank_mask and random.random() < self.blank_mask_prob:
            mask_image_np = np.zeros_like(mask_image_np)
            gt_image_np = image_np

        image_np = image_np[start_row : start_row + crop_size, start_col : start_col + crop_size]
        gt_image_np = gt_image_np[start_row : start_row + crop_size, start_col : start_col + crop_size]
        mask_image_np = mask_image_np[start_row : start_row + crop_size, start_col : start_col + crop_size]
        shadow_mask_image_np = shadow_mask_image_np[
            start_row : start_row + crop_size,
            start_col : start_col + crop_size,
        ]
        ori_mask_image_np = ori_mask_image_np[start_row : start_row + crop_size, start_col : start_col + crop_size]

        if self.flip_augmentation and random.random() > 0.5:
            image_np = np.flip(image_np, axis=1)
            gt_image_np = np.flip(gt_image_np, axis=1)
            mask_image_np = np.flip(mask_image_np, axis=1)
            shadow_mask_image_np = np.flip(shadow_mask_image_np, axis=1)
            ori_mask_image_np = np.flip(ori_mask_image_np, axis=1)

        image_np, gt_image_np, mask_image_np, shadow_mask_image_np, ori_mask_image_np = self.random_rotate_sample(
            image_np,
            gt_image_np,
            mask_image_np,
            shadow_mask_image_np,
            ori_mask_image_np,
        )

        wo_color_aug_gt_np = gt_image_np
        wo_color_aug_image_np = image_np
        if self.color_augmentation:
            image_np, gt_image_np = self.conservative_augment_image(image_np, gt_image_np)

        input_tensor = self.process_to_tensor(image_np, self.input_size, is_mask=False)
        gt_tensor = self.process_to_tensor(gt_image_np, self.input_size, is_mask=False)
        mask_tensor = self.process_to_tensor(mask_image_np, self.input_size, is_mask=True)
        shadow_mask_tensor = self.process_to_tensor(shadow_mask_image_np, self.input_size, is_mask=True)
        prompt = "remove the instance of object"
        if self.prompt_col is not None and row[self.prompt_col] is not None:
            prompt = str(row[self.prompt_col])

        if self.return_minimal:
            return {
                "input": input_tensor * 2.0 - 1.0,
                "gt": gt_tensor * 2.0 - 1.0,
                "mask": mask_tensor,
                "shadow_mask": shadow_mask_tensor,
                "prompt": prompt,
            }

        ori_mask_tensor = self.process_to_tensor(ori_mask_image_np, self.input_size, is_mask=True)
        wo_color_aug_gt_tensor = self.process_to_tensor(wo_color_aug_gt_np, self.input_size, is_mask=False)
        wo_color_aug_image_tensor = self.process_to_tensor(wo_color_aug_image_np, self.input_size, is_mask=False)

        input_wo_shadow = ori_mask_tensor * input_tensor + (1 - ori_mask_tensor) * gt_tensor
        obj_only_tensor = mask_tensor * input_tensor
        color_aug_obj_ori_gt_tensor = ori_mask_tensor * input_tensor + (1 - ori_mask_tensor) * wo_color_aug_gt_tensor
        masked_image_tensor = (1 - mask_tensor) * input_tensor

        outputs = {
            "images": gt_tensor * 2.0 - 1.0,
            "input": input_tensor * 2.0 - 1.0,
            "gt": gt_tensor * 2.0 - 1.0,
            "mask": mask_tensor,
            "shadow_mask": shadow_mask_tensor,
            "masked_image": masked_image_tensor,
            "input_wo_shadow": input_wo_shadow * 2.0 - 1.0,
            "ori_mask_tensor": ori_mask_tensor * 2.0 - 1.0,
            "obj_only_tensor": obj_only_tensor,
            "aug_obj_ori_gt": color_aug_obj_ori_gt_tensor * 2.0 - 1.0,
            "wo_color_aug_image": wo_color_aug_image_tensor * 2.0 - 1.0,
            "prompt": prompt,
        }

        return outputs
