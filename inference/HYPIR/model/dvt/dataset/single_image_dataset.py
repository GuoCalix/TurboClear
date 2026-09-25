from typing import Tuple, Union
import random
import math
import importlib
from omegaconf import OmegaConf

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from typing import Mapping, Any, Tuple, Callable, Literal

from HYPIR.dataset.utils import augment, random_crop_arr, center_crop_arr, load_file_meta
from HYPIR.utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from HYPIR.utils.common import instantiate_from_config

Image.MAX_IMAGE_PIXELS = None
np.random.seed(0)

def get_obj_from_str(string: str, reload: bool=False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class SingleImageDataset:
    """
    Returns the randomly augmented view of a single image.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        config: str = None,
        base_transform: transforms.Compose = None,
        final_transform: transforms.Compose = None,
        num_views: int = 768,
        blur_kernel_size = 21,
        kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        sinc_prob = 0.1,
        blur_sigma = [0.2, 3],
        betag_range = [0.5, 4],
        betap_range = [1, 2],

        blur_kernel_size2 = 21,
        kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        sinc_prob2 = 0.1,
        blur_sigma2 = [0.2, 1.5],
        betag_range2 = [0.5, 4],
        betap_range2 = [1, 2],

        final_sinc_prob = 0.8,
    ):
        self.size = size
        self.config = config
        self.base_transform = base_transform
        self.final_transform = final_transform
        self.num_views = num_views
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def set_image(self, img: Union[str, np.ndarray]):
        if not isinstance(img, np.ndarray):
            img = np.array(Image.open(img).convert("RGB"))
        self.image = img
        self.original_image = F.resize(
            self.base_transform(self.image),
            size=self.size,
            interpolation=Image.BICUBIC,
            antialias=True,
        )
        self.set_degragation()

    def set_degragation(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        config = OmegaConf.load(self.config)

        batch_transform = instantiate_from_config(config.data_config.train.batch_transform)

        data = {
            "hq": self.original_image.unsqueeze(0),
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
        }

        self.data: dict = batch_transform(data)
        self.data = {k: v.squeeze(0) for k, v in self.data.items()}

    def __getitem__(self, index):
        # pixel coords are in the range of [0, 1]
        # pixel coords are in feature-resolution, i.e., (H/patch_size, W/patch_size)
        aug_view, pixel_coords = self.final_transform(self.data["LQ"])
       
        return {
            "transformed_view": aug_view,
            "pixel_coords": pixel_coords,
            "full_image": self.original_image.clone(),
            **self.data
        }

    def __len__(self):
        return self.num_views
