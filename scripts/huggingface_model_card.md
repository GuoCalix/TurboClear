---
library_name: diffusers
pipeline_tag: image-to-image
tags:
  - image-inpainting
  - object-removal
  - sdxl
  - one-step-diffusion
license: apache-2.0
---

# TurboClear weights

This repository contains the first TurboClear checkpoint release used by the
public inference code.

## Files

- `sdxl/state_dict.pth`: one-step DMD student generator state.
- `fusion/fusion_module.pth`: learnable spatial fusion head trained after the
  generator.

The SDXL/ObjectClear base model is not duplicated here. Check its license and
download it separately before inference. OBER data and evaluation images are
not redistributed.

## Local inference

Download the files, then pass the local `sdxl` directory as `WEIGHT_PATH` and
`fusion/fusion_module.pth` as `FUSION_MODULE_PATH` to
`inference/inference_turboclear.sh`.
