---
library_name: diffusers
pipeline_tag: image-to-image
tags:
  - image-inpainting
  - object-removal
  - sdxl
  - one-step-diffusion
---

# TurboClear weights

This repository contains the first TurboClear checkpoint release used by the
public inference code.

## Files

- `sdxl/state_dict.pth`: one-step DMD student generator state.
- `sdxl/fake_state_dict.pth`: fake score model state used by DMD training.
- `fusion/fusion_module.pth`: learnable spatial fusion head trained after the
  generator.

The SDXL/ObjectClear base model is not duplicated here. Check its license and
download it separately before inference. OBER data and evaluation images are
not redistributed.

## Local inference

Download the files, then pass the local `sdxl` directory as `WEIGHT_PATH` and
`fusion/fusion_module.pth` as `FUSION_MODULE_PATH` to
`inference/inference_turboclear.sh`.
