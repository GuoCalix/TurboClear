<div align="center">

# TurboClear

### One-Step Object-Effect Removal via Region-Calibrated Distribution Matching and Fusion

<p>
<a href="https://guocalix.github.io/">Jiawei Guo</a><sup>1</sup>,
<a href="#">Junxian Li</a><sup>1</sup>,
<a href="#">Yixin Tang</a><sup>1</sup>,
<a href="#">Bingya Zhang</a><sup>2</sup>,
<a href="#">Jiaxin Lu</a><sup>2</sup>,
<a href="https://yulunzhang.com/">Yulun Zhang</a><sup>1,&dagger;</sup>,
<a href="https://shangchenzhou.com/">Shangchen Zhou</a><sup>3,&dagger;</sup>
<br>
<sup>1</sup> Shanghai Jiao Tong University &nbsp;&nbsp;
<sup>2</sup> Honor Device Co., Ltd &nbsp;&nbsp;
<sup>3</sup> Imperial College London
<br>
<sup>&dagger;</sup> Corresponding authors
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-24292f?style=flat-square&logo=googlechrome&logoColor=white)](https://guocalix.github.io/TurboClear/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.01288)
[![PDF](https://img.shields.io/badge/PDF-Download-b31b1b?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2608.01288)
[![Model Weights](https://img.shields.io/badge/Model_Weights-Hugging_Face-ffcc4d?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/JGuo666/TurboClear)

**TurboClear removes target objects and their associated visual effects with a single denoising step while preserving unaffected background regions.**

![TurboClear qualitative results](https://raw.githubusercontent.com/GuoCalix/TurboClear/page/assets/images/qualitative-strip.webp)

</div>

## Highlights

- **One-step removal.** TurboClear performs object-effect removal with one SDXL UNet evaluation.
- **Region-Calibrated Distribution Matching.** RDM calibrates the distillation signal across affected and preserved regions.
- **Learnable Spatial Fusion.** LSF combines removal-oriented generation with an identity-preserving reference stream.
- **Efficient inference.** TurboClear substantially reduces denoising computation while maintaining competitive visual quality.

## Overview

![TurboClear overview](https://raw.githubusercontent.com/GuoCalix/TurboClear/page/assets/images/pipeline.webp)

TurboClear addresses the spatial asymmetry of object-effect removal: regions influenced by the target object should be regenerated, while the remaining image should stay unchanged. RDM provides region-aware supervision during one-step distillation, and LSF learns spatial gates for lightweight fusion at inference time.

## Code layout

This first public release keeps the three experiment stages separate:

- `inference/`: one-step inference and learnable fusion, launched via
  `inference/inference_turboclear.sh`.
- `training/`: masked-effect DMD one-step training, based on
  `train_dmd_1step_masked_effect.sh`.
- `post_training/`: frozen-generator learnable fusion training, based on
  `train_objectclear_fusion_1step.sh`.

The training and post-training code use the OBER dataset and an ObjectClear/SDXL
base model. The dataset and base model are not included in this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build before installing the remaining
dependencies when the default PyPI wheel does not match your system.

## Inference

Set the input, mask, base-model, student-checkpoint, and fusion-checkpoint
paths, then run:

```bash
INPUT_DIR=/path/to/images \
MASK_DIR=/path/to/masks \
BASE_MODEL_PATH=/path/to/ObjectClear \
WEIGHT_PATH=/path/to/checkpoint-25000-sdxl \
FUSION_MODULE_PATH=/path/to/fusion_module.pth \
bash inference/inference_turboclear.sh
```

The student checkpoint directory must contain `state_dict.pth`. The fusion
checkpoint can be either `fusion_module.pth` or its containing directory.

## Training

Configure the environment variables below, or edit the YAML configuration:

```bash
export TURBOCLEAR_DATASET_PATH=/path/to/OBER/data
export TURBOCLEAR_VALIDATION_DATASET_PATH=/path/to/OBER-Test
export TURBOCLEAR_BASE_MODEL_PATH=/path/to/ObjectClear
export TURBOCLEAR_GENERATOR_CHECKPOINT=/path/to/pretrained/student/checkpoint
export TURBOCLEAR_OUTPUT_DIR=./outputs/dmd
bash training/train_dmd_1step_masked_effect.sh
```

The fusion post-training stage uses the same dataset and base model variables,
and additionally reads `TURBOCLEAR_FUSION_CHECKPOINT` when resuming:

```bash
export TURBOCLEAR_OUTPUT_DIR=./outputs/fusion
bash post_training/train_objectclear_fusion_1step.sh
```

Both scripts use `accelerate`; adjust the GPU count and accelerator config for
your machine.

## Release status

The source code is available in this repository. Download the released model
files from the [TurboClear Hugging Face repository](https://huggingface.co/JGuo666/TurboClear)
before running inference. OBER images and masks are not redistributed here.

## License

The TurboClear source code is released under the [Apache License 2.0](LICENSE).
Third-party components and pretrained models retain their respective licenses.

## Citation

```bibtex
@article{guo2026turboclear,
  title={TurboClear: One-Step Object-Effect Removal via Region-Calibrated Distribution Matching and Fusion},
  author={Guo, Jiawei and Li, Junxian and Tang, Yixin and Zhang, Bingya and Lu, Jiaxin and Zhang, Yulun and Zhou, Shangchen},
  journal={arXiv preprint arXiv:2608.01288},
  year={2026}
}
```

## Acknowledgements

TurboClear was inspired by [ObjectClear](https://github.com/jixin0101/ObjectClear)
and [DMD2](https://github.com/tianweiy/DMD2). We thank the authors for their
open-source implementations and research contributions.
