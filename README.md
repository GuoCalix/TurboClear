<div align="center">

<img src="https://raw.githubusercontent.com/GuoCalix/TurboClear/page/assets/turboclear-mark.svg" width="88" alt="TurboClear logo">

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
- `training/`: masked-effect DMD one-step training, launched via
  `train_dmd_1step_masked_effect.sh`.
- `post_training/`: frozen-generator learnable fusion training, launched via
  `train_objectclear_fusion_1step.sh`.

The training and post-training code use the
[OBER dataset](https://huggingface.co/datasets/sczhou/OBERDataset_ObjectClear)
and an [ObjectClear/SDXL base model](https://huggingface.co/jixin0101/ObjectClear).
The training dataset and base model are downloaded separately. This repository
includes 12 [ObjectClear example image/mask pairs](inputs/README.md) for inference.

## Installation (Conda)

Use Linux with an NVIDIA CUDA GPU supporting bfloat16 and a driver compatible
with CUDA 12.4. The default inference processes images at 512 × 512 with batch
size 1. CPU and macOS inference are not supported by the supplied launcher.

```bash
git clone https://github.com/GuoCalix/TurboClear.git
cd TurboClear
conda env create -f environment.yml
conda activate turboclear
python -m pip check
```

The portable [environment.yml](environment.yml) and pinned
[requirements.txt](requirements.txt) are derived from the original research
Conda environment:

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch / torchvision | 2.5.1+cu124 / 0.20.1+cu124 |
| CUDA wheel runtime | 12.4 |
| Diffusers | 0.31.0 |
| Transformers / Tokenizers | 4.41.0 / 0.19.1 |
| Accelerate / PEFT | 1.13.0 / 0.10.0 |
| Hugging Face Hub | 0.28.1 |

Only the headless OpenCV package is installed, avoiding conflicting `cv2`
packages. Full image-quality evaluation additionally requires
`python -m pip install -r requirements-eval.txt`; quickstart does not use it.

## Quickstart

From the repository root, in the activated Conda environment:

```bash
python scripts/quickstart.py
```

This checks CUDA/bfloat16 support, downloads the inference weights into
`./model_weights/`, and runs all 12 bundled image/mask pairs using the same
Python interpreter. It does **not** download OBER. Predictions are written to
`outputs/quickstart/pred/`, with timing in `outputs/quickstart/latency.csv`.
The default pipeline uses one-step inference with Learnable Spatial Fusion (LSF).
By default, inference runs at 512 × 512 and predictions are resized back to each
original image's dimensions before saving. Use `--save-resize model` to save
the 512 × 512 predictions instead.

To run inference at the original resolution, padding to multiples of 8 and
cropping away that padding before saving:

```bash
python scripts/quickstart.py --local-files-only --resize-mode pad_to_multiple
```

Native resolution inference uses more GPU memory for larger images. Alternatively,
`--resize-mode short_side` preserves the aspect ratio with a target short side of
512 pixels (dimensions aligned to multiples of 8), then restores the original
dimensions when saving.

Without activating the environment in your shell, use:

```bash
conda run -n turboclear --no-capture-output python scripts/quickstart.py
```

For a single-image smoke test or a subsequent offline run:

```bash
python scripts/quickstart.py --max-samples 1
python scripts/quickstart.py --local-files-only
```

Use your own paired images and masks by matching their filename stems. White
mask pixels select the object to remove; extensions may differ between a pair.

```bash
python scripts/quickstart.py \
  --input-dir /path/to/images --mask-dir /path/to/masks \
  --output-dir /path/to/results
```

## Downloads

Download weights without running inference:

```bash
python scripts/download_weights.py
```

The default destinations are anchored to the TurboClear repository directory,
regardless of the current working directory:

```text
TurboClear/
├── model_weights/
│   ├── ObjectClear/                # Complete ObjectClear/SDXL inference components
│   └── TurboClear/
│       ├── sdxl/state_dict.pth      # One-step student
│       └── fusion/fusion_module.pth
├── datasets/                       # Created only when OBER is requested
└── inputs/
    ├── imgs/
    └── masks/
```

The downloader pins Hugging Face revisions, fetches the required base-model
components and the two released TurboClear checkpoints, and reuses completed
files on repeated runs. The student checkpoint alone is approximately 10.3 GB;
allow additional disk space for the ObjectClear base model. No separate SDXL
repository download is needed. Inference does not need `fake_state_dict.pth`.

To choose another model directory, pass the same option to both commands:

```bash
python scripts/download_weights.py --weights-dir /path/to/model_weights
python scripts/quickstart.py --weights-dir /path/to/model_weights
```

Alternatively, `--cache-dir /path/to/huggingface/hub` uses Hugging Face's snapshot
cache layout instead of the local model directory. `--cache-dir` and
`--weights-dir` are mutually exclusive. `--local-files-only` verifies existing
files and avoids network access.

OBER is large and is **opt-in**, for training/post-training only:

```bash
python scripts/download_weights.py --with-dataset
# Optional custom destinations:
python scripts/download_weights.py --with-dataset \
  --weights-dir /path/to/model_weights --datasets-dir /path/to/datasets
```

By default, OBER parquet shards are saved under `./datasets/OBER/data/`.
The dataset retains its upstream license; see its linked Hugging Face card.

## Inference launcher

Quickstart sets the paths automatically. To call the shell launcher directly,
use absolute paths because it changes into the inference directory:

```bash
INPUT_DIR="$PWD/inputs/imgs" \
MASK_DIR="$PWD/inputs/masks" \
BASE_MODEL_PATH="$PWD/model_weights/ObjectClear" \
WEIGHT_PATH="$PWD/model_weights/TurboClear/sdxl" \
FUSION_MODULE_PATH="$PWD/model_weights/TurboClear/fusion/fusion_module.pth" \
OUTPUT_DIR="$PWD/outputs/inference" \
bash inference/inference_turboclear.sh
```

Set `CUDA_VISIBLE_DEVICES` to select the GPU (default: `0`). The student directory
must contain `state_dict.pth`. The fusion path may be the file or its directory.

## Training and post-training

Activate the same Conda environment and download OBER with `--with-dataset`.
Run these commands from the repository root:

```bash
export TURBOCLEAR_DATASET_PATH="$PWD/datasets/OBER/data"
export TURBOCLEAR_BASE_MODEL_PATH="$PWD/model_weights/ObjectClear"
export TURBOCLEAR_VALIDATION_DATASET_PATH=""
export TURBOCLEAR_GENERATOR_CHECKPOINT=""
export TURBOCLEAR_OUTPUT_DIR="$PWD/outputs/dmd"
bash training/train_dmd_1step_masked_effect.sh
```

An empty generator checkpoint initializes full-UNet training from ObjectClear.
Set `TURBOCLEAR_GENERATOR_CHECKPOINT` to a compatible checkpoint when continuing
from an existing generator. Released inference weights are not a complete
optimizer/training-state resume checkpoint.

Validation is disabled in this minimal example. To enable it, set
`TURBOCLEAR_VALIDATION_DATASET_PATH` to a prepared OBER validation directory with
`image/`, `mask/`, `GT/`, and optionally `effect_mask/`, paired by filename stem.
The download command fetches raw parquet shards; it does not decode this
validation directory. The training loader excludes the test parquet, so do not
point validation at the downloaded training directory or the excluded test file.

For LSF post-training, keep the dataset/base-model variables above and provide
a trained one-step student (the released student can be used):

```bash
export TURBOCLEAR_GENERATOR_CHECKPOINT="$PWD/model_weights/TurboClear/sdxl"
export TURBOCLEAR_OUTPUT_DIR="$PWD/outputs/fusion"
bash post_training/train_objectclear_fusion_1step.sh
```

To initialize from existing fusion weights, set `TURBOCLEAR_FUSION_CHECKPOINT`
to `fusion_module.pth`. Both stage launchers default to eight GPUs: edit
`training/config/accelerate_fsdp_8gpu.yaml` (or set `ACCELERATE_CONFIG` to an
absolute config path) for DMD training; set `NUM_PROCESSES` for fusion training.
The YAML files in each stage's `config/` directory define batch size, losses,
validation, and checkpoint settings. Training's LPIPS loss may download
pretrained VGG weights into the Torch Hub cache on first use; this is separate
from the inference model downloader.

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
