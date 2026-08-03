<div align="center">

# TurboClear

### One-Step Object-Effect Removal via Region-Calibrated Distribution Matching and Fusion

<p>
<a href="https://guocalix.github.io/">Jiawei Guo</a><sup>1</sup>,
Junxian Li<sup>1</sup>,
Yixin Tang<sup>1</sup>,
Bingya Zhang<sup>2</sup>,
Jiaxin Lu<sup>2</sup>,
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
![Paper](https://img.shields.io/badge/Paper-TBD-8c8c8c?style=flat-square&logo=arxiv&logoColor=white)
![Model Weights](https://img.shields.io/badge/Model_Weights-TBD-8c8c8c?style=flat-square&logo=huggingface&logoColor=white)

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

## Release Status

| Resource | Status |
| --- | --- |
| Paper | TBD |
| Model weights | TBD |
| Code | TBD |
| Usage guide | TBD |

## Citation

TBD

## Acknowledgements

TBD
