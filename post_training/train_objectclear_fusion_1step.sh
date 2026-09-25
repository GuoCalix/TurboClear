#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-config/objectclear_fusion_1step.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch --multi_gpu --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  train.py --config "${CONFIG_PATH}"
