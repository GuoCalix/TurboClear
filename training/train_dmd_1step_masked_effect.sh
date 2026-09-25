#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-config/objectclear_dmd_1step_masked_effect.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-config/accelerate_fsdp_8gpu.yaml}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" train.py --config "${CONFIG_PATH}"
