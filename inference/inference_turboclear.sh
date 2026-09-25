#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Set these variables for a local run. BASE_MODEL_PATH may also be a
# Hugging Face model id supported by diffusers.
INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to a directory of input images}"
MASK_DIR="${MASK_DIR:?Set MASK_DIR to a directory of object masks}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-jixin0101/ObjectClear}"
WEIGHT_PATH="${WEIGHT_PATH:?Set WEIGHT_PATH to the TurboClear student checkpoint}"
FUSION_MODULE_PATH="${FUSION_MODULE_PATH:?Set FUSION_MODULE_PATH to fusion_module.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
TIMESTEP_SPACING="${TIMESTEP_SPACING:-fixed}"
ALPHA_FUSION_THRESHOLD="${ALPHA_FUSION_THRESHOLD:-0.5}"

EXTRA_ARGS=()
if [[ "${TORCH_COMPILE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--torch_compile --torch_compile_mode reduce-overhead)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" validate_objectclear.py \
  --config configs/objectclear_train.yaml \
  --input_dir "${INPUT_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --resize_mode square \
  --save_resize model \
  --base_model_path "${BASE_MODEL_PATH}" \
  --text_encoder_dtype fp32 \
  --object_encoder_dtype fp32 \
  --postfuse_dtype fp32 \
  --output_dir "${OUTPUT_DIR}" \
  --weight_path "${WEIGHT_PATH}" \
  --prompt "remove the instance of object" \
  --distill_steps 1 \
  --scheduler_type ddim \
  --timestep_spacing "${TIMESTEP_SPACING}" \
  --fixed_timestep 399 \
  --one_step_output_type pred_x0 \
  --precision bf16 \
  --fusion_precision fp32 \
  --fusion_validation_mode training \
  --attention_capture_mode single \
  --alpha_fusion_threshold "${ALPHA_FUSION_THRESHOLD}" \
  --seed 231 \
  --batch_size 1 \
  --num_workers 2 \
  --max_samples "${MAX_SAMPLES}" \
  --warmup_runs "${WARMUP_RUNS}" \
  --optimize_latency \
  --inference_only \
  --learnable_fusion \
  --fusion_module_path "${FUSION_MODULE_PATH}" \
  "${EXTRA_ARGS[@]}"
