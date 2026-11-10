#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DELAY=${1:?Usage: $0 DELAY GPU}
GPU=${2:?Usage: $0 DELAY GPU}
PYTHON=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
POLICY=${XVLA_POLICY:-/data/yhr/output/online_filtering/ckpt/xvla50_corrected_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model}
OUT_ROOT=${XVLA_EVAL_ROOT:-/data/yhr/output/online_filtering/eval/xvla_long50_fixed}/delay${DELAY}

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
export LIBERO_CONFIG_PATH="${ROOT}/research/config/libero_runtime"

groups=("0,4,8" "1,5,9" "2,6" "3,7")

run_worker() {
  local worker=$1
  local out="${OUT_ROOT}/worker_${worker}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPU,0" "$PYTHON" \
    "$ROOT/research/scripts/eval_libero_synthetic_delay.py" \
    --policy-path "$POLICY" --output-dir "$out" \
    --suite "libero_spatial,libero_object,libero_goal,libero_10" \
    --task-ids "${groups[$worker]}" --episodes 5 \
    --delay-min "$DELAY" --delay-max "$DELAY" --replan-interval 20 \
    --control-fps 20 --allow-queue-underrun --seed 1000 --device cuda \
    >"${out}/worker.log" 2>&1
}

pids=()
for worker in "${!groups[@]}"; do
  run_worker "$worker" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
