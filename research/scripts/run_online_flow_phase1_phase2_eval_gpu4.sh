#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/research/scripts/run_libero_suite_demos.sh"
PYTHON_BIN=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
OUT_ROOT=/data/yhr/output/online_filtering/eval
LIBERO_CONFIG_PATH="$ROOT/research/config/libero_runtime"
PHASE1=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow/phase1_no_innovation_fullmask_p025_4gpu/checkpoints/030000/pretrained_model
PHASE2=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow/phase2_no_innovation_annealed_fullmask_2gpu/checkpoints/030000/pretrained_model

for path in "$PYTHON_BIN" "$PHASE1/model.safetensors" "$PHASE2/model.safetensors"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
mkdir -p "$OUT_ROOT"

run_eval() {
  local name="$1"
  local checkpoint="$2"
  local cuda_device="$3"
  POLICY=custom \
  CUSTOM_POLICY_NAME="$name" \
  CUSTOM_POLICY_PATH="$checkpoint" \
  PYTHON_BIN="$PYTHON_BIN" \
  CUDA_DEVICE="$cuda_device" \
  WORKERS=5 \
  SHARD_COUNT=5 \
  EPISODES=50 \
  DELAY_MIN=1 \
  DELAY_MAX=10 \
  REPLAN_INTERVAL=20 \
  CONTROL_FPS=20 \
  OUT_ROOT="$OUT_ROOT" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  bash "$RUNNER"
  "$PYTHON_BIN" "$ROOT/research/scripts/consolidate_libero_worker_outputs.py" \
    --input-dir "$OUT_ROOT/$name"
}

run_eval phase1_noinnovation_fullmask_30k_fps20 "$PHASE1" 4 &
phase1_pid=$!
run_eval phase2_noinnovation_annealed_30k_fps20 "$PHASE2" 5 &
phase2_pid=$!

status=0
wait "$phase1_pid" || status=1
wait "$phase2_pid" || status=1
if (( status != 0 )); then
  echo "At least one evaluation failed; inspect the worker logs." >&2
  exit "$status"
fi

echo "Evaluation complete: $OUT_ROOT"
