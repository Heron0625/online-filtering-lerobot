#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/research/scripts/run_libero_suite_demos.sh"
PYTHON_BIN=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
OUT_ROOT=/data/yhr/output/online_filtering/eval/fixed_delay_sweep_fps20
PHASE_A=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow/phase1_no_innovation_fullmask_p025_4gpu/checkpoints/030000/pretrained_model
PHASE_2=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow/phase2_no_innovation_annealed_fullmask_2gpu/checkpoints/030000/pretrained_model

run_model() {
  local label="$1"
  local checkpoint="$2"
  local gpu="$3"
  local delay
  for delay in 1 5 10; do
    echo "=== ${label} fixed delay=${delay} on GPU ${gpu} ==="
    POLICY=custom \
    CUSTOM_POLICY_NAME="${label}_delay${delay}" \
    CUSTOM_POLICY_PATH="$checkpoint" \
    PYTHON_BIN="$PYTHON_BIN" \
    CUDA_DEVICE="$gpu" \
    WORKERS=5 \
    SHARD_COUNT=5 \
    EPISODES=10 \
    DELAY_MIN="$delay" \
    DELAY_MAX="$delay" \
    REPLAN_INTERVAL=20 \
    CONTROL_FPS=20 \
    OUT_ROOT="$OUT_ROOT" \
    LIBERO_CONFIG_PATH="$ROOT/research/config/libero_runtime" \
      bash "$RUNNER"
  done
}

mkdir -p "$OUT_ROOT"
run_model phase_a "$PHASE_A" 4 &
phase_a_pid=$!
run_model phase_2 "$PHASE_2" 5 &
phase_2_pid=$!

status=0
wait "$phase_a_pid" || status=1
wait "$phase_2_pid" || status=1
exit "$status"
