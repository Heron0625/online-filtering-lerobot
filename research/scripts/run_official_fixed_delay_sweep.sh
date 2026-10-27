#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/research/scripts/run_libero_suite_demos.sh"
PYTHON_BIN=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
OUT_ROOT=/data/yhr/output/online_filtering/eval/fixed_delay_sweep_fps20
OFFICIAL=/data/zhoukai/pretrain/pi05_libero_finetuned
GPU="${GPU:-2}"

for delay in 1 5 10; do
  echo "=== Official PI0.5 fixed delay=${delay} on GPU ${GPU} ==="
  POLICY=custom \
  CUSTOM_POLICY_NAME="official_delay${delay}" \
  CUSTOM_POLICY_PATH="$OFFICIAL" \
  PYTHON_BIN="$PYTHON_BIN" \
  CUDA_DEVICE="$GPU" \
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
