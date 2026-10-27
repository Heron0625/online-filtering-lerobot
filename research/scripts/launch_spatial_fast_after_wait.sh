#!/usr/bin/env bash
# One-shot delayed launcher: Stage B plus the five-worker Stage-A LIBERO evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAIT_SECONDS="${WAIT_SECONDS:-3600}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast}"
STAGE_A_CKPT="$TRAIN_ROOT/stage_a/checkpoints/030000/pretrained_model"
EVAL_ROOT="${EVAL_ROOT:-/data/zhoukai/robotmine/eval/libero_continuous_online_spatial_fast}"
LOG_DIR="$TRAIN_ROOT/launch_logs"
mkdir -p "$LOG_DIR"

echo "[$(date --iso-8601=seconds)] waiting ${WAIT_SECONDS}s before one completion check"
sleep "$WAIT_SECONDS"

for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  if [[ ! -f "$STAGE_A_CKPT/$required" ]]; then
    echo "[$(date --iso-8601=seconds)] Stage A checkpoint incomplete: $STAGE_A_CKPT/$required missing" >&2
    exit 1
  fi
done

if [[ -e "$TRAIN_ROOT/stage_b_online_teacher" ]]; then
  echo "[$(date --iso-8601=seconds)] Stage B output already exists; refusing to overwrite" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Stage A checkpoint complete; launching Stage B on GPU0"
GPU_ID=0 BATCH_SIZE=32 bash research/scripts/run_stage_b_v3_1xa100.sh \
  > "$LOG_DIR/stage_b.log" 2>&1 &
echo $! > "$LOG_DIR/stage_b.pid"

echo "[$(date --iso-8601=seconds)] launching Stage A evaluation: GPU0 workers 0,1; GPU1 workers 2,3,4"
CUDA_DEVICE=0 WORKERS=2 SHARD_COUNT=5 WORKER_OFFSET=0 POLICY=stage_a \
  OUT_ROOT="$EVAL_ROOT" STAGE_A_POLICY="$STAGE_A_CKPT" \
  bash research/scripts/run_libero_suite_demos.sh \
  > "$LOG_DIR/stage_a_eval_gpu0.log" 2>&1 &
echo $! > "$LOG_DIR/stage_a_eval_gpu0.pid"

CUDA_DEVICE=1 WORKERS=3 SHARD_COUNT=5 WORKER_OFFSET=2 POLICY=stage_a \
  OUT_ROOT="$EVAL_ROOT" STAGE_A_POLICY="$STAGE_A_CKPT" \
  bash research/scripts/run_libero_suite_demos.sh \
  > "$LOG_DIR/stage_a_eval_gpu1.log" 2>&1 &
echo $! > "$LOG_DIR/stage_a_eval_gpu1.pid"

echo "[$(date --iso-8601=seconds)] launch submitted"
