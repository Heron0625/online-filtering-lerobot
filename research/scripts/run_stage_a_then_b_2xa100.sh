#!/usr/bin/env bash
# Run Stage A and then automatically start Stage B on 2x A100-80GB.
#
# If Stage A is already running, this script waits for its final checkpoint.
# If Stage A has not started, this script starts it first.
# Usage:
#   bash research/scripts/run_stage_a_then_b_2xa100.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${LEROBOT_PYTHON_BIN:-/home/zhoukai/miniforge3/envs/lerobot/bin}:$PATH"
export PYTHONPATH=src
export DINOV3_PATH="${DINOV3_PATH:-/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m}"
export LIBERO_ROOT="${LIBERO_ROOT:-/data/zhoukai/dataset/libero}"
export TRAIN_ROOT="${TRAIN_ROOT:-/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast}"

STAGE_A_OUT="$TRAIN_ROOT/stage_a"
STAGE_A_CKPT="$STAGE_A_OUT/checkpoints/030000/pretrained_model"
STAGE_B_OUT="$TRAIN_ROOT/stage_b_online_teacher"

mkdir -p "$TRAIN_ROOT"

stage_a_checkpoint_complete() {
  [[ -f "$STAGE_A_CKPT/model.safetensors" ]] \
    && [[ -f "$STAGE_A_CKPT/config.json" ]] \
    && [[ -f "$STAGE_A_CKPT/train_config.json" ]]
}

stage_a_is_running() {
  pgrep -f -- "lerobot_train.*--output_dir=$STAGE_A_OUT" >/dev/null
}

if stage_a_checkpoint_complete; then
  echo "[pipeline] Reusing completed Stage A checkpoint."
elif stage_a_is_running; then
  echo "[pipeline] Stage A is already running."
  echo "[pipeline] Waiting for the complete checkpoint: $STAGE_A_CKPT"
  while ! stage_a_checkpoint_complete; do
    if ! stage_a_is_running; then
      echo "ERROR: Stage A stopped before producing a complete final checkpoint." >&2
      exit 1
    fi
    sleep 60
  done
elif [[ ! -d "$STAGE_A_OUT" ]]; then
  echo "[pipeline] Stage A does not exist; starting Stage A first."
  bash research/scripts/run_stage_a_2xa100.sh
else
  echo "ERROR: Stage A output exists but training is not running and its final checkpoint is incomplete." >&2
  echo "       Inspect or resume: $STAGE_A_OUT" >&2
  exit 1
fi

if ! stage_a_checkpoint_complete; then
  echo "ERROR: Stage A checkpoint is incomplete: $STAGE_A_CKPT" >&2
  exit 1
fi

if [[ -e "$STAGE_B_OUT" ]]; then
  echo "ERROR: $STAGE_B_OUT already exists. Move/remove it or resume it explicitly." >&2
  exit 1
fi

LOG="$TRAIN_ROOT/stage_b_online_teacher_train_$(date +%Y%m%d_%H%M%S).log"
echo "[pipeline] Stage A complete: $STAGE_A_CKPT"
echo "[stage_b full] 2 GPUs, 3000 steps -> $STAGE_B_OUT"
echo "log: $LOG"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes=2 \
  --multi_gpu \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$LIBERO_ROOT" \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$STAGE_A_CKPT" \
  --policy.dinov3_model_id="$DINOV3_PATH" \
  --policy.tokenizer_name=/data/zhoukai/pretrain/paligemma-tokenizer \
  --policy.training_stage=stage_b \
  --policy.max_inference_delay=10 \
  --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps=3000 \
  --policy.stage_b_min_gap=1 \
  --policy.stage_b_max_gap=10 \
  --policy.posterior_identity_dropout=0.1 \
  --policy.freeze_dinov3=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$STAGE_B_OUT" \
  --job_name=online_filtering_pi05_stage_b_online_teacher \
  --optimizer.lr=1e-6 \
  --batch_size=32 \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers=4 \
  --steps=3000 \
  --save_freq=500 \
  --log_freq=20 \
  --env_eval_freq=0 \
  --seed=1000 \
  2>&1 | tee "$LOG"
