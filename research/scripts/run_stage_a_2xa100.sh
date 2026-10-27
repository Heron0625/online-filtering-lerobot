#!/usr/bin/env bash
# Clean online Stage A training for online_filtering_pi05 on 2x A100-80GB.
# Usage:
#   bash research/scripts/run_stage_a_2xa100.sh           # full 30k steps, global batch 32
#   bash research/scripts/run_stage_a_2xa100.sh smoke     # 20-step sanity check (1 GPU)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${LEROBOT_PYTHON_BIN:-/home/zhoukai/miniforge3/envs/lerobot/bin}:$PATH"
export PYTHONPATH=src
export PI05_PRETRAINED="${PI05_PRETRAINED:-/data/zhoukai/pretrain/pi05_libero_finetuned}"
export DINOV3_PATH="${DINOV3_PATH:-/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m}"
export LIBERO_ROOT="${LIBERO_ROOT:-/data/zhoukai/dataset/libero}"
export TRAIN_ROOT="${TRAIN_ROOT:-/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast}"

MODE="${1:-full}"

COMMON_ARGS=(
  --dataset.repo_id=lerobot/libero
  --dataset.root="$LIBERO_ROOT"
  --policy.type=online_filtering_pi05
  --policy.pretrained_path="$PI05_PRETRAINED"
  --policy.dinov3_model_id="$DINOV3_PATH"
  --policy.tokenizer_name=/data/zhoukai/pretrain/paligemma-tokenizer
  --policy.training_stage=stage_a
  --policy.fast_spatial_tokens=true
  --policy.fast_num_tokens=64
  --policy.slow_visual_dropout_probability=0.5
  --policy.slow_visual_dropout_min=0.5
  --policy.slow_visual_dropout_max=0.75
  --policy.max_inference_delay=10
  --policy.stage_a_clean_probability=1.0
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.freeze_dinov3=true
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.device=cuda
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
  --policy.push_to_hub=false
  --job_name=online_filtering_pi05_stage_a
  --env_eval_freq=0
  --seed=1000
)

for required_file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  if [[ ! -f "$PI05_PRETRAINED/$required_file" ]]; then
    echo "ERROR: missing official LIBERO checkpoint file: $PI05_PRETRAINED/$required_file" >&2
    exit 1
  fi
done

mkdir -p "$TRAIN_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  OUT="$TRAIN_ROOT/stage_a_smoke"
  rm -rf "$OUT"
  echo "[stage_a smoke] official LIBERO init, 1 GPU, 20 steps -> $OUT"
  CUDA_VISIBLE_DEVICES=0 lerobot-train \
    "${COMMON_ARGS[@]}" \
    --output_dir="$OUT" \
    --batch_size=8 \
    --accelerator.gradient_accumulation.steps=1 \
    --num_workers=2 \
    --steps=20 \
    --save_freq=10 \
    --log_freq=5
else
  OUT="$TRAIN_ROOT/stage_a"
  if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists. Move/remove it or resume explicitly." >&2
    exit 1
  fi
  LOG="$TRAIN_ROOT/stage_a_train_$(date +%Y%m%d_%H%M%S).log"
  echo "[stage_a full] official LIBERO init: $PI05_PRETRAINED"
  echo "[stage_a full] 2 GPUs, per-GPU batch 16, global batch 32, 30000 steps -> $OUT"
  echo "log: $LOG"
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes=2 \
    --multi_gpu \
    -m lerobot.scripts.lerobot_train \
    "${COMMON_ARGS[@]}" \
    --output_dir="$OUT" \
    --batch_size=24 \
    --accelerator.gradient_accumulation.steps=1 \
    --num_workers=8 \
    --steps=30000 \
    --save_freq=10000 \
    --log_freq=50 \
    2>&1 | tee "$LOG"
fi
