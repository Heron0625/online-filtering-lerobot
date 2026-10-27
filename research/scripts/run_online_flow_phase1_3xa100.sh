#!/usr/bin/env bash
# Canonical two-condition online-flow training on physical GPUs 4,5.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export LD_LIBRARY_PATH=/data/yhr/anaconda/anaconda3/lib
export PYTHONPATH=src

PI05_PRETRAINED=/data/zhoukai/pretrain/pi05_libero_finetuned
DINOV3_PATH=/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m
TOKENIZER_PATH=/data/zhoukai/pretrain/paligemma-tokenizer
LIBERO_ROOT=/data/zhoukai/dataset/libero
TRAIN_ROOT=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow
OUT=$TRAIN_ROOT/phase1_canonical_2gpu
LOG=$TRAIN_ROOT/phase1_canonical_2gpu_$(date +%Y%m%d_%H%M%S).log

for path in /data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python \
  "$PI05_PRETRAINED" "$DINOV3_PATH" "$TOKENIZER_PATH" "$LIBERO_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 1
  fi
done
if [[ -e "$OUT" ]]; then
  echo "ERROR: output already exists: $OUT" >&2
  exit 1
fi

mkdir -p "$TRAIN_ROOT"
echo "[two-condition phase1] GPUs=4,5 processes=2 per-GPU batch=16 global batch=32"
echo "[two-condition phase1] output=$OUT log=$LOG"

CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
  --num_processes=2 \
  --multi_gpu \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$LIBERO_ROOT" \
  --dataset.video_backend=torchcodec \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$PI05_PRETRAINED" \
  --policy.dinov3_model_id="$DINOV3_PATH" \
  --policy.tokenizer_name="$TOKENIZER_PATH" \
  --policy.training_stage=online_flow \
  --policy.online_flow_phase=canonical \
  --policy.fast_spatial_tokens=true \
  --policy.fast_num_tokens=64 \
  --policy.slow_visual_dropout_probability=0.5 \
  --policy.slow_visual_dropout_min=0.5 \
  --policy.slow_visual_dropout_max=0.75 \
  --policy.max_inference_delay=10 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.freeze_dinov3=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=two_condition_online_flow_phase1 \
  --env_eval_freq=0 \
  --seed=1000 \
  --batch_size=16 \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=50 \
  >> "$LOG" 2>&1
