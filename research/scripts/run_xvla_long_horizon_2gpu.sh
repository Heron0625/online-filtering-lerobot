#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export PYTHONPATH=src

SOURCE=${XVLA_SOURCE:-/data/yhr/pretrain/online_filtering_baselines/xvla-libero-long50-corrected-v2-init}
OUT=${XVLA_OUT:-/data/yhr/output/online_filtering/ckpt/xvla_long_horizon50_fixed_v3_3view_nw2_2gpu_b32}
GPUS=${XVLA_GPUS:-4,5}
BATCH_SIZE=${XVLA_BATCH_SIZE:-32}
STEPS=${XVLA_STEPS:-30000}
DATASET=${XVLA_DATASET:-/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked}

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --multi_gpu --num_processes=2 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$DATASET" \
  --dataset.video_backend=pyav \
  --policy.path="$SOURCE" \
  --policy.num_image_views=3 \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --policy.dtype=float32 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=xvla_long_horizon50 \
  --env_eval_freq=0 \
  --batch_size="$BATCH_SIZE" \
  --num_workers=2 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=50
