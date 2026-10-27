#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export PYTHONPATH=src

SOURCE=${XVLA_SOURCE:-/data/yhr/pretrain/online_filtering_baselines/xvla-online-filtering-init}
OUT=${XVLA_STAGE_A_OUT:-/data/yhr/output/online_filtering/ckpt/xvla_online_filtering/stage_a}
GPUS=${XVLA_GPUS:-4,5,6,7}
PROCESSES=${XVLA_PROCESSES:-4}
BATCH_SIZE=${XVLA_BATCH_SIZE:-2}
ACCUMULATION=${XVLA_ACCUMULATION:-16}
STEPS=${XVLA_STAGE_A_STEPS:-30000}

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --multi_gpu --num_processes="$PROCESSES" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/data/yhr/pretrain/recovered_zhoukai/dataset/libero \
  --dataset.video_backend=pyav \
  --policy.path="$SOURCE" \
  --policy.training_stage=stage_a \
  --policy.max_inference_delay=10 \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=xvla_online_filtering_stage_a \
  --env_eval_freq=0 \
  --batch_size="$BATCH_SIZE" \
  --accelerator.gradient_accumulation.steps="$ACCUMULATION" \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=50

