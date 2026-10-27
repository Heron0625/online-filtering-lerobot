#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export PYTHONPATH=src

SOURCE=${XVLA_STAGE_A_SOURCE:?Set XVLA_STAGE_A_SOURCE to a Stage-A pretrained_model directory}
OUT=${XVLA_STAGE_B_OUT:-/data/yhr/output/online_filtering/ckpt/xvla_online_filtering/stage_b}
GPUS=${XVLA_GPUS:-4,5,6,7}
PROCESSES=${XVLA_PROCESSES:-4}
BATCH_SIZE=${XVLA_BATCH_SIZE:-2}
ACCUMULATION=${XVLA_ACCUMULATION:-16}
STEPS=${XVLA_STAGE_B_STEPS:-30000}

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --multi_gpu --num_processes="$PROCESSES" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/data/yhr/pretrain/recovered_zhoukai/dataset/libero \
  --dataset.video_backend=pyav \
  --policy.path="$SOURCE" \
  --policy.training_stage=stage_b \
  --policy.max_inference_delay=25 \
  --policy.dinov3_model_id=/data/yhr/pretrain/dinov3-vits16-pretrain-lvd1689m \
  --policy.freeze_dinov3=true \
  --policy.fast_num_tokens=64 \
  --policy.online_flow_ema_decay=0.999 \
  --policy.online_flow_rollout_steps=9 \
  --policy.slow_visual_dropout_probability=0.25 \
  --policy.slow_visual_dropout_final_probability=0.0 \
  --policy.slow_visual_dropout_anneal_start_step=6000 \
  --policy.slow_visual_dropout_anneal_steps=18000 \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=xvla_online_filtering_stage_b \
  --env_eval_freq=0 \
  --batch_size="$BATCH_SIZE" \
  --accelerator.gradient_accumulation.steps="$ACCUMULATION" \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=50

