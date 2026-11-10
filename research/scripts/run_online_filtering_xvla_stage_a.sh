#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

SOURCE=${XVLA_SOURCE:-/data/yhr/pretrain/online_filtering_baselines/xvla50-online-filtering-stage-a-corrected-v2-init}
OUT=${XVLA_STAGE_A_OUT:-/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031}
GPUS=${XVLA_GPUS:-4,5}
PROCESSES=${XVLA_PROCESSES:-2}
BATCH_SIZE=${XVLA_BATCH_SIZE:-32}
ACCUMULATION=${XVLA_ACCUMULATION:-1}
STEPS=${XVLA_STAGE_A_STEPS:-30000}
DATASET=${XVLA_DATASET:-/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked}

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --multi_gpu --num_processes="$PROCESSES" \
  --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no --main_process_port=29647 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$DATASET" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0 \
  --policy.path="$SOURCE" \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.tokenizer_name=/data/yhr/pretrain/online_filtering_baselines/bart-large-tokenizer \
  --policy.tokenizer_max_length=50 \
  --policy.training_stage=stage_a \
  --policy.max_inference_delay=25 \
  --policy.dinov3_model_id=/data/yhr/pretrain/dinov3-vits16-pretrain-lvd1689m \
  --policy.freeze_dinov3=true \
  --policy.fast_num_tokens=64 \
  --policy.slow_visual_dropout_probability=0.0 \
  --policy.slow_visual_dropout_final_probability=0.25 \
  --policy.slow_visual_dropout_anneal_start_step=500 \
  --policy.slow_visual_dropout_anneal_steps=2000 \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.dtype=float32 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=xvla_online_filtering_stage_a \
  --env_eval_freq=0 \
  --eval_steps=0 --wandb.enable=false \
  --batch_size="$BATCH_SIZE" \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps="$ACCUMULATION" \
  --num_workers=2 \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=10

