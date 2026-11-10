#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${XVLA_GPUS:-4,5}"
SOURCE=${XVLA_SOURCE:-/data/yhr/pretrain/online_filtering_baselines/xvla-libero}
DATASET=${XVLA_DATASET:-/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked}
OUT=${XVLA_OUT:-/data/yhr/output/online_filtering/ckpt/xvla30_corrected_v2_2gpu_b32_30k_20261030}
exec accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=no --main_process_port=29645 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$DATASET" \
  --dataset.video_backend=pyav \
  --policy.path="$SOURCE" \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.tokenizer_name=/data/yhr/pretrain/online_filtering_baselines/bart-large-tokenizer \
  --policy.tokenizer_max_length=50 \
  --policy.num_image_views=3 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --policy.dtype=float32 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name=xvla30_corrected_v2 \
  --env_eval_freq=0 --batch_size=32 --num_workers=2 --steps=30000 \
  --save_checkpoint=true --save_freq=10000 --log_freq=10
