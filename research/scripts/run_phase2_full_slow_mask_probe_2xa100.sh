#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export LD_LIBRARY_PATH=/data/yhr/anaconda/anaconda3/lib
export PYTHONPATH=src

SOURCE=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow/phase2_rollout_only_mixed_schedule_2gpu/checkpoints/030000/pretrained_model
TRAIN_ROOT=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow
OUT=${PROBE_OUT:-$TRAIN_ROOT/phase2_full_slow_mask_p025_probe_3k}
LOG=${PROBE_LOG:-$TRAIN_ROOT/phase2_full_slow_mask_p025_probe_3k.log}
GPUS=${PROBE_GPUS:-4,5}

[[ -e "$SOURCE/model.safetensors" ]] || { echo "ERROR: missing source checkpoint: $SOURCE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 1; }

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --multi_gpu --num_processes=2 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/data/zhoukai/dataset/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$SOURCE" \
  --policy.dinov3_model_id=/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m \
  --policy.tokenizer_name=/data/zhoukai/pretrain/paligemma-tokenizer \
  --policy.training_stage=online_flow \
  --policy.online_flow_phase=ema_rollout \
  --policy.online_flow_ema_decay=0.999 \
  --policy.online_flow_rollout_steps=9 \
  --policy.online_flow_canonical_weight=0.0 \
  --policy.online_flow_rollout_weight=1.0 \
  --policy.fast_spatial_tokens=true \
  --policy.fast_num_tokens=64 \
  --policy.slow_visual_dropout_probability=0.25 \
  --policy.slow_visual_dropout_min=1.0 \
  --policy.slow_visual_dropout_max=1.0 \
  --policy.max_inference_delay=10 \
  --policy.optimizer_lr=5e-6 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=3000 \
  --policy.freeze_dinov3=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=phase2_full_slow_mask_p025_probe \
  --env_eval_freq=0 \
  --seed=1000 \
  --batch_size=16 \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers=8 \
  --steps=3000 \
  --save_checkpoint=true \
  --save_freq=3000 \
  --log_freq=20 \
  2>&1 | tee "$LOG"
