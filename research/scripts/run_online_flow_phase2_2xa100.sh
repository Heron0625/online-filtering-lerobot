#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export LD_LIBRARY_PATH=/data/yhr/anaconda/anaconda3/lib
export PYTHONPATH=src

PHASE1=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow/phase1_canonical_2gpu/checkpoints/030000/pretrained_model
TRAIN_ROOT=/data/yhr/output/online_filtering/ckpt/two_condition_online_flow
OUT=${PHASE2_OUT:-$TRAIN_ROOT/phase2_rollout_only_mixed_schedule_2gpu}
LOG=${PHASE2_LOG:-$TRAIN_ROOT/phase2_rollout_only_mixed_schedule_2gpu_$(date +%Y%m%d_%H%M%S).log}
STEPS=${PHASE2_STEPS:-30000}
BATCH_SIZE=${PHASE2_BATCH_SIZE:-16}
SAVE_FREQ=${PHASE2_SAVE_FREQ:-10000}
SAVE_CHECKPOINT=${PHASE2_SAVE_CHECKPOINT:-true}
GPUS=${PHASE2_GPUS:-4,5}
PROCESSES=${PHASE2_PROCESSES:-2}

[[ -e "$PHASE1/model.safetensors" ]] || { echo "ERROR: missing Phase-1 checkpoint: $PHASE1" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$TRAIN_ROOT"

echo "[two-condition phase2] GPUs=$GPUS processes=$PROCESSES per-GPU batch=$BATCH_SIZE"
echo "[two-condition phase2] output=$OUT log=$LOG"

LAUNCH_ARGS=(--num_processes="$PROCESSES")
if (( PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  "${LAUNCH_ARGS[@]}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/data/zhoukai/dataset/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$PHASE1" \
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
  --policy.slow_visual_dropout_probability=0.0 \
  --policy.slow_visual_dropout_min=0.5 \
  --policy.slow_visual_dropout_max=0.75 \
  --policy.max_inference_delay=25 \
  --policy.optimizer_lr=5e-6 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.freeze_dinov3=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=two_condition_online_flow_phase2 \
  --env_eval_freq=0 \
  --seed=1000 \
  --batch_size="$BATCH_SIZE" \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  >> "$LOG" 2>&1
