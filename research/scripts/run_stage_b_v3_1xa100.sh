#!/usr/bin/env bash
# Stage B v3 on one configurable GPU. Uses shared-noise online teachers under old/new conditions.
# Usage:
#   bash research/scripts/run_stage_b_v3_1xa100.sh
#   GPU_ID=1 BATCH_SIZE=32 STEPS=3000 bash research/scripts/run_stage_b_v3_1xa100.sh
#   GPU_IDS=0,1 NUM_PROCESSES=2 BATCH_SIZE=32 bash research/scripts/run_stage_b_v3_1xa100.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${LEROBOT_PYTHON_BIN:-/home/zhoukai/miniforge3/envs/lerobot/bin}:$PATH"
export PYTHONPATH=src
export DINOV3_PATH="${DINOV3_PATH:-/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m}"
export LIBERO_ROOT="${LIBERO_ROOT:-/data/zhoukai/dataset/libero}"
export TRAIN_ROOT="${TRAIN_ROOT:-/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast}"

GPU_ID="${GPU_ID:-1}"
GPU_IDS="${GPU_IDS:-$GPU_ID}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-3000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"

STAGE_A_CKPT="${STAGE_A_CKPT:-$TRAIN_ROOT/stage_a/checkpoints/030000/pretrained_model}"
STAGE_B_OUT="${STAGE_B_OUT:-$TRAIN_ROOT/stage_b_scale_balanced_transport}"

if [[ ! -f "$STAGE_A_CKPT/model.safetensors" ]]; then
  echo "ERROR: missing Stage A checkpoint: $STAGE_A_CKPT" >&2
  exit 1
fi
if [[ -e "$STAGE_B_OUT" ]]; then
  echo "ERROR: $STAGE_B_OUT already exists. Move/remove it or resume explicitly." >&2
  exit 1
fi

mkdir -p "$TRAIN_ROOT"
LOG="$TRAIN_ROOT/stage_b_condition_transport_train_$(date +%Y%m%d_%H%M%S).log"
echo "[stage_b online teacher] GPUs $GPU_IDS, per-GPU batch $BATCH_SIZE, global batch $((BATCH_SIZE * NUM_PROCESSES)), $STEPS steps -> $STAGE_B_OUT"
echo "[stage_b condition transport] shared noise is teacher-only; updater sees old/new conditions and particle"
echo "log: $LOG"

if (( NUM_PROCESSES == 1 )); then
  LAUNCH=(lerobot-train)
else
  LAUNCH=(accelerate launch --num_processes="$NUM_PROCESSES" --multi_gpu -m lerobot.scripts.lerobot_train)
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" "${LAUNCH[@]}" \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$LIBERO_ROOT" \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$STAGE_A_CKPT" \
  --policy.dinov3_model_id="$DINOV3_PATH" \
  --policy.tokenizer_name=/data/zhoukai/pretrain/paligemma-tokenizer \
  --policy.training_stage=stage_b \
  --policy.fast_spatial_tokens=true \
  --policy.fast_num_tokens=64 \
  --policy.max_inference_delay=10 \
  --policy.posterior_identity_dropout=0.0 \
  --policy.posterior_use_learned_gate=false \
  --policy.posterior_relative_weight=1.0 \
  --policy.posterior_cosine_weight=0.1 \
  --policy.posterior_relative_floor=1e-6 \
  --policy.stage_b_endpoint_probability=0.0 \
  --policy.posterior_endpoint_weight=1.0 \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.freeze_dinov3=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$STAGE_B_OUT" \
  --job_name=online_filtering_pi05_stage_b_condition_transport \
  --optimizer.lr=1e-6 \
  --batch_size="$BATCH_SIZE" \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=20 \
  --env_eval_freq=0 \
  --seed=1000 \
  2>&1 | tee "$LOG"
