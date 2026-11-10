#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

GPU=${GPU:?Set physical GPU index}
OUT=${OUT:?Set unique output directory}
PORT=${PORT:?Set unique accelerate port}
ABLATION_NAME=${ABLATION_NAME:?Set descriptive job name}
MASK_FINAL=${MASK_FINAL:-0.25}
MASK_START=${MASK_START:-500}
MASK_STEPS=${MASK_STEPS:-2000}
FAST_ADAPTERS_ENABLED=${FAST_ADAPTERS_ENABLED:-true}
FAST_CURRENT_ONLY=${FAST_CURRENT_ONLY:-false}
BATCH_SIZE=${BATCH_SIZE:-8}

PHASE_A=/data/yhr/output/online_filtering/ckpt/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101/checkpoints/030000/pretrained_model
DATASET=/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked
for path in "$PHASE_A" "$DATASET"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES="$GPU"

exec accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 \
  --dynamo_backend=no --main_process_port="$PORT" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$DATASET" --dataset.video_backend=pyav --dataset.eval_split=0 \
  --policy.path="$PHASE_A" \
  --policy.training_stage=stage_a --policy.max_inference_delay=25 \
  --policy.fast_adapters_enabled="$FAST_ADAPTERS_ENABLED" \
  --policy.fast_current_only="$FAST_CURRENT_ONLY" \
  --policy.slow_visual_dropout_probability=0.0 \
  --policy.slow_visual_dropout_final_probability="$MASK_FINAL" \
  --policy.slow_visual_dropout_anneal_start_step="$MASK_START" \
  --policy.slow_visual_dropout_anneal_steps="$MASK_STEPS" \
  --policy.dtype=float32 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name="$ABLATION_NAME" \
  --env_eval_freq=0 --eval_steps=0 --wandb.enable=false \
  --seed=1000 --batch_size="$BATCH_SIZE" --num_workers=2 \
  --accelerator.mixed_precision=bf16 --accelerator.gradient_accumulation.steps=1 \
  --steps=30000 --save_checkpoint=true --save_freq=10000 --log_freq=10
