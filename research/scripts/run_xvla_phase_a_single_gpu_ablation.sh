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
PORT=${PORT:-29648}
MASK_FINAL=${MASK_FINAL:?Set slow visual dropout final probability}
MASK_START=${MASK_START:-500}
MASK_STEPS=${MASK_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-8}
ACCUMULATION=${ACCUMULATION:-8}
MAIN_DELAY_PROB=${MAIN_DELAY_PROB:-0.5}
MEDIUM_DELAY_PROB=${MEDIUM_DELAY_PROB:-0.25}
FAST_ADAPTER_LAYERS=${FAST_ADAPTER_LAYERS:-'[3,7,11,15,19,23]'}
ABLATION_NAME=${ABLATION_NAME:-xvla_phase_a_mask_${MASK_FINAL}}
SOURCE=/data/yhr/pretrain/online_filtering_baselines/xvla50-online-filtering-stage-a-corrected-v2-init
DATASET=/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked

for path in "$SOURCE" "$DATASET"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES="$GPU"

exec accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 \
  --dynamo_backend=no --main_process_port="$PORT" \
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
  --policy.fast_adapter_layers="$FAST_ADAPTER_LAYERS" \
  --policy.slow_visual_dropout_probability=0.0 \
  --policy.slow_visual_dropout_final_probability="$MASK_FINAL" \
  --policy.slow_visual_dropout_anneal_start_step="$MASK_START" \
  --policy.slow_visual_dropout_anneal_steps="$MASK_STEPS" \
  --policy.main_delay_probability="$MAIN_DELAY_PROB" \
  --policy.medium_delay_probability="$MEDIUM_DELAY_PROB" \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.dtype=float32 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name="$ABLATION_NAME" \
  --env_eval_freq=0 --eval_steps=0 --wandb.enable=false \
  --seed=1000 --batch_size="$BATCH_SIZE" --num_workers=2 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps="$ACCUMULATION" \
  --steps=30000 --save_checkpoint=true --save_freq=10000 --log_freq=10
