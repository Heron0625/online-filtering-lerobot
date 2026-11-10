#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/yhr/project/robot/online_filtering
PREPROCESSED=/data/yhr/pretrain/xvla_official/libero_xvla_preprocessed
DATASET=/data/yhr/pretrain/xvla_official/lerobot_libero_20hz_absolute
SOURCE=/data/yhr/pretrain/online_filtering_baselines/xvla-libero-long50-fixed-v2-init
OUTPUT=/data/yhr/output/online_filtering/ckpt/xvla_long_horizon50_lerobot20hz_2gpu_b32
PY=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
ACCELERATE=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/accelerate

cd "$ROOT"
while true; do
  completed=$(find "$PREPROCESSED" -mindepth 2 -maxdepth 2 -name '*.hdf5' -type f 2>/dev/null | wc -l)
  partial=$(find "$PREPROCESSED" -name '*.incomplete' -type f 2>/dev/null | wc -l)
  echo "preprocessed tasks: ${completed}/40, active partials: ${partial}"
  if [[ "$completed" -eq 40 && "$partial" -eq 0 ]]; then
    break
  fi
  sleep 60
done

if [[ ! -f "$DATASET/meta/info.json" ]]; then
  "$PY" research/scripts/convert_xvla_preprocessed_to_lerobot.py \
    --source-root "$PREPROCESSED" \
    --output-root "$DATASET" \
    --repo-id local/xvla_libero_20hz_absolute
fi

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=src
exec "$ACCELERATE" launch --multi_gpu --num_processes=2 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/xvla_libero_20hz_absolute \
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
  --output_dir="$OUTPUT" \
  --job_name=xvla_long_horizon50_lerobot20hz \
  --env_eval_freq=0 \
  --batch_size=32 \
  --num_workers=4 \
  --steps=60000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=50
