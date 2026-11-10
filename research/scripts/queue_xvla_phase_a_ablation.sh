#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?physical GPU index required}
MASK_FINAL=${2:?slow visual dropout probability required}
OUT=${3:?output path required}
PORT=${4:?accelerate port required}
MASK_START=${5:-500}
MASK_STEPS=${6:-2000}
MAIN_DELAY_PROB=${7:-0.5}
MEDIUM_DELAY_PROB=${8:-0.25}
ROOT=/data/yhr/project/robot/online_filtering

echo "Waiting for the matched Phase A evaluation to finish before using GPU $GPU."
while [[ ! -f "$ROOT/research/outputs/pi05_phase_a_full_r25_20261108/comparison.json" \
     || ! -f "$ROOT/research/outputs/xvla_phase_a_full_r25_20261108/comparison.json" ]] \
   || tmux has-session -t pi05_phase_a_r25_1108 2>/dev/null \
   || tmux has-session -t xvla_phase_a_r25_1108 2>/dev/null; do
  sleep 60
done

while true; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU + 1))p" | tr -d ' ')
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 48000 )); then
    break
  fi
  echo "GPU $GPU has ${free_mib:-unknown} MiB free; waiting for at least 48000 MiB."
  sleep 60
done

echo "Starting XVLA Phase A ablation on GPU $GPU, slow-mask final probability $MASK_FINAL."
GPU="$GPU" MASK_FINAL="$MASK_FINAL" MASK_START="$MASK_START" MASK_STEPS="$MASK_STEPS" \
  MAIN_DELAY_PROB="$MAIN_DELAY_PROB" MEDIUM_DELAY_PROB="$MEDIUM_DELAY_PROB" \
  OUT="$OUT" PORT="$PORT" BATCH_SIZE=8 ACCUMULATION=8 \
  bash "$ROOT/research/scripts/run_xvla_phase_a_single_gpu_ablation.sh"
