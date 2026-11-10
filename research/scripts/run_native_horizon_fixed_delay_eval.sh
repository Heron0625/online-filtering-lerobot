#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL=${1:?Usage: $0 openvla|pi0fast [gpu]}
GPU=${2:-4}

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
export LIBERO_CONFIG_PATH="${ROOT}/research/config/libero_runtime"

suites=(libero_spatial libero_object libero_goal libero_10)

if [[ "$MODEL" == openvla ]]; then
  PYTHON=/data/yhr/anaconda/anaconda3/envs/openvla_oft/bin/python
  CHECKPOINT=/data/yhr/pretrain/online_filtering_baselines/openvla-oft-libero-all
  OUT_ROOT=/data/yhr/output/online_filtering/eval/openvla_oft_native_fixed_delay
  delays=(1 4 7)
  groups=("0,4,8" "1,5,9" "2,6" "3,7")

  run_worker() {
    local delay=$1 worker=$2 suite
    local out="${OUT_ROOT}/delay${delay}/worker_${worker}"
    mkdir -p "$out"
    # The upstream loader may update checkpoint AutoClass metadata.
    sleep $((worker * 12))
    for suite in "${suites[@]}"; do
      CUDA_VISIBLE_DEVICES="$GPU,0" "$PYTHON" \
        "$ROOT/research/scripts/eval_openvla_oft_libero_delay.py" \
        --checkpoint "$CHECKPOINT" --output-dir "$out" --suite "$suite" \
        --task-ids "${groups[$worker]}" --episodes 50 \
        --delay-min "$delay" --delay-max "$delay" --replan-interval 0 --seed 1000 \
        >"${out}/${suite}.log" 2>&1
    done
  }
elif [[ "$MODEL" == pi0fast ]]; then
  PYTHON=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
  CHECKPOINT=/data/yhr/pretrain/online_filtering_baselines/pi0fast-libero
  OUT_ROOT=/data/yhr/output/online_filtering/eval/pi0fast_native_fixed_delay
  delays=(1 5 9)
  groups=("0,5" "1,6" "2,7" "3,8" "4,9")
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

  run_worker() {
    local delay=$1 worker=$2
    local out="${OUT_ROOT}/delay${delay}/worker_${worker}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$GPU,0" "$PYTHON" \
      "$ROOT/research/scripts/eval_libero_synthetic_delay.py" \
      --policy-path "$CHECKPOINT" --output-dir "$out" \
      --suite "libero_spatial,libero_object,libero_goal,libero_10" \
      --task-ids "${groups[$worker]}" --episodes 50 \
      --delay-min "$delay" --delay-max "$delay" --replan-interval 0 \
      --control-fps 20 --allow-queue-underrun --seed 1000 --device cuda \
      >"${out}/worker.log" 2>&1
  }
else
  echo "Unknown model: $MODEL" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
for delay in "${delays[@]}"; do
  pids=()
  for worker in "${!groups[@]}"; do
    run_worker "$delay" "$worker" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  if (( status != 0 )); then
    echo "$MODEL delay=$delay failed; inspect ${OUT_ROOT}/delay${delay}/worker_* logs" >&2
    exit "$status"
  fi
done
