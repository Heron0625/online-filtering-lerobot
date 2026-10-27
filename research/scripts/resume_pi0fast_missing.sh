#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python}"
GPU="${GPU:-4}"
ROOT="${ROOT:-/data/yhr/output/online_filtering/eval/pi0fast_libero_all_dynamic_delay1_10_replan0}"
POLICY="${POLICY:-/data/yhr/pretrain/online_filtering_baselines/pi0fast-libero}"

export CUDA_VISIBLE_DEVICES="${GPU},0"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
export LIBERO_CONFIG_PATH="${REPO_ROOT}/research/config/libero_runtime"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Only these task summaries are absent from the interrupted 2,000-episode run.
groups=(
  "libero_spatial:5 libero_10:5"
  "libero_object:0 libero_10:0"
  "libero_object:5 libero_10:6"
  "libero_goal:0 libero_10:4"
  "libero_goal:5 libero_10:9"
)

run_group() {
  local group_id="$1"
  local item suite task worker task_dir log
  for item in ${groups[$group_id]}; do
    suite="${item%%:*}"
    task="${item##*:}"
    worker=$((task % 5))
    task_dir="${ROOT}/worker_${worker}/${suite}/task_${task}"
    log="${ROOT}/resume_group_${group_id}_${suite}_task_${task}.log"
    mkdir -p "${task_dir}"
    "${PYTHON_BIN}" "${REPO_ROOT}/research/scripts/eval_libero_synthetic_delay.py" \
      --policy-path "${POLICY}" \
      --output-dir "${task_dir}" \
      --suite "${suite}" \
      --task-id "${task}" \
      --episodes 50 \
      --delay-min 1 \
      --delay-max 10 \
      --dynamic-delay \
      --replan-interval 0 \
      --control-fps 20 \
      --allow-queue-underrun \
      --seed $((1000 + worker * 100000)) \
      --device cuda >"${log}" 2>&1
  done
}

pids=()
for group_id in "${!groups[@]}"; do
  run_group "${group_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
