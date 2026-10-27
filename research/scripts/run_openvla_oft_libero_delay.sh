#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN=/data/yhr/anaconda/anaconda3/envs/openvla_oft/bin/python
CHECKPOINT=/data/yhr/pretrain/online_filtering_baselines/openvla-oft-libero-all
OUT_ROOT=/data/yhr/output/online_filtering/eval/openvla_oft_libero_all_dynamic_delay1_10_replan0
export CUDA_VISIBLE_DEVICES=5,0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH="${REPO_ROOT}/research/config/libero_runtime"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

suites=(libero_spatial libero_object libero_goal libero_10)
task_groups=("0,4,8" "1,5,9" "2,6" "3,7")

run_worker() {
  local worker="$1" suite
  local worker_dir="${OUT_ROOT}/worker_${worker}"
  mkdir -p "${worker_dir}"
  # The upstream loader updates AutoClass metadata in config.json; stagger
  # startup so multiple workers never write that file simultaneously.
  sleep $((worker * 12))
  for suite in "${suites[@]}"; do
    "${PYTHON_BIN}" "${REPO_ROOT}/research/scripts/eval_openvla_oft_libero_delay.py" \
      --checkpoint "${CHECKPOINT}" --output-dir "${worker_dir}" --suite "${suite}" \
      --task-ids "${task_groups[$worker]}" --episodes 50 --delay-min 1 --delay-max 10 \
      --replan-interval 0 --seed 1000 --save-videos \
      >"${worker_dir}/${suite}.log" 2>&1
  done
}

pids=()
for worker in "${!task_groups[@]}"; do
  run_worker "${worker}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
