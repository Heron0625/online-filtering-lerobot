#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
POLICY=/data/yhr/pretrain/online_filtering_baselines/pi0fast-libero
ROOT=/data/yhr/output/online_filtering/eval/pi0fast_libero_all_dynamic_delay1_10_replan0
export CUDA_VISIBLE_DEVICES=4,0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
export LIBERO_CONFIG_PATH="${REPO_ROOT}/research/config/libero_runtime"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_task() {
  local suite="$1" task="$2" start="$3" count="$4" seed="$5" log="$6"
  local out="${ROOT}/worker_$((task % 5))/${suite}/task_${task}"
  "${PYTHON_BIN}" "${REPO_ROOT}/research/scripts/eval_libero_synthetic_delay.py" \
    --policy-path "${POLICY}" --output-dir "${out}" --suite "${suite}" --task-id "${task}" \
    --episode-start "${start}" --episodes "${count}" --delay-min 1 --delay-max 10 --dynamic-delay \
    --replan-interval 0 --control-fps 20 --allow-queue-underrun --seed "${seed}" --device cuda \
    >"${ROOT}/${log}" 2>&1
}

case "${1:?group required}" in
  a)
    run_task libero_spatial 5 41 9 1000 tail_spatial_5.log
    run_task libero_10 5 0 50 1000 tail_libero10_5.log
    ;;
  b)
    run_task libero_goal 0 18 32 1000 tail_goal_0.log
    run_task libero_10 4 0 50 401000 tail_libero10_4.log
    ;;
  c)
    run_task libero_10 6 8 42 101000 tail_libero10_6.log
    ;;
esac
