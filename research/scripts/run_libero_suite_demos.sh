#!/usr/bin/env bash
# Run a complete delayed-inference LIBERO evaluation for one policy on all four suites.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhoukai/miniforge3/envs/lerobot/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DELAY_MIN="${DELAY_MIN:-1}"
DELAY_MAX="${DELAY_MAX:-10}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-20}"
CONTROL_FPS="${CONTROL_FPS:-20}"
EPISODE_LENGTH="${EPISODE_LENGTH:-}"
ALLOW_QUEUE_UNDERRUN="${ALLOW_QUEUE_UNDERRUN:-0}"
# LIBERO's standard evaluation uses all 50 initial states for every task.
# Override with EPISODES=N only for a smoke test.
EPISODES="${EPISODES:-50}"
WORKERS="${WORKERS:-1}"
# When launching workers across multiple GPUs, use one global shard space.  For example,
# SHARD_COUNT=5, WORKER_OFFSET=2, WORKERS=3 launches global workers 2,3,4.
SHARD_COUNT="${SHARD_COUNT:-$WORKERS}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
POLICY="${POLICY:-stage_a}"
OUT_ROOT="${OUT_ROOT:-/data/zhoukai/robotmine/eval/libero_continuous_online_clean_causal}"
OFFICIAL_POLICY="${OFFICIAL_POLICY:-/data/zhoukai/pretrain/pi05_libero_finetuned}"
STAGE_A_POLICY="${STAGE_A_POLICY:-/data/zhoukai/robotmine/ckpt/online_filtering_pi05_clean_causal/stage_a/checkpoints/030000/pretrained_model}"
STAGE_B_POLICY="${STAGE_B_POLICY:-/data/zhoukai/robotmine/ckpt/online_filtering_pi05_clean_causal/stage_b_online_teacher/checkpoints/003000/pretrained_model}"
CUSTOM_POLICY_NAME="${CUSTOM_POLICY_NAME:-custom}"
CUSTOM_POLICY_PATH="${CUSTOM_POLICY_PATH:-}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/research/config/libero_runtime}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ "$CUDA_DEVICE" == "0" ]]; then
  CUDA_VISIBLE_DEVICES_LIST="0"
else
  # Keep the requested physical GPU as logical cuda:0 for the policy, while robosuite
  # receives a visible-device list containing the physical id it validates at import time.
  CUDA_VISIBLE_DEVICES_LIST="${CUDA_DEVICE},0"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || {
  echo "Missing LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml" >&2
  exit 2
}

SUITES=(libero_spatial libero_object libero_goal libero_10)

run_policy() {
  local policy_name="$1"
  local policy_path="$2"
  echo "=== ${policy_name} | suites=${SUITES[*]} | workers=${WORKERS} on cuda:${CUDA_DEVICE} ==="
  local queue_args=()
  local episode_args=()
  if [[ "${ALLOW_QUEUE_UNDERRUN}" == "1" ]]; then
    queue_args+=(--allow-queue-underrun)
  fi
  if [[ -n "${EPISODE_LENGTH}" ]]; then
    episode_args+=(--episode-length "${EPISODE_LENGTH}")
  fi
  if (( WORKERS <= 1 )); then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}" "${PYTHON_BIN}" \
      "${REPO_ROOT}/research/scripts/eval_libero_synthetic_delay.py" \
      --policy-path "${policy_path}" \
      --output-dir "${OUT_ROOT}/${policy_name}" \
      --suite "$(IFS=,; echo "${SUITES[*]}")" \
      --all-tasks \
      --episodes "${EPISODES}" \
      --delay-min "${DELAY_MIN}" \
      --delay-max "${DELAY_MAX}" \
      --dynamic-delay \
      --replan-interval "${REPLAN_INTERVAL}" \
      --control-fps "${CONTROL_FPS}" \
      "${episode_args[@]}" \
      "${queue_args[@]}" \
      --seed 1000 \
      --device cuda
    return
  fi

  local pids=()
  local worker
  for ((local_worker = 0; local_worker < WORKERS; local_worker++)); do
    worker=$((WORKER_OFFSET + local_worker))
    if (( worker < 0 || worker >= SHARD_COUNT )); then
      echo "worker index ${worker} is outside SHARD_COUNT=${SHARD_COUNT}" >&2
      return 2
    fi
    local task_ids=()
    local task_id
    for ((task_id = worker; task_id < 10; task_id += SHARD_COUNT)); do
      task_ids+=("${task_id}")
    done
    if (( ${#task_ids[@]} == 0 )); then
      continue
    fi
    local task_ids_csv
    task_ids_csv="$(IFS=,; echo "${task_ids[*]}")"
    local worker_dir="${OUT_ROOT}/${policy_name}/worker_${worker}"
    mkdir -p "${worker_dir}"
    echo "worker_${worker}: task_ids=${task_ids_csv} log=${worker_dir}/worker.log"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}" "${PYTHON_BIN}" \
      "${REPO_ROOT}/research/scripts/eval_libero_synthetic_delay.py" \
      --policy-path "${policy_path}" \
      --output-dir "${worker_dir}" \
      --suite "$(IFS=,; echo "${SUITES[*]}")" \
      --task-ids "${task_ids_csv}" \
      --episodes "${EPISODES}" \
      --delay-min "${DELAY_MIN}" \
      --delay-max "${DELAY_MAX}" \
      --dynamic-delay \
      --replan-interval "${REPLAN_INTERVAL}" \
      --control-fps "${CONTROL_FPS}" \
      "${episode_args[@]}" \
      "${queue_args[@]}" \
      --seed $((1000 + worker * 100000)) \
      --device cuda \
      > "${worker_dir}/worker.log" 2>&1 &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "${policy_name} failed; check ${OUT_ROOT}/${policy_name}/worker_*/worker.log" >&2
    return "${status}"
  fi

  "${PYTHON_BIN}" "${REPO_ROOT}/research/scripts/aggregate_libero_worker_summaries.py" \
    --input-dir "${OUT_ROOT}/${policy_name}" \
    --output "${OUT_ROOT}/${policy_name}/summary_all.json"
}

case "${POLICY}" in
  official)
    run_policy official "${OFFICIAL_POLICY}"
    ;;
  stage_a)
    run_policy stage_a "${STAGE_A_POLICY}"
    ;;
  stage_b)
    run_policy stage_b "${STAGE_B_POLICY}"
    ;;
  custom)
    if [[ -z "${CUSTOM_POLICY_PATH}" ]]; then
      echo "POLICY=custom requires CUSTOM_POLICY_PATH" >&2
      exit 2
    fi
    run_policy "${CUSTOM_POLICY_NAME}" "${CUSTOM_POLICY_PATH}"
    ;;
  *)
    echo "Unknown POLICY=${POLICY}; expected one of: official, stage_a, stage_b, custom" >&2
    exit 2
    ;;
esac

echo "All LIBERO suite demos are under: ${OUT_ROOT}"
