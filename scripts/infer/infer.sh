#!/usr/bin/env bash
# bash scripts/infer/infer.sh [config.yaml] [--check-config|--check|--dry-run]
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath -- "${1:-$SCRIPT_DIR/config.yaml}")"
MODE="${2:-}"
case "$MODE" in ''|--check-config|--check|--dry-run) ;; *) echo "未知参数：$MODE"; exit 1;; esac
readarray -t SETTINGS < <(/usr/bin/python3 - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)['runtime']
for key in ('ros_setup', 'workspace_setup', 'ros_python', 'model_python', 'log_dir', 'cache_dir'):
    print(cfg[key])
print(cfg.get('policy_repo', ''))
PY
)
[[ ${#SETTINGS[@]} == 7 ]] || { echo '无法读取 YAML runtime 配置'; exit 1; }
REPO="$(realpath -- "${SETTINGS[6]:-$SCRIPT_DIR/../..}")"
[[ -f "$REPO/src/lerobot/policies/pi05/modeling_pi05.py" ]] || { echo "无效 policy_repo：$REPO"; exit 1; }
export PYTHONPATH="$REPO/src"
echo "模型源码：$REPO"
# Models/tokenizers must not fill the small /home partition.
mkdir -p -- "${SETTINGS[5]}"
export HF_HOME="${SETTINGS[5]}"
export TOKENIZERS_PARALLELISM=false
if [[ "$MODE" == --check-config ]]; then
    uv run --no-project --python "${SETTINGS[3]}" python "$SCRIPT_DIR/infer.py" --config "$CONFIG" --check-config
    exit 0
fi
if [[ "$MODE" == --check ]]; then
    uv run --no-project --python "${SETTINGS[3]}" python "$SCRIPT_DIR/policy_worker.py" --config "$CONFIG" --check
    exit 0
fi
TEMP_DIR="$(mktemp -d -t rb75_infer.XXXXXXXX)"
mkdir -p -- "${SETTINGS[4]}"
LOG_DIR="$(mktemp -d -- "${SETTINGS[4]}/run_XXXXXXXX")"
WORKER_PID=''
cleanup() {
    if [[ -n "$WORKER_PID" ]]; then
        # Isolated process group includes uv and the policy Python child.
        kill -TERM -- "-$WORKER_PID" 2>/dev/null || true
        for ((i=0; i<30; i++)); do
            kill -0 "$WORKER_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL -- "-$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "模型日志：$LOG_DIR/policy_worker.log"
# Start before sourcing ROS: Python 3.12 model and Python 3.10 ROS stay separate.
setsid uv run --no-project --python "${SETTINGS[3]}" python "$SCRIPT_DIR/policy_worker.py" \
    --config "$CONFIG" --socket "$TEMP_DIR/policy.sock" >"$LOG_DIR/policy_worker.log" 2>&1 &
WORKER_PID=$!
source "${SETTINGS[0]}"
source "${SETTINGS[1]}"
ARGS=(--config "$CONFIG" --socket "$TEMP_DIR/policy.sock")
[[ "$MODE" != --dry-run ]] || ARGS+=(--dry-run)
if ! "${SETTINGS[2]}" "$SCRIPT_DIR/infer.py" "${ARGS[@]}"; then
    echo "推理失败，模型日志末尾："
    tail -n 25 -- "$LOG_DIR/policy_worker.log"
    exit 1
fi
