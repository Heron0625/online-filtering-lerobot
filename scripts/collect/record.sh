#!/usr/bin/env bash
# Usage: bash scripts/collect/record.sh [config.yaml] [--check | --convert SESSION]
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(realpath -- "$CONFIG")"
# 初始仅依赖系统 PyYAML；ROS overlay 必须由兼容 Python 3.10 的解释器加载。
readarray -t SETTINGS < <(/usr/bin/python3 - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)['runtime']
for k in ('ros_setup', 'workspace_setup', 'collector_python', 'converter_python'):
    print(c[k])
PY
)
[[ ${#SETTINGS[@]} == 4 ]] || { echo '无法读取 YAML runtime 配置'; exit 1; }
if [[ "${2:-}" == --convert ]]; then
    [[ -n "${3:-}" ]] || { echo '--convert 需要作业目录参数'; exit 1; }
    export PYTHONPATH="$(realpath -- "$SCRIPT_DIR/../../src")"
    exec uv run --no-project --python "${SETTINGS[3]}" python "$SCRIPT_DIR/record.py" convert --session "$3"
fi
[[ -z "${2:-}" || "${2:-}" == --check ]] || { echo "未知参数：$2"; exit 1; }
source "${SETTINGS[0]}"
source "${SETTINGS[1]}"
if [[ "${2:-}" == --check ]]; then
    exec "${SETTINGS[2]}" "$SCRIPT_DIR/record.py" check --config "$CONFIG"
fi
# ROS Python 环境采集；脚本结束时再用 uv 启动独立的 LeRobot 转换环境。
exec "${SETTINGS[2]}" "$SCRIPT_DIR/record.py" record --config "$CONFIG"
