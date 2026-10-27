#!/usr/bin/env bash
# bash scripts/reply/reply.sh [config.yaml] [--check]
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath -- "${1:-$SCRIPT_DIR/config.yaml}")"
readarray -t SETTINGS < <(/usr/bin/python3 - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)['runtime']
for key in ('ros_setup', 'workspace_setup', 'ros_python', 'dataset_python'):
    print(cfg[key])
PY
)
[[ ${#SETTINGS[@]} == 4 ]] || { echo '无法读取 YAML runtime 配置'; exit 1; }
[[ -z "${2:-}" || "${2:-}" == --check ]] || { echo "未知参数：$2"; exit 1; }
# 数值预处理独立于 ROS 环境：不解码相机视频，不发送控制命令。
PREPARED="$(mktemp -d -t rb75_reply.XXXXXXXX)"
trap 'rm -rf -- "$PREPARED"' EXIT
PYTHONPATH="$(realpath -- "$SCRIPT_DIR/../../src")" uv run --no-project \
    --python "${SETTINGS[3]}" python "$SCRIPT_DIR/reply.py" prepare --config "$CONFIG" --prepared "$PREPARED"
if [[ "${2:-}" == --check ]]; then
    echo '数据预检通过，未连接或控制机械臂。'
    exit 0
fi
source "${SETTINGS[0]}"
source "${SETTINGS[1]}"
"${SETTINGS[2]}" "$SCRIPT_DIR/reply.py" play --config "$CONFIG" --prepared "$PREPARED"
