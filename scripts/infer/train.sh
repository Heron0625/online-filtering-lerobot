#!/usr/bin/env bash
# Run on the CLOUD server in its configured LeRobot environment.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath -- "${1:-$SCRIPT_DIR/config.yaml}")"
ARGS=(--config "$CONFIG")
case "${2:-}" in '') ;; --print) ARGS+=(--print);; *) echo "未知参数：$2"; exit 1;; esac
export PYTHONPATH="$(realpath -- "$SCRIPT_DIR/../../src")"
# Cloud Python comes from the activated environment; no local machine path is assumed.
TRAIN_INTERPRETER="${TRAIN_PYTHON:-$(command -v python3)}"
uv run --no-project --python "$TRAIN_INTERPRETER" python "$SCRIPT_DIR/train.py" "${ARGS[@]}"
