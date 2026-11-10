#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---dry-run}"
CONFIG="$(realpath -- "${2:-$SCRIPT_DIR/config.yaml}")"
readarray -t SETTINGS < <(/usr/bin/python3 - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for key in ('model_python', 'ros_python', 'ros_setup', 'workspace_setup', 'policy_repo', 'cache_dir'):
    print(cfg['runtime'][key])
PY
)
[[ ${#SETTINGS[@]} == 6 ]] || { echo 'Invalid configuration'; exit 1; }
export PYTHONPATH="${SETTINGS[4]}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${SETTINGS[5]}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
case "$MODE" in
  --server|--check)
    ARGS=(--config "$CONFIG")
    [[ "$MODE" != --check ]] || ARGS+=(--check)
    exec uv run --no-project --python "${SETTINGS[0]}" python -u "$SCRIPT_DIR/server.py" "${ARGS[@]}"
    ;;
  --check-config)
    exec "${SETTINGS[1]}" "$SCRIPT_DIR/client.py" --config "$CONFIG" --check-config
    ;;
  --dry-run|--execute)
    source "${SETTINGS[2]}"
    source "${SETTINGS[3]}"
    exec "${SETTINGS[1]}" "$SCRIPT_DIR/client.py" --config "$CONFIG" "$MODE"
    ;;
  *) echo 'Usage: bash run.sh --server|--check|--check-config|--dry-run|--execute [config.yaml]'; exit 1 ;;
esac
