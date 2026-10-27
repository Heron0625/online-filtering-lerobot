#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE=1 POLICY=stage_b WORKERS=1 \
  bash /home/zhoukai/robotmine/online-filtering-lerobot/research/scripts/run_libero_suite_demos.sh
