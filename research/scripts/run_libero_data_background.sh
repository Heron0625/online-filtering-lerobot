#!/usr/bin/env bash
set -euo pipefail
cd /data/yhr/project/robot/online_filtering
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBERO_CONFIG_PATH="$PWD/research/config/libero_runtime"
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
exec nice -n 15 ionice -c 2 -n 7 /data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python \
 research/scripts/libero_data_background.py "$@" \
 --manifest /data/yhr/pretrain/xvla_official/libero_download_tree.json \
 --raw /data/yhr/pretrain/xvla_official/libero_hdf5_raw
