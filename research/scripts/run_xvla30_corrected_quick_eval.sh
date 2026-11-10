#!/usr/bin/env bash
set -euo pipefail
cd /data/yhr/project/robot/online_filtering
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=3,0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=2
export LIBERO_CONFIG_PATH="$PWD/research/config/libero_runtime"
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
PY=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
OUT=/data/yhr/output/online_filtering/eval/xvla30_corrected_10k_vs_official_d0_r30_quick40
CKPT=/data/yhr/output/online_filtering/ckpt/xvla30_corrected_v2_2gpu_b32_30k_20261030/checkpoints/010000/pretrained_model
groups=(0,4,8 1,5,9 2,6 3,7)
suites=(libero_spatial libero_object libero_goal libero_10)
horizons=(280 280 300 520)
run_worker() {
 local worker=$1 model suite out policy
 for model in corrected10k official30; do
  policy="$CKPT"
  if [[ "$model" == official30 ]]; then policy=/data/yhr/pretrain/online_filtering_baselines/xvla-libero; fi
  for i in "${!suites[@]}"; do
   suite=${suites[$i]}
   out="$OUT/$model/worker_$worker/$suite"
   mkdir -p "$out"
   "$PY" research/scripts/eval_libero_synthetic_delay.py \
    --policy-path "$policy" --output-dir "$out" --suite "$suite" \
    --task-ids "${groups[$worker]}" --episodes 1 --episode-length "${horizons[$i]}" \
    --delay-min 0 --delay-max 0 --replan-interval 30 --control-fps 20 --seed 1000 --device cuda \
    > "$out/worker.log" 2>&1
  done
 done
}
pids=()
for worker in "${!groups[@]}"; do run_worker "$worker" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
