#!/usr/bin/env bash
set -euo pipefail
cd /data/yhr/project/robot/online_filtering
MODE=${1:?Usage: $0 zero40|delay25_400}
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=2
export LIBERO_CONFIG_PATH="$PWD/research/config/libero_runtime"
export LIBERO_ASSETS_PATH=/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets
PY=/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python
CKPT=/data/yhr/output/online_filtering/ckpt/xvla50_corrected_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model
OUT=/data/yhr/output/online_filtering/eval/xvla50_corrected_30k_${MODE}_r20
suites=(libero_spatial libero_object libero_goal libero_10)
horizons=(280 280 300 520)
if [[ "$MODE" == zero40 ]]; then
 delay=0; episodes=1
 groups=(0,4,8 1,5,9 2,6 3,7)
 gpus=(4 5 6 7)
elif [[ "$MODE" == delay25_400 ]]; then
 delay=25; episodes=10
 groups=(); gpus=()
 for worker in {0..39}; do groups+=("$((worker % 10))"); gpus+=("$((3 + worker % 5))"); done
else
 exit 2
fi
run_worker() {
 local worker=$1 suite out
 export CUDA_VISIBLE_DEVICES="${gpus[$worker]},0"
 local indices=(0 1 2 3)
 if [[ "$MODE" == delay25_400 ]]; then indices=("$((worker / 10))"); sleep "$((worker / 5 * 8))"; fi
 for i in "${indices[@]}"; do
  suite=${suites[$i]}; out="$OUT/worker_$worker/$suite"
  mkdir -p "$out"
  "$PY" research/scripts/eval_libero_synthetic_delay.py \
   --policy-path "$CKPT" --output-dir "$out" --suite "$suite" \
   --task-ids "${groups[$worker]}" --episodes "$episodes" --episode-length "${horizons[$i]}" \
   --delay-min "$delay" --delay-max "$delay" --replan-interval 20 --control-fps 20 \
   --allow-queue-underrun --seed 1000 --device cuda > "$out/worker.log" 2>&1
 done
}
pids=()
for worker in "${!groups[@]}"; do run_worker "$worker" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
