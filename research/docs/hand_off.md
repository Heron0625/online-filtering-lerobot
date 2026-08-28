# Online Filtering PI0.5 训练迁移手册

本文用于把当前 `online_filtering_pi05` 实现迁移到另一台机器，并依次完成 Stage A、Stage B 训练。

## 1. 必须迁移的内容

不要只在新机器 clone 官方 LeRobot。本实现包含尚未进入上游的新增和修改文件，必须迁移当前完整工作树。

至少需要：

```text
lerobot/                                  当前完整代码仓库
pi05_base/                               PI0.5 Base 权重，约 14 GB
dinov3-small/ 或 dinov3-base/            DINOv3 权重
libero/                                  LeRobot 格式 LIBERO 数据集
```

当前机器上的来源为：

```text
/data4/yanghuoren/project/lerobot
/data4/yanghuoren/pretrain/pi05_base
/data4/yanghuoren/pretrain/dinov3-small
/data4/yanghuoren/pretrain/dinov3-base
/data4/yanghuoren/dataset/libero
```

建议先在当前机器提交到私有分支，或者用 `rsync` 迁移完整目录。由于当前实现包含 untracked 文件，单独迁移 `git diff` 不够。

示例：

```bash
rsync -a --info=progress2 \
  /data4/yanghuoren/project/lerobot/ \
  USER@TRAIN_HOST:/path/to/lerobot/

rsync -a --info=progress2 \
  /data4/yanghuoren/pretrain/pi05_base/ \
  USER@TRAIN_HOST:/path/to/pretrain/pi05_base/

rsync -a --info=progress2 \
  /data4/yanghuoren/pretrain/dinov3-small/ \
  USER@TRAIN_HOST:/path/to/pretrain/dinov3-small/

rsync -a --info=progress2 \
  /data4/yanghuoren/dataset/libero/ \
  USER@TRAIN_HOST:/path/to/dataset/libero/
```

## 2. 新机器目录约定

后续命令假定：

```bash
export LEROBOT_ROOT=/path/to/lerobot
export PI05_BASE=/path/to/pretrain/pi05_base
export DINOV3_PATH=/path/to/pretrain/dinov3-small
export LIBERO_ROOT=/path/to/dataset/libero
export TRAIN_ROOT=/path/to/outputs/online_filtering_pi05
```

所有命令都从仓库根目录执行：

```bash
cd "$LEROBOT_ROOT"
```

## 3. 环境安装

要求：

- Python 3.12；
- PyTorch 能识别目标 NVIDIA GPU；
- ffmpeg/PyAV 能解码 LIBERO 视频；
- 推荐 80 GB GPU；显存不足时降低 batch size 并使用梯度累积；
- Stage B 的两条 teacher rollout 比 Stage A 明显更慢。

优先按仓库锁文件安装：

```bash
uv sync --locked --extra test --extra dev
```

若使用 conda，先建立 Python 3.12 环境，再在该环境中执行 `uv sync`。本文命令可将 `uv run` 替换为对应环境中的可执行程序。

检查 GPU 和导入：

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
uv run python -c "from lerobot.policies.factory import get_policy_class; print(get_policy_class('online_filtering_pi05'))"
```

如果 tokenizer 没有包含在本地 checkpoint/cache 中，PI0.5 仍可能访问 `google/paligemma-3b-pt-224`。此时需要先接受 Hugging Face gated model 协议并登录：

```bash
hf auth login
```

## 4. 迁移后完整性检查

### 4.1 权重和数据

```bash
test -f "$PI05_BASE/model.safetensors"
test -f "$PI05_BASE/config.json"
test -f "$DINOV3_PATH/config.json"
test -f "$LIBERO_ROOT/meta/info.json"
test -f "$LIBERO_ROOT/meta/stats.json"
```

LIBERO 应包含 457 个仓库文件；文件数可能因传输工具生成的临时文件略有变化：

```bash
find "$LIBERO_ROOT" -type f | wc -l
du -sh "$PI05_BASE" "$DINOV3_PATH" "$LIBERO_ROOT"
```

### 4.2 静态检查和 CPU 测试

```bash
uv run ruff check \
  src/lerobot/policies/pi05/online_filtering.py \
  src/lerobot/policies/pi05/modeling_pi05.py \
  src/lerobot/policies/online_filtering_pi05 \
  tests/policies/pi0_pi05/test_online_filtering_components.py \
  tests/policies/pi0_pi05/test_online_filtering_pi05_stage_a.py

PYTHONPATH=src uv run pytest -q \
  tests/policies/pi0_pi05/test_online_filtering_components.py \
  tests/policies/pi0_pi05/test_online_filtering_pi05_stage_a.py
```

当前基线为：

```text
40 passed
```

退出时可能出现 `multiprocess.resource_tracker` 的清理警告；只要 pytest 返回码为 0，就不影响测试结论。

### 4.3 真实资产 smoke test

正式训练前必须运行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python \
  research/scripts/smoke_test_online_filtering_pi05.py \
  --dataset-root "$LIBERO_ROOT" \
  --pi05-path "$PI05_BASE" \
  --dinov3-path "$DINOV3_PATH" \
  --device cuda \
  --max-delay 2 \
  --teacher-steps 1
```

成功标准：

- Stage A loss 为有限值；
- fast adapters 获得梯度；
- Stage B loss 为有限且非零值；
- posterior updater 获得梯度；
- frozen teacher gradient tensors 为 0。

当前机器最近一次结果：

```text
Stage A loss:                       1.3947141171
Stage A adapter gradient tensors:  84
Stage B loss:                       5.6765731e-07
Stage B updater gradient tensors:  37
Frozen teacher gradient tensors:   0
```

## 5. 数据窗口约定

`max_inference_delay=D_max` 会让 dataloader 为每个样本读取：

```text
image/state: [0, 1, ..., D_max]
action:      [0, 1, ..., 49]
```

LIBERO FPS 为 10，因此 `D_max=10` 对应最多约 1 秒的观测窗口。Stage A 和 Stage B 必须使用相同的：

- `max_inference_delay`；
- DINOv3 型号；
- fast token 数、adapter 层和 bridge 结构；
- normalization mapping；
- PI0.5 action chunk 长度。

当前方案使用：

```text
chunk_size=50
max_inference_delay=10
fast_num_tokens=16
fast_adapter_layers=(2,5,8,11,14,17)
posterior_slow_tokens=4
absolute observation.state，包括 gripper
MEAN_STD state/action normalization
```

## 6. Stage A 训练

Stage A 学习：

$$
K_d^t(X_d^{t-1}\mid X_d^t,C_d).
$$

训练完整 action chunk 的 PI0.5 flow loss，同时训练 action expert、fast resampler 和 fast adapters。slow VLM 与 DINOv3 默认冻结。

先用较小 batch 验证显存。单张 80 GB GPU 的保守起始命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run lerobot-train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$LIBERO_ROOT" \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$PI05_BASE" \
  --policy.dinov3_model_id="$DINOV3_PATH" \
  --policy.training_stage=stage_a \
  --policy.max_inference_delay=10 \
  --policy.freeze_dinov3=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$TRAIN_ROOT/stage_a" \
  --job_name=online_filtering_pi05_stage_a \
  --batch_size=4 \
  --accelerator.gradient_accumulation.steps=8 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=50 \
  --env_eval_freq=0 \
  --seed=1000
```

有效 batch size 是 `batch_size × gradient_accumulation.steps × GPU 数量`。首次运行建议先改成：

```text
steps=20
save_freq=10
num_workers=2
```

确认 loss、显存和 checkpoint 保存均正常后再启动正式任务。

Stage A 预期可训练模块包括：

```text
model.paligemma_with_expert.gemma_expert
model.action_in_proj
model.action_out_proj
model.fast_resampler
model.fast_adapters
```

DINOv3 和 slow PaliGemma/Gemma prefix 应保持冻结。

## 7. Stage B 训练

Stage B 学习 shared-noise bridge：

$$
B_d^t(X_d^t\mid X_r^t,C_d),
\qquad
Q_d^t=K_d^t\circ B_d^t.
$$

必须从 Stage A checkpoint 初始化，不能重新从 PI0.5 Base 开始。设 Stage A checkpoint 为：

```bash
export STAGE_A_CKPT="$TRAIN_ROOT/stage_a/checkpoints/030000/pretrained_model"
```

实际 step 目录名称以磁盘为准：

```bash
find "$TRAIN_ROOT/stage_a/checkpoints" -maxdepth 2 -type f -name model.safetensors
```

Stage B 会冻结完整 Stage-A teacher，只训练：

```text
posterior_updater
posterior_fast_proj
posterior_slow_resampler
```

保守起始命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run lerobot-train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$LIBERO_ROOT" \
  --policy.type=online_filtering_pi05 \
  --policy.pretrained_path="$STAGE_A_CKPT" \
  --policy.dinov3_model_id="$DINOV3_PATH" \
  --policy.training_stage=stage_b \
  --policy.max_inference_delay=10 \
  --policy.teacher_integration_steps=10 \
  --policy.stage_b_min_gap=1 \
  --policy.stage_b_max_gap=10 \
  --policy.posterior_identity_dropout=0.1 \
  --policy.freeze_dinov3=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --output_dir="$TRAIN_ROOT/stage_b" \
  --job_name=online_filtering_pi05_stage_b \
  --batch_size=1 \
  --accelerator.gradient_accumulation.steps=16 \
  --num_workers=4 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=20 \
  --env_eval_freq=0 \
  --seed=1000
```

`teacher_integration_steps=10` 表示每个 batch 为 old/new filtration 分别运行 10 步 frozen teacher，计算成本较高。调试时可以设为 1，但正式训练应使用与计划部署接近的积分精度，并做 5/10/20 steps 消融。

Stage B 的关键监控项：

```text
loss
posterior_endpoint_loss
mean_old_delay
mean_new_delay
mean_gap
identity_fraction
```

`mean_gap` 应覆盖 1 到 10，而不是永久等于 1。`identity_fraction` 长期均值应接近 0.1。

## 8. 恢复训练

恢复同一阶段时，指向 checkpoint 根目录或其中的 `train_config.json`：

```bash
uv run lerobot-train \
  --config_path="$TRAIN_ROOT/stage_a/checkpoints/030000/pretrained_model/train_config.json" \
  --resume=true
```

若实际 `train_config.json` 位于 checkpoint 的其他层级，使用：

```bash
find "$TRAIN_ROOT/stage_a/checkpoints" -name train_config.json
```

Stage A 切换到 Stage B 不属于 resume。它是一个新训练任务，应使用 `policy.pretrained_path=<Stage-A pretrained_model>` 和新的 output directory。

## 9. 多 GPU

先确认单 GPU 的 20-step 训练通过，再启用多 GPU。仓库训练器由 Accelerate 管理；使用新机器现有的 accelerate/torchrun 启动方式，并保持上述模型参数不变。

不要一开始启用 FSDP、`torch.compile` 和多 worker 视频解码。建议按以下顺序增加复杂度：

1. 单 GPU、`num_workers=2`、20 steps；
2. 单 GPU 调整 batch 与 gradient accumulation；
3. 多 GPU DDP；
4. 确有显存需要时再启用 FSDP；
5. 稳定后再测试 `policy.compile_model=true`。

Stage B 中 teacher 是冻结的，但仍需要保存 activation/执行两条 rollout；多 GPU 时先观察每卡显存和 step time。

## 10. Checkpoint 验收

每个正式 checkpoint 至少应包含：

```text
pretrained_model/config.json
pretrained_model/model.safetensors
pretrained_model/policy_preprocessor.json
pretrained_model/policy_postprocessor.json
pretrained_model/train_config.json
training_state/
```

Stage A checkpoint 中应存在：

```text
model.fast_resampler.*
model.fast_adapters.*
```

Stage B checkpoint 还应存在：

```text
posterior_updater.*
posterior_fast_proj.*
posterior_slow_resampler.*
```

可以快速检查：

```bash
uv run python - <<'PY'
from safetensors.torch import load_file
import os

path = os.environ["STAGE_A_CKPT"] + "/model.safetensors"
keys = load_file(path).keys()
for prefix in ("model.fast_resampler.", "model.fast_adapters."):
    print(prefix, sum(key.startswith(prefix) for key in keys))
PY
```

## 11. 常见问题

### CUDA OOM

依次降低 `batch_size`、增大梯度累积、保持 gradient checkpointing、降低 dataloader workers。不要通过缩短 `max_inference_delay` 来临时解决正式训练 OOM，因为它会改变模型要覆盖的物理时延范围。

### Stage B loss 恒为 0

最常见原因是错误地从原始 PI0.5 Base 初始化 Stage B，或者 Stage A fast adapters 没有训练成功。确认 `policy.pretrained_path` 指向 Stage A checkpoint，并检查其中存在非初始的 `model.fast_adapters.*` 权重。

### 数据只返回单帧

确认 policy type 是 `online_filtering_pi05`，并且 `max_inference_delay>0`。该 config 的 modality-specific delta indices 才会让 dataloader 读取 `[0,...,D_max]` 图像和 state。

### 相机数或 feature key 不匹配

不要硬编码新机器上的相机 key。先查看：

```bash
uv run python - <<'PY'
import json, os
with open(os.environ["LIBERO_ROOT"] + "/meta/info.json") as f:
    info = json.load(f)
print(info["features"].keys())
PY
```

当前 LIBERO smoke 使用 `observation.images.image` 和 `observation.images.image2`。

### tokenizer 访问失败

接受 `google/paligemma-3b-pt-224` 协议并执行 `hf auth login`，或者把对应 tokenizer cache 一并迁移。

### 视频解码错误

确认 ffmpeg/PyAV 可用，并先运行真实资产 smoke。多 worker 下才失败时，将 `num_workers` 降为 0 或 2 定位问题。

## 12. 训练完成后的模型含义

最终部署使用 Stage B checkpoint。单个 flow step 的计算为：

$$
X_r^t
\xrightarrow{B_{r\rightarrow d}^t}
X_d^t
\xrightarrow{K_d^t}
X_d^{t-1}.
$$

在线观测采用容量 4 的 latest-ready buffer；过旧 packet 丢弃；DINOv3 不阻塞 action expert；推理完成后按实测 $D$ 丢弃新 chunk 的过期前缀，并对尚未执行的动作使用 $R=4$ 线性 merge。

理论与实现说明见：

- `research/docs/online_filtering_flow_v2.md`
- `research/scripts/smoke_test_online_filtering_pi05.py`

`online_filtering_flow_v2.md` 是当前符号和方案的权威版本；旧 deployment 文档包含早期设计记录，不应用作训练实现依据。

## 13. 最小执行清单

```text
[ ] 迁移完整工作树，而不是只 clone 上游仓库
[ ] 校验 PI0.5、DINOv3、LIBERO 文件
[ ] 安装 Python 3.12/PyTorch/LeRobot 环境
[ ] policy 注册导入成功
[ ] 40 个 CPU 测试通过
[ ] 真实资产 smoke 的 Stage A/Stage B 均通过
[ ] Stage A 先跑 20 steps
[ ] Stage A 正式训练并保存 checkpoint
[ ] Stage B 从 Stage A checkpoint 启动，不使用 resume
[ ] 检查 Stage B 只有 posterior modules 获得梯度
[ ] 保存最终 Stage B checkpoint 和完整 train_config
```
