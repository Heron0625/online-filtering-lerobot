# RB75 双臂 π0.5 / online-filtering 本地推理

本目录将 `runtime.policy_repo` 指定源码中的 `pi05` 或 `online_filtering_pi05` 接到 `collect` 的四相机、16 维状态及 `reply` 使用的 ROS2 从臂控制链路。当前指向大盘的 online-filtering-lerobot（feature/clean-online-filtering-pi05-xvla 分支，带 prefix KV cache 优化）。使用原生 LeRobot checkpoint，不兼容直接放入 OpenPI/JAX 权重。

worker 根据 checkpoint 的 `config.json.type` 自动选择模型，严格加载所有权重；缺失、不匹配或损坏均报错，不会退回随机权重。`dtype` 使用配置字符串 `bfloat16/float32`，与当前模型接口一致。

online-filtering checkpoint 还需要本机 DINOv3 架构配置：将 `model.dinov3_config` 指向含 `config.json` 的本地 DINOv3 目录（必须与训练 backbone 配置一致）；完整原生 checkpoint 已包含 DINO 权重，不需要重复下载权重。若训练 tokenizer 路径在本机不存在，将 `model.tokenizer_name` 指向本地 tokenizer 目录。两项默认 null，沿用 checkpoint/processor 中的路径。

当前 online-filtering 走相同的分块停等 IPC：每块处理初始 fast observation，不在块内注入新观测。它使用优化后的模型计算接口，但不等同于 research 中连续异步在线更新的执行器。XVLA 未接入本 RB75 接口。

完整流程：本地采集并回放验证 → 将作业中的 `lerobot/` 数据集上传云端 → 云端微调 → 将完整的 `pretrained_model/` 导出目录下载本机 → 检查模型 → 只预测验证 → 真机执行。

目前实现的是分块停等推理：读取新观测，计算一块动作，按训练数据的频率执行前 `execute_steps` 步，然后停住、重新观察和预测。计算期间不继续执行旧块；按 f 的检测独立于模型进程。块间会有模型计算造成的停顿，这不是 RTC 连续异步控制，也不保证模型每秒预测 50 次。当前 `action_hz: 50` 表示每个动作步对应 20 ms；`command_hz: 50` 表示控制命令频率。必须与训练数据 FPS 一致，不改变现有驱动的真实反馈频率。

## 本机配置

修改同目录 `config.yaml`：

- `model.checkpoint`：云端导出的本地 `pretrained_model/` 绝对路径。默认在剩余空间较大的 `/media/rpp/.../home/robuster/wsvla/models/` 下。
- `model.task`：具体任务指令，与训练数据 task 保持一致；默认复制了采集的任务描述，正式采集和训练前需明确实际任务。
- `model.device/dtype`：默认 `cuda/bfloat16`。本机 RTX 3060 12 GB 的显存和实际速度需要用训练权重测量，无法仅凭接口代码保证能装下四相机模型。显存不足会终止模型进程，控制端停止后续动作。
- `model.execute_steps`：每次执行预测块的前多少步，默认 10，在 50Hz 下为 0.2 秒。必须不超过模型 chunk_size。
- `control.enabled`：默认 **false**，只预测、不发关节和夹爪命令。确认模型后改 true 才执行真机动作。
- `control.max_duration_s`：单轮最长执行时间，默认 60 秒；按 f 可提前结束。
- `streams/cameras`：与采集同样的单位、顺序和 ROS 话题。四路 camera key 为 head/chest/right/left。
- `robot/arms/runtime`：沿用回放的 IP、驱动和话题；真机模式启动缺失的桥接、从臂和夹爪驱动，复用已存在的匹配实例。**相机沿用已经运行的数采相机驱动，不在这里启动。**不要同时运行遥操作、回放或另一个推理控制器。
- `runtime.model_python`：LeRobot/PyTorch 环境；`runtime.ros_python`：系统 ROS Python。两者通过权限受限临时目录中的 Unix socket 传递数组，不要求把 ROS 安装进模型 Python。
- `runtime.policy_repo`：模型源码仓库绝对路径；启动脚本据此设置模型进程的 PYTHONPATH，并打印实际路径，不再强制使用 scripts/infer 所在仓库的旧模型代码。
- `runtime.cache_dir`：模型/tokenizer 缓存放到大分区，避免占满 `/home`。初次加载 tokenizer 可能需要访问 Hugging Face；离线使用前先在该缓存下完成一次模型检查。

先只检查 YAML，不加载模型，不连接机器人：

```bash
bash scripts/infer/infer.sh scripts/infer/config.yaml --check-config
```

下载模型后，离线于机器人做一次真实模型试算（零图像/零状态仅用于软件检查，其结果不会发给机器人）：

```bash
bash scripts/infer/infer.sh scripts/infer/config.yaml --check
```

输入驱动和四相机已启动时，只预测：

```bash
bash scripts/infer/infer.sh scripts/infer/config.yaml --dry-run
```

在 YAML 设置 `control.enabled: true` 后执行真机推理：

```bash
bash scripts/infer/infer.sh
```

也可以在本目录中直接 `bash infer.sh`。终端打印模型日志路径；模型就绪后提示按 → 开始、按 f 结束。开始前需要把机器人放在训练任务的合理起始状态；推理没有已知 episode 第一帧，**不会自动对齐到模型的第一个预测姿态**。等待模型、计算动作和执行期间均检测 f；Ctrl+C 也退出。结束后重新运行脚本可以开始下一轮。

只预测模式不会自动启动驱动，不会发送动作；仍然需要真实四相机和双臂状态，不能仅靠空输入启动。真机模式自动启动夹爪驱动时，其自身可能进行初始化。

## 数据与 checkpoint 约定

状态和动作均为：左 7 关节、左夹爪、右 7 关节、右夹爪。模型输入/输出关节单位为弧度，控制发布转换成度。状态夹爪来自位置反馈，动作夹爪来自采集的命令值，二者不会混用；图像按 RGB、CHW、float32 `[0,1]` 输入，与 LeRobot 训练接口一致。

从训练服务器下载整个 `checkpoints/last/pretrained_model/` 的内容，需包含：

- `config.json`、`model.safetensors`。
- `policy_preprocessor.json`、`policy_postprocessor.json`。
- 两个 processor 配置引用的所有 `.safetensors` 等状态文件（归一化统计等）。

不要只复制一个权重文件，也不要把基础模型原封不动用于 RB75 控制。接口严格要求 state/action 为 16 维，以及四路图像键/尺寸匹配。LIBERO 的两相机、7 维动作权重不能直接用于本接口。预处理和反归一化使用训练导出的 pipeline；相对动作启用时使用同一次观测作为绝对动作恢复的基准。支持标准单帧 PI05 和 online_filtering_pi05，启用 MEM 或 RTC 的模型会明确拒绝。

模型输出整块检查后才开始执行：维度/数值、夹爪范围、首动作偏移、关节速度；执行中检查反馈有效期、跟踪偏差、输入新鲜度和控制冲突。异常动作拒绝执行，不截断成“看似有效”的动作。默认速度上限较低，若触发应先核对单位、任务和模型表现。

## 云端微调入口

云端使用与本机相同版本的仓库代码，安装训练及 PI05 依赖。例如在已准备的 Python 3.12+ 环境中：

```bash
uv pip install -e '.[training,pi]'
```

初次使用 PaliGemma tokenizer 按仓库 `docs/source/pi05.mdx` 完成 Hugging Face 访问授权。不要在本机小容量 `/home` 下下载基础模型；云端缓存位置按服务器空间设置。

将采集作业里的完整 `lerobot/` 上传云端。多个作业先合并为一个 LeRobot 数据集，不能直接把多个作业父目录当作 dataset.root。修改云端配置的 `training.dataset_root`、`repo_id`、`output_dir`、批大小和步数；`dataset_root` 指向含 `meta/`、`data/`、`videos/` 的那一级。

先打印命令，再开始训练（都在云端已激活的 LeRobot 环境中）：

```bash
bash scripts/infer/train.sh scripts/infer/config.yaml --print
bash scripts/infer/train.sh scripts/infer/config.yaml
```

训练入口检查数据集 FPS 与 control.action_hz 一致（当前目标 50），以及 16 维与特征顺序，默认以 `lerobot/pi05_base` 微调 action expert，使用 MEAN_STD 归一化，四相机键保持采集原名；统计随 processor 保存。初始 batch_size=4、steps=30000 只是可调整的起始配置，实际显存需求和收敛情况取决于云端 GPU 与数据。训练时间、效果和真实模型加载尚未在本次任务中验证。

## 已验证范围

本目录的模拟测试不会加载大模型或连接机器人：

```bash
PYTHONPATH=src uv run --no-project \
  --python /home/rpp/miniconda3/envs/lerobot/bin/python \
  python scripts/infer/test_infer.py -v
```

检查输入颜色/维度、动作单位与速度、超限整块拒绝、控制时序、只预测不发送、f 取消、IPC 分包和训练参数。接口复用 `../reply/reply.py` 和 `../collect/record.py`，应随仓库保留这两个目录。目前没有已训练的 RB75 PI05 权重，真实模型试算、显存占用、云端训练和真机闭环仍需在权重准备后验证。
