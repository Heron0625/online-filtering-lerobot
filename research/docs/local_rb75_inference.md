# 本机 RB75 推理接口

版本化控制入口：本仓库的 `scripts/infer/infer.sh`，配置文件在同目录的 `config.yaml`。依赖的采集、回放模块也一并纳入本仓库的 `scripts/collect`、`scripts/reply`。原位置 `/home/rpp/project_my/lerobot/scripts/infer/infer.sh` 的本机副本仍可使用；模型源码由 `runtime.policy_repo` 指向本仓库，Python 由 `runtime.model_python` 指向大盘 `nas_data/conda/envs/lerobot/bin/python`。配置中的设备地址、ROS 路径和数据路径为本机设置，在其他机器使用时需修改。

本仓库当前分支 `feature/clean-online-filtering-pi05-xvla`，跟踪同名 origin 分支。worker 依据原生 checkpoint 的 config.type 加载 `pi05` 或 `online_filtering_pi05`，使用本分支的 prefix-KV 缓存实现。不调用可能回退随机权重的转换 loader；safetensors 严格检查完整原生权重，失败直接报错。

## 权重与输入

- `model.checkpoint`：完整 RB75 `pretrained_model` 目录，包括配置、模型、前后处理及其统计文件。
- 四相机 head/chest/right/left，RGB 640×480；state/action 16 维，顺序为左7关节、左夹爪、右7关节、右夹爪。
- 模型内 resize 至配置尺寸；语言 tokenize、归一化、相对动作恢复使用训练导出的 processors。
- `model.dtype` 是字符串 `bfloat16` 或 `float32`。
- `model.tokenizer_name` 可覆盖训练服务器的 tokenizer 路径。
- online-filtering 的 `model.dinov3_config` 可覆盖为本地 DINOv3 配置目录，必须与训练架构一致。DINO 权重从完整原生 checkpoint 加载，不另行下载。未覆盖时沿用 checkpoint.dinov3_model_id，仅读取本地缓存。

两相机、7维动作的 LIBERO checkpoint 不匹配本机 RB75 接口。没有真实 RB75 权重时，不把随机模型接入控制入口。

## 离线检查

```bash
bash scripts/infer/infer.sh scripts/infer/config.yaml --check-config
```

准备完整 checkpoint、tokenizer 和 DINO 配置后，`--check` 会运行一次零图像/零状态推理，不连接机器人。当前默认 RB75 checkpoint 路径尚不存在，真实权重加载需准备模型后再验证。

```bash
PYTHONPATH=src uv run --no-project --python /media/rpp/2955b89d-64af-4f7f-bd14-ae4d766151b2/home/robuster/wsvla/nas_data/conda/envs/lerobot/bin/python python scripts/infer/test_infer.py -v
```

当前12项测试覆盖分块输出、类型分派、严格权重加载与失败路径、颜色/维度、IPC、取消和控制检查；模型构造分派使用 mocks，不冒充真实 checkpoint 试算。

## 执行方式

控制端仍是分块停等：读取观测→预测→执行前 execute_steps 个动作→再次读取。online-filtering 每块使用初始 fast 观测，当前 IPC 不传递块内新观测、capture_step 或 executed_prefix，尚不是 research 异步连续在线更新执行器。XVLA 未接入本机接口。

`control.enabled` 保持 false；本次接口对齐未启动驱动或连接机械臂。完整模型随机权重的纯计算延时见 `research/results/local_latency_clean/README.md`。
