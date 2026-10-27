# 本机离线推理延时（2026-10-27）

RTX 3060 12GB；随机初始化完整模型；batch=1；四路 640×480 RGB，模型缩放到 224×224；state/action=16；语言固定 200 tokens（100 valid）；50-step chunk；10 次去噪；BF16 配置（仓库保留视觉路径等部分为 FP32）；不启用 compile。

每项预热 3 次、测量 20 次，perf_counter 计时前后同步 CUDA。输入已在 GPU；包含模型内图像预处理，不包含模型加载、文本 tokenize、相机、CPU→GPU 传输、IPC、动作反归一化或机械臂控制。输出检查为 finite。仅代表当前代码路径的计算延时，不代表训练权重精度或部署端到端性能。

| Model | Measurement | Mean ms | P50 ms | P95 ms | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| pi05 | chunk_no_new_observations | 641.74 | 641.53 | 645.63 | 8.93 |
| online_flow | chunk_no_new_observations | 3355.75 | 3356.02 | 3385.66 | 9.12 |
| online_flow | fast_observation_encoding | 13.31 | 13.14 | 14.33 | 8.93 |
| online_flow | chunk_one_synchronous_new_observation | 3528.06 | 3549.06 | 3614.99 | 9.12 |

online_flow 使用 training_stage=online_flow、64 spatial fast tokens、DINOv3 ViT-S/16（4 register tokens）。无新观测的 chunk 包含初始 fast 编码；单次 fast 编码包含四相机 DINO 和 resampler；一次更新的 chunk 在第 5 次去噪前同步编码并加入 capture_step=1 的模拟观测，不是异步部署基准，也不是 metrics.md 的真实 checkpoint 复现。

当前 modeling_pi05.py 的 denoise_step 在 fast 条件启用时使用联合 prefix/expert 层循环，每次传 past_key_values=None，导致重复语言主干计算；这是当前 online-flow chunk 相对 PI05 慢的主要结构性原因，未对实现做性能优化。

结果原始样本：pi05.json、online_flow.json。脚本：research/scripts/benchmark_local_latency.py。运行时设置 PYTHONPATH=src，使用 nas_data/conda/envs/lerobot 的 Python，经 uv run --no-project 启动。
