# Cached online inference 本机复测

日期：2026-10-27。远端分支 origin/feature/clean-online-filtering-pi05-xvla，提交 `ec5f32a97568577331fa40acdfbaffa67e8c53fe`。代码在相邻独立 worktree online-filtering-latency-clean，原工作区未切换、未合并。通过 PYTHONPATH 指向该 worktree/src；已验证 modeling_pi05.__file__ 与缓存辅助函数存在。

与上次相同脚本和环境：RTX3060 12GB，随机权重完整模型，四路 640×480 输入，模型 224×224，state/action=16，语言固定 200 tokens（100 valid），50-step chunk，10 次去噪，BF16 配置（部分模块为 FP32），不 compile。online_flow 使用64 spatial fast tokens，DINOv3 ViT-S/16。每项3次预热、20次计时，GPU同步，输出finite检查通过。

| Model | Measurement | Mean ms | P95 ms | Peak allocated GiB |
|---|---|---:|---:|---:|
| pi05 | chunk_no_new_observations | 645.73 | 650.87 | 8.93 |
| online_flow | chunk_no_new_observations | 732.83 | 735.78 | 9.10 |
| online_flow | fast_observation_encoding | 12.84 | 13.45 | 8.93 |
| online_flow | chunk_one_synchronous_new_observation | 754.57 | 756.47 | 9.10 |

旧版 online_flow 无更新3355.75ms，新版732.83ms：4.58倍加速，耗时降低78.2%。同步一次新观测旧版3528.06ms，新版754.57ms。fast编码旧版13.31ms，新版12.84ms。

测量含模型内图像预处理，输入已在GPU；不含加载、tokenizer、相机采集、CPU→GPU传输、IPC、动作反归一化和机械臂控制。新观测测试为中途一次同步编码与assimilation，不是异步端到端部署测试。随机权重仅用于计算延时，不评估模型精度。

复现命令（在原仓库根目录）：

```bash
PYTHONPATH=../online-filtering-latency-clean/src uv run --no-project --python /media/rpp/2955b89d-64af-4f7f-bd14-ae4d766151b2/home/robuster/wsvla/nas_data/conda/envs/lerobot/bin/python python research/scripts/benchmark_local_latency.py --model online_flow --output research/results/local_latency_clean/online_flow.json
```

pi05基线将 --model 改为 pi05，并更换 output。
