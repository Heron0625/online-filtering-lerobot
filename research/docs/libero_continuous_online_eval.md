# LIBERO 连续在线观测评测

从本次修改开始，默认评测协议为 `continuous_online_v1`。旧的单次切换结果不能直接作为这个协议下的结果。

每次重规划固定 slow 观测和新 chunk 的动作时间原点。旧 chunk 继续执行；在每个去噪步查询之前，推进仿真到该 flow 步对应的控制时刻，采集实际 image/proprio，编码 packet，提交到 generation 对应的 fast buffer。模型接收最新就绪 packet 后：Stage A 更新 fast condition 并硬固定已执行前缀；Stage B 先调用 updater 修正当前 particle，再继续在新 condition 下去噪。一次去噪中可以接收多次 packet。

新观测的两张参考图像始终是 slow 原点图像与当前真实图像，与训练的 fast 编码一致。控制步数 d 始终相对于当前 slow 原点，而不是相对于上一次 packet。已执行前缀来自此期间实际发出的旧 chunk 动作，使用 policy-normalized 坐标。

## 时钟及公平性

本 runner 使用确定性的仿真控制/flow 时钟，不是墙钟异步性能测试。总 delay 每次重规划可随机采样；第 k 个去噪步（共 N 步）之前，旧 chunk 已执行 `floor(k * delay / (N - 1))` 个控制步。这样最后一个去噪步之前可收到 delay 时刻的观测，且不需要先推进到未来再倒装观测。

默认 `--online-observation-interval=1`，每个执行控制步都采集。多个 packet 在同一 flow 查询前就绪时，latest-ready buffer 只接收最新的一个；所有采集和实际融合分别写入 events。因此 delay=1 本身只会产生一次新控制步观测；delay=8、N=10 可发生八次融合。不能为了增加融合次数凭空重复或者生成未来观测。

Official 使用相同的旧 chunk 执行步数、重规划间隔及随机 delay，但不接收 fast condition。Stage A 不调用未训练的 updater。Stage B 不再调用单次切换 helper。旧的 switch-step 选项不再控制融合时刻；summary 中对应范围为空，以连续 events 为准。

`mean_inference_s` 排除仿真推进、观测预处理和 MP4 渲染耗时，包含 fast packet 编码、updater 和去噪。`mean_simulated_pipeline_s` 包含整个仿真管线耗时。两者都不会反向决定合成 delay。

## 资源与启动

资源恢复：

```bash
/home/zhoukai/miniforge3/envs/lerobot/bin/python research/scripts/restore_libero_resources.py
```

该脚本仅在自己的下载进程中清除不可用 SOCKS 代理：HF 模型/数据通过 hf-mirror.com；DINO 和 PaliGemma tokenizer 通过 ModelScope。LIBERO 数据和 official 权重固定原来的 revision。下载可断点续传。

- 数据：`/data/zhoukai/dataset/libero`
- assets：`/data/zhoukai/dataset/libero-assets`
- official：`/data/zhoukai/pretrain/pi05_libero_finetuned`
- DINO：`/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m`
- tokenizer：`/data/zhoukai/pretrain/paligemma-tokenizer`

assets 路径仅通过评测进程内 cache 指定，不修改共享 LIBERO 安装或其他用户配置。

```bash
CUDA_DEVICE=1 WORKERS=1 POLICY=stage_b bash research/scripts/run_libero_suite_demos.sh
```

默认输出到 `/data/zhoukai/robotmine/eval/libero_continuous_online_clean_causal`，避免混入旧协议输出。全量仍是四个 suite、每个 task 50 个初始状态。MP4 保留 success/failure 文件名及两行时间线注释；events 中的 `online_observation` 和 `online_assimilation` 分别记录采集与融合。

`--max-control-steps` 只供快速 smoke，episode 会标记 `smoke_test_capped=true`，不能作为完整成功率数据。

## 本次范围

本次只修复评测与资源路径；不修改 Stage B 的 teacher 目标或训练超参数。训练目前只监督单次 old/new teacher pair，连续 updater 组合的有效性仍需单独验证。
