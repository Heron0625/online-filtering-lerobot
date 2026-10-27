# Spatial fast Stage A

新版训练由 `research/scripts/run_stage_a_2xa100.sh` 启用，仍以官方 LIBERO finetuned 权重初始化，双卡每卡 batch 16、30k steps、BF16、gradient checkpointing，无梯度累积。输出默认改为 `/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast`，不会覆盖 clean_causal 结果。单卡 Stage B 与 A→B 脚本使用同一新根目录。

## Fast

- `fast_spatial_tokens=true`，`fast_num_tokens=64` 表示**视觉** token 总数。两路相机各 32 个 query，只读取自身相机 patch，不能竞争另一相机的配额。
- 在 DINO 的 initial/current/difference patch 特征上加入显式归一化二维位置和相机 embedding。
- 当前 proprio 经独立 MLP 产生 1 个 token，绕过视觉 resampling；加 modality 和绝对相对控制步 d embedding。因此 fast 输出总长度为 **65**。
- 视觉 query 同样使用 d embedding。原 action expert adapter 层与因果 mask 保留，已执行位置不读取 fast 修正。
- 不使用人工缺陷数据，不强制观测变化必须导致动作变化。

旧 config 的 `fast_spatial_tokens` 缺省为 false，保留旧模型结构和 checkpoint 形状。新结构不能直接续训旧 Stage A optimizer；应按新脚本从 official 初始化。Stage B 从新版 Stage A 的配置与权重继承结构，不能单独改回旧 resampler。

## Slow 视觉限制

Stage A 训练时每个样本 50% 概率启用屏蔽，启用时每路相机独立随机屏蔽 50%～75% 视觉 token，至少保留部分视觉。语言不屏蔽，fast 初始参考图像不屏蔽，其余样本用完整 slow 图像。

在 vision tower 输出之后、VLM 语言融合之前置零被屏蔽 embedding，并用有效 token mask 阻止它们成为可见 key/value。固定长度，不添加 learnable mask token。每次 forward 只采样一次，checkpoint 重计算使用同一 mask。注意 vision tower 内部已经进行过视觉融合，因此不是严格删除对应图像像素信息。

推理及 Stage B 冻结 teacher 不启用这种训练 dropout；Stage B 一对旧／新 teacher 都使用相同完整 slow condition。

## 已执行前缀

LIBERO 连续评测现在在 policy postprocessor 和 env action processor 之后，记录实际传给 `env.step` 的控制指令；使用 checkpoint 的 action normalizer 转回模型坐标，放入新观测 packet。不能继续直接记录旧模型 chunk 的归一化输出，否则后处理裁剪／离散化可能丢失。

这里只支持 LIBERO 已有的非 relative-action 路径；缺少 normalizer 或开启额外 relative-action 转换时显式报错，避免静默填错前缀。环境内部动作饱和、实际执行位移不等同于下发指令；物理偏差由 image/proprio 表示。

Stage A 在收到 packet 时硬替换前缀，Stage B updater 内部同样替换，每一步 Euler 去噪之后再次硬固定。前缀仍不计算 loss。训练中的真实示范前缀与推理中的下发前缀遵循同一“控制指令历史”语义。

## 启动

```bash
bash research/scripts/run_stage_a_2xa100.sh
```

本次修改不自动启动训练或重跑完整评测。结构正确与 token 增多不等于已验证闭环收益，需要后续训练与评测确认。
