# Online Filtering 项目总交接（2026-10-30）

> 当前双时刻方案与阶段配置以 [Online Filtering 当前方案（2026-11-01）](online_filtering_current_method_20261101.md) 为准。PI0.5 新 Phase A 必须使用 `training_stage=online_flow, online_flow_phase=canonical`；`training_stage=stage_a` 是旧单时刻路径。XVLA 的 `stage_a` 则是当前双时刻路径。

## 1. 环境与仓库

- 仓库：`/data/yhr/project/robot/online_filtering`
- 当前分支：`feature/clean-online-filtering-pi05-xvla`
- 当前 HEAD：`fcc0975 Implement clean online filtering for PI0.5 and X-VLA`
- Python：`/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python`
- 训练输出：`/data/yhr/output/online_filtering/ckpt`
- 评测输出：`/data/yhr/output/online_filtering/eval`
- 当前 `/data` 剩余约 `675 GB`。
- 项目迁移过目录和环境，不要使用旧的 `/data/zhoukai/...` 命令，除非只是读取仍存在的预训练权重或数据。

工作区有尚未提交的 X-VLA 修复和脚本。接手后第一步运行：

```bash
git status --short
git diff --stat
```

不要覆盖或回退这些改动。远端已有 `fcc0975`，但当前工作区改动晚于该提交。

## 2. 研究问题与最终实验定义

目标是在 slow VLM/action expert 推理期间继续接收 fast observation，并修正尚未执行的 action suffix。延时定义为从 slow observation 采集到新 chunk 可用之间经过的控制步数：

$$
\tau=\frac{d}{f_{control}}.
$$

LIBERO 使用：

$$
f_{control}=20\ \mathrm{Hz},\qquad 1\ \mathrm{step}=50\ \mathrm{ms}.
$$

因此 delay `1/5/10/15/20/25` 分别对应 `50/250/500/750/1000/1250 ms`。论文和表格应报告物理时间，不能跨不同 Hz 直接比较 delay step。

推理的因果时间线：在时刻 $$t$$ 采集 slow observation，推理期间继续执行旧 chunk；经过 $$d$$ 步后，新 chunk 从与当前时刻对齐的位置开始执行：

$$
a_{t+d}^{new}=A_t[d].
$$

主张应拆成两项验证：

1. 同一 baseline 的成功率随物理延时增加而下降；
2. 相同 backbone、控制频率、delay 和 replan schedule 下，Online Filtering 的成功率高于 baseline。

## 3. PI0.5 方法演进与已完成实现

### 3.1 Phase A

Phase A 从 PI0.5 LIBERO checkpoint 全参微调。输入包含 slow prefix、采样 delay、对应时刻的 fast observation。训练仍为单步 flow-matching；delay 前的已执行 prefix 被 hard commit，loss 只作用于 suffix。

已完成的关键实现：

- fast observation 使用两个 LIBERO 相机和 proprio；
- DINOv3 fast encoder；
- spatial fast tokens，避免早期过强 token 压缩；
- delay embedding、action-position mask 和 prefix commit；
- slow visual stochastic mask/dropout，用于降低 slow-only 捷径；
- 在线推理期间可持续推送 fast packets；
- prefix KV cache，避免每个去噪步重复运行完整语言主干；
- fast token 的显式 `new-old innovation` 分支后来被删除，避免无充分依据的冗余输入。

早期诊断确认 fast observation 对 action 的影响偏弱。采用 spatial fast token、slow mask 和 adapter 修正后，Phase A 可以正常训练和评测。

### 3.2 Phase 2

Phase 2 的目标不是保持 Phase A 不变，而是从 Phase A 初始化，最终学习 condition 随在线观测演进的 flow。Phase A 只提供 canonical flow 和良好初值。

已实现的 Phase 2 训练包括：

- frozen/EMA canonical teacher；
-同一 base noise、同一 flow time 下构造 old/new paired particles；
- rollout 1--4 个在线切换步骤；
-执行 prefix hard commit；
-只优化 rollout/switch recovery 目标，不再人为加入 Phase A canonical loss；
- slow mask 随 Phase 2 训练退火，避免长期破坏 slow context；
-最终有效版本移除了显式 innovation token，并采用 annealed slow mask。

曾尝试或讨论但未作为最终方案保留的内容：

- 单输入与双输入混合格式；
- Stage B 额外 canonical loss；
- 强制 endpoint/global trajectory loss；
-依赖私有 base noise 的 updater；
-固定大量 identity transition；
-显式 `new-old` fast token。

### 3.3 clean cross-attention 分支

当前分支 `fcc0975` 实现了不兼容旧权重的 clean 版本：PI0.5 和 X-VLA 均使用 zero-init cross-attention 注入 online fast condition，并实现 prefix caching。该版本是为了完整方法设计，不保证兼容早期 checkpoint。

注意：`metrics.md` 中成功的 PI0.5 Phase A/Phase 2 指标主要来自此前经过诊断的 no-innovation/annealed 训练，不应自动声称为当前 clean cross-attention 分支的结果。clean 分支需要单独训练和评测后才能替换主表。

## 4. PI0.5 已完成训练与结果

统一指标文件：[`metrics.md`](./metrics.md)。不要另建第二份数值表。

最重要结果如下。

动态 delay `1--10`、20 Hz、2000 episodes：

- Phase A 30k：`91.70%`
- Phase 2 mixed 30k：`89.80%`，已否决
- Phase 2 no-innovation annealed 30k：`91.95%`
- Phase 2 no-innovation annealed 60k：`92.05%`

固定 delay、20 Hz、每点400 episodes：

| Delay | Official PI0.5 | Phase A | Phase 2 |
|---:|---:|---:|---:|
| 1 | 96.00% | 95.50% | 97.50% |
| 5 | 90.25% | 90.50% | 92.50% |
| 10 | 81.50% | 82.25% | 87.00% |

该结果支持两项核心主张：延时显著降低 Official PI0.5，Phase 2 在中高延时下恢复一部分成功率。delay=10 时 Phase 2 相对 Official 提升 `5.5` 个百分点。

Official PI0.5 大延时结果也已完成：

| Delay | 物理延时 | 成功率 |
|---:|---:|---:|
| 15 | 750 ms | 72.75% |
| 20 | 1000 ms | 68.25% |
| 25 | 1250 ms | 61.25% |

所有正式 fixed-delay 结果均按20 Hz运行。此前把 PI0.5 解释为10 Hz是错误归因，不需要重测这些主结果。

## 5. LIBERO 评测脚本与已修问题

核心脚本：`research/scripts/eval_libero_synthetic_delay.py`。

它实现：

-固定或动态 delay；
-推理期间旧 chunk 继续执行；
-在线 observation packet 与 flow step 的因果对齐；
-新 chunk 从 `chunk[delay:]` 接管；
-queue underrun 计数；
-annotated mp4 和事件记录；
-同一 seed 下 checkpoint-independent delay sampling。

曾发现的评测错误/风险：

-模型输入和 delay 方式变化后不能直接复用旧 eval；
-错误 camera preprocessing 会让 Phase A 成功率低于10%；修正后恢复正常；
-不同模型 episode length 不一致曾让 X-VLA 得到不公平的高分，后来按 PI0.5 长度统一；
-短 action chunk 在大 delay 下会产生 queue underrun，hold/padding action 会污染延时退化结论；
-`delay == replan_interval` 的边界检查目前偏严格，主实验若要使用该边界应单独修正并测试。

主实验不能把 queue underrun 当作普通 latency degradation。应保证：

$$
replan\_interval+delay\le chunk\_size.
$$

## 6. 已评测 baseline

已搭建并运行过：

- Official PI0.5
- X-VLA
- π0-FAST
- OpenVLA-OFT

动态 delay 主表中已有：

- X-VLA，PI0.5-matched episode horizon：`96.15%`
- π0-FAST：`57.40%`
- OpenVLA-OFT：`53.40%`

X-VLA 的大延时 `15/20/25` 虽有 `92.50/85.25/66.75%`，但对应 queue underrun 分别为 `18,281/48,181/120,081`，不能作为干净的主表结论。这也是把 X-VLA chunk 从30扩到50的动机。

π0-FAST 和 OpenVLA-OFT 原生短 chunk 在大 delay 下同样有严重 underrun。不要直接用“全部失败”或 hold-action 结果声称模型不耐延时。

## 7. X-VLA 官方复现与当前事实

官方仓库：`/data/yhr/project/robot/X-VLA-official`。

官方 X-VLA LIBERO：

-原始 HDF5 `control_freq=20`；
-连续读取 simulator samples，无 stride；
-官方 checkpoint action chunk 为30步；
-官方推理环境为20 Hz；
-action 是绝对 EE position + rotation6D + gripper。

现有 LeRobot LIBERO：

-路径：`/data/yhr/pretrain/recovered_zhoukai/dataset/libero`
-metadata 和视频时间轴写为10 fps；
-但轨迹行实际保留连续20 Hz控制样本，没有减半。

已做实证：同一 `turn on the stove` 任务，原始 HDF5总计4460步，LeRobot版本4454步；若真为10 Hz，应约为2230步。因此训练应继续使用连续行，推理按20 Hz解释。不要简单把 metadata 改成20，否则 LeRobot 的 timestamp sampling 可能重复或跳行。

### 7.1 已完成的 X-VLA 修复

-官方 checkpoint 已转换为 LeRobot policy；
-官方30-step checkpoint 在正确设置下可达到正常/接近官方的成功率；
-两个真实相机加一个 empty camera，共3 views；
-修复 `empty_cameras` 被重复计数成4 views；
-修复图像 float/normalization processor；
-7维 LIBERO relative action 与对应 future EE state 在 batch 内转换成官方20维 absolute EE6D controller goal；
-state/action normalization 当前均为 `IDENTITY`，几何换算不会作用于归一化后的伪坐标；
-增加 future state delta indices 和 episode-tail padding mask；
-实现30→50 action positional embedding 扩展，保留前30位置，新增20位置由第30位置初始化，并正确后移 condition positions；
-加入真实 target 和 positional remap 单元测试。

关键文件：

- `src/lerobot/policies/xvla/configuration_xvla.py`
- `src/lerobot/policies/xvla/modeling_xvla.py`
- `src/lerobot/policies/xvla/processor_xvla.py`
- `src/lerobot/policies/xvla/utils.py`
- `tests/policies/xvla/test_xvla_libero_targets.py`
- `research/scripts/prepare_xvla_long_horizon_init.py`

### 7.2 尚未解决的 X-VLA 困难

30→50 finetune 的旧实验成功率明显低于预期。后续曾把问题归因为10/20 Hz不一致，但该归因已经被否定：最新旧实验使用的连续样本本来就是20 Hz物理轨迹。

旧实验更早的版本确实存在 relative/absolute target、视图数量、processor 和 position remap 错误；最终 v3 在训练前已包含大部分修复，但性能仍不理想。因此不能直接重复同一长训练并期待 Hz 修复带来提升。

接手后优先做以下最小诊断：

1. Official30 checkpoint，20 Hz，执行30步，复现官方成功率；
2. 仅做30→50初始化、不训练，仍只执行前30步，检查输出前30步与 Official30 数值 parity 和成功率；
3. 50-init 分别以 `replan=30` 和 `replan=50` 评测，区分“初始化破坏”与“新增20步质量差”；
4. 用一个真实 batch，把 LeRobot 在线转换出的20维 target 与官方 X-VLA HDF5 handler target逐元素比较；
5. 只有上述通过，才启动50-step finetune；先保存1k/5k并快速 eval，不要直接盲跑60k。

此前三轮已判定有问题的 X-VLA-50 checkpoint 和 smoke 已清理，释放约89 GB。不要在文档里引用已删除 checkpoint 路径。

### 7.3 已停止的冗余数据路线

曾启动下载40个原始 HDF5并通过 MuJoCo replay生成256px绝对动作数据，但 replay很慢。停止时约有18/40 raw、2/40 preprocessed。下载和 preprocess tmux 已停止，残留数据仍在：

- `/data/yhr/pretrain/xvla_official/libero_hdf5_raw`
- `/data/yhr/pretrain/xvla_official/libero_xvla_preprocessed`

由于现有 LeRobot 数据已证实保留连续20 Hz样本，这条 replay路线当前不是必需项。除非第7.2节第4项发现真实 target 不一致，否则不要恢复下载/replay。

## 8. X-VLA + Online Filtering

当前 clean 分支已经搭建 X-VLA + Online Filtering：

- old/new fast condition；
-zero-init cross-attention；
-prefix cache；
-Phase A/Phase 2 配置与 processor；
-初始化脚本 `research/scripts/prepare_online_filtering_xvla_init.py`；
-单元测试 `tests/unit/policies/test_online_filtering_xvla.py`。

但它还没有建立可信的完整训练结果。正确顺序必须是：先得到可靠的标准 X-VLA-50 baseline，再训练 X-VLA-50 + Online Filtering。否则无法判断收益来自 Online Filtering 还是 baseline 实现差异。

## 9. 真机配置

真机计划使用 PI0.5 backbone，不是 X-VLA。完整设定见 [`real_robot_25hz_training_config.md`](./real_robot_25hz_training_config.md)。核心参数：

```text
data_hz = 25
control_hz = 25
chunk_size = 50
replan_step = 10
max_inference_delay = 30
```

端侧实测推理约 `700--1000 ms`，对应25 Hz下约18--25步。训练 delay 主分布应覆盖该区间。推理异步运行，25 Hz指 chunk action 的执行频率，不是每秒运行25次完整 VLA。

## 10. 其他 benchmark

讨论并开始准备 RoboCasa365，原因是 LeRobot已有相关环境支持，可减少脚手架工作。当前没有形成可写入主表的 RoboCasa完整结果。不要声称已经完成 RoboCasa训练/评测。

仿真扩展建议：先完成 LIBERO 的方法闭环和 X-VLA backbone generalization，再决定 RoboCasa全量或子集。真机实验仍是论文的重要组成部分。

## 11. 当前运行状态

-当前没有新的 X-VLA转换或训练进程；刚才重复的 LeRobot→absolute离线转换已停止，不完整的102 MB输出已删除。
-旧 tmux session 很多，部分只保留历史窗口。不要仅根据 `tmux ls` 判断任务仍在运行；同时检查：

```bash
ps -eo pid,etimes,cmd | rg 'lerobot_train|eval_libero|accelerate'
nvidia-smi
```

- `rm75_pi05_gripper_position` 当前显示 attached，属于真机/其他实验，未经确认不要停止。

## 12. 接手优先级

1. 保存当前工作区 diff，运行 X-VLA相关单元测试。
2. 完成第7.2节的 Official30→50-init 最小诊断。
3. 若 parity通过，启动短50-step finetune并在1k/5k做 delay=0 快速评测。
4. 标准 X-VLA-50恢复正常后，再训练 X-VLA + Online Filtering Phase A/Phase 2。
5. clean PI0.5 cross-attention 分支需要单独训练和评测，不能直接继承旧主表结论。
6. 所有新结果统一补入 `research/docs/metrics.md`，记录 control Hz、delay、replan interval、chunk size、episode数和 queue underruns。

## 13. 不要重复的错误

-不要把 LeRobot LIBERO metadata 的10 fps直接当作轨迹物理频率；
-不要仅修改 metadata 为20；
-不要在不同 Hz 下比较相同 delay step；
-不要把 queue underrun/hold action造成的失败解释为纯延时退化；
-不要默认旧 eval 脚本适配新的模型输入；
-不要无条件信任 converted checkpoint 的 processor JSON；
-不要在标准 X-VLA-50尚未复现时训练 X-VLA + Online Filtering；
-不要把旧 PI0.5指标归到尚未评测的 clean cross-attention 分支；
-不要重复恢复已否决的 explicit innovation、mixed Phase 2 或 canonical Phase-2 loss。
