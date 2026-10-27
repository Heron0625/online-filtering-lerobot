# Online Filtering PI0.5：Stage A / B 交接（2026-09-20）

## 1. 研究目标与动机

目标不是普通的 action chunk 重规划，而是在一个已经开始去噪、且部分动作已被真实执行的 action chunk 内，持续接收快于 slow VLM 的图像和 proprioception，并对**尚未执行**的动作作因果修正。

真实部署中的两类困难应区分：

1. 已下发的动作无法撤回，且实际执行可能和计划不同；
2. 新观测带来旧观测不可得的新信息，尤其在接近物体、抓取和精细放置阶段，后续动作本应被修正。

当前 LIBERO 主要复现第 2 类：环境状态会推进，持续收到真实的环境图像与 proprio；仿真并未刻意模拟执行器动作误差。我们的核心假设是：相对于只等待一个完整 chunk 的基线，及时利用 fast condition 应使策略对状态变化反应更好，而非更犹豫、过早开夹爪或降低抓取精度。

## 2. 符号与目标数学对象

一个 action chunk 长度为 \(H=50\)，在 flow time \(s\) 的 particle 为 \(X_s\)。slow condition 包含初始图像、语言等长上下文；fast condition 含当前多相机图像、proprio 和以控制步计的 delay \(d\)。

一次新 packet 到达时，旧条件为 \(C_{old}\)，新条件为 \(C_{new}\)，其中 \(d_{new}\ge d_{old}\)。已执行 prefix 为 \(A_{<d_{new}}^{exec}\)。必须先执行硬约束：

\[
\bar X_s^{old}=\operatorname{Commit}(X_s^{old}, A_{<d_{new}}^{exec}).
\]

然后仅在可执行 suffix 上完成条件切换：

\[
U(\bar X_s^{old}, C_{old}, C_{new},s,d_{old},d_{new})\approx X_s^{new}.
\]

训练 teacher 采用同一初始噪声 \(Z\) 来定义配对：

\[
X_s^{old}=\Phi_s(Z;C_{old}),\qquad X_s^{new}=\Phi_s(Z;C_{new}).
\]

这里 shared \(Z\) **仅是离线构造 paired teacher target 的工具，不能输入 Stage-B updater**。理想情况下，固定旧 condition 的 deterministic flow map 可写成

\[
X_s^{new}=\Phi_s^{C_{new}}\circ(\Phi_s^{C_{old}})^{-1}(X_s^{old}),
\]

因此 updater 学的是由当前 particle 和新旧 condition 决定的 re-anchoring / transport map；不应依赖生成时的私有随机数。

## 3. Stage A：目的、设计与当前实现

### 3.1 阶段性目的

Stage A 不是 bridge。它需要得到一个可用的、对 fast observation 与 delay 有因果响应的条件 flow policy，作为后续 Stage-B 的 frozen teacher。给定固定 condition \(C\)，它仍训练标准 PI0.5 imitation flow，而不是训练人工 defect path。

当前结论是：先不要在 Stage A 混入人造缺陷动作。若只独立扰动 GT prefix 而不对应真实新状态/观测，物理语义不一致，且容易损害正常 IL 能力。当前更可信的归因路径是让 fast condition 的时间语义、空间表达、mask 和 prefix 处理正确。

### 3.2 训练目标

从 LIBERO 数据中选择可用 delay \(d\)，令前 \(d\) 个 action 位置不可修改；仅对 suffix 做 PI0.5 flow matching：

\[
X_t=tZ+(1-t)A,\qquad v^*=Z-A,
\]

\[
\mathcal L_A=\operatorname{MSE}_{i\ge d}(v_\theta(X_t,t,C_d),v^*).
\]

这里 \(d\) 的单位是**控制步数**，不是墙钟秒数；数据集 FPS 用于对齐采样时刻，实际部署可用控制周期换算/量化到 step，不要求执行频率等于采集 FPS。

### 3.3 模型与信息注入

- PI0.5 action expert 为主策略；fast DINOv3 backbone 冻结。
- fast branch 接收两个相机的 DINO patch 特征、初始/当前差异以及当前 proprio。
- 当前空间版采用 `fast_spatial_tokens=true`、`fast_num_tokens=64`：每个相机 32 个带二维位置编码的视觉 token，外加一个 proprio token；避免早期默认 16-token 压缩丢失定位信息。
- fast adapter 注入 action expert 的第 `(2, 5, 8, 11, 14, 17)` 层。
- delay 同时用于 fast token 的 delay embedding、adapter gate，以及 action 位置 mask；位置 \(<d\) 的 fast correction 被硬置零。
- slow visual dropout：50% 样本启用；启用样本在 slow vision token 融合前随机丢弃 50--75% token。不是 learnable mask token，而是直接以 attention mask 排除 token。目的是减少 slow-only 捷径，但不改标签。
- 当前 Stage-A 动作 prefix 在评测中来自实际 postprocessor 下发给环境的动作，再转回 policy 归一化坐标；不是直接复用模型未执行的预测值。

### 3.4 推理语义

每个 chunk 内，packet 可在多次 denoising query 前到达。Stage A 直接将新的 fast condition 交给随后的 action-expert denoise step；已执行 prefix 仍硬提交，不允许改写。这个“中途直接切 condition”的做法是实用 baseline，但严格说是 stitched conditional ODE，不保证当前 particle 已处于新条件 flow 的正确边缘分布；这正是 Stage B 存在的数学动机。

### 3.5 关键实现位置

- 配置：`src/lerobot/policies/online_filtering_pi05/configuration_online_filtering_pi05.py`
- Stage-A loss：`OnlineFilteringPI05Policy._forward_stage_a`，位于 `modeling_online_filtering_pi05.py`
- fast encoder / resampler / adapter / prefix commit：`src/lerobot/policies/pi05/online_filtering.py`
- 连续 packet 推理：`OnlineFilteringPI05Policy.predict_action_chunk`
- 当前空间版 checkpoint：
  `/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast/stage_a/checkpoints/030000/pretrained_model`

## 4. Stage B：目的、设计与当前实现

### 4.1 不要混淆的三个对象

1. `old particle`：当前 flow time 下、旧 fast condition 得到的 live \(X_s^{old}\)；
2. `new teacher particle`：同一 \(Z\) 和同一 flow time 下、将条件改为新 packet 后的 \(X_s^{new}\)；
3. `executed prefix`：物理系统已经执行的动作，不是 teacher 或 updater 任意生成的目标。

Stage B 不输出一个全新 chunk，也不处理“下一个 flow 时间步”本身；它在同一个 chunk、同一个当前 flow time，把对象 1 重锚定到对象 2，同时 hard commit 对象 3。

### 4.2 当前 teacher 构造

每个 batch：

1. 从合法时刻采样 \(d_{old}<d_{new}\)，10% 行设为 identity，即 `old_delay=new_delay`；
2. 采样同一 `base_noise` \(Z\) 和离散的中间 switch step（1--9 / 10），得到对应 \(s\)；
3. 冻结 Stage-A teacher 在 old/new 两个 fast condition 下，从同一 \(Z\) 积分到相同 \(s\)；每一步均 hard commit 对应执行 prefix；
4. 得到 `(old_particle, target_particle)`；
5. updater 用 old particle、old/new fast token、它们的 innovation token、slow context、flow time、old/new delay 和 prefix phase 预测 target particle。

loss 是只在 \(i\ge d_{new}\) 的真实 action 维度上计算的 MSE：

\[
\mathcal L_B=\operatorname{MSE}_{i\ge d_{new}}(U(\bar X_s^{old},C_{old},C_{new}),X_s^{new}).
\]

### 4.3 updater 的可见信息与结构

- `PosteriorResidualUpdater` 约 2.9M 可训练参数；Stage B 冻结完整 Stage-A teacher。
- slow prefix 经 `posterior_slow_resampler` 压缩为 4 token。
- old fast、new fast、innovation `(new-old)` 分别经投影并带有 semantic role embedding 后拼接为 condition token。
- action particle 经 Transformer 编码；每个 horizon position 有三类 phase embedding：旧 packet 到达前已执行、`[d_old,d_new)` 期间新增执行、尚未执行 suffix。
- per-position residual gate；有 concrete executed action 时，\(i<d_{new}\) gate 为零并 hard commit。
- **当前版本不把 base noise / \(Z\) 输入 updater。**

当前启动训练：

```bash
tmux attach -t train2
# 已在运行：GPU 0,1；每卡 batch 32，全局 batch 64；3000 steps
```

输出目录：
`/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast/stage_b_condition_only_transport`

启动脚本：`research/scripts/run_stage_b_v3_1xa100.sh`。它必须显式传入 `fast_spatial_tokens=true` 与 `fast_num_tokens=64`，否则 CLI 默认会错误构建成 16-token 非空间 fast branch，无法正确加载上述 Stage-A checkpoint。

## 5. 指标解释与成功判据

当前 Stage-B 日志来自 `_forward_stage_b`：

| 指标 | 定义 | 健康趋势 |
|---|---|---|
| `teacher_delta_mse` | \(\|X_s^{new}-X_s^{old}\|^2\) | 表示切换难度；不应单独最小化 |
| `posterior_endpoint_loss` | \(\|\hat X_s-X_s^{new}\|^2\) | 应低于 teacher delta |
| `student_delta_mse` | \(\|\hat X_s-X_s^{old}\|^2\) | 不应长期远小于 teacher delta |
| `delta_cosine_similarity` | teacher/student residual 余弦 | 应明显大于 0 |
| `bridge_gain` | \(1-\text{endpoint}/\text{teacher_delta}\) | 应持续大于 0 |
| `posterior_gate_mean` | 未执行 suffix 上 residual gate 的平均值 | 单独不代表成功；需结合 residual 对齐 |

初始化时 zero-initialized residual head 会自然得到：endpoint \(\approx\) teacher delta、student delta \(\approx0\)、gain \(\approx0\)。这在最初几十 step 正常；若 warmup 后（建议至少 step 300--500）仍如此，则 updater 坍缩为 identity，训练无效。

## 6. LIBERO 结果与评测协议

### 6.1 官方无延时参考（用户提供的 LeRobot / OpenPI 报告）

| Suite | LeRobot | OpenPI |
|---|---:|---:|
| LIBERO Spatial | 97.0% | 98.8% |
| LIBERO Object | 99.0% | 98.2% |
| LIBERO Goal | 98.0% | 98.0% |
| LIBERO-10 | 96.0% | 92.4% |
| 平均 | 97.5% | 96.85% |

这些是原始/无本项目动态延时协议的参考上限，不能和下述动态延时结果直接相减解释为模型退化。

### 6.2 当前最可信的空间 fast Stage-A 全量结果

文件：
`/data/zhoukai/robotmine/eval/libero_continuous_online_spatial_fast/stage_a/summary_all.json`

协议：`continuous_online_v1`；四个 suite 每 task 50 episode，共 2000 episode；每 20 控制步启动一个 chunk；每个 chunk 的初始 delay 均匀随机于 1--10；chunk 内每个控制步接收/assimilation 一个新 online packet；Stage A 无 updater。

| Suite | 成功 / 总数 | 成功率 |
|---|---:|---:|
| Spatial | 425 / 500 | 85.0% |
| Object | 470 / 500 | 94.0% |
| Goal | 471 / 500 | 94.2% |
| LIBERO-10 | 442 / 500 | 88.4% |
| 总计 | 1808 / 2000 | **90.4%** |

视频路径记录在 summary 的 `video_path`，并带有 `success` / `failure` 文件名和 online observation / chunk switch 注释事件。

### 6.3 历史 Stage-A（非当前 checkpoint；仅作背景）

`/data/zhoukai/robotmine/eval/stage_a_organized/summary_all.json` 对应旧 `online_filtering_pi05_clean_causal` checkpoint、旧评测实现：Spatial 89.2%、Object 96.4%、Goal 95.0%、LIBERO-10 84.8%、总计 91.35%。它没有当前 continuous-online 协议的全部修正，因此**不得与 90.4% 直接做方法优劣比较**。

历史全量对比的定性现象是：动态 delay 明显降低官方 chunk baseline；一些 Stage-A/Stage-B 版本在前三个 suite 有小幅改善，但 LIBERO-10 变差，且视频中曾出现频繁重规划导致抓取阶段犹豫或过早开夹爪。尚不能宣称方法有效。

### 6.4 Stage B 结果状态（2026-09-21 更新）

- 旧 Stage-B checkpoint 曾训练完成，但诊断显示 identity collapse，不能作为有效 bridge 结果。
- condition-only 版本也发生 identity collapse；随后实现了 scale-balanced residual 版本：关闭 learned residual gate，加入 per-sample relative residual loss、cosine loss，并将 identity sample 比例暂时设为 0。
- scale-balanced 版本已完成 3000 steps，但**正式随机 batch 训练仍然 identity collapse**，不能进入 LIBERO 闭环评测。
- 最新 checkpoint：
  `/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast/stage_b_scale_balanced_transport/checkpoints/003000/pretrained_model`
- 本版本没有可报告的闭环成功率。不应将它与当前 spatial-fast Stage A 做任务成功率比较。

## 7. 已观察到的 Stage-B 困境

旧 Stage-B 完训日志典型范围：

```text
teacher_delta_mse       ≈ 4e-5 -- 1.7e-4
student_delta_mse       ≈ 1e-6 -- 4e-6
posterior_endpoint_loss ≈ teacher_delta_mse
delta_cosine_similarity ≈ 0
bridge_gain             ≈ 0
```

condition-only 训练在 step 20 / 40 也出现相同早期形态：

```text
step 40:
teacher_delta_mse       = 2.664e-4
posterior_endpoint_loss = 2.665e-4
student_delta_mse       = 1.355e-7
delta_cosine_similarity = 0.003
bridge_gain             = -0.001
posterior_gate_mean     = 0.684
```

这说明 prediction 几乎等于 old particle；不是 gate 完全关闭（gate 约 0.68），而是 residual head 没有学出 teacher 所需的方向和幅度。

### 7.1 Scale-balanced residual 实验及结论（2026-09-21）

为排除普通 endpoint MSE 偏好 identity 的问题，最新版本做了以下修改：

- `posterior_use_learned_gate=false`，suffix 上使用固定 hard availability gate；
- `posterior_identity_dropout=0.0`；
- 加入按单样本 teacher residual energy 归一化的 relative residual loss；
- 加入 cosine loss，并保留小尺度 endpoint loss；
- 增加 `delta_magnitude_ratio` 指标；
- 将 teacher-pair 构造与 updater loss 分离，允许缓存固定 pair 做快速 overfit。

固定 8 个真实 LIBERO samples、固定 noise/delay/switch time 的 deterministic overfit 在 300 steps 后达到：

```text
bridge_gain                    = 0.351
delta_cosine_similarity        = 0.623
delta_magnitude_ratio          = 0.587
correct_condition_bridge_gain  = 0.351
stale_condition_bridge_gain    = 0.178
shuffled_condition_bridge_gain = 0.292
```

这证明 updater 容量、反向传播和 condition 路径并未完全断开；但固定 batch 只能证明可记忆性，不能证明跨样本泛化。

正式 3000-step 随机 batch 训练从头到尾没有获得有效 transport。最终窗口为：

```text
posterior_endpoint_loss  = 2.852e-4
teacher_delta_mse        = 2.852e-4
posterior_relative_loss  = 0.987
posterior_cosine_loss    = 0.930
student_delta_mse        = 4.221e-8
delta_cosine_similarity  = 0.015
bridge_gain              = 0.000
delta_magnitude_ratio    = 0.053
gradient_norm            = 387.653
```

训练全过程的 relative loss 约为 1、cosine 约为 0、bridge gain 约为 0，并非先学会后退化。梯度范数长期约 300--500，而 optimizer gradient clipping 为 1.0。当前最可信的解释是：per-sample 小能量归一化与 residual 接近零时的 cosine 项产生病态大梯度；随机 batch 下梯度方向互相抵消，并在每步被强裁剪。固定 batch 可以沿重复方向逐渐记忆，但不能泛化。该实验因此仍判定为 **identity collapse**。

注意：loss 的绝对量级 \(10^{-4}\) 本身并不反常，因为它是归一化 action particle 的 MSE；不能据此直接把 loss 放大或直接加大学习率。当前 scheduler peak 配置约为 \(2.5\times10^{-5}\)，日志中的早期 \(10^{-6}\) 是 warmup 的瞬时值。

## 8. 已知诊断、不可接受的解释与后续建议

已有诊断脚本与结果：

- `research/scripts/validate_stage_b_checkpoint.py`
- `research/scripts/diagnose_online_gripper_transitions.py`
- `/data/zhoukai/robotmine/eval/stage_b_validation/real_diagnostic_report.json`
- `research/docs/stage_b_targeted_diagnosis.md`

历史诊断显示 Stage-A teacher 的纯 observation 改变在某些样本小于 delay/prefix 联合改变，但这**不是**“Stage B 不该处理 condition switch”的理由。Stage B 的定义就是针对同一 action chunk 内、条件和已执行进度共同变化的严格 re-anchoring。

下一位 agent 不应继续训练或闭环评测当前 checkpoint。建议按以下顺序推进：

1. 先离线缓存数万组 teacher pairs，包括 raw old/new fast tokens、slow prefix embeddings、old/target particle、delay、flow time、suffix mask；避免每次实验重复运行 4B frozen teacher。
2. 统计 `(flow-time bucket, delay-gap bucket, action dimension)` 的 teacher residual RMS，以训练集统计量定义稳定的 bucket scale；不要再除以每个样本自身可能接近零的 residual MSE。
3. 预测 bucket-standardized residual，例如 `delta_hat = sigma_bucket * tanh(f_theta(...))`；第一版去掉 cosine loss，仅使用稳定的 standardized residual regression 和小权重 endpoint loss。
4. 记录 clipping 前/后的梯度范数与每个模块的梯度，目标是避免几乎每一步都触发 global norm clipping。
5. 在 held-out cached pairs 上报告 bridge gain、delta cosine、magnitude ratio；固定 batch overfit 只作为 wiring test。
6. 在 held-out pairs 上做正确 new condition、stale、shuffled、observation-only、delay/prefix-only 消融。正确条件必须显著优于 stale/shuffled，才说明 updater 使用了新 observation。
7. 只有 held-out offline transport 明显成立后，才做昂贵的 2000-episode LIBERO 闭环评测。

不要把“直接在后续 Euler step 改用新 Stage-A condition”当成 Stage B 的严格解：它虽可作为 Stage-A baseline，却把旧条件轨迹上的 particle 接到新条件 vector field 上，通常不等价于新条件下正确的 \(X_s\) 分布。

## 9. 当前代码/训练状态（2026-09-21）

- 当前仓库存在未提交修改；勿用 reset/checkout 覆盖。
- scale-balanced Stage-B 已训练完毕；当前没有仍在运行的 Stage-B 训练进程。`tmux train2` 只保留结束后的 shell/session。
- 训练脚本：`research/scripts/run_stage_b_v3_1xa100.sh`。
- 完整训练日志：
  `/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast/stage_b_condition_transport_train_20260920_113832.log`
- scale-balanced checkpoints 每 500 steps 保存于：
  `/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast/stage_b_scale_balanced_transport/checkpoints/`
- Stage-B 可训练模块仅为 updater、fast projection、condition role embeddings 和 slow context resampler；Stage-A teacher 全冻结。
- 当前单步训练很慢（约 14--15 s/step），原因是每个 batch 需通过冻结的完整 VLA 分别积分 old/new teacher 到随机 mid-flow time；双卡按 data parallel 分摊样本，但每张卡仍需运行自己 batch 内的 paired teacher。
- 相关组件测试：

```bash
PYTHONPATH=src /home/zhoukai/miniforge3/envs/lerobot/bin/python -m pytest \
  tests/policies/pi0_pi05/test_online_filtering_components.py \
  tests/policies/pi0_pi05/test_online_filtering_pi05_stage_a.py \
  tests/policies/pi0_pi05/test_continuous_online_generation.py -q
```

最近代码修改后该集合通过（55 tests）。

## 10. 交接给其他用户时的资产说明

仓库副本包含代码、文档、测试和 git working tree（包括未提交修改），但大模型、LIBERO 数据、训练 checkpoint 与评测视频位于仓库外部的 `/data/zhoukai/...`，不会因复制仓库目录而自动复制。接手者至少需要重新配置或取得以下资产：

- LIBERO dataset：`/data/zhoukai/dataset/libero`
- DINOv3 weights：`/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m`
- tokenizer：`/data/zhoukai/pretrain/paligemma-tokenizer`
- Stage-A teacher：`/data/zhoukai/robotmine/ckpt/online_filtering_pi05_spatial_fast/stage_a/checkpoints/030000/pretrained_model`
- Stage-B checkpoints/log：见第 9 节
- Stage-A evaluation：`/data/zhoukai/robotmine/eval/libero_continuous_online_spatial_fast/stage_a/`

若新用户无权读取 `/data/zhoukai`，必须另行复制这些外部资产，并修改脚本中的默认路径。不要把当前 Stage-B 003000 checkpoint 当作有效方法结果；它仅适合复现实验失败和继续诊断。
