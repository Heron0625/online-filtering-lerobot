# Online Filtering Flow：LIBERO 训练与部署规格

> 状态：工程设计草案  
> 理论文档：[online_filtering_flow.md](./online_filtering_flow.md)

## 1. 目标与基线选择

第一目标是在 LeRobot 的 LIBERO 环境中实现一个完整的异步闭环 flow policy：机器人执行旧 chunk 的同时生成新 chunk，重叠期间到达的 observation/action 被用于更新新 chunk，推理完成后裁掉 elapsed prefix，从第 $d+1$ 个动作开始执行。

推荐以 PI0.5 为 backbone，原因包括：

- 默认 action horizon 为 50；
- 已有 LIBERO base checkpoint；
- 已有 Training-Time RTC clean-prefix 支持；
- 已有 RTC action queue、delay tracking 和 guidance；
- 可直接与 RTC、Training-Time RTC 和 Legato-style continuation 比较。

## 2. LIBERO 数据接口

当前推荐数据集为 `lerobot/libero`：

- 两路 $256\times256$ RGB：agent view 和 wrist view；
- 8 维 state；
- 7 维 relative action；
- 50-step action target 由 policy 的 `action_delta_indices` 读取。

闭环训练需要额外读取最大 delay 内的连续观测：

```python
image_observation_delta_indices = list(range(max_delay + 1))
state_observation_delta_indices = list(range(max_delay + 1))
action_delta_indices = list(range(chunk_size))
```

建议 batch schema：

```text
images:          [B, D+1, num_cameras, C, H, W]
states:          [B, D+1, state_dim]
actions:         [B, H, action_dim]
arrival_mask:    [B, K, D+1]
elapsed_mask:    [B, K, H]
valid_obs_mask:  [B, D+1]
valid_act_mask:  [B, H]
delay:           [B]
```

必须固定数据时序约定：

$$
o_t\rightarrow a_t\rightarrow o_{t+1}.
$$

以推理开始时刻 $t$ 为新 chunk 原点，当 delay 为 $d$ 时：

- observation filtration：$o_{t:t+d}$；
- executed overlap actions：$a_{t:t+d-1}$；
- elapsed action prefix：$X_{1:d}$；
- 推理完成后首个可执行动作：$X_{d+1}$。

## 3. Policy 模块

### 3.1 Slow semantic encoder

初始帧和语言由 PI0.5 VLM 编码：

$$
z^{slow}=\operatorname{VLM}(o_t,l).
$$

第一阶段可在一次 chunk generation 内固定该特征。

### 3.2 Fast observation encoder：DINOv3

Fast observation path 固定采用 DINOv3 ViT-S/16：

```text
model_id: facebook/dinov3-vits16-pretrain-lvd1689m
input_size: 224 x 224
patch_size: 16
hidden_dim: 384
backbone_parameters: approximately 21M
```

选择 ViT-S/16 而不是更大的 DINOv3 版本，是因为 fast path 的目标是在 PI0.5 VLM 推理期间多次更新。它需要提供局部几何和变化特征，而不负责完整任务语义。

DINOv3 消费重叠期间最新的两路 RGB 和 state：

$$
z_d^{fast}
=
E_{DINO}
(I_{t+d}^{agent},I_{t+d}^{wrist},q_{t+d},d).
$$

每个 camera 的 $224\times224$ 输入产生 $14\times14$ patch grid。只使用归一化后的 patch tokens，CLS/register/storage tokens 按 DINOv3 配置显式剥离。

初始帧的 DINOv3 feature 只计算一次并缓存：

$$
F_0^c=E_{DINO}(I_t^c),
$$

新帧到达时计算：

$$
F_d^c=E_{DINO}(I_{t+d}^c),
$$

其中 $c\in\{agent,wrist\}$。构造每个 camera 的变化特征：

$$
R_d^c
=
\operatorname{Concat}
\left[
F_d^c,
F_0^c,
F_d^c-F_0^c
\right].
$$

由于两路 camera 的原始 patch tokens 数量较大，不能直接在每个 PI0.5 action-expert layer 中全部 cross-attend。使用一个小型 resampler：

$$
Z_d^{fast}
=
\operatorname{Resampler}_\psi
\left(
R_d^{agent},R_d^{wrist},q_{t+d},d
\right),
$$

将它们压缩为固定的 8--16 个 fast tokens，并投影到 PI0.5 action expert 的 hidden dimension。

建议训练策略：

1. 初始实现冻结完整 DINOv3，只训练 resampler、projection、gated cross-attention 和 posterior updater；
2. 基线收敛后，可解冻 DINOv3 最后 1--2 个 block，以主干学习率的 $0.01$--$0.1$ 倍微调；
3. 保留 frozen-DINOv3 ablation，防止把收益错误归因于增加可训练视觉参数；
4. 图像预处理采用 DINOv3 对应的 `AutoImageProcessor`，不能直接沿用 PI0.5/PaliGemma 的 normalization；
5. 每个新 observation 只运行一次 DINOv3，所得 fast tokens 写入带 timestamp 的 cache；在新帧到来之前，多个 flow solver steps 复用同一份 tokens。

第一版使用初始帧和最新帧，不保留所有完整视觉 token history。后续可替换为 recurrent fast-state 或固定窗口 temporal encoder。

#### 3.2.1 注入 PI0.5 action expert 的位置

不把 $Z_d^{fast}$ 拼接到 PI0.5 的 PaliGemma/VLM prefix。现有 PI0.5 在每个 Gemma layer 中对 slow prefix stream 和 action-expert suffix stream 做联合 attention，并在推理时缓存 slow prefix KV。如果将每帧变化的 DINOv3 tokens 加入 prefix，新 observation 到达后会使 prefix KV cache 失效，并迫使昂贵的慢分支重新计算。

采用仅作用于 action expert 的 cross-attention adapter：

```text
existing PI0.5 joint attention (slow prefix <-> action suffix)
    -> action-stream residual
    -> DINOv3 fast cross-attention adapter
    -> action-expert MLP
```

具体插入到每个选定 action-expert layer 的 joint-attention residual 之后、post-attention layer norm/MLP 之前。PaliGemma slow stream 不经过该 adapter。

设第 $\ell$ 层 action hidden states 为：

$$
H_A^\ell\in\mathbb R^{B\times H\times D_A},
$$

resampler 输出为：

$$
Z_d^{fast}\in\mathbb R^{B\times N_f\times D_A},
\qquad N_f\in\{8,16\}.
$$

fast cross-attention 为：

$$
Q^\ell=W_Q^\ell\operatorname{LN}(H_A^\ell),
$$

$$
K_f^\ell=W_K^\ell Z_d^{fast},
\qquad
V_f^\ell=W_V^\ell Z_d^{fast},
$$

$$
C_f^\ell
=
\operatorname{MHA}
(Q^\ell,K_f^\ell,V_f^\ell).
$$

只允许最新 observation 修正尚未执行的位置：

$$
M_{d,j}^{fast}
=
\mathbf 1[j\ge d].
$$

更新 action stream：

$$
\widetilde H_{A,j}^\ell
=
H_{A,j}^\ell
+
M_{d,j}^{fast}
\gamma_{\ell,j}
W_O^\ell C_{f,j}^\ell.
$$

gate 依赖 flow time、相对 action horizon 和 observation age：

$$
\gamma_{\ell,j}
=
\gamma_\ell
\left(
\tau,
\frac{j-d}{H-d},
\Delta t_{obs}
\right).
$$

$\gamma_\ell$ 或 $W_O^\ell$ 使用零初始化，使新增模块在训练开始时严格退化为原始 PI0.5，避免随机 fast adapter 破坏已有 action policy。

第一版不必在全部 18 个 Gemma expert layers 中添加 adapter。建议每 3 层注入一次，即 6 个注入点：

```text
expert layer indices: 2, 5, 8, 11, 14, 17
```

需要消融 3、6、18 个注入点。$Z_d^{fast}$ 每个 observation 只计算和缓存一次，但各层保留独立的轻量 $Q/K/V/O$ projections。

#### 3.2.2 为什么不采用另外两种简单注入

- **不直接拼接 VLM prefix**：会使动态 observation 更新失效 slow-prefix KV cache，并把 fast path 的成本传递给 PaliGemma 分支；
- **不只使用 FiLM/AdaRMS**：全局调制适合传递“是否接触”等低维状态，但会过早压缩 DINOv3 的空间 patch 信息；
- **采用 cross-attention adapter**：action horizon 中的每个未来动作位置可以选择不同的图像区域，同时 fast K/V 可随新帧独立刷新。

state、delay、observation age 等低维量仍可额外进入 AdaRMS/FiLM；DINOv3 spatial tokens 主要通过 cross-attention 注入。

### 3.3 Action-prefix encoder

已执行 overlap actions 独立编码：

$$
z_d^{act}
=
E_{act}(a_{t:t+d-1},d).
$$

该编码器与 clean-prefix mask 同时使用：前者向模型提供执行历史，后者保证 action tensor 中 elapsed coordinates 的对齐。

### 3.4 Conditional generation flow

generation vector field 接收：

```text
noisy free suffix
flow time
slow semantic tokens
latest fast observation tokens
executed action prefix tokens
elapsed/queued masks
observation timestamps and age
```

### 3.5 Re-anchoring teacher 与 student

理论 teacher 使用 conditional flow 的 inverse-forward composition：

$$
U_{s\rightarrow s+1}^{\tau}
=
\Phi_{s+1}^{\tau\leftarrow0}
\circ
(\Phi_s^{\tau\leftarrow0})^{-1}.
$$

runtime 从初始化开始保存每个 sample 的 base noise $Z$，因此 exact teacher 通常直接计算：

$$
X_\tau^+
=
\Phi_{s+1}^{\tau\leftarrow0}(Z),
$$

无需实际逆积分旧 flow。inverse-forward 形式用于证明 pushforward 正确性。

部署 student 使用蒸馏后的：

- one-step update map $T_\phi$；或
- 少量 assimilation steps 的 innovation flow $g_\phi$。

## 4. 训练阶段

### 4.1 统一符号

动作块长度为 $H=50$，使用 0-based 下标：

$$
A=(a_0,a_1,\ldots,a_{H-1}).
$$

推理期间已经执行 $d$ 个 overlap actions 时：

- elapsed prefix：$A_{0:d-1}$；
- 最新闭环观测：$o_d$；
- 尚未执行的随机 suffix：$A_{d:H-1}$；
- 推理此刻完成时，下一条实际执行的是 $a_d$。

信息集定义为：

$$
\mathcal F_d
=
\{o_0,\ldots,o_d; a_0^{exec},\ldots,a_{d-1}^{exec}\}.
$$

基础 policy 的目标分布为：

$$
p(A_{d:H-1}\mid\mathcal F_d).
$$

### 4.2 总体训练结构

训练分为两个必需的模型阶段，以及一个可选联合微调阶段：

1. **Stage A：filtration-conditioned action flow**。学习每个信息集 $\mathcal F_d$ 下正确的动作 posterior；
2. **Stage B/C：exact re-anchoring teacher pair generation 与 innovation distillation**。学习新 observation 到达时如何快速从旧 posterior 转移到新 posterior；
3. **Stage D：可选短闭环 unroll**。在前两个阶段稳定后缩小训练--部署差异。

理论正确性由 Stage A 加 exact shared-base re-anchoring 定义。Stage C 的 student 用于降低推理成本，不重新定义目标 posterior。

### Stage A：条件 flow family

随机采样：

- trajectory anchor $t$；
- elapsed delay $d\in[0,D_{max}]$；
- flow time $\tau$；
- camera arrival/dropout schedule；
- episode-valid masks。

模型条件为：

$$
\mathcal F_{t+d}
=
\{o_{t:t+d},a_{t:t+d-1}\}.
$$

flow state 只包含可修改 suffix：

$$
Y_d=A_{d+1:H}.
$$

已执行 prefix 不参与随机 flow，而是作为条件输入。对不同 $d$，可采用：

- 固定最大长度加 validity mask；或
- rolling horizon，在尾部读取更多未来 action 以保持长度固定。

训练目标为 conditional flow-matching loss，仅作用于 valid free suffix。

对每个 batch，形式化步骤如下：

1. 从 LIBERO demonstration 采样 trajectory 起点 $t$；
2. 读取 $o_{t:t+D_{max}}$ 和 $a_{t:t+H-1}$；
3. 随机采样 delay $d\in[0,D_{max}]$；
4. 构造严格因果的信息集 $\mathcal F_d$，禁止使用 $o_{t+d+1:}$；
5. 用 PI0.5 VLM 编码 $o_t$ 和 language，得到 $z^{slow}$；
6. 用 DINOv3 编码 $o_t$ 与 $o_{t+d}$，经 residual resampler 得到 $z_d^{fast}$；
7. 采样 flow time $\tau$ 和 base noise $\epsilon$；
8. 对 $j<d$ 的 elapsed prefix 保持 clean，对 $j\ge d$ 的 suffix 加噪；
9. action expert 预测 velocity；
10. 只在 $j\ge d$ 的 valid suffix 上计算 flow-matching loss。

定义 mask：

$$
m_{d,j}
=
\mathbf 1[j\ge d].
$$

线性 interpolation 为：

$$
X_{\tau,j}
=
\begin{cases}
a_j,&j<d,\\
(1-\tau)a_j+\tau\epsilon_j,&j\ge d.
\end{cases}
$$

velocity target 为：

$$
u_{\tau,j}=\epsilon_j-a_j.
$$

Stage A loss 为：

$$
\mathcal L_A
=
\frac{
\sum_j m_{d,j}
\|v_{\theta,j}-u_{\tau,j}\|^2
}{
\sum_jm_{d,j}
}.
$$

Stage A 的目的只有一个：让同一个 PI0.5-based flow 对任意 $d$ 都能表示 $p(A_{d:H-1}\mid\mathcal F_d)$。

### Stage B：精确 re-anchoring pair 生成

冻结或使用 EMA conditional-flow teacher。对相邻 filtrations $\mathcal F_s$ 和 $\mathcal F_{s+1}$：

1. 采样共享 base noise $Z$；
2. 在旧条件下积分到随机 $\tau$，得到 $X_\tau^-$；
3. 在新条件下使用相同 $Z$ 积分到同一 $\tau$，得到 $X_\tau^+$；
4. 保存 pair、filtration delta、flow time 和 masks。

pair 可以在线生成，也可以离线缓存。需比较生成成本、存储成本和 teacher checkpoint 漂移。

更具体地，对相邻信息集：

$$
\mathcal F_d
=
\{o_{0:d},a_{0:d-1}\},
$$

$$
\mathcal F_{d+1}
=
\{o_{0:d+1},a_{0:d}\},
$$

二者新增的闭环信息是：

$$
\Delta\mathcal F_d=\{a_d^{exec},o_{d+1}\}.
$$

使用同一个 base noise $Z$ 生成：

$$
X_\tau^-
=
\Phi_\theta^{\tau\leftarrow0}(Z\mid\mathcal F_d),
$$

$$
X_\tau^+
=
\Phi_\theta^{\tau\leftarrow0}(Z\mid\mathcal F_{d+1}).
$$

$X_\tau^+$ 是 posterior updater 的监督目标。位置 $0:d$ 在更新后已成为 elapsed prefix，程序直接用真实动作覆盖；teacher/student loss 只作用于 $j>d$ 的剩余 suffix。

### Stage C：innovation distillation

训练 $T_\phi$ 或 $g_\phi$ 将 $X_\tau^-$ 更新为 $X_\tau^+$。损失包括：

$$
\mathcal L
=
\mathcal L_{bridge}
+\alpha\mathcal L_{endpoint}
+\beta\mathcal L_{committed}
+\gamma\mathcal L_{boundary}
+\eta\mathcal L_{teacher\ consistency}.
$$

需要对新 observation 做随机 dropout，使 student 在没有新帧时退化为 identity update。

推荐 student 预测 residual：

$$
\Delta X_\tau
=
T_\phi
(X_\tau^-,\tau,o_{d+1},a_d^{exec},d),
$$

$$
\widehat X_\tau^+
=
X_\tau^-+r_d\odot\Delta X_\tau,
$$

其中：

$$
r_{d,j}=\mathbf 1[j>d].
$$

基本蒸馏损失为：

$$
\mathcal L_C
=
\frac{
\sum_jr_{d,j}
\|\widehat X_{\tau,j}^+-X_{\tau,j}^+\|^2
}{
\sum_jr_{d,j}
}.
$$

Stage C 的目的不是重新学习 action posterior，而是逼近 Stage A 定义的 exact re-anchoring operator。

### Stage D：短闭环 unroll

联合或交替训练 generation 与 innovation modules，随机 unroll 若干事件：

```text
generation step
observation arrival
prefix growth
re-anchoring
generation step
```

不必一开始 unroll 完整 50 steps，但训练时的事件顺序、mask 更新和推理 runtime 必须一致。

Stage D 只作为最终小学习率微调。建议固定或使用 EMA Stage-A teacher，避免 posterior target 随联合训练漂移。

### 4.1 当前代码中的两阶段映射

当前 `online_filtering_pi05` policy 将上述 Stage B pair generation 和 Stage C
innovation distillation 合并为第二个可执行训练模式：

```text
training_stage=stage_a
    训练 filtration-conditioned PI0.5 flow、fast resampler 和 expert adapters

training_stage=stage_b
    冻结完整 Stage-A teacher
    在线生成 shared-base teacher pairs
    只训练 posterior_updater 和 posterior_fast_proj
```

这里的合并只是工程上的 online pair generation，不改变监督目标。每个 Stage-B batch：

1. 选择同时有效的相邻 filtration $d,d+1$；
2. 采样一次共享 base noise $Z$ 和一次 PI0.5 flow time $\tau$；
3. 按 PI0.5 的时间约定，从噪声端 $t=1$ 分别积分到同一个 $\tau$；
4. 得到 $X_\tau^-$ 和 $X_\tau^+$；
5. 用真实 $a_d^{exec}$ 硬覆盖新 committed coordinate；
6. updater 只允许修改 $j>d$；
7. 在真实 action dimensions 和 valid suffix 上计算 endpoint loss。

实现的总损失为：

$$
\mathcal L_B
=
\lambda_e\mathcal L_{endpoint}
+\lambda_c\mathcal L_{committed}
+\lambda_b\mathcal L_{boundary}.
$$

其中 committed loss 是结构不变量的诊断项：updater 会直接提交 $a_d^{exec}$，并通过
mask 保证更早的动作不变。boundary loss 监督第一个仍未执行的位置 $d+1$。随机
`posterior_identity_dropout` 会把新 observation 替换为旧 filtration，并把 target 设为
hard-committed old particle，从而训练“无新信息时为 identity”的性质。

Stage-A checkpoint 作为 Stage-B 的 `policy.pretrained_path`。从普通 PI0.5 或 Stage-A
checkpoint 加载 processor 时，代码会强制 state prompt 使用时间窗口第 0 帧，避免
`state[:, -1]` 引入未来信息泄漏。

关键配置：

```text
policy.training_stage=stage_a|stage_b
policy.teacher_integration_steps=10
policy.posterior_hidden_dim=256
policy.posterior_num_heads=8
policy.posterior_num_layers=2
policy.posterior_endpoint_weight=1.0
policy.posterior_committed_weight=1.0
policy.posterior_boundary_weight=0.1
policy.posterior_identity_dropout=0.1
```

### 4.2 本地真实数据 smoke test

仓库提供：

```text
research/scripts/smoke_test_online_filtering_pi05.py
```

它执行以下检查：

1. 从本地 `lerobot/libero` 解码真实两路 AV1 observation window 和 50-step action chunk；
2. 使用 LIBERO mean/std 归一化 state/action；
3. 严格加载本地 PI0.5 checkpoint 和 DINOv3 Small；
4. Stage A forward/backward，并验证 fast adapters 获得 finite gradients；
5. 在同一模型上冻结 Stage-A teacher；
6. Stage B shared-$Z$ teacher generation 与 updater backward；
7. 验证 teacher 没有梯度、committed loss 为零。

运行命令：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 python \
  research/scripts/smoke_test_online_filtering_pi05.py \
  --dataset-root /data4/yanghuoren/dataset/libero \
  --pi05-path /data4/yanghuoren/pretrain/pi05_base \
  --dinov3-path /data4/yanghuoren/pretrain/dinov3-small \
  --device cuda \
  --max-delay 2 \
  --teacher-steps 1
```

2026-08-26 的实测结果：

```text
Stage A loss:                 1.1413328648
Stage A adapter grad tensors: 84
Stage B loss:                 0.0193795674
Stage B endpoint:             0.0075417333
Stage B committed:            0.0
Stage B boundary:             0.1183783412
Stage B updater grad tensors: 31
Frozen teacher grad tensors:  0
```

该脚本使用合成的合法 token IDs，以便把模型/数据/损失 smoke 与 gated PaliGemma
tokenizer 的外部访问权限分开。标准 `lerobot-train` 仍使用真实语言 tokenizer，并可通过：

```text
policy.tokenizer_name=/path/to/local/paligemma-tokenizer
```

指定本地 tokenizer。若从原 PI0.5 checkpoint 加载 processor，训练入口也会把这个配置
覆盖进反序列化后的 `tokenizer_processor`。

## 5. 异步 LIBERO runtime

标准同步 `lerobot-eval` 不足以验证该方法，需要专用 runner。

### 5.1 Control/environment loop

- 以固定 simulated control frequency 调用 `env.step`；
- 从 thread-safe action queue 读取旧 chunk 动作；
- 写入 observation ring buffer；
- 为 action、observation 和 simulator step 记录单调 timestamp；
- 推理未完成时绝不暂停环境。

### 5.2 Inference loop

- 在指定 trigger step 获取初始 observation；
- 计算 slow semantic tokens；
- 启动 action flow；
- 每个 solver step 读取最新 fast feature；
- 检测 elapsed-prefix growth 和 observation arrival；
- 执行 re-anchoring student update；
- 推理完成时以实测 delay $d$ 裁掉 $X_{1:d}$；
- 从 $X_{d+1}$ 原子替换 action queue 的可替换部分。

### 5.3 Reset contract

每个 episode reset 时必须清空：

- action queue；
- observation ring buffer；
- VLM/fast-feature cache；
- elapsed-prefix state；
- latency tracker；
- inference worker 的 pending result；
- RTC/debug state。

## 6. 配置项

建议新增：

```text
policy.max_inference_delay
policy.observation_history
policy.fast_encoder_type
policy.fast_token_count
policy.reanchor_mode          # exact | map | flow
policy.reanchor_steps
policy.reanchor_loss_weight
policy.prefix_conditioning
policy.random_frame_drop
policy.random_latency_min
policy.random_latency_max
policy.use_shared_base_coupling
eval.async_control_frequency
eval.synthetic_inference_delay
eval.record_timestamps
```

## 7. 测试

### 单元测试

- observation/action 的 off-by-one 对齐；
- episode boundary padding；
- elapsed mask 随 delay 单调增长；
- $d=0$ 和 $d=H-1$；
- completion 后从 $X_{d+1}$ 执行；
- re-anchoring 不修改 elapsed prefix；
- 无新 observation 时 update 为 identity；
- reset 不泄漏上一 episode 状态。

### 数值测试

- Gaussian conditional flow 上验证 re-anchoring 后的 mean/covariance；
- inverse-forward reconstruction error；
- ODE step 数与 posterior transport error；
- student 与 exact teacher 的 Wasserstein/MMD 距离；
- 多次 observation events 后的误差累积。

### Benchmark

- synchronous policy；
- naive asynchronous policy；
- RTC；
- Training-Time RTC；
- condition switching without re-anchoring；
- exact inverse-forward re-anchoring；
- distilled re-anchoring map；
- distilled innovation flow；
- full restart from the latest observation。

除标准 LIBERO 外，需要加入 action noise、camera delay/dropout、目标扰动或 LIBERO-plus，以显式测量闭环新观测的价值。

## 8. 实现顺序

1. 为 PI0.5 dataset config 增加多时刻 observation delta indices；
2. 实现 batch timeline、elapsed mask 和 episode-valid mask；
3. 实现 fast observation/action-prefix encoders；
4. 训练 filtration-conditioned flow family；
5. 在 Gaussian toy problem 验证 exact re-anchoring；
6. 用 conditional-flow teacher 生成 paired posterior samples；
7. 蒸馏 re-anchoring student；
8. 实现 asynchronous LIBERO runner；
9. 进行标准 LIBERO 和扰动 benchmark；
10. 加入短闭环 unroll 和误差传播实验。

## 9. Implementation Readiness Audit

本节记录在当前 LeRobot 工作区中开始实现之前的实际状态。数据集和 pretrained weights 不计入缺口。

### 9.1 已具备的基础设施

- PI0.5 已实现 50-step flow-matching action expert；
- PI0.5 已支持 Training-Time RTC clean prefix、per-action flow time 和 postfix-only loss；
- PI0.5 已有 slow-prefix KV cache inference path；
- dataset factory 已支持 image/state modality-specific delta indices 和正时间 offset；
- RTC 已有 thread-safe action queue、latency tracking、guidance 和 debug utilities；
- LIBERO environment、processor、训练文档和 evaluation entry point 已存在；
- 当前 Transformers 版本包含 `DINOv3ViTModel` 和 `AutoImageProcessor`；
- DINOv3 ViT-S/16 不需要新增独立 Python package，可通过 Transformers 加载。

### 9.2 当前 Conda 环境状态

Conda executable：

```text
/data4/yanghuoren/conda/condabin/conda
```

目标环境：

```text
/data4/yanghuoren/conda/envs/lerobot
```

已确认：

```text
Python        3.12.0
PyTorch       2.11.0+cu130
Transformers  5.5.4
DINOv3 API    available
torchcodec    available
```

尚未满足：

- 环境中的 `lerobot 0.6.1` 不是当前 workspace 的 editable install；
- `hf-libero` 未安装；
- `pytest` 未安装；
- `ruff` 未安装；
- 当前执行会话中 `torch.cuda.is_available()` 为 false；
- 当前执行会话中 `nvidia-smi` 无法连接 NVIDIA driver。

开始开发前需要安装当前工作区及开发依赖。遵循项目约定，优先在目标 Conda 环境中使用 `uv`/editable workflow；等价依赖范围至少覆盖：

```text
.[pi,libero,test,dev]
```

GPU 状态必须在实际训练节点单独复核；如果只是当前受限容器不可见，不影响 CPU 单元测试和代码编写。

### 9.3 需要新增的代码模块

建议保持原始 PI0.5 parity path 不变，通过 opt-in 配置或独立 policy package 添加功能。尚需实现：

1. **Config**
   - max overlap delay；
   - future observation delta indices；
   - DINOv3 model id、freeze/unfreeze 策略；
   - fast token count 和 adapter layer indices；
   - Stage A/Stage C training mode；
   - posterior updater 类型和 checkpoint；
   - async runtime 参数。
2. **Batch timeline builder**
   - 随机采样 $d$；
   - 构造 $\mathcal F_d$；
   - 生成 elapsed/free/valid masks；
   - 处理 episode boundary 和 off-by-one 对齐。
3. **DINOv3 fast encoder**
   - 两路 camera preprocessing；
   - initial/latest feature extraction；
   - CLS/register token stripping；
   - camera/2D position embeddings；
   - residual feature construction；
   - 8--16 token resampler。
4. **PI0.5 fast adapters**
   - 在 expert joint-attention residual 与 MLP 之间插入；
   - action-time causal mask；
   - zero-init gate/output projection；
   - per-layer fast KV cache；
   - 保持未启用时与原 PI0.5 数值路径一致。
5. **Stage-A loss path**
   - 将当前 RTC 随机 prefix 扩展为与 observation filtration 对齐的 prefix；
   - suffix-only flow loss；
   - DINOv3/latest-observation conditioning；
   - delay/frame-drop randomization。
6. **Exact re-anchoring teacher**
   - 保存共享 base noise $Z$；
   - 在 $\mathcal F_d$ 和 $\mathcal F_{d+1}$ 下积分到相同 $\tau$；
   - 生成 masked $(X_\tau^-,X_\tau^+)$ pairs；
   - 支持在线生成和离线 cache。
7. **Posterior updater**
   - 接收旧 action particle、新 observation、新执行 action、flow time 和 masks；
   - 预测 suffix residual；
   - hard-preserve elapsed prefix；
   - 支持 one-step map 或少步 innovation flow。
8. **Training entry point**
   - Stage A 与 Stage C 的 mode/config dispatch；
   - Stage C 加载并冻结 Stage-A teacher；
   - 独立 optimizer parameter groups；
   - checkpoint 中保存 teacher/student/DINO/resampler 配置。
9. **Asynchronous LIBERO runner**
   - control/environment worker；
   - inference worker；
   - observation ring buffer；
   - action queue atomic replacement；
   - elapsed delay 和 timestamp tracking；
   - episode reset contract。
10. **Tests**
    - config/delta indices；
    - timeline/mask；
    - DINO token shape/cache；
    - adapter identity at zero initialization；
    - frozen prefix；
    - teacher shared-$Z$ pairing；
    - Gaussian posterior re-anchoring；
    - async runner 的第 $d+1$ 动作接管。

### 9.4 开始编码前仍需固定的设计决定

大部分方法已经形式化，但以下决定会直接影响文件结构和 checkpoint compatibility，编码前应明确：

1. **Policy registration**：扩展 `pi05` 的 opt-in config，还是注册独立的 `online_filtering_pi05` policy type；
2. **Posterior updater architecture**：one-step residual Transformer，还是少步 flow；
3. **Stage-B pair generation**：训练时在线生成，还是先离线缓存；
4. **Suffix layout**：固定 50-position tensor 加 mask，还是随 $d$ 缩短并在尾部滚动补齐；
5. **Async LIBERO delay**：以真实 wall-clock、人工 simulator steps，还是两种模式都支持；
6. **第一里程碑范围**：先完成 Stage A 和 exact re-anchoring，还是一次性包含 distilled updater 和 async runner。

推荐默认决策：

```text
policy type:          independent online_filtering_pi05
updater:              one-step residual Transformer
teacher pairs:        online generation first, optional cache later
suffix layout:        fixed H=50 tensor with elapsed/free masks
async delay:          deterministic simulated-step mode + measured wall-clock mode
first milestone:      Stage A + exact re-anchoring + Gaussian tests
```

采用这些默认值后，方法层面没有必须继续澄清的阻塞项，可以开始编写配置、timeline builder 和单元测试。
