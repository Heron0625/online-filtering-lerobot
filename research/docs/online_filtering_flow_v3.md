# Online Filtering Flow v3：基于真实观测的结构化旧计划扰动

> 历史方案说明：本文记录含人工缺陷的 v3 设计与实验。当前 `online_filtering_pi05_clean_causal` 实验已暂停人工缺陷：Stage A 只用真实轨迹的 clean flow，控制步数 `d` 同时驱动观测嵌入、committed-prefix 硬固定、因果 attention 和 suffix loss mask；Stage B 的 old/new teacher 与 updater 使用相同的 prefix 约束。以下 corruption/switch-flow 章节不再描述当前训练配置。

## 1. 文档目标

v3 不再假设干净 demonstration 中相邻观测天然产生显著不同的 action posterior。它把问题明确改写为：

> 给定一个存在结构化缺陷的旧计划，以及数据集中与干净专家动作对齐的新图像和 proprio，学习把旧 action particle 修复到干净 action posterior。

该方案第一阶段不依赖 simulator rollout，也不在每个训练 step 运行两个完整 VLM teacher。真实图像、proprio 和目标动作都来自数据集；人工构造的只是旧计划误差通道。

本文重点回答四个问题：

1. 什么样的旧计划 corruption 在概率上成立；
2. 为什么不能只独立扰动已经过去的前 $d$ 个动作；
3. corruption 应来自手工模型、官方 PI0.5、Stage A，还是它们的组合；
4. Stage A 和 Stage B 应如何修改并验证。

## 2. v2 实验暴露的问题

相同的 40-task、动态 delay $1\ldots10$、单 rollout 协议得到：

| 方法 | Spatial | Object | Goal | LIBERO-10 | 总成功率 |
|---|---:|---:|---:|---:|---:|
| official PI0.5 | 9/10 | 9/10 | 8/10 | 8/10 | 85.0% |
| Stage A | 6/10 | 7/10 | 10/10 | 8/10 | 77.5% |
| Stage B，实际启用 bridge | 5/10 | 7/10 | 7/10 | 4/10 | 57.5% |

Stage B 训练后期的典型量级为：

```text
teacher_delta_mse = 4e-5 ... 1.7e-4
student_delta_mse = 1e-6 ... 4e-6
endpoint_loss     ≈ teacher_delta_mse
```

这表示 frozen Stage-A teacher 在新旧 fast condition 下只产生很小的差异，而 student bridge 大部分时间接近 identity。当前 Stage B 既没有显著复现 teacher delta，又会在长时闭环中累积小扰动。

根因来自当前 Stage A 数据构造：同一干净 action chunk 配合不同 delay 和不同 fast observation，target 始终相同。在标准 flow-matching MSE 下，忽略 fast condition 是一个合理的低风险解。

## 3. v3 的结论边界

不经过 simulator 的方案可以严格定义为一个 conditional inverse problem，但不能声称它已经精确模拟了物理执行误差。

它解决的是：

```text
错误或过时的 action plan
        +
与正确专家动作对齐的真实 image/proprio
        ↓
恢复干净 action plan
```

它暂时不严格解决：

```text
错误动作真实执行
        ↓ robot/environment dynamics
偏离后的物理状态和图像
        ↓
恢复动作
```

前者可完全使用离线数据实现，并适合验证 posterior correction 是否可学；后者最终仍需要 simulator 或真实机器人扰动评测确认。训练不必一开始就依赖 simulator，但论文或报告中必须区分这两个 claim。

## 4. 概率模型

定义：

- $C_0$：旧的 slow context，包括 instruction、初始图像和 proprio；
- $C_d=(I_d,q_d,d)$：时刻 $d$ 的真实 fast image、proprio 和 delay；
- $A^+$：数据集中的干净专家 action chunk；
- $A^-$：人工构造或模型产生的旧计划；
- $q_\psi(A^-\mid A^+,C_0,d)$：旧计划 corruption channel。

Stage B 要学习：

$$
p(A^+\mid A^-,C_0,C_d,d).
$$

联合分布写成：

$$
p_{\mathrm{data}}(C_0,C_d,A^+)
q_\psi(A^-\mid A^+,C_0,d).
$$

因此，只要明确给出并验证 $q_\psi$，不通过 simulator 仍然是数学上完整的训练问题。风险不在于它“不合法”，而在于：

$$
q_\psi(A^-\mid A^+,C_0,d)
\not\approx
p_{\mathrm{deploy}}(A^-\mid A^+,C_0,d).
$$

也就是训练 corruption 与部署时旧计划误差不匹配。

## 5. 为什么不能只独立扰动前 $d$ 步

将动作块写成：

$$
A=(A_{1:d},A_{d+1:H}).
$$

如果只构造：

$$
A^-_{1:d}=A^+_{1:d}+\epsilon,
\qquad
A^-_{d+1:H}=A^+_{d+1:H},
$$

那么部署时真正执行的 suffix 已经是干净的。对 suffix 的最优 bridge 是 identity：

$$
T_\phi(A^-_{d+1:H},C_d)=A^-_{d+1:H}=A^+_{d+1:H}.
$$

如果反而在完整 chunk 上计算 loss，模型主要学会修复即将被丢弃的 prefix；这与控制目标不一致。

要让前缀缺陷在统计上影响 suffix，corruption 必须具有跨时间相关性。引入 episode 内共享潜变量：

$$
z\sim p(z),
\qquad
\epsilon_j=G_j(d)z+\eta_j.
$$

于是：

$$
\operatorname{Cov}(\epsilon_j,\epsilon_k)\ne0.
$$

一个最小例子是衰减 bias：

$$
A_j^-=A_j^++w_j(d)b+\eta_j,
\qquad b\sim\mathcal N(0,\Sigma_b),
$$

$$
w_j(d)=
\begin{cases}
1,&j\le d,\\
\exp(-(j-d)/\tau),&j>d.
\end{cases}
$$

这不是精确动力学，但它明确表达了“早期控制偏差会持续影响旧计划的后续部分”。

## 6. Corruption 来源比较

### 6.1 仅使用手工结构化 corruption

形式包括：

- 低频 additive bias；
- action gain；
- 时间平移、动作重复或丢失；
- position/rotation 分组的低秩扰动；
- gripper 指令延迟；
- 随 horizon 衰减的累计误差。

优点：成本低、强度可控、可覆盖部署错误的分布尾部，而且不依赖某个 checkpoint。

缺点：容易偏离真实 policy error，bridge 可能只学会人工噪声纹理，也很难凭直觉确定跨 action 维和跨时间 covariance。

因此手工 corruption 适合作为覆盖分布尾部的组成部分，不宜成为唯一来源。

### 6.2 使用 official PI0.5 输出

定义对齐后的 residual：

$$
R_{\pi05,d}
=
\operatorname{Align}\left(A_{\pi05}(C_0),d\right)-A^+.
$$

优点：

- official PI0.5 是系统 slow backbone 的来源；
- 它在 $C_0$ 下产生的计划接近真实推理启动时的初始计划；
- 不受当前 Stage-A fast branch 被忽略的问题污染；
- residual 是模型自然产生的，而不是人为选择形状。

缺点：

- official checkpoint 同样在 LIBERO 数据分布上训练，训练集 residual 可能过小；
- 它不包含新 fast branch、delay condition 和 bridge 的误差；
- sampling noise 与系统误差混在一起，必须固定或共享 noise 后再统计；
- official PI0.5 输出不等于动作在真实物理系统中的执行结果。

因此 PI0.5 residual 更适合代表“初始计划误差”，而不是完整的物理执行误差。

### 6.3 使用当前 Stage A 输出

定义：

$$
R_{A,d}
=
\operatorname{Align}\left(A_{\mathrm{StageA}}(C_0,C_d,d),d\right)-A^+.
$$

优点：

- 与实际送入 Stage-B bridge 的模型分布最接近；
- 包含 action expert、fast adapter、delay embedding 和当前积分器的联合误差；
- 可以直接测量部署模型最常见的 failure modes。

缺点：

- 当前 Stage A 的 fast condition 已被证明可能被系统性忽略；
- 用它作为唯一 corruption 来源会把当前设计缺陷固化进下一阶段；
- Stage A 一旦重新训练，旧 residual distribution 会过时；
- 同一训练集上的 residual 可能同样很小；
- 它仍然是模型计划误差，不是 physical execution error。

因此不能把 Stage A 输出视为 ground-truth teacher。它只能作为 deployment-matched negative/input distribution。

### 6.4 GT 与模型输出的角色不能互换

v3 固定：

```text
A+ / clean target：始终以数据集 GT 为锚点
PI0.5 或 Stage A：只用于生成 A- 或拟合 corruption statistics
```

不能把 Stage A 输出作为 $A^+$，否则 Stage A 忽略 fast condition 的问题会再次成为 Stage B 的监督上限。

## 7. 当前决定：第一版只使用规则化人工 corruption

第一版不混入 official PI0.5 residual 或旧 Stage-A residual。这样做不是断言人工 corruption 最接近部署分布，而是先隔离变量，验证“新 condition 能否修正旧 particle”这一核心机制。corruption channel 固定为：

$$
q_\psi(A^-\mid A^+,d)
=
\sum_{k=1}^{K}\pi_kq_k(A^-\mid A^+,d),
$$

其中 $q_k$ 是下文定义的结构化人工规则。模型 checkpoint 的 residual 只保留为后续 distribution-gap audit，不进入第一版训练数据。Identity 情形用于独立验证，不设置与 clean flow 数学重复的训练分支。

所有 corruption 都在 action normalization 之后生成，并用 suffix RMS 统一标定强度：

$$
r=
\sqrt{
\frac{1}{(H-d)D}
\sum_{j>d}\|A_j^- - A_j^+\|_2^2
}.
$$

第一版将 $r$ 分为 mild、medium、strong 三档；具体边界应先根据训练集 normalized action 的分位数确定，不能直接用原始机器人单位拍常数。

### 7.1 Clean/switch 必须共享 noise coupling

同一个训练 pair 的 clean endpoint、人工旧计划和 switch particle 必须使用同一个 $Z$。否则 $X_s^-$ 与对应的 clean path 同时混入 sampling noise 差异，模型会把随机采样差异误认为需要由新观测修正的 innovation。

至少保存：

```text
sample_id, task_id, t0, delay, switch_time
corruption_type, corruption_strength, noise_seed
A_gt, A_corrupted, suffix_mask
image/proprio keys
```

### 7.2 Corruption 不能只记录一个标量 MSE

需要统计：

- 每个 action 维度的 bias 和 RMS；
- $H\times H$ 时间 covariance；
- action-dimension covariance；
- residual PCA spectrum；
- delay-conditioned residual；
- success/failure-conditioned residual；
- suffix-only residual；
- gripper timing error。

若 residual 的前几个 PCA mode 能解释大部分方差，就用低秩 mode 采样：

$$
R\approx\sum_{k=1}^{K}\alpha_kU_k.
$$

这比逐 token 独立 Gaussian noise 更接近真实模型错误。

## 8. 一个更强的无 simulator 数据来源：跨 demonstration coupling

LIBERO 同一 task 通常有多条成功 demonstration。可以在同一 task 内寻找相近但不完全相同的真实状态：

$$
(C_0^{(i)},C_d^{(i)},A^{(i)})
\quad\text{和}\quad
(C_0^{(j)},C_d^{(j)},A^{(j)}).
$$

要求 instruction/task 相同、初始 phase 相近、$q_0^{(i)}$ 与 $q_0^{(j)}$ 接近、当前状态存在可控差异，而且两条轨迹最终都成功。

构造：

```text
old plan：trajectory i 的旧计划
new image/proprio：trajectory j 的真实观测
clean target：trajectory j 的真实未来动作
```

这种 cross-trajectory pair 仍不是严格反事实轨迹，但它同时使用真实 observation 和真实 recovery-like action，比“同一轨迹加独立白噪声”更接近状态 innovation。匹配必须使用 task phase 或 dynamic time warping，不能随机拼接两条轨迹。

该来源应作为独立消融，不应在第一版直接与所有 corruption 混在一起，否则难以判断收益来自哪里。

## 9. Stage A v3

### 9.1 独立于 Stage B 的唯一职责

先完全不考虑 bridge。Stage A 的目标是得到一个 **filtration-conditioned action flow**：给定推理触发时刻的 slow context，以及物理时刻 $d$ 已经获得的最新 image/proprio，它能够从标准 Gaussian noise **重新生成**当前 filtration 下正确的 action posterior。

记：

$$
C_d=(P_0,I_d,q_d,d),
$$

其中 $P_0$ 是触发时刻计算并缓存的 slow prefix。Stage A 定义：

$$
p_A(A_{d+1:H}\mid C_d),
$$

以及对应的 flow kernel：

$$
K_d^t(X_d^{t-1}\mid X_d^t,C_d).
$$

它的独立使用方式是：

```text
取得最新 filtration C_d
从新的 Gaussian noise Z 开始完整运行 action flow
得到 new chunk
丢弃已经过期的前 d 个位置
执行 suffix d:H
```

因此 Stage A 是一个允许“每次新观测到达后从头重算”的 reference policy。只要计算预算无限，它本身就应该能够完成控制。Stage B 后续只是为了避免每次 filtration 更新都从噪声重新积分；Stage B 不应反过来定义 Stage A 学什么。

### 9.2 Stage A 的两项职责

Stage A 同时承担：

1. **clean generation**：从 fresh Gaussian noise 和当前 condition 生成干净动作 posterior；
2. **condition switch**：去噪中途收到新 image/proprio 后，将人工旧计划产生的 particle 转向干净 GT endpoint。

$A^-$ 不必作为额外 token 显式输入；在 switch 分支中，它通过当前 action particle 进入 action expert。Stage A 不应学习“在旧 condition 下输出人工坏动作”，人工缺陷只定义 switch 分支的起始 particle 分布，所有监督终点仍为 GT $A^+$。

在无扰动的成功 demonstration 中，新观测只是确认系统仍沿原轨迹运动，因此：

$$
p_A(A_{d+1:H}\mid C_d)
\approx
p_A(A_{d+1:H}\mid C_{d-1})
$$

完全可能是正确结果。Stage A 的职责是“条件正确”，不是“条件变化时必须输出不同动作”。

### 9.3 Action 坐标与 clean GT target

Stage A 保持 trigger-time action 坐标。完整 GT chunk 为：

$$
A^+=(A_1^+,\ldots,A_H^+).
$$

物理 delay 为 $d$ 时，位置 $1\ldots d$ 已经执行，真正需要建模的是：

$$
A_{d+1:H}^+.
$$

clean flow path 为：

$$
X_+^t=\alpha_tZ+(1-\alpha_t)A^+,
\qquad
U_+^t=Z-A^+.
$$

基础 flow loss 保持：

$$
\mathcal L_{A,\mathrm{clean}}
=
\left\|V_\theta(X_+^t,t,C_d)-U_+^t\right\|^2_{j>d}.
$$

prefix 可以保留在 particle 中作为序列上下文，但主 loss 必须只作用于尚未执行的 suffix。这样不同 delay 虽然共享同一个完整 demonstration chunk，监督的有效随机变量并不相同：

$$
d=1:\quad A_{2:H},
$$

$$
d=5:\quad A_{6:H}.
$$

GT 始终是唯一 clean endpoint。Stage A、PI0.5 或后续 bridge 的输出都不能替代 GT 成为 $A^+$。

### 9.4 数学上错误的直接混用

下面的组合不是严格 flow matching：

$$
X^t=\alpha_tZ+(1-\alpha_t)A^-,
\qquad
U^t=Z-A^+.
$$

输入点位于通向 $A^-$ 的插值路径，velocity 却来自通向 $A^+$ 的另一条路径。它可以作为经验 off-manifold denoising loss，但不能继续解释为原始 conditional flow objective。

因此不能把 structured corruption 直接塞进标准 clean-flow input，同时复用 clean-flow velocity。必须把 clean flow 和 switch flow 定义为两条各自闭合的概率路径：

```text
clean flow：fresh noise + C+ → A+
switch flow：old/corrupted particle + C+ → A+
```

### 9.5 数学一致的 switch flow

令新观测在 flow time $s$ 到达，时间方向仍为 $t=1$ 噪声端、$t=0$ 动作端。由人工旧计划形成切换时刻的 particle：

$$
X_s^-=\alpha_sZ+(1-\alpha_s)A^-.
$$

从该点到干净 endpoint 定义局部路径：

$$
Y_t=
\frac{t}{s}X_s^-
+
\left(1-\frac{t}{s}\right)A^+,
\qquad 0\le t\le s.
$$

它满足 $Y_s=X_s^-$、$Y_0=A^+$，速度为：

$$
U_{\mathrm{switch}}
=
\frac{\mathrm dY_t}{\mathrm dt}
=
\frac{X_s^- - A^+}{s}.
$$

对应损失：

$$
\mathcal L_{A,\mathrm{switch}}
=
\left\|
V_\theta(Y_t,t,C_d,d,s)
-
\frac{X_s^- - A^+}{s}
\right\|^2_{j>d}.
$$

这里 input point、路径、endpoint 和 target velocity 来自同一条路径，因此仍是合法的 conditional flow matching。人工性只影响训练联合分布 $q_\psi$ 与部署分布的匹配程度，不破坏 flow 的数学定义。

### 9.6 人工缺陷的具体生成规则

LIBERO action 已核实为 7 维 relative control：前三维是 delta-EE position，随后三维是 delta-EE rotation，最后一维是 gripper。第一版仅扰动前六个连续维度，并保持 gripper GT 不变。每个样本只选择一种低频、跨时间相关的规则，不使用逐 token 独立高斯噪声。

#### A. 持续偏置与平滑衰减

采样组内共享方向 $b$ 和衰减尺度 $\tau_c$：

$$
A_j^-=A_j^++\lambda w_j(d)b,
$$

$$
w_j(d)=
\begin{cases}
1,&j\le d,\\
\exp(-(j-d)/\tau_c),&j>d.
\end{cases}
$$

对于 delta-EE action，这表示一段连续控制指令具有方向性偏置；其物理效果会随执行累积，因此必须使用 mild 为主的强度分布，并让偏置在 suffix 中衰减。

#### B. 低频 drift

采样 2--3 个时间基函数 $B_k(j)$ 和共享系数 $z_k$：

$$
\Delta A_j=\lambda\sum_kB_k(j)z_k,
\qquad
A_j^-=A_j^++\Delta A_j.
$$

$B_k$ 使用常数、线性项和一个低频正弦项，随后对 chunk 做平滑。该规则产生逐渐过冲或欠冲，而不是高频抖动。

#### C. 时间滞后

采样很小的整数 lag $\ell$，对有效 suffix 做插值移位：

$$
A_j^-=\operatorname{Interp}(A^+,j-\ell).
$$

边界位置使用最近有效动作，不循环回绕。它模拟旧 chunk 的相位落后。第一版限制 $\ell$ 很小，避免把任务阶段完全错位。

第一版不人工破坏 gripper。gripper 是近似离散的开合命令，不能与连续 EE 六维共用 RMS 缩放；若后续加入，应使用单独的事件时间模型和独立 loss/强度定义。

生成后执行四项过滤：

1. 已提交 prefix 不进入 loss，但 corruption 必须在 $j>d$ 的 suffix 上仍有非零能量；
2. 按 suffix RMS 将强度缩放到采样档位；
3. clip 到 normalized action 的合法范围；
4. 丢弃几乎 identity 或超过训练集合理动作范围的样本。

### 9.7 Stage A 的混合训练采样

原始干净数据和人工缺陷必须混在同一次 Stage-A 训练中，不能先训完 clean 再只训 defect，否则容易遗忘原策略。第一版建议：

```text
60% clean flow：其中 d=0 与 d>0 各占一半
40% corrupted switch flow：d ~ deployment delay distribution
```

总损失为：

$$
\mathcal L_A
=
\lambda_{\mathrm{clean}}\mathcal L_{A,\mathrm{clean}}
+
\lambda_{\mathrm{switch}}\mathcal L_{A,\mathrm{switch}}.
$$

采样比例和 loss 权重分开配置。起步令两种样本的单 token loss 权重相同，再根据 clean no-regression 和 condition gain 调整；不能因为 switch velocity 数值较大就让它支配全部梯度。

$d=0$ clean 样本负责保持原 PI0.5 能力；$d>0$ clean 样本训练 stale slow prefix 与当前 fast observation 的正常组合。所有分支都以 GT 为唯一 endpoint，并按 $j>d$ 做 suffix mask。

如果 episode 尾部不足以提供完整 chunk，必须使用有效 action mask，不能用 padding 参与 loss。

训练前期使用 mild 为主的 curriculum，随后增加 medium；strong 始终保持少数。第一版可从 corrupted switch batch 内 `mild/medium/strong = 60/30/10` 开始，但最终按 held-out 恢复性能调整。

### 9.8 Stage A 的独立验收标准

Stage A 完成后、训练 Stage B 之前，必须单独通过：

1. **$d=0$ no-regression**：标准 LIBERO 成功率不应显著低于初始化 PI0.5；
2. **delayed full-resampling**：对每个 $d$ 从 fresh noise 完整推理，再执行 suffix，闭环性能应稳定；
3. **时间对齐**：执行位置必须从 $d$ 开始，不能重复已执行 prefix；
4. **GT flow loss**：held-out suffix flow loss 随训练下降；
5. **condition correctness**：正确 fast condition 的 held-out loss 不高于 stale/shuffled condition；
6. **模态消融**：分别测 image-only、proprio-only 和 image+proprio。
7. **switch recovery**：相同人工旧 particle 下，正确 fast condition 必须优于 stale/shuffled condition；
8. **identity stability**：$A^-=A^+$ 时，switch 不应无故改变 clean endpoint。

Stage A 的核心产品不是一个 bridge teacher checkpoint，而是一个可独立运行的 reference family：

$$
\boxed{\{p_A(A_{d+1:H}\mid P_0,I_d,q_d,d)\}_{d=0}^{D_{\max}}}.
$$

### 9.9 Condition sensitivity 只是诊断，不是目标本身

诊断包括：

```text
velocity_delta_mse(correct fast vs stale fast)
loss(correct fast)
loss(shuffled fast)
fast-token gradient norm
adapter output/input RMS ratio
```

若 clean demonstration 中 fast condition 仍不重要，这本身可能是数据事实；switch 分支负责提供需要 correction 的训练情形，但必须通过 shuffled-condition 测试确认模型没有只学会固定的 corruption 逆变换。

但如果 shuffled fast condition 与正确 condition 在所有 held-out 样本上完全没有差别，则说明数据无法识别 fast branch，或实现没有把 fast token 有效注入 action expert。此时应先解决 Stage A，不能直接进入 Stage B。

### 9.10 Delay、dataset FPS 与 runtime control rate

当前实现中的 $d$ 是 **已过去/已提交的 action-control steps 数量**，不是抽象秒数。数据加载器将索引 $d$ 按数据集 FPS 转换为时间戳：

$$
\Delta t=\frac{d}{f_{\mathrm{dataset}}}.
$$

因此 suffix mask 使用零基索引 $j\ge d$；第 $d$ 项是新观测可用后第一项尚可执行的动作。该 mask 的依据应始终是 runtime 中实际已经提交了多少个 action，而不能仅由 wall-clock latency 乘训练 FPS 猜测。

现有 action chunk 同时编码了动作值和采集时的时间离散化。若 runtime 以不同 control FPS 直接逐项播放同一个 chunk，轨迹速度和 chunk 的物理 horizon 都会改变；这不要求推理调用频率与采集 FPS 完全相同，但 action execution rate 默认应与训练数据 FPS 一致。若要支持任意更慢或更快的执行速度，应显式加入 runtime $\Delta t$/control-rate condition，并对 observation/action trajectory 做时间重采样；不能只降低循环 sleep 而仍把 $d$ 当作原训练步数。

## 10. Stage B v3

Stage B 的职责不是在 action chunk 已经生成完毕后修补 endpoint，而是在一次 ODE
积分尚未结束时消费新观测。设新观测在离散去噪第 $k$ 步到达，PI0.5 使用
$N$ 个 Euler steps，则切换发生在严格位于区间内部的

$$
s_k=1-\frac{k}{N},\qquad k\in\{1,\ldots,N-1\}.
$$

updater 在同一个 $s_k$ 上把旧 condition 的 particle 重锚定到新 condition 的
particle；随后 action expert 使用新 image/proprio 和新 delay，从 $s_k$ 继续积分到
0。一次生成期间若继续收到更新的 packet，同一操作可以重复发生，不需要重新加载模型，
也不需要等当前 chunk 完成。

### 10.1 共享噪声的双 condition online teacher

在实际推理网格上采样 $s_k$ 和共享噪声 $Z$。冻结 Stage-A action expert，分别在
旧、新 fast condition 下从同一个 $Z$ 积分到同一个 $s_k$：

$$
X_{s_k}^{old}=\Phi_{1\rightarrow s_k}(Z;C_r),
$$

$$
X_{s_k}^{new}=\Phi_{1\rightarrow s_k}(Z;C_d).
$$

于是：

$$
\Delta_*^{s_k}=X_{s_k}^{new}-X_{s_k}^{old}.
$$

两个分支沿 batch 维合并，在 `no_grad` 下运行完整 Stage-A velocity field。slow prefix
共享；old/new fast image、proprio 和 delay 不同。Updater 显式接收带 role 的 old tokens、
new tokens 和 `new-old` innovation tokens。特别地，训练不采样 $t=0$：
completion-time bridge 与目标 runtime 不同，而且在 bridge 后没有剩余 action-expert
积分可以利用新 condition。

### 10.2 人工 defect 暂不混入

第一版只学习真实的 $X_{s_k}^{old}\rightarrow X_{s_k}^{new}$。不再用
$sZ+(1-s)(A^++\epsilon)$ 冒充 stale particle，也不添加独立人工 corruption。
先用 `teacher_delta_mse` 检验 Stage A 是否确实对新旧 fast condition 产生可学习的
轨道差异；只有 natural bridge 已经收敛后，才把 defect 作为独立增强分支加入。

### 10.3 Suffix residual bridge 与后续积分

定义：

$$
\widehat X_{s}^{new}=X_s^{old}+R_\phi(X_s^{old},C_r,C_d,r,d,s).
$$

Updater 的 condition tokens 显式包含 slow prefix、old fast、new fast 和 fast innovation。
没有可靠新信息时 residual 应接近零。

主要 loss：

$$
\mathcal L_{\mathrm{endpoint}}
=
\left\|\widehat X_{+,d:H}^t-X_{+,d:H}^t\right\|_2^2,
$$

$$
\mathcal L_{\Delta}
=
\left\|(\widehat X_+^t-X_-^t)_{d:H}-\Delta_{*,d:H}^t\right\|_2^2.
$$

为避免小 delta 被绝对 MSE 淹没：

$$
\mathcal L_{\mathrm{relative}}
=
\frac{\mathcal L_{\mathrm{endpoint}}}
{\left\|X_{+,d+1:H}^t-X_{-,d+1:H}^t\right\|_2^2+\epsilon}.
$$

clean identity 样本额外约束：

$$
\mathcal L_{\mathrm{id}}=\|T_\phi(X^t,C_d)-X^t\|_2^2.
$$

当前已实现的第一版只优化最直接、可辨识的同时间重锚定目标：

$$
\mathcal L_{B,v3}=\mathcal L_{\mathrm{endpoint}}.
$$

其中 10% 样本令 $r=d$，因共享噪声和相同 condition 而严格满足
$X_s^{old}=X_s^{new}$，使相同网络同时学习 identity。delta/relative/ranking
项保留为诊断或后续消融，不能在没有证据时同时加入并掩盖主目标。bridge 输出
$\widehat X_+^{s_k}$ 后，冻结的 Stage-A action expert 在新 condition 下继续完成
$s_k\rightarrow0$ 的普通 clean flow；因此无需为每个历史切换时刻再训练一套 expert。

### 10.4 Condition-use 约束与诊断

同一个 $X_-^t$ 分别输入正确和 batch-shuffled 的 $C_d$，首先监控：

$$
\operatorname{condition\_gain}
=
L_{\mathrm{shuffle}}-L_{\mathrm{correct}}.
$$

只有当该值长期接近零时，才考虑加入较弱 ranking loss。不能无条件强迫不同帧产生大 correction，因为相邻帧确实可能支持相同动作。

## 11. 人工 defect 暂不进入 Stage B 第一版

当前 Stage B 只使用 frozen Stage A 产生的 natural old/new condition pair。规则化人工
corruption、official PI0.5 residual 和额外 Stage-A residual 均保留为后续增强，不作为
第一版的训练前置条件。

### Experiment R0：输出与 GT 的差异到底有多大

在 held-out episodes 上，对 official PI0.5 和 Stage A 使用相同 noise seeds，统计：

```text
suffix action MSE
per-dimension RMS
temporal covariance
PCA explained variance
delay buckets
success/failure buckets
```

需要避免只在训练 episode 上测量，否则会低估误差。

### 后续 Experiment R1：source-only bridge

分别训练相同预算的：

```text
B-struct：仅 structured synthetic
B-pi05：仅 PI0.5 residual
B-stageA：仅 Stage-A residual
```

比较 clean no-regression 和 corrupted-plan recovery，不直接先训练 mixture。

### 后续 Experiment R2：扩展 corruption source

只有人工规则版已经证明机制有效、且 R0 显示明显覆盖缺口后，才考虑引入模型 residual，并将每种新来源作为独立消融。

预期角色是：

```text
PI0.5 residual：更稳定的初始计划误差基线
Stage-A residual：更接近实际 bridge 输入，但不可作为 clean target
structured noise：覆盖模型 residual 未覆盖的尾部
GT：唯一 clean anchor
```

## 12. 训练与评测分布必须匹配

Stage B v2 训练和 completion-time 评测不匹配。当前实现改为从有效 observation
window 中采样任意 $r<d$ 的 filtration pair，并将 bridge time 与部署的离散推理网格
对齐。单次 teacher pair 覆盖第一次 $0\rightarrow d$ 和后续 $r\rightarrow d$
的条件组合，但不等于已经监督或验证了连续 updater 组合后的 particle 分布。
训练与评测必须分别报告 gap 分桶：

```text
old/new：有效 observation 中任意 old < new，gap ∈ [stage_b_min_gap, stage_b_max_gap]
90%：natural old/new Stage-A particle pair
10%：identity（old delay = new delay）
```

flow time（当前 N=10）：

```text
t ∈ {0.9, 0.8, ..., 0.1}，等概率采样
t=0 和 t=1 均不采样
```

当前 LIBERO runner 已改为 `continuous_online_v1`：在每个 Euler step 之前推进
旧 chunk 执行、采集真实观测，并通过同一个 latest-ready buffer 消费连续 packet。
Stage A 仅更新 condition；Stage B 每次融合都调用 updater，然后继续去噪。
总 delay 动态采样，控制/flow 时钟采用确定性线性映射；不再随机指定单次 switch_step。
这是受控仿真时钟评测，不是墙钟异步性能测量。旧的单次切换结果不能直接混入新协议。
详细资源路径、启动方式和时钟定义见 [连续在线评测说明](libero_continuous_online_eval.md)。
真实异步入口同样在每个 Euler step 查询 buffer。后续应记录真实
packet-arrival flow-time histogram；若明显不均匀，再考虑按该经验分布采样。

## 13. 验收指标

### 13.1 训练指标

必须记录：

```text
identity_loss
bridge_endpoint_loss
bridge_relative_loss
bridge_gain = 1 - endpoint_loss / teacher_delta_mse
teacher_delta_mse
student_delta_mse
delta_cosine_similarity
relative_delta_norm
condition_gain
gate_mean / gate_by_corruption_strength
```

基本健康条件：

```text
bridge_gain > 0
endpoint_loss 显著小于 teacher_delta_mse
delta_cosine_similarity > 0
relative_delta_norm 不长期接近 0
condition_gain > 0
clean gate 接近 0
```

### 13.2 闭环评测

必须分开报告：

1. clean LIBERO：要求不显著低于 Stage A；
2. synthetic corrupted-plan LIBERO：验证 action repair；
3. image-only、proprio-only、image+proprio 消融；
4. delay $1\ldots10$ 分桶；
5. mild、medium、strong corruption 分桶；
6. 最终 simulator/real disturbance 测试，用于验证 physical recovery claim。

每 task 单 rollout 只适合作为开发检查。最终结论至少需要多 seeds、多 rollouts 和置信区间。

## 14. 推荐实施顺序

1. 实现低频、跨时间相关的人工 corruption 和 suffix RMS 分桶；
2. 为 Stage A 加 committed-prefix mask、clean/switch 两种合法路径及独立日志；
3. 在同一训练中混合 clean 和 corrupted switch batch，并将 identity switch 留作验证；
4. 先做短程训练，检查 clean no-regression、switch recovery、identity stability 和 condition gain；
5. 通过后完成 Stage-A 全量训练，并做 clean/corrupted-plan 双评测；
6. 训练 Stage B 的同-flow-time reanchoring，并由 Stage-A expert 在新 condition 下完成剩余积分；
7. 只有发现人工分布覆盖不足时，才审计 PI0.5/Stage-A residual 并作为后续独立消融；
8. 最后用 simulator 或真实系统验证 action corruption 是否转化为 physical recovery。

## 15. v3 核心结论

1. Stage A 同时训练 clean generation 和 condition switch，不需要运行两个完整 teacher；GT 始终是唯一 clean endpoint。
2. 不通过 simulator 也能定义数学上完整的 corrupted-plan posterior，但其 claim 是 plan repair，不是严格 physical recovery。
3. 只独立扰动前 $d$ 步不能监督可执行 suffix，必须使用跨时间相关的 corruption。
4. 第一版只使用可控的规则化人工 corruption，不混入 PI0.5 或旧 Stage-A 输出。
5. clean flow 和 switch flow 必须分别具有一致的 input path、endpoint 和 velocity target，不能交叉复用 target。
6. 原始 clean 数据和人工 defect 数据在同一次 Stage-A 训练中混合，identity switch 作为独立验证，防止 clean policy 退化。
7. Stage A 的成功标准不是绝对 loss 很小，而是 clean policy 不退化、defect particle 可恢复、identity 稳定，并且修正明确使用 image/proprio。
8. Stage B 只在严格内部 flow time 做重锚定；新观测到达后继续执行剩余去噪，并允许一次生成期间多次更新。
