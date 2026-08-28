# Online Filtering PI0.5：统一符号、概率分解与训练流程

> 本文只使用一套符号。上标表示 flow 时刻；下标 $d$ 表示物理时延和条件版本；动作位置使用 $j$。全文不再使用另一套 filtration 符号。

## 1. 物理时间与动作块

### 1.1 推理触发时刻

旧动作块已经执行了 $m$ 步时，机器人获得新观测并触发一次新动作块推理。将这一物理时刻定义为：

$$
d=0.
$$

触发时模型获得：

$$
(\text{text},q_0,I_0),
$$

其中：

- $q_0$ 是初始 proprioception；
- $I_0$ 是初始多相机图像的统称；
- text 是任务指令。

### 1.2 新动作块

新动作块长度为 $H$：

$$
A=(A_1,A_2,\ldots,A_H).
$$

动作下标 $j$ 表示相对本轮推理触发时刻的物理执行位置。对应关系为：

$$
A_j
\longleftrightarrow
\text{旧动作块执行到第 }m+j\text{ 步的物理时刻}.
$$

### 1.3 物理时延

当旧动作块已经执行到第 $m+d$ 步时，称当前物理时延为 $d$。此时：

$$
A_1,\ldots,A_d
$$

对应的物理时间已经过去，不能再由新动作块执行。下一条可以执行的动作是：

$$
A_{d+1}.
$$

本文先采用固定完成时延的简化设定：新动作块在 $d=D$ 时生成完成。因此：

$$
A_{1:D}
$$

不能执行，真正交给机器人的是：

$$
A_{D+1:H}.
$$

更一般地，$D$ 可以解释为训练支持的最大时延，而单次运行的实际完成时延记为 $D_{finish}\le D$。本文公式先使用固定 $D$，不影响方法本身。

---

## 2. Flow 时间

flow 推理按离散顺序运行：

$$
T,T-1,\ldots,1,0.
$$

其中：

$$
T:\text{噪声端},
\qquad
0:\text{动作端}.
$$

定义归一化 flow 系数：

$$
\alpha_t=\frac{t}{T}.
$$

因此：

$$
\alpha_T=1,
\qquad
\alpha_0=0.
$$

后续公式中使用整数 flow 时刻 $t$，线性插值使用 $\alpha_t$。这样既保留 $T\rightarrow0$ 的直观推理顺序，也不会把整数 $t$ 错误地直接代入 $[0,1]$ 插值公式。

---

## 3. Slow prefix 与当前条件

### 3.1 Slow prefix

初始输入经过 PaliGemma/Gemma：

$$
P
=
\operatorname{Gemma}
(\text{text},q_0,I_0).
$$

$P$ 称为 slow prefix。一次新动作块推理期间，$P$ 固定不变。

### 3.2 后续观测

在物理时延 $d$，机器人获得当前观测。为了不在理论公式中展开 DINOv3、resampler 和 proprio projection，统一将编码后的当前快速观测条件记为：

$$
I_d.
$$

这里的 $I_d$ 是“模型在物理时延 $d$ 可使用的 fast observation condition”，可以包含：

- 当前多相机图像特征；
- 当前 proprioception $q_d$；
- 相对初始图像的视觉变化；
- observation timestamp 或 age。

因此，后文中的 $I_d$ 不再专指未经编码的 RGB tensor。

### 3.3 条件变量 $C_d$

定义物理时延 $d$ 下模型的完整条件：

$$
C_d
=
\left(
P,
I_d,
d,
A_{1:d}^{exec}
\right).
$$

其中：

$$
A_{1:d}^{exec}
=
(A_1^{exec},\ldots,A_d^{exec}).
$$

当 $d=0$ 时，已执行动作前缀为空：

$$
A_{1:0}^{exec}=\varnothing.
$$

从此以后，所有当前条件都写成 $C_d$。不会再在推导中额外引入另一套信息集符号。

---

## 4. 动作粒子符号

定义：

$$
X_d^t
$$

表示：

> 在物理条件 $C_d$ 下，flow 时刻 $t$ 的完整动作块粒子。

其中：

- 上标 $t$：flow 时刻；
- 下标 $d$：物理时延和条件版本。

若需要表示粒子内部第 $j$ 个动作位置，则写为：

$$
X_{d,j}^t.
$$

动作位置 $j$ 和物理时延 $d$ 是不同的索引，不能都写在同一个下标含义中。

---

## 5. 原始 PI0.5

### 5.1 单次训练 forward

训练数据给出目标动作块：

$$
A=(A_1,\ldots,A_H).
$$

采样：

$$
Z\sim\mathcal N(0,I),
$$

以及 flow 时刻 $t$。构造：

$$
X^t
=
\alpha_tZ
+
(1-\alpha_t)A.
$$

线性 path 的连续 flow velocity target 为：

$$
U=Z-A.
$$

PI0.5 预测：

$$
V_\theta(X^t,t,P).
$$

训练 loss 为：

$$
\mathcal L_{PI05}
=
\mathbb E
\left[
\left\|
V_\theta(X^t,t,P)-(Z-A)
\right\|^2
\right].
$$

训练时每个 batch 只采样一个 $t$ 并执行一次网络 forward，不会从 $T$ 完整积分到 0。

### 5.2 Flow 视角

PI0.5 直接训练的是 velocity field：

$$
V_\theta(X^t,t,P).
$$

### 5.3 离散 transition 视角

数值求解器使用 velocity field 诱导出一步 transition：

$$
K^t
\left(
X^{t-1}\mid X^t,P
\right).
$$

PI0.5 不是显式预测 DDPM 高斯分布参数；这里的 $K^t$ 是 vector field 和 solver 共同诱导的 Markov kernel。对 deterministic Euler solver，它是一个 Dirac kernel。

---

## 6. Stage A：当前条件下的 generation kernel

### 6.1 Stage-A 粒子

在物理时延 $d$，前 $d$ 个动作已经执行。因此训练粒子为：

$$
X_{d,j}^t
=
\begin{cases}
A_j^{exec},&j\le d,\\
\alpha_tZ_j+(1-\alpha_t)A_j,&j>d.
\end{cases}
$$

也就是：

```text
X_d^t = clean executed prefix + noisy unexecuted suffix
```

### 6.2 Stage-A velocity

模型预测：

$$
V_\theta
\left(
X_d^t,t,C_d
\right).
$$

其中 $C_d$ 已经包含 slow prefix、当前 fast observation、物理 delay 和已执行动作前缀。

### 6.3 Stage-A loss

只在尚未执行的位置 $j>d$ 上计算：

$$
\mathcal L_A
=
\frac{
\sum_{j=d+1}^{H}
\left\|
V_{\theta,j}(X_d^t,t,C_d)-(Z_j-A_j)
\right\|^2
}{H-d}.
$$

实际实现还只保留真实 action dimensions，并排除 episode padding。

### 6.4 Stage-A 学到的 kernel

从离散 transition 视角，Stage A 学到：

$$
\boxed{
K_d^t
\left(
X_d^{t-1}\mid X_d^t,t,C_d
\right)
}
$$

它回答：

> 输入已经是属于当前条件 $C_d$ 的 flow 粒子 $X_d^t$ 时，怎样生成同一条件下的下一 flow 粒子 $X_d^{t-1}$？

它完成：

$$
(d,t)\longrightarrow(d,t-1).
$$

物理条件版本 $d$ 不变，只推进 flow 时刻。

---

## 7. 在线运行时真正拥有的粒子

当新条件 $C_d$ 刚刚到达时，模型手中通常不是 $X_d^t$，而是旧条件下已经生成到 flow 时刻 $t$ 的粒子：

$$
X_{d-1}^t.
$$

所以 Stage-A kernel 的标准输入：

$$
X_d^t
$$

在运行时并不存在。

如果直接把旧粒子送入 Stage-A kernel：

$$
K_d^t
\left(
X_d^{t-1}\mid X_{d-1}^t,t,C_d
\right),
$$

就违反了 Stage A 的训练输入分布，因为：

$$
X_{d-1}^t
\not\sim
p(X_d^t\mid C_d).
$$

这就是需要补齐的问题。

---

## 8. 我们真正想要的 kernel

目标是：

$$
\boxed{
Q_d^t
\left(
X_d^{t-1}\mid X_{d-1}^t,t,C_d
\right)
}
$$

它回答：

> 输入是旧条件下的当前 flow 粒子 $X_{d-1}^t$，并且新条件 $C_d$ 已经到达时，怎样得到新条件下的下一 flow 粒子 $X_d^{t-1}$？

它同时完成：

$$
(d-1,t)\longrightarrow(d,t-1).
$$

即同时改变条件版本和 flow 时刻。

---

## 9. 用全概率公式连接已知项和目标项

在目标 kernel 中插入同一 flow 时刻、当前条件下的中间变量：

$$
X_d^t.
$$

根据全概率公式：

$$
Q_d^t
\left(
X_d^{t-1}\mid X_{d-1}^t,t,C_d
\right)
$$

$$
=
\int
p
\left(
X_d^{t-1}\mid
X_d^t,
X_{d-1}^t,
t,
C_d
\right)
p
\left(
X_d^t\mid
X_{d-1}^t,
t,
C_d
\right)
dX_d^t.
$$

给定 $X_d^t$ 和 $C_d$ 后，正常的下一 flow step 不再需要旧粒子 $X_{d-1}^t$。采用条件独立：

$$
X_d^{t-1}
\perp
X_{d-1}^t
\mid
(X_d^t,t,C_d).
$$

因此第一项化为 Stage-A kernel：

$$
p
\left(
X_d^{t-1}\mid
X_d^t,
X_{d-1}^t,
t,
C_d
\right)
$$

$$
=
K_d^t
\left(
X_d^{t-1}\mid X_d^t,t,C_d
\right).
$$

定义第二项：

$$
\boxed{
B_d^t
\left(
X_d^t\mid X_{d-1}^t,t,C_d
\right)
}
$$

由于上标 $t$ 已经标明 bridge 所在的 flow 时刻，后续可简写为用户约定的形式：

$$
\boxed{
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right)
}
$$

最终：

$$
\boxed{
Q_d^t
\left(
X_d^{t-1}\mid X_{d-1}^t,t,C_d
\right)
=
\int
K_d^t
\left(
X_d^{t-1}\mid X_d^t,t,C_d
\right)
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right)
dX_d^t
}
$$

这条公式是整套方法的核心。

---

## 10. 确切缺失的对象

Stage A 已经提供：

$$
K_d^t
\left(
X_d^{t-1}\mid X_d^t,t,C_d
\right).
$$

真正缺失的是：

$$
\boxed{
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right)
}
$$

它回答：

> 在 flow 时刻不变的情况下，怎样把旧条件 $C_{d-1}$ 下的具体粒子 $X_{d-1}^t$，更新成当前条件 $C_d$ 下的具体粒子 $X_d^t$？

它完成：

$$
(d-1,t)\longrightarrow(d,t).
$$

可以称为：

- posterior bridge；
- re-anchoring kernel；
- condition-transition kernel；
- observation-assimilation kernel。

本文统一称为 posterior bridge。

---

## 11. 为什么 Stage-A marginals 不能唯一确定 $B_d^t$

Stage A 可以定义两个 marginal：

$$
p(X_{d-1}^t\mid C_{d-1}),
$$

$$
p(X_d^t\mid C_d).
$$

但两个 marginal 不能唯一确定条件 coupling：

$$
p(X_d^t\mid X_{d-1}^t,C_d).
$$

例如，假设新旧 marginal 都是：

$$
p(-1)=p(+1)=\frac12.
$$

以下两个 coupling 都满足相同 marginals：

$$
X_d^t=X_{d-1}^t,
$$

以及：

$$
X_d^t=-X_{d-1}^t.
$$

所以，仅仅知道“旧 posterior 是什么”和“新 posterior 是什么”，不足以知道某个旧粒子应该对应哪个新粒子。还必须选择或学习一个 coupling。

---

## 12. 贝叶斯公式能提供什么

需要求的对象始终是：

$$
B_d^t(X_d^t\mid X_{d-1}^t,C_d).
$$

不对 $C_d$ 添加任何额外的内部条件结构，直接应用贝叶斯公式：

$$
B_d^t
(x^+\mid x^-,C_d)
$$

$$
=
\frac{
p(C_d\mid x^+,x^-)
p(x^+\mid x^-)
}{
p(C_d\mid x^-)
}.
$$

其中：

$$
x^-=X_{d-1}^t,
\qquad
x^+=X_d^t.
$$

这个公式没有引入其他信息集或隐藏的条件记号。这里把 $C_d$ 当作被观测到的随机条件变量。公式在形式上成立，但并不意味着其中各项容易建模。

但它显示出直接贝叶斯求解需要两个未知对象：

1. 新增观测/执行信息的 likelihood：

   $$
   p(C_d\mid x^+,x^-);
   $$

2. 新旧粒子之间的 coupling prior：

   $$
   p(x^+\mid x^-).
   $$

PI0.5 没有显式 world model 或 image likelihood，因此不能直接计算第一项；Stage-A marginals 也不能唯一决定第二项。

因此，贝叶斯公式能够说明缺什么，但不能仅凭 Stage A 自动算出 $B_d^t$。

---

## 13. Shared-base coupling

### 13.1 用共同 base noise 定义 joint

采样同一个：

$$
Z\sim\mathcal N(0,I).
$$

旧条件下生成：

$$
X_{d-1}^t
=
\Phi_{d-1}^{t\leftarrow T}(Z;C_{d-1}).
$$

当前条件下生成：

$$
X_d^t
=
\Phi_d^{t\leftarrow T}(Z;C_d).
$$

同一个 $Z$ 定义了 paired sample：

$$
(X_{d-1}^t,X_d^t).
$$

因此它定义了一个合法 joint distribution，并进一步定义：

$$
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right).
$$

### 13.2 Exact re-anchoring

若相关 free suffix 上的 flow map 可逆，则：

$$
Z
=
\Phi_{d-1}^{T\leftarrow t}(X_{d-1}^t;C_{d-1}).
$$

所以：

$$
X_d^t
=
\Phi_d^{t\leftarrow T}
\circ
\Phi_{d-1}^{T\leftarrow t}
(X_{d-1}^t).
$$

定义：

$$
R_d^t
=
\Phi_d^{t\leftarrow T}
\circ
\Phi_{d-1}^{T\leftarrow t}.
$$

则 deterministic bridge 为：

$$
B_d^t(x^+\mid x^-,C_d)
=
\delta
\left(
x^+-R_d^t(x^-)
\right).
$$

这不是由 Bayes 公式唯一推出的 coupling，而是我们利用 conditional flow 的共同 base space 选择的一个明确、合法、可计算的 coupling。

### 13.3 已执行坐标

从 $d-1$ 更新到 $d$ 时，动作位置 $d$ 从随机变量变成已执行事实：

$$
X_{d,d}^t=A_d^{exec}.
$$

所以 re-anchoring 只需要在剩余 suffix：

$$
j>d
$$

上进行。位置 $j\le d$ 由 hard commitment 直接覆盖，不要求连续可逆 transport。

---

## 14. Stage B：学习 $B_d^t$

### 14.1 Teacher pair

冻结 Stage-A flow，采样共享 $Z$ 和 flow 时刻 $t$：

$$
X_{d-1}^t
=
\Phi_{d-1}^{t\leftarrow T}(Z;C_{d-1}),
$$

$$
X_d^t
=
\Phi_d^{t\leftarrow T}(Z;C_d).
$$

得到监督 pair：

$$
(X_{d-1}^t,X_d^t,C_d).
$$

### 14.2 Student

训练 posterior updater：

$$
T_\phi
\left(
X_{d-1}^t,t,C_d
\right)
\approx
X_d^t.
$$

从 kernel 角度：

$$
p_\phi
\left(
X_d^t\mid X_{d-1}^t,C_d
\right)
\approx
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right).
$$

### 14.3 Hard commitment

student 输出定义为：

$$
\widehat X_{d,j}^t
=
\begin{cases}
A_j^{exec},&j\le d,\\
X_{d-1,j}^t+\Delta X_{d,j}^t,&j>d.
\end{cases}
$$

### 14.4 Loss

主要 endpoint loss 为：

$$
\mathcal L_{bridge}
=
\frac{1}{H-d}
\sum_{j=d+1}^{H}
\left\|
\widehat X_{d,j}^t-X_{d,j}^t
\right\|^2.
$$

当前实现还包括：

- committed diagnostic loss；
- 第一个可执行位置 $A_{d+1}$ 的 boundary loss；
- 无新增视觉信息时的 identity dropout。

---

## 15. 目标 kernel 的最终实现

有了 Stage-A kernel：

$$
K_d^t
\left(
X_d^{t-1}\mid X_d^t,t,C_d
\right),
$$

以及 Stage-B bridge：

$$
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right),
$$

目标 kernel 为二者复合：

$$
Q_d^t=K_d^t\circ B_d^t.
$$

deterministic 实现为：

$$
\widehat X_d^t
=
T_\phi(X_{d-1}^t,t,C_d),
$$

$$
X_d^{t-1}
=
\Psi_{\theta,d}^{t\rightarrow t-1}
\left(
\widehat X_d^t,t,C_d
\right).
$$

即：

```text
输入旧条件粒子 X_(d-1)^t
→ Stage B 在相同 flow 时刻更新条件版本
→ 得到当前条件粒子 X_d^t
→ Stage A 推进一个 flow step
→ 得到 X_d^(t-1)
```

---

## 16. Fast observation 的具体实现

理论公式把 fast condition 统一记为 $I_d$。当前代码使用：

$$
F_0^c=E_{DINO}(I_{0,raw}^c),
$$

$$
F_d^c=E_{DINO}(I_{d,raw}^c).
$$

$I_{d,raw}^c$ 只在这一实现小节表示相机 $c$ 的原始图像；经过下述编码和 resampling 得到的整体 fast condition 仍记为理论符号 $I_d$。

对每个相机和 patch 构造：

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

加入 camera embedding，并用带当前 proprio 和 delay condition 的 learned queries 做 cross-attention resampling：

$$
Q_d
=
Q^{learn}
+
W_qq_d
+
\operatorname{MLP}_d(d),
$$

$$
I_d
=
\operatorname{LN}
\left(
Q_d
+
\operatorname{MHA}(Q_d,R_d,R_d)
\right).
$$

DINOv3 Small 的两路相机共 392 个 patch tokens，resampler 将其压缩为 16 个 action-expert-width fast tokens。

这些 fast tokens 通过 cross-attention adapters 注入 action expert，不进入固定的 slow prefix。

---

## 17. 跨多个物理时刻的更新

如果 observation 并非从 $d-1$ 更新到 $d$，而是从旧条件版本 $r<d$ 直接跳到 $d$，需要：

$$
B_{r\rightarrow d}^t
\left(
X_d^t\mid X_r^t,C_d
\right).
$$

可以选择：

1. 依次复合相邻 bridges：

   $$
   B_{r\rightarrow d}^t
   =
   B_d^t\circ\cdots\circ B_{r+1}^t;
   $$

2. 训练时随机采样物理 gap，直接学习跨步 bridge。

当前代码以相邻 $d-1\rightarrow d$ 为基本训练单元。

---

## 18. 两阶段训练流程

### 18.1 Stage A

```text
输入 demonstration action A
→ 采样物理时延 d
→ 构造条件 C_d
→ 采样 Z 和 flow 时刻 t
→ 构造 X_d^t：clean prefix + noisy suffix
→ 预测 V_theta(X_d^t,t,C_d)
→ 只在 j>d 上计算 flow-matching loss
→ 得到 K_d^t
```

### 18.2 Stage B

```text
冻结 Stage A
→ 采样相邻条件 C_(d-1), C_d
→ 采样共享 Z 和 flow 时刻 t
→ 生成 paired X_(d-1)^t, X_d^t
→ 训练 T_phi(X_(d-1)^t,t,C_d) ≈ X_d^t
→ 得到 B_d^t
```

### 18.3 在线组合

```text
X_(d-1)^t
→ B_d^t：条件更新
→ X_d^t
→ K_d^t：flow 更新
→ X_d^(t-1)
```

---

## 19. 与代码下标的对应

本文为了符合物理直觉，动作位置使用 1-based：

$$
A_1,\ldots,A_H.
$$

代码使用 0-based：

```text
document A_j     ↔ code action[:, j-1]
document A_1:d   ↔ code action[:, 0:d]
document A_(d+1) ↔ code action[:, d]
```

因此 delay 为 $d$ 时：

- 文档中不可执行的是 $A_{1:d}$；
- 代码中的 elapsed mask 是 `position < d`；
- 文档中下一动作是 $A_{d+1}$；
- 代码中下一动作是 `action[:, d]`。

---

## 20. 当前实现状态

已经实现：

- Stage-A conditional generation kernel；
- clean executed prefix 和 suffix-only flow loss；
- DINOv3 fast observation 和 resampler；
- action-expert fast adapters；
- shared-base teacher pairs；
- one-step posterior bridge student；
- hard commitment、endpoint、boundary 和 identity losses；
- Stage-B teacher freeze；
- CPU 理论测试和真实 LIBERO GPU smoke test。

尚未完成：

- observation 在 flow 积分中持续到达的完整异步 runner；
- 跨任意物理 gap 的 bridge 训练；
- 高效 slow-prefix KV-cache teacher；
- 长程 Stage-A/Stage-B benchmark 训练。

---

## 21. 核心结论

已知的是：

$$
\boxed{
K_d^t
\left(
X_d^{t-1}\mid X_d^t,t,C_d
\right)
}
$$

想要的是：

$$
\boxed{
Q_d^t
\left(
X_d^{t-1}\mid X_{d-1}^t,t,C_d
\right)
}
$$

根据全概率公式，两者之间缺少：

$$
\boxed{
B_d^t
\left(
X_d^t\mid X_{d-1}^t,C_d
\right)
}
$$

并且：

$$
\boxed{
Q_d^t
=
K_d^t\circ B_d^t
}
$$

所以，本方法需要补齐的不是另一个普通 denoising step，而是：

> 同一 flow 时刻上，从旧物理条件粒子到当前物理条件粒子的 posterior bridge。
