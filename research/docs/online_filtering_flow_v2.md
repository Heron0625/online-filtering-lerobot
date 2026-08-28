# Online Filtering Flow v2

## 1. 目标

在动作块异步生成期间持续吸收真实观测，并在不重复执行重叠动作的前提下修正尚未执行的动作后缀。

模型只处理 observation filtering：

$$
X_{d-1}^t\xrightarrow{B_d^t}X_d^t\xrightarrow{K_d^t}X_d^{t-1}.
$$

重叠动作不进入训练或概率模型。新旧动作块的连续性由部署端线性 merge 处理。

## 2. 时间轴与符号

动作块长度为 $H$：

$$
A=(A_1,\ldots,A_H).
$$

旧块执行 $m$ 步时取得 $I_0$ 并触发新块推理，记为物理时刻 $d=0$。物理时延 $d$ 表示旧块已经执行到 $m+d$。新块在 $d=D$ 时完成，因此：

$$
A_{1:D}^{\mathrm{new}}
$$

已经过期，不能执行；可执行后缀为：

$$
\boxed{A_{D+1:H}^{\mathrm{new}}.}
$$

本文使用三种索引：

- 下标 $d$：物理观测时刻，$d=0,\ldots,D$；
- 上标 $t$：flow 时刻，$t=T,T-1,\ldots,0$；
- 粒子内部下标 $j$：动作位置，$j=1,\ldots,H$。

$$
X_{d,j}^t
$$

表示物理时刻 $d$ 的条件下，flow 时刻 $t$ 的完整粒子中第 $j$ 个动作位置。主推导使用完整随机变量 $X_d^t$，不引入已执行动作 mask。

## 3. 条件变量

初始输入经过 Gemma 得到 slow prefix：

$$
P=\operatorname{Gemma}(\text{text},\text{proprio},I_0).
$$

不区分图像是原图还是编码后的 token；只使用 action expert 最终取得的条件。

物理时刻 $d$ 的 fast observation 为 $I_d$。定义：

$$
\boxed{C_d=(t,P,I_d,d).}
$$

rollout 内部处于 flow 时刻 $s$ 时写成：

$$
C_d^s=(s,P,I_d,d).
$$

其中物理条件 $(P,I_d,d)$ 固定，flow-time 输入随积分步骤变化。

### 3.1 Fast observation encoder

同一物理时刻 $d$ 的 fast observation 包含多视角图像与同步采集的 proprio：

$$
O_d^{\mathrm{fast}}
=
\left(I_d^{(1)},\ldots,I_d^{(V)},q_d\right).
$$

#### 多视角 DINOv3

将 $V$ 个视角合并到 batch 维，用一次 DINOv3 forward 并行编码：

$$
[B,V,C,H,W]
\longrightarrow
[BV,C,H,W].
$$

第 $v$ 个相机产生：

$$
F_d^{(v)}
=
\operatorname{DINO}
\left(I_d^{(v)}\right)
\in\mathbb R^{N_v\times D_v}.
$$

每个视觉 token 加入相机身份与采集年龄编码：

$$
\widetilde F_{d,n}^{(v)}
=
F_{d,n}^{(v)}
+e_{\mathrm{camera}}^{(v)}
+e_{\mathrm{age}}^{(v)}.
$$

不同相机分别保留 capture timestamp，不能假设硬件采集完全同步。第一版要求所有配置的相机视角都有效；任一视角缺失、解码失败或超出同步容差时，整个 fast-observation packet 作废。

#### Fast proprio

初始 proprio $q_0$ 已经进入 slow prefix；fast 分支只使用当前时刻完整的绝对机器人状态：

$$
p_d=q_d=\operatorname{Norm}(\text{observation.state}_d).
$$

`observation.state` 包含 LIBERO 中可用的完整 proprio state，包括 gripper state。其归一化统计量与 PI0.5 数据预处理保持一致。绝对 state 由轻量 MLP 编成一个 token：

$$
F_d^q
=
\operatorname{MLP}_q
\left(p_d\right)
\in\mathbb R^{1\times D_{\mathrm{fast}}}.
$$

第一版固定 $N_q=1$，不使用额外的大型 proprio encoder，也不使用相对 state 或速度。相对 state 与 $\dot q_d$ 留作后续消融；若未来启用 $\dot q_d$，LIBERO 离线训练按数据集 FPS 计算，在线部署按 capture timestamp 计算。

#### Resampler 融合

拼接多视角视觉 tokens 与 proprio tokens：

$$
S_d
=
\left[
\widetilde F_d^{(1)};
\cdots;
\widetilde F_d^{(V)};
F_d^q
\right].
$$

使用固定数量的 learned queries 做 cross-attention resampling：

$$
\boxed{
E_d
=
\operatorname{Resampler}
\left(Q_{\mathrm{fast}},S_d,M_d^{\mathrm{camera}}\right)
\in\mathbb R^{N_{\mathrm{fast}}\times D_{\mathrm{action}}}.
}
$$

Resampler 同时完成 token 数量压缩、特征维度对齐和视觉/proprio 融合。理论推导中的 $I_d$ 从此表示这个最终 fast condition $E_d$，而不是特指原始 RGB。

因此实现中的条件为：

$$
\boxed{
C_d
=
\left(t,P,E_d,d,\operatorname{age}_d\right).
}
$$

这里：

$$
\operatorname{age}_d
=
\tau^{\mathrm{query}}-\tau_d^{\mathrm{capture}}.
$$

为保持后续公式简洁，仍将其统一简写为 $C_d$。

## 4. 原始 PI0.5 Flow Matching

给定真实动作块 $A$，采样：

$$
t\sim\operatorname{Uniform}\{1,\ldots,T\},
\qquad Z\sim\mathcal N(0,I),
\qquad \alpha_t=\frac{t}{T}.
$$

构造：

$$
X^t=\alpha_tZ+(1-\alpha_t)A,
\qquad U^t=Z-A.
$$

标准 loss 为：

$$
\mathcal L_{\mathrm{PI0.5}}
=
\mathbb E\left[
\left\|V_\theta(X^t,t,P)-(Z-A)\right\|_2^2
\right].
$$

离散转移为：

$$
p_\theta(X^{t-1}\mid X^t,t,P),
$$

确定性实现记为：

$$
X^{t-1}=\psi_\theta(X^t,t,P).
$$

Euler step 为：

$$
\psi_\theta(X^t,t,P)=X^t-\frac1T V_\theta(X^t,t,P).
$$

## 5. Stage A：加入当前观测

Stage A 的 forward noising 不变：

$$
X_d^t=\alpha_tZ+(1-\alpha_t)A.
$$

将 $I_d$ 和 $d$ 注入 action expert：

$$
V_\theta(X_d^t,C_d)=V_\theta(X_d^t,t,P,I_d,d).
$$

训练目标仍为 $Z-A$：

$$
\boxed{
\mathcal L_A
=
\mathbb E_{A,Z,t,d}
\left[
\left\|V_\theta(X_d^t,C_d)-(Z-A)\right\|_2^2
\right].
}
$$

Stage A 学到：

$$
\boxed{
K_d^t(x'\mid x)
=
p_\theta(X_d^{t-1}=x'\mid X_d^t=x,C_d).
}
$$

## 6. 在线条件错位

新观测 $I_d$ 到达时，系统保存的是旧条件粒子 $X_{d-1}^t$，而 $K_d^t$ 要求输入当前条件粒子 $X_d^t$。

真正需要的是：

$$
\boxed{
Q_d^t(x'\mid x^-)
=
p(X_d^{t-1}=x'\mid X_{d-1}^t=x^-,C_d).
}
$$

其中：

$$
x^-=X_{d-1}^t,
\qquad x^+=X_d^t,
\qquad x'=X_d^{t-1}.
$$

## 7. 推导缺失的 Bridge

插入中间变量 $X_d^t$：

$$
p(X_d^{t-1}\mid X_{d-1}^t,C_d)
=
\int p(X_d^{t-1},X_d^t\mid X_{d-1}^t,C_d)\,dX_d^t.
$$

链式分解：

$$
\begin{aligned}
&p(X_d^{t-1},X_d^t\mid X_{d-1}^t,C_d)\\
&=p(X_d^{t-1}\mid X_d^t,X_{d-1}^t,C_d)
p(X_d^t\mid X_{d-1}^t,C_d).
\end{aligned}
$$

采用条件独立假设：

$$
\boxed{X_d^{t-1}\perp X_{d-1}^t\mid X_d^t,C_d.}
$$

定义缺失的 bridge：

$$
\boxed{
B_d^t(x^+\mid x^-,C_d)
=
p(X_d^t=x^+\mid X_{d-1}^t=x^-,C_d).
}
$$

则：

$$
\boxed{
Q_d^t(x'\mid x^-)
=
\int K_d^t(x'\mid x^+)B_d^t(x^+\mid x^-,C_d)\,dx^+.
}
$$

即：

$$
\boxed{Q_d^t=K_d^t\circ B_d^t.}
$$

## 8. Shared-noise Bridge 监督

冻结训练完成的 Stage-A teacher。定义：

$$
\Phi_\theta^t(Z;P,I_d,d)
$$

为从 $X^T=Z$ 开始，在物理条件 $(P,I_d,d)$ 下积分到 $t$ 的结果：

$$
\Phi_\theta^t(\cdot;P,I_d,d)
=
\psi_\theta^{t+1}(\cdot;C_d^{t+1})
\circ\cdots\circ
\psi_\theta^T(\cdot;C_d^T).
$$

从同一轨迹取 $(I_{d-1},I_d)$，只采样一次：

$$
Z\sim\mathcal N(0,I).
$$

用相同 $Z$ 运行两条 teacher rollout：

$$
x^-=\Phi_\theta^t(Z;P,I_{d-1},d-1),
$$

$$
x^+=\Phi_\theta^t(Z;P,I_d,d).
$$

于是 shared-noise pair 为：

$$
\boxed{
(x^-,x^+)
=
\left(
\Phi_\theta^t(Z;P,I_{d-1},d-1),
\Phi_\theta^t(Z;P,I_d,d)
\right).
}
$$

不能用相同 $A,Z$ 分别做解析 noising 来构造 pair，否则两者都等于 $\alpha_tZ+(1-\alpha_t)A$，只能监督恒等映射。

## 9. Stage B：训练 Bridge

使用 residual bridge：

$$
\boxed{
\widehat x^+
=T_\phi(x^-,C_d)
=x^-+\Delta_\phi(x^-,C_d).
}
$$

即输入 $x^-$ 和 $C_d$，输出 $x^+$。teacher pair 阻断梯度，基本 loss 为：

$$
\boxed{
\mathcal L_B
=
\mathbb E_{d,t,Z}
\left[
\left\|T_\phi(x^-,C_d)-x^+\right\|_2^2
\right].
}
$$

确定性 bridge 对应：

$$
B_{\phi,d}^t(x^+\mid x^-,C_d)
=
\delta\left(x^+-T_\phi(x^-,C_d)\right).
$$

实现中，$x^-$ 先按 action position 编成 particle tokens。条件端不只使用一个 pooled vector，而是保留完整的 $N_{\mathrm{fast}}$ 个 fast tokens；slow prefix 用独立 learned-query resampler 压缩为 4 个 tokens。二者拼接后供 bridge 的每层 cross-attention 使用：

$$
S_d^{B}
=
\left[
\operatorname{SlowResampler}_4(P);
\operatorname{Proj}_{B}(E_d)
\right].
$$

因此 bridge 的实际形式为：

$$
\widehat x^+
=x^-+operatorname{Decoder}_{\phi}
\left(x^-,S_d^B,t,r,d,\operatorname{age}_d\right).
$$

这样 fast resampler 负责把 DINOv3 patch 数压到固定规模，而 bridge 仍能按动作位置查询不同的观测 token；不会在进入 bridge 前再次做全局平均。

在 $t=T$ 时两条轨迹都等于 $Z$，因此增加：

$$
\mathcal L_{\mathrm{id}}
=
\mathbb E\left[\left\|T_\phi(Z,C_d)-Z\right\|_2^2\right].
$$

$$
\boxed{
\mathcal L_{\mathrm{StageB}}
=
\mathcal L_B+\lambda_{\mathrm{id}}\mathcal L_{\mathrm{id}}.
}
$$

## 10. 得到目标 Transition

在线时：

$$
x^+=T_\phi(x^-,C_d),
\qquad
x'=\psi_\theta(x^+,C_d).
$$

因此：

$$
\boxed{
X_d^{t-1}
=
\psi_\theta\left(T_\phi(X_{d-1}^t,C_d),C_d\right).
}
$$

Euler 实现为：

$$
\boxed{
X_d^{t-1}
=
T_\phi(X_{d-1}^t,C_d)
-\frac1T
V_\theta\left(T_\phi(X_{d-1}^t,C_d),C_d\right).
}
$$

若观测从 $r$ 直接跳到 $d$，则使用跨 gap pair：

$$
x^-=\Phi_\theta^t(Z;P,I_r,r),
\qquad
x^+=\Phi_\theta^t(Z;P,I_d,d),
$$

训练 $T_\phi(x^-,C_d,r,d)\approx x^+$。相邻更新 $r=d-1$ 是其特例。

## 11. 异步 Fast-observation Buffer

### 11.1 $D$ 只在完成时可知

部署时不能预先假定终端推理时延 $D$。触发推理时记录控制周期：

$$
n_{\mathrm{trigger}}.
$$

推理期间只计算当前 elapsed steps：

$$
d=n_{\mathrm{now}}-n_{\mathrm{trigger}}.
$$

action expert 完成新块后，才得到实测时延：

$$
\boxed{
D=n_{\mathrm{finish}}-n_{\mathrm{trigger}}.
}
$$

第 12 节根据这个实测 $D$ 丢弃过期动作并对齐 merge。

### 11.2 Producer--consumer

DINOv3 和 action expert 异步运行，二者不互相等待：

```text
camera + proprio producer
    → batched multi-view DINOv3
    → proprio MLP
    → resampler
    → timestamped embedding buffer

action expert consumer
    → 查询最新 ready packet
    → 有更新则执行 B
    → 始终继续执行 K
```

slow prefix 在每次 chunk 开始时只计算一次，并沿用原 PI0.5 的 Gemma prefix KV cache。后续每个 flow step 只计算 action postfix；新 fast packet 只经过 bridge 与 action-expert adapter，不触发 Gemma prefix 重算。

producer 调度采用“一项 running + 一项可替换 pending”：DINOv3 忙时，新帧覆盖尚未开始的旧 pending 帧；正在编码的任务不中断。编码失败会记录错误并继续处理最新 pending 帧。

每个 buffer item 至少保存：

$$
\boxed{
\mathcal E_i
=
\left(
E_i,
n_i,
q_i,
\tau_i^{\mathrm{capture}},
\tau_i^{\mathrm{ready}},
M_i^{\mathrm{camera}},
g_i
\right).
}
$$

其中 $g_i$ 是 episode/reset generation ID，用于阻止旧 episode 的异步结果污染当前推理。物理时刻由 capture control-step 决定：

$$
d_i=n_i-n_{\mathrm{trigger}},
$$

不能用 embedding ready time 替代 capture time。

第一版使用容量为 4 的线程安全 ring buffer：

$$
\boxed{N_{\mathrm{buffer}}=4.}
$$

buffer 满时覆盖最旧 packet。配置最大允许观测年龄 $a_{\max}$：

$$
\operatorname{age}_i
=
\tau^{\mathrm{query}}-\tau_i^{\mathrm{capture}}
\le a_{\max}.
$$

超过最大 age 的 packet 直接丢弃。第一版要求所有相机均有效：

$$
\boxed{M_i^{\mathrm{camera}}=\mathbf 1.}
$$

### 11.3 Latest-ready 消费规则

设当前粒子使用的最新物理条件版本为 $r$。查询时已经完成且比 $r$ 更新的 packets 为：

$$
\mathcal E_{\mathrm{ready}}
=
\left\{
\mathcal E_i:
\tau_i^{\mathrm{ready}}\le\tau^{\mathrm{query}},
d_i>r,
g_i=g_{\mathrm{current}},
\operatorname{age}_i\le a_{\max},
M_i^{\mathrm{camera}}=\mathbf 1
\right\}.
$$

若集合非空，选择 capture step 最新的 packet：

$$
\mathcal E^\star
=
\arg\max_{\mathcal E_i\in\mathcal E_{\mathrm{ready}}}d_i,
$$

并执行跨 gap bridge：

$$
X_r^t
\xrightarrow{B_{r\rightarrow d^\star}^t}
X_{d^\star}^t
\xrightarrow{K_{d^\star}^t}
X_{d^\star}^{t-1}.
$$

若集合为空，不等待 DINOv3，直接推进：

$$
X_r^t\xrightarrow{K_r^t}X_r^{t-1}.
$$

如果多个 packets 积压，第一版直接跳到最新 packet；Stage B 必须相应训练随机 observation gaps。action expert 已完成后才到达的 packets 直接丢弃。

### 11.4 第一版的 proprio 到达策略

第一版不让 proprio 单独触发 bridge。与图像采集时刻对齐的 $q_d$ 和多视角视觉特征共同组成 $E_d$，resampler 完成后以一个 packet 触发一次 bridge。

这样避免额外训练 proprio-only、vision-only 和乱序 modality assimilation。未来若需要更高 proprio 反应频率，再增加独立的 proprio-only bridge。

## 12. 部署端线性 Merge

重叠动作不进入 Stage A、Stage B、$K_d^t$ 或 $B_d^t$。模型生成完整新块，推理完成后再处理衔接。

### 12.1 丢弃过期前缀

在 $d=D$ 时直接丢弃：

$$
A_{1:D}^{\mathrm{new}}.
$$

这些动作绝不重新执行。

### 12.2 对齐未来动作

对完成推理后的第 $k$ 个未来控制时刻：

$$
A_k^{\mathrm{old,align}}=A_{m+D+k}^{\mathrm{old}},
$$

$$
A_k^{\mathrm{new,align}}=A_{D+k}^{\mathrm{new}}.
$$

只在旧块仍有有效未来动作的位置进行融合。

### 12.3 线性 Crossfade

第一版固定 merge horizon：

$$
\boxed{R=4.}
$$

线性权重为：

$$
\lambda_k=\frac{k}{R+1},
\qquad k=1,\ldots,R.
$$

融合动作：

$$
\boxed{
A_k^{\mathrm{merge}}
=(1-\lambda_k)A_k^{\mathrm{old,align}}
+\lambda_kA_k^{\mathrm{new,align}}.
}
$$

在 $k>R$ 后完全采用新块：

$$
A_k^{\mathrm{merge}}=A_k^{\mathrm{new,align}}.
$$

若旧块剩余有效长度小于 $R$，令：

$$
R_{\mathrm{eff}}
=
\min(R,\text{旧块剩余有效长度}),
$$

并按 $R_{\mathrm{eff}}$ 重新计算权重。merge 是 action queue 的确定性后处理，不参与训练或反向传播。

## 13. 实现与验证状态

当前代码已经实现：

- 完整 chunk 的 Stage-A flow-matching loss；
- frozen Stage-A teacher、shared-noise 跨 gap pair 与 Stage-B endpoint loss；
- 多视角单次 batched DINOv3 forward，以及包含 gripper 的绝对 `observation.state` token；
- 完整 fast-token cross-attention、4 个 slow-prefix bridge tokens 和 PI0.5 prefix KV cache；
- 容量 4、最大 age、latest-ready、stale generation、整包相机有效性检查；
- one-running/latest-pending 异步编码调度；
- 在线 $B\rightarrow K$ 组合、实测 $D$、过期前缀丢弃和 $R=4$ linear merge。

CPU 单元测试覆盖 Gaussian shared-noise 理论、组件形状与梯度、buffer、调度器、变长 gap 和 controller。真实本地 smoke test 已使用 PI0.5 Base、DINOv3 Small 与 LIBERO batch 完成 Stage A/Stage B forward-backward。

尚未完成的是机器人或 LIBERO 环境中的长时序闭环成功率评测；这属于实验验证，不是训练图或 policy wiring 缺失。

## 14. 完整流程

### Stage A

$$
V_\theta(X_d^t,C_d)\approx Z-A
\quad\Longrightarrow\quad
K_d^t(X_d^{t-1}\mid X_d^t,C_d).
$$

### Stage B

$$
\left(
\Phi_\theta^t(Z;P,I_{d-1},d-1),
\Phi_\theta^t(Z;P,I_d,d)
\right)
$$

监督：

$$
T_\phi(X_{d-1}^t,C_d)\approx X_d^t.
$$

### 在线 filtering

$$
X_{d-1}^t
\xrightarrow{T_\phi(\cdot,C_d)}
X_d^t
\xrightarrow{\psi_\theta(\cdot,C_d)}
X_d^{t-1}.
$$

### 动作提交

```text
丢弃 new_chunk[1:D]
对仍有效的 old/new future actions 做 4 步线性 crossfade
crossfade 结束后完全执行 new_chunk
```

## 15. 核心结论

Stage A 解决 $X_d^t\rightarrow X_d^{t-1}$；Stage B 补齐 $X_{d-1}^t\rightarrow X_d^t$。两者复合为：

$$
\boxed{Q_d^t=K_d^t\circ B_d^t.}
$$

动作重叠不进入模型。部署端丢弃已经过期的前 $D$ 个动作，只对尚未执行的新旧动作做短窗口线性 merge。
