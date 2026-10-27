# Two-Condition Online Flow 执行方案

## 1. 目标

用一个统一模型替代现有的单输入 Stage A 和独立 Stage B updater。每个去噪步始终接收：

$$
(X_s,s,C_{old},d_{old},C_{new},d_{new}).
$$

没有新 packet 时 old 与 new 相同；有新 packet 时直接处理 condition transition。部署时只有这一个模型。

## 2. 时延协议

训练复刻当前 `continuous_online_v1`：

- action chunk 为 50 steps，denoising 为 10 steps，replan interval 为 20 control steps；
- 每个 chunk 的总 delay：$D\sim\mathrm{Uniform}\{1,\ldots,10\}$；
- 第 $k$ 个去噪查询前的控制进度：
  $$
  d_k=\left\lfloor\frac{kD}{9}\right\rfloor,\quad k=0,\ldots,9;
  $$
- 每个控制步产生 observation；同一查询前有多个 packet 时只使用 latest-ready packet；
- 相邻查询的 $d_k$ 相同时构造 identity transition。

训练支持范围为 $d\in[0,10]$，暂不覆盖范围外 delay。

## 3. Phase 1：Canonical Two-Condition FM

从 official PI0.5 LIBERO checkpoint 重训，不继承现有单输入 Stage-A checkpoint。

1. 采样 $D$ 并生成 $d_0,\ldots,d_9$。
2. 采样 flow step $k$，令 $s_k=1-k/10$、$d_{old}=d_{k-1}$、$d_{new}=d_k$；$k=0$ 时两者均为 0。
3. 从 LIBERO 时序 batch 取得对应的 old/new image 和 proprio。
4. 使用标准 FM：
   $$
   X_s=sZ+(1-s)A,\qquad v^*=Z-A.
   $$
5. 仅在 $i\ge d_{new}$ 的有效 action 维度计算 FM loss。

所有 condition transition 由此共享一个 canonical flow 坐标，不再先训练两个独立 flow 再学习事后 bridge。

## 4. Phase 2：EMA Online-Rollout Fine-Tuning

Phase 1 结束后初始化同结构的 student 和 EMA teacher。每个 optimizer step 后更新：

$$
\bar\theta\leftarrow\beta\bar\theta+(1-\beta)\theta.
$$

第一版使用 $\beta=0.999$。

1. 按第 2 节采样 causal delay schedule。
2. EMA 从噪声开始，使用相邻 old/new condition 执行 online rollout。
3. delay 推进后 hard commit 数据集中对应的 policy-normalized action prefix；prefix 不计 loss。
4. 从 EMA rollout 访问的 particle $\tilde X_s$ 采样训练点。
5. student 的 suffix recovery target 为：
   $$
   v^*_{online}=\frac{\tilde X_s-A}{s}.
   $$
6. 同时保留 canonical FM loss，防止基础去噪能力退化。

先实现 1--4 个去噪步的 truncated EMA rollout，通过离线验证后再扩展到完整 10 步。EMA 负责产生 on-policy particle，数据集动作 $A$ 负责提供正确性监督。

## 5. 实现范围

- 新增 two-condition training mode，不覆盖现有 Stage A/B 代码和 checkpoint。
- old/new fast token 使用显式 role embedding，并输入 $d_{old},d_{new}$。
- 训练、EMA rollout 和推理共享同一个 two-condition forward 接口。
- 推理保留 latest-ready buffer 和 hard prefix commit；新模式不调用 posterior updater。
- 先补单元测试，再做真实资产 smoke test。

## 6. 验收顺序

1. shape、mask、identity transition、delay schedule 和 hard commit 单测。
2. Phase 1 小 batch overfit：FM loss 明显下降。
3. Phase 2 固定 schedule overfit：online recovery loss 下降，EMA 数值稳定。
4. held-out offline rollout：correct-new 优于 stale 和 shuffled-new。
5. LIBERO smoke：每 suite 少量 episode，检查 packet/delay/prefix 时序。
6. 全部通过后才进行 2000-episode 闭环评测。

## 7. 环境与资产

- Python：`/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python`（已验证 Python 3.12.14）。
- 已验证存在：
  - `/data/zhoukai/pretrain/pi05_libero_finetuned`
  - `/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m`
  - `/data/zhoukai/pretrain/paligemma-tokenizer`
  - `/data/zhoukai/dataset/libero`

## 8. 执行边界

本文通过审阅后再修改代码、运行训练或闭环评测。实施时保留现有未提交修改，不覆盖原 Stage A/B 实验资产。
