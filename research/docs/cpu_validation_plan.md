# Online Filtering Flow：CPU 测试与理论验证计划

> 范围：不依赖 LIBERO 数据集、PI0.5/DINOv3 预训练权重或 GPU。

## 1. 验证目标

CPU 阶段必须回答以下问题：

1. overlap timeline 和第 $d+1$ 个动作接管是否无 off-by-one；
2. naive condition switching 是否确实不能得到新 posterior；
3. shared-base-noise re-anchoring 是否得到解析正确的新 posterior；
4. DINO token resampler 与 fast adapter 的 shape、mask、cache contract 是否正确；
5. zero-init adapter 是否严格退化为原 action expert；
6. Stage-A suffix-only loss 与 Stage-C updater mask 是否正确；
7. 多次 observation update 后误差如何累积。

## 2. Gaussian 理论验证

构造线性 Gaussian 系统：

$$
A\sim\mathcal N(m_0,P_0),
\qquad
y_s=H_sA+\eta_s,
\qquad
\eta_s\sim\mathcal N(0,R_s).
$$

使用解析 Bayesian update 计算：

$$
P_{s+1}
=
(P_s^{-1}+H_s^\top R_s^{-1}H_s)^{-1},
$$

$$
m_{s+1}
=
P_{s+1}
(P_s^{-1}m_s+H_s^\top R_s^{-1}y_s).
$$

需要测试：

- affine conditional flow 的 endpoint mean/covariance；
- ODE forward 与解析 map 一致；
- inverse-forward reconstruction；
- flow 中途直接切换 condition 后终点分布错误；
- shared-$Z$ re-anchoring 与新 posterior 一致；
- 连续多次 observation update 的归纳不变量；
- Euler steps 对 mean/covariance 和 reconstruction error 的影响。

## 3. Timeline 与 mask

- $H=50,d=10$ 时 elapsed 为 indices `[0, 10)`，下一动作 index 为 `10`；
- 严格采用 $o_t\rightarrow a_t\rightarrow o_{t+1}$；
- $\mathcal F_d$ 只含 observations `[0, d]` 和 actions `[0, d)`；
- elapsed/free masks 互补并随 $d$ 单调变化；
- $d=0$、$d=H-1$ 和非法 $d\ge H$；
- episode padding 不参与 observation attention 或 action loss。

## 4. Fast observation components

在不加载权重时使用 synthetic DINO patch features：

- 双 camera patch shape；
- initial/current/residual 拼接；
- camera embedding；
- resampler 输出 `[B, 8|16, action_expert_width]`；
- camera/token mask；
- float32/bfloat16；
- timestamp 不变时 cache reuse，变化时 invalidation。

## 5. Action-expert fast adapter

- zero-init 输出与输入严格一致；
- elapsed positions correction 为零；
- free positions 对 fast tokens 敏感；
- disabled path 与原路径一致；
- 只允许 adapter/resampler 参数获得梯度；
- 不同 adapter layer 配置正确构建。

## 6. Stage A / Stage C

Stage A：

- clean elapsed prefix；
- suffix-only flow loss 与手算结果一致；
- 修改不可见未来 observation 不改变输入/loss；
- 不同 delay 均能 forward/backward；
- checkpoint round trip。

Stage C：

- teacher pair 使用相同 $Z$ 和 $\tau$；
- 相同 filtration 时 update 为 identity；
- updater 不修改 elapsed prefix；
- synthetic teacher 上训练后 endpoint MSE 下降；
- 连续 update 后 mean/covariance 误差受控。

## 7. 异步 runtime contract

使用 deterministic fake environment：

- inference delay 期间旧 chunk 持续执行；
- delay 为 $d$ 时从新 chunk index $d$ 接管；
- updater 收到配对的 $(a_d^{exec},o_{d+1})$；
- action queue 原子替换；
- 过期 inference result 不覆盖新结果；
- reset 清空 queue、observation buffer、base noise、feature cache 和 pending result。

## 8. CPU 阶段不覆盖

- DINOv3 pretrained feature 对 LIBERO 的有效性；
- GPU latency、显存和吞吐；
- 实际 LIBERO success rate；
- wall-clock camera/control synchronization；
- 外部扰动下的真实恢复能力。

这些在 CPU contract 全部通过后进入 GPU/LIBERO 阶段。

## 9. 当前验证结果

执行环境：Python 3.12、PyTorch 2.11、CPU。

新增测试：

```text
tests/policies/pi0_pi05/test_online_filtering_components.py
27 passed
```

PI0.5/flow regression：

```text
tests/policies/pi0_pi05/test_pi05_training_time_rtc.py
tests/policies/common/test_flow_matching.py
14 passed
```

静态检查：

```text
ruff check: passed
git diff --check: passed
```

Gaussian 数值验证脚本：

```text
research/scripts/validate_gaussian_reanchoring.py

naive mean error:       1.941170
shared-Z mean error:    0.010933
naive covariance error: 0.586841
shared-Z cov error:     0.015956
```

结果表明，在该解析反例上直接切换条件无法恢复新 posterior，而 shared-$Z$ re-anchoring 的经验 mean/covariance 与解析新 posterior 一致。

当前 Conda 环境退出 pytest 时会由第三方 `multiprocess.resource_tracker` 打印 Python 3.12 清理异常，但测试进程 exit code 为 0，所有断言均通过。该异常与本方法代码无关。
