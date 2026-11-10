# XVLA Online Filtering 实现核对与观测利用诊断（2026-11-01）

核对对象为本次 LIBERO XVLA50 Phase A / Phase B 最终30k权重。
此前400回合各档评测结果保留在 [metrics.md](metrics.md)。本次没有重训、替换权重或改写原评测产物。
所有诊断证据保存于 `research/outputs/xvla_implementation_audit_20261101/`。

## 1. 已核对的链路

- 两阶段均使用 old/new 两时刻条件，摄像头与 proprio 在同一 delay 索引取样。
- 10次去噪使用 `floor(k*D/9)` 控制时钟；runner 检查 packet assimilation trace 与时钟一致。
- 训练/推理动作前缀hard commit、suffix loss与padding/无效target mask均存在。
- 当前XVLA Stage B在teacher和student预测前补齐新增prefix，更新后继续commit。
- 评测从新块索引delay切换，未从0重新执行；正式结果queue underruns均为0。
- slow/fast图像预处理分开：Florence保留官方输入约定，DINO局部ImageNet归一化。
- 关闭所有fast adapter输出投影后，在线重写transformer路径与原生transformer在实际数据上逐元素一致，最大误差 **0.0**（A/B均通过）。
  此项只验证共享输入下transformer数值一致，不证明完整闭环、训练数据与仿真状态分布完全一致。
- trainer在加载student后创建EMA teacher，每次实际optimizer update后更新teacher。

## 2. 发现并修复的实现问题

### 2.1 Slow visual dropout在图文融合后发生，语言token泄漏旧视觉

原实现先用初始图像和语言经过Florence encoder，再在action transformer中屏蔽primary/aux视觉key。
但留下的语言token已经通过Florence attention吸收初始视觉；后续action transformer还可从语言token恢复旧视觉。
所以0.25的视觉dropout **不等于** 25%的样本完全看不到slow视觉，无法按原意充分消除slow-only捷径。

实测8个真实片段，打乱slow图像后，原始融合语言特征变化RMS：Phase A **1.4399**，Phase B **1.4689**。
这是旧视觉进入语言特征的直接证据，不是其造成成功率差异的因果证明。

修复：在Florence图文encoder之前按样本清零视觉features并mask primary image keys，aux features同步清零；
action transformer原有视觉key mask继续保留。dropout采样前移到slow encoding之前。
默认无dropout路径不改变；不允许在已经融合的缓存slow context上事后声称做完整视觉dropout。
真实Phase B checkpoint复测：开启新mask后，改变图像导致的保留语言特征最大差异 **0.0**。

**本次已训A/B权重使用旧dropout。修复推理代码不能让已有权重自动重新学习fast反馈；需要新的训练才能验证修复收益。**

### 2.2 XVLA标准推理API在没有callback时退回父类，绕过fast adapters

原 `predict_action_chunk()` 在未传 `observation_callback` 时调用原生XVLA生成路径。
这会使标准同步调用不使用训练出的fast分支，部分额外kwargs也会传到不支持它们的父类接口。

修复：缺省callback作为空回调，仍运行delay-zero的online路径，使用初始fast packet。
加入有/无callback的回归测试。

**此前正式LIBERO评测明确传入online callback，初始chunk也已使用空回调，因此这项问题不解释此前正式结果。**
不要把这一修复说成已恢复之前400回合评测中的fast分支。

### 2.3 PI0.5 EMA teacher新增prefix锁定滞后一查询

此前已记录：PI0.5 `_forward_online_flow_ema_rollout` 原来只在proposal后commit；新delay增长时，teacher预测前particle仍含新增prefix的噪声/旧预测。
本次修复为teacher预测前按active rows补齐prefix，并保留proposal后的commit。
增加回归测试逐次核对teacher/student输入prefix。
**这不是本次XVLA成绩的原因；XVLA启动本轮Phase B之前已经有同类修复。**

## 3. 离线观测利用检查（已完成）

脚本：`research/scripts/diagnose_xvla_online_implementation.py`。
从完整数据集沿索引分散选取8个非padding且target有效的片段；它们是训练分布诊断样本，不是held-out成功率测试。
两个checkpoint使用相同索引、噪声和query，固定slow context和action particle，只改变fast条件。
总delay25，检查query0/5/9；query9对应old22/new25、flow time0.1。

| query9操作 | Phase A输出变化RMS | Phase B输出变化RMS |
|---|---:|---:|
| 跨样本打乱fast图像，保留proprio | 1.37e-6 | 9.47e-7 |
| fast图像冻结到触发时刻 | 1.13e-6 | 8.36e-7 |
| fast proprio冻结到触发时刻 | 4.88e-6 | 1.95e-6 |
| 图像与proprio一起冻结 | 4.76e-6 | 2.05e-6 |
| 将fast adapters输出投影置零 | 1.11e-3 | 9.88e-4 |

RMS在整个20D原始endpoint输出上计算，包含不同量纲和gripper logits，**不能直接换算成米或角度**。
query0打乱图像也只有约1e-6的输出变化，说明不是仅因query9 particle接近目标导致不敏感。

在这些样本上，内容消融的loss几乎不变；关闭adapter的变化显著大于替换其视觉/状态输入的变化。
这支持“当前fast分支对观测内容依赖很弱，可能主要学到近似固定的修正”的解释。
不能据8个训练片段声称所有环境状态下都完全忽略fast，也不能把固定偏置假设当作已证实的内部机制。

原始证据：`phase_a.json`, `phase_b.json`；修复后无dropout路径及mask复测：`phase_b_fixed_code.json`。

## 4. 闭环对照（后台运行）

脚本：`research/scripts/run_xvla_implementation_audit_eval.py`。
Phase B最终30k checkpoint，GPU2/6/7，每卡5模型，最多15路并行。
每组40任务各1个初始状态，4组共160回合：

1. 零delay、正常fast。
2. delay25、正常fast。
3. delay25、fast图像/proprio始终冻结为本次触发时刻内容，packet计时与prefix继续更新。
4. delay25、adapter输出置零；仍执行相同online观测与prefix约束。

20Hz，replan20，suite horizons280/280/300/520。
新增 `--diagnostic-paired-seed` 在每个episode重置采样seed，避免上一task的不同episode长度污染下一task噪声序列。
同task的同一chunk序号通常使用相同噪声；不同闭环轨迹不保证在相同物理时刻进入同一chunk序号。
这是独立诊断协议，不能将零delay40回合当成匹配正式400回合的充分对照，也不混入正式主表。

- tmux：`xvla_audit_1101`，`progress`窗口。
- 状态日志：`research/outputs/xvla_implementation_audit_20261101/closed_loop.log`。
- 自动最终汇总：该目录下`closed_loop/comparison.json`。
- 未出现最终comparison前不报告整组成功率；日志异常需检查具体worker，不把已退出当完成。

## 5. 结论与后续判断

目前不能再简单归因为“方案理论对但训练不够”。已发现确实削弱预期训练机制的dropout实现问题，
并测得已训模型对fast内容的弱依赖。另一方面，未发现正式评测完全没有使用fast路径，或重写transformer数值错误。
下一步先读闭环正常/冻结/关闭adapter差异，再决定是否从baseline重训正确dropout的Phase A。
仅延长当前Phase B或只在推理时换mask，不能验证修复训练机制是否有效。

## 6. 验证记录

本次相关回归测试 **82 passed**（XVLA online/控制目标、PI0.5 online训练与连续生成、fast/runtime组件）。
新增覆盖：融合前dropout阻断图像→语言泄漏、有/无callback的XVLA同步路径、dropout采样与编码顺序、PI0.5 teacher新增prefix。
Ruff与`git diff --check`通过。真实Phase B权重已验证融合前mask语言特征差异为0。
闭环诊断已产出完整episode视频与task summary，剩余任务由tmux继续执行。


## 7. 四组闭环诊断完成

Phase B旧权重：zero 37/40，online25 38/40，stale25 38/40，noadapters25 37/40；全部queue underruns=0。
normal25与stale25逐task成功/失败完全一致，分歧0/40。
这支持旧权重对新观测内容利用不足，但40回合仍是诊断样本；不能据此声称delay25优于零延时。

修复dropout的Phase A10k对照于2026-11-01北京时间22:15完成，耗时3h02m，最终训练loss0.172。
该loss混合了有/无slow视觉的样本，不能和旧泄漏版loss直接比较，也不能作为成功率。
输出：`/data/yhr/output/online_filtering/ckpt/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101/checkpoints/010000/pretrained_model`。
观测敏感性检查输出将保存为 `phase_a_dropoutfix_10k.json`，成功率评测尚未完成。


## 8. 修复版10k与旧版10k的同预算离线对照

同8个数据片段、同噪声、同query，均为10k权重。query9（old22/new25）结果：

|权重|正确fast loss|打乱图像loss|打乱图像输出变化RMS|冻结fast loss|
|---|---:|---:|---:|---:|
|phase_a_old_10k|0.0175965|0.0175965|1.7e-06|0.0175961|
|phase_a_dropoutfix_10k|0.0175141|0.0189449|0.0264|0.0170007|

修复后模型对视觉内容明显更敏感，打乱图像导致误差上升；但冻结fast未导致误差上升，且正常输入误差与旧版接近。不能把内容敏感性等同于利用新鲜观测的校准收益。样本仅8个训练片段，修复版成功率尚未测定。

## 9. 修复版Phase B30k离线检查及正式评测启动（2026-11-03）

同8片段、同噪声和query；权重：`xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102/checkpoints/030000/pretrained_model`。
这是canonical particle上的离线依赖性探针，不是最终在线rollout的完整分布。

| query | 正常loss | 关闭adapters | 打乱fast图像 | fast重复起点图像/状态 |
|---|---:|---:|---:|---:|
| 0 | 0.0037689 | 0.0058841 | 0.0044752 | 0.0037689 |
| 5 | 0.0034931 | 0.0046463 | 0.0040600 | 0.0029853 |
| 9 | 0.0037586 | 0.0051821 | 0.0036931 | 0.0030707 |

三处关闭adapter均增大误差，fast分支在当前探针上有作用；打乱图像引起输出RMS变化0.0544/0.0389/0.0143。
但query9打乱图像未增大误差、重复起点观测仍较好，因此不能声称最新观测稳定改善预测。
关闭adapter的原生transformer数值一致性仍为最大误差0。
原始输出：`research/outputs/xvla_implementation_audit_20261101/phase_b_dropoutfix_30k.json`。

已按正式400回合协议启动delay15/20/25各400回合，GPU0/6/7各5模型，共15路并行。
Tmux `xvla50_dropoutfix_eval_parallel`，输出与collector路径见metrics末尾本轮章节。
评测完成后自动汇总并更新metrics；当前不报告成功率。

本轮pool首次启动发现GPU0的可见列表被构造为`0,0`，导致CUDA invalid device ordinal并回退CPU。
在任何结果汇总前中断该批，修正GPU0只使用`0`（其余卡仍附加0满足robosuite EGL设备检查），重启评测。
离线诊断本来使用CUDA_VISIBLE_DEVICES=0，无此问题；正式本轮结果以重启后的pool为准。
