# PI05 old/new fast 条件与真实推理时钟核查（2026-11-10）

核查依据：当前仓库训练/推理源码、下载 checkpoint 的 config.json/train_config.json、用户提供的在线推理日志。
导出配置不能独立证明训练作业运行时的源码版本；以下关于训练调度的结论针对当前仓库实现，未将配置视为服务器源码快照。

## 结论

- 推理确实向 expert 输入“上一去噪步使用的 current fast”和“本去噪步最新 ready fast”两组条件，没有漏掉 old fast。
- `assimilated=[[0,9],[1,12]]` 只记录新包的 `[flow_step,capture_step]`，并没有展示 old/new 完整配对。
- 这组日志对应 `(old=0,new=9)`、`(old=9,new=12)`。如果下一步没有新包，则下一步使用 `(12,12)`，不是继续使用 `(9,12)`。
- 真实时间上的因果性正确，但当前训练的线性观测时钟与部署存在明显偏移；“接口正常工作”不能等价为“与训练分布一致”。

## 当前 checkpoint

- `training_stage=online_flow`，`online_flow_phase=ema_rollout`。
- `num_inference_steps=10`，`max_inference_delay=30`。
- delay 范围 `[1,10] / [11,17] / [18,25] / [26,30]`，概率 `.15/.20/.50/.15`。
- `online_flow_rollout_steps=4`，`online_flow_canonical_weight=0`，`online_flow_rollout_weight=1`。
- export 的全局 EMA 配置 `ema.enable=false`，不要把 online_flow 的 EMA teacher 与部署平滑或全局 EMA 权重导出混淆。

## 训练管线

`configuration_online_filtering_pi05.py:195` 为四相机和 state 请求 `[0..max_delay]` 的时间窗口。
`datasets/factory.py:87` 将这些控制步索引除以 dataset FPS，变成物理采样时间。
模型 `_make_fast_tokens` 从相同 delay 索引取得各路图像和 state，生成该物理时刻的 fast 条件。

`modeling_online_filtering_pi05.py:587` 的 `_make_online_condition_tokens` 为 old/new 加不同 role embedding，
并嵌入 `(old_delay,new_delay,new_delay-old_delay)`，最后沿 token 维拼接。expert 接收到两组条件。

canonical（该文件 923 行）使用：

```text
d[k] = floor(k * D / (N-1))
old_delay = d[max(k-1,0)]
new_delay = d[k]
flow_time = 1-k/N
```

EMA rollout（990 行及 1016 行）也使用相同的线性 schedule。
teacher 每一步用 `old_step=max(step-1,0)`；student recovery 使用 `old_steps=recovery_step-1`。
teacher 前后都会硬锁新观测时刻之前的 GT 动作前缀，student loss 只覆盖可执行后缀。
当前 checkpoint 的 recovery_step 从 1..4 采样；不能从此推断继承的 backbone/初始化权重没有其他时间步知识。

以 D=23、N=10 为例：

```text
delay schedule: 0,2,5,7,10,12,15,17,20,23
old/new pairs:  (0,0),(0,2),(2,5),(5,7),...
```

所有训练 schedule 的 k=0 都为 `(0,0)`。D<=30 时，相邻 delay 增量最多 ceil(30/9)=4。
虽然 metrics 中保留 jitter/burst/stall fraction 名称，当前采样函数只设 `mode=0`，实际不生成这些时钟。
`online_flow_switch_probability` 的配置值也不能据此视为已经训练随机到达调度。

## 部署管线与差异

`predict_action_chunk` 在 1378 行先同步计算 slow prefix/KV cache，随后编码初始 fast，才进入 1422 行的去噪循环。
与此同时 server 的 DINO worker 持续编码新包。
因此，第一次去噪查询可能已经能取得控制步 9、10 或 11 的特征。
具体等待还可能包含预处理、初始 fast 编码、线程/GPU 调度，不能仅凭 assimilated 日志把整段归因于 slow prefix。

每步 1424 行先保存 `previous_fast_tokens=current_fast_tokens`，再消费新包，
1497 行将 previous/current 一起构造条件。没有新包时 previous/current 相同，符合“上一去噪步使用的 fast”语义。
首次查询没有上一去噪步，old 以初始 fast@0 引导。

例如实际日志 `[0,9],[1,12]`：

```text
k=0, flow_time=1.0: old=0, new=9
k=1, flow_time=0.9: old=9, new=12
```

它们不等于上述训练时对应去噪时刻的配对。尤其第 0 步 `(0,9)` 在当前线性训练采样器中不会出现。
当前最新包跳过部分编码结果是 latest-ready 设计允许的行为，但其分布也没有被当前线性训练采样完整覆盖。

观测 age 虽在推理中计算/传递，用于 buffer 过期筛选，但当前 online_flow FastObservationAdapter 本身只接受 tokens/delays，
没有直接使用该 age 数值。不要认为 age 记录已经自动补偿了 flow/control 时钟错配。

## 判断与后续方向

1. 保留 old/new 双条件结构；这部分符合设计。
2. latest-ready 查询在真实异步系统中是合理的因果调度，但当前 checkpoint 对这套时序的适配不能由现有日志证明。
3. 不应把 capture_step=9 强行改成 0，也不应修改真实采集时间来伪装对齐；这会破坏动作前缀和执行时间语义。
4. 仅强制第 0 步不消费新包也不充分：可能只是把 `(0,9)` 的大跳变推迟到第 1 步。
5. 优先测量预处理、slow prefix、初始 fast、每个 flow step 的耗时和每步完整 old/new delay；
   用真实轨迹构建训练 schedule，覆盖前置耗时、同包复用、丢帧和观测跳跃，保持训练/推理相同的前缀承诺规则。
6. 优化前置计算有助于缩小差异，但不能代替训练与实际调度的匹配验证。不能仅凭本次审计认定该差异就是小停顿的原因。

## 训练期间如何对齐并覆盖真实 delay（建议实施方案）

本节是后续训练代码应实现的方案，不代表当前采样器已经支持。核心目标是：
**训练与部署使用同一个“观测何时采集、何时完成编码、在哪个去噪步被消费”的调度语义，
并覆盖 `(flow_step, old_delay, new_delay)` 的联合分布。仅覆盖最终总 delay 的直方图不够。**

### 1. 统一时间原点，区分四个量

每个新 chunk 以其初始观测的物理时刻为原点。客户端命令历史中该时刻对应的已下发序号为 origin。
以下所有步数均为相对该 origin 的控制步；25 Hz 下名义周期为 40 ms，但有调度抖动时应按真实命令时间定位，
不能用网络包序号、DINO 次数或简单的四舍五入替代客户端执行序号。

| 符号 | 定义 | 在训练中如何使用 |
|---|---|---|
| `q[k]` | 第 k 次去噪查询发生时，客户端已下发的控制步数 | 检查因果性、衡量前置等待及逐步计算时间 |
| `d[k]` | 第 k 次查询实际使用的 fast 观测的采集步数 | 选择图像/state、构造 new fast 与 delay embedding |
| `age[k]` | 查询时刻减去该 fast 的采集时刻，单位秒 | 模拟过期筛选和诊断；当前 adapter 并未直接学习这个数值 |
| `E_return` | 新块在 client 真正应用时，已下发的控制步数 | 评价返回时裁剪量、可执行后缀和队列余量 |

`q[k]`、`d[k]`、`E_return` 不应混成一个“总 delay”。编码和传输存在延迟时通常 `d[k] <= q[k]`；
最后一次查询之后还有最后一轮模型计算、反归一化、传输和客户端调度，所以 `d[N-1]` 通常也不等于 `E_return`。
`replan_steps` 决定上一块执行多少步后触发请求，不直接加进新块的 d[k]；触发之后实际消耗的时间必须统计。
不要把输入时间改成 server 收包时间或 DINO 完成时间，也不要将 d[0]=9 重编号成 0。

当前输入侧只有保守的 host 接收时刻，不是已同步的硬件曝光时刻；训练 trace 首先复现这一现有语义。
后续若引入可靠的曝光时间或改变四相机同步规则，应同时更新 trace 采集、训练采样和部署索引。

### 2. 先采集整条部署 trace，而不是只采集 assimilated 列表

在目标 GPU、实际四相机输入、相同 batch=1、精度和 10 次去噪设置下记录若干完整 rollout，覆盖正常运行、
冷启动和负载变化。每个 chunk 至少保存：

- request_id、origin、25 Hz 控制周期、源码/权重版本。
- 初始观测时间、server 接收时间、预处理完成时间、slow prefix 开始/完成时间、初始 fast 完成时间。
- 每个候选观测的 capture_step、capture_timestamp、ready_timestamp、generation、有效相机标志；
  ready 必须是 CUDA event 确认完成之后的时刻。
- 每个去噪步的查询时刻、flow_time、实际 old_delay/new_delay，以及是否复用了上一份条件。
- client 实际切换时刻和 E_return、旧队列余量、最终下发动作及 hold 段。

异机时间要映射到同一时钟并保存映射误差。只记录 `[0,9],[1,12],[4,15]` 不足以还原吞吐或年龄：
中间哪些步复用了条件、包何时 ready、何时被覆盖，都不能从这三个事件唯一确定。

### 3. 首选复用 latest-ready 规则回放生成训练 schedule

训练不必真的等待几百毫秒或在线运行 DINO。可用记录好的调度 trace，在离线数据上构造对应的时间索引。
给定一次 trace 的去噪查询时刻 Q[k] 和各包的采集 C[j]/编码完成 R[j]：

1. 初始 current 指向初始 fast@0。
2. 在第 k 步，先保存 old=current。
3. 按部署的 bounded-buffer、generation、有效性、max_age 与版本规则，从当时实际可见的候选包中
   取 capture_step 最大的 ready 包；只能使用 R[j] <= Q[k] 的包。
4. 有更新就将 current 切到该包，否则沿用原 current。训练的 new=current。
5. 用 old/new 的物理索引，从同一 episode 的时间窗口提取图像和 state。

在当前部署中，过期规则决定一个新包能否被接纳；没有可接纳新包时继续复用已有 current，
不要在训练中额外引入部署没有的“缓存一过期立即清空”策略。
同样要复现容量限制和 latest-only 覆盖，不能事后从所有完成过的包中任意挑一个历史包。

得到的 d[k] 单调不减，可以重复，也可以跳跃；第一项允许大于 0。
时间戳必须满足因果性，相同观测不可通过修改时间戳伪装成更新的观测。
slow prefix 和初始 prompt 始终来自初始观测，不能因为抽到 d[0]=9 就把 slow 图像或 prompt state 也换到第 9 步。

实测 trace 有限时，再用离散事件模拟补充覆盖：抽样前置计算时间 B、各去噪步耗时 L[k]、
相机/控制相位、编码耗时、传输与排队延迟，生成 Q[k] 和 R[j] 后仍用相同的查询规则选择包。
若各项时间以同一原点计量，可写成 `Q[k] = B + sum(L[i], i<k)`。
B 包含去噪前真正等待的所有步骤，不应只设成 slow prefix 的耗时。
不要直接独立随机抽每一步 d[k]，这会丢失相邻步骤、负载和丢帧之间的相关性。

### 4. old/new 的边界必须显式修正，尤其是第 0 步

采用新 schedule 后，正确配对是：

```text
new_delay[k] = d[k]
old_delay[0] = 0                # 初始 fast@0，独立于 d[0]
old_delay[k] = d[k-1]           # k > 0，上一去噪步实际使用的 fast
flow_time[k] = 1 - k/N         # 流时间与物理观测 delay 独立
```

例如下面是一条用于说明语义的完整 schedule，并非从现有事件日志唯一推导出的 trace：

```text
k:          0      1       2        3        4        5        6        7        8        9
new_delay:  9     12      12       14       15       15       17       18       18       20
old/new:  (0,9) (9,12) (12,12) (12,14) (14,15) (15,15) (15,17) (17,18) (18,18) (18,20)
```

不能只把 `_sample_rollout_schedule` 的首项改成 9，再保留 `old_step=max(step-1,0)`：
那样第 0 步会错误地配成 `(9,9)`。初始 fast@0 应单独缓存，或在 schedule 前显式放置一个初始哨兵索引，
teacher、canonical 和 student recovery 三处都通过同一个配对函数取得 old/new。

### 5. canonical 与 EMA rollout 两阶段都要使用新时钟

**canonical 阶段：** 从上述 schedule 中抽一个去噪步 k，允许 k=0；用对应 old/new 条件训练。
对新观测步 d=d[k]，动作索引 i<d 的 particle 固定为已执行动作，i>=d 的 particle 为
`t*noise+(1-t)*A`，目标速度为 `noise-A`，只对有效动作后缀计算 loss。
因此 `(flow_time=1,old=0,new=9)` 必须能在训练中真正被抽到，而不是只在日志中声明支持。

**EMA rollout 阶段：** teacher 从噪声开始，逐步回放同一条 schedule。
每次预测速度前先 commit 新增加的执行前缀，积分后再次 commit；每行样本停止 rollout 后不能被其他样本的后续步污染。
student 的 recovery 条件必须对应同一条 schedule 的上一步/本步，目标仍为 `(particle-A)/t`，只监督有效后缀。

当前 recovery 只抽 1..4，而部署会执行 0..9。建议让 recovery 覆盖 1..9，并由 canonical 分支覆盖 k=0；
采样分配按部署统计平衡早期大跳变、后期更新和 identity 步，训练成本需要实测。
不要直接取 recovery=N：该点 t=0，会使 recovery 目标退化或数值不稳定。

实现时可以先进行一段采用新 schedule 的 canonical 适配，再接新 schedule 的 EMA rollout；
也可以显式实现 `loss=lambda_canonical*L_canonical+lambda_rollout*L_rollout` 的混合训练。
**当前 EMA rollout 函数只计算 rollout loss，直接把 config 的 online_flow_canonical_weight 改成正数并不会自动加上 canonical loss。**
若采用混合方案，必须修改训练分支并验证两项 loss 和梯度都实际生效。
更换 schedule 的起步权重、优化器重置/续训方式和 loss 权重要通过对照实验确定，不能保证仅恢复现有 checkpoint 就能消除错配。

### 6. 执行前缀、离线动作与有效后缀必须一致

训练提取 fast@d 时，图像和 state 都取同一物理索引；动作前缀取同一 episode 的 `A[0:d]`，
经相同 action normalizer 后 commit，loss 排除 i<d 及 padding。不要用 state 代替命令前缀。
训练窗口需要有对应 d 时刻的真实有效观测，且 d<chunk_size；尾部缺帧、无未来动作的样本应重采样或正确掩码。

当前推理按照 capture_step=d 锁定前缀，而不是查询时已经执行的 q[k]；训练首先应复现这一现有契约。
如果以后希望锁到 q[k]，必须把“观测步数”与“commit 步数”拆成两个显式变量并同步修改训练与部署，
不能暗中把 fast@d 的 delay 改成 q[k]。

部署前缀包含线性 merge、最终关节滤波/单步限幅以及 hold 后的真实下发命令。
离线 demonstrations 的 GT 动作前缀只是这个量的代理，不会自动覆盖所有闭环偏差。
若要用真实策略执行前缀适配，需要采集与这些命令对应的闭环图像/state 和可信目标；
不能把另一条轨迹的平滑前缀随意拼到当前 demonstration，仍把原 GT 后缀当成必然正确的监督。

### 7. 覆盖范围与采样比例怎么定

优先按完整 trace 或保留相邻相关性的 trace 片段重采样，覆盖当前硬件的主体分布；再补压力样本与低 delay 样本。
一个可作为实验起点、尚未验证的比例是：70% 实测 trace、20% 因果一致的抖动/拥塞/丢帧模拟、10% 低 delay/无更新对照。
比例应根据验证集覆盖率和任务表现调整，不是固定推荐超参数。

至少检查下面这些类型确实进入训练 batch，而不只是配置或 metric 名称中出现：

| 类型 | 要覆盖的现象 |
|---|---|
| 前置延迟 | d[0]=0，以及实测中常见的 d[0]=8..12；范围应以完整采样统计为准 |
| 连续更新 | 相邻去噪步取到不同 fast，并带真实 old/new gap |
| 同包复用 | 无新 ready 包时 old=new；不能强迫每一步都换图 |
| 跳跃更新 | latest-only 跳过中间包，出现大于原线性 schedule 的增量 |
| 长尾延迟 | 前置计算或中间某一步变慢、包过期、编码队列拥塞 |
| 低 delay | 初始 (0,0)、小 gap 和快速返回，防止只适配慢设备后退化 |

`max_inference_delay=30` 是当前观测索引覆盖范围，不等于保证任意相位/任意调度都已学会。
训练时检查有效观测的最大 d，评估时另检查 E_return；在当前部署中 E_return>30 还会触发控制端停止。
若实测尾部超出 30，应明确选择降低时延或扩大训练/部署范围并重新验证，而不是将 trace 的 delay 静默 clamp 到 30。
扩大范围也需检查数据窗口、chunk 可执行后缀以及 delay embedding 的归一化尺度，不能只改 YAML 的一个数。

### 8. 具体改动点与验收标准

建议把调度、配对和有效性逻辑抽成共享组件，由以下位置调用：

| 改动点 | 必须达到的行为 |
|---|---|
| `sample_online_flow_transitions` | canonical 从 trace/事件模拟抽 k 与 old/new，覆盖 k=0,new>0 |
| `_sample_rollout_schedule` | 返回完整的 latest-ready schedule 和 recovery；模式确实实现而非只打标签 |
| `_forward_online_flow_ema_rollout` teacher | k=0 显式用初始 fast@0；每步前后 hard commit |
| 同函数 student | old/new 选取与 teacher/部署一致，扩展 recovery 覆盖，必要时显式加入 canonical loss |
| 数据窗口与配置 | 图像/state/action 对齐、padding 合法、索引上限与训练硬件/频率一致 |
| server/client trace | 同一时钟下记录每步 old/new、ready 与 query，并标明原始曝光时间是否可信 |

完成训练前后必须验收：

1. 同一 trace 输入离线调度器与真实推理调度回放，逐去噪步得到完全相同的 old/new 索引。
2. 覆盖 `(0,9)->(9,12)->(12,12)`、无更新、跳包、过期、跨 generation 和 episode 尾部案例，且没有消费未来包。
3. 验证 teacher/student 的 old/new token 内容、role 与 delay embedding 都对应正确帧，而不仅是数值索引相同。
4. 验证每次新增前缀都被正确锁定、后缀 loss/padding mask 正确、目标和梯度有限。
5. 按去噪步分别报告 d[0]、new_delay、gap、identity 比例的训练/部署分布，另外报告 age 与 E_return；
   检查实测 trace 中每类关键事件在训练和留出验证集中均有覆盖。
6. 在留出的 trace 和任务/episode 上比较旧线性训练与新时钟训练：先离线评估，再做受控闭环对照，
   同时统计任务成功率、队列 hold、最终指令速度/变化和跟踪误差，避免把强平滑带来的慢动作误判为模型更好。
7. 保存源码 commit、训练配置、trace 版本、数据 FPS 与动作归一化信息，保证之后能复现这套时钟。

## 已完成的代码核查验证

新增回归用实际 `predict_action_chunk` 方法验证 buffered/direct 两种到达方式：

```text
arrivals: (k=0,d=9), (k=1,d=12), (k=3,d=15)
actual condition pairs: (0,9),(9,12),(12,12),(12,15)
```

另验证实际训练 schedule 函数在 D=1/9/23/30 时从 0 开始线性推进、recovery 取值 1..4。
联合已有多次观测吸收与 teacher 前缀锁定测试，20 项通过。审计未更改部署的数值计算或观测调度。
