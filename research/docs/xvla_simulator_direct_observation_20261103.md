# XVLA simulator 查询时刻直接取观测（2026-11-03）

## 时序定义

真机保留异步latest-ready缓冲；受控simulator可以在每次denoising query前同步取当前观测。
总控制delay D、N次查询时，query k的环境相对控制时刻为 `floor(k*D/(N-1))`。
先执行旧chunk推进到这个时刻，再处理当前图像/状态、通过DINO/resampler得到condition。
保留上一查询condition组成old/new，锁定已执行prefix，再运行当前去噪查询。
**不能把环境一次推进到D后再运行全部去噪，否则会改变训练使用的观测/flow时钟。**

## 原路径实际行为

XVLA只有每个policy实例私有的一个`_pending_online_packet`槽，不是跨模型共享的观测池。
runner同步执行每个控制step、编码每帧，并不断覆盖该槽；query前取最新capture_step。
没有异步编码线程，也没有XVLA wall-clock age gate。
所以query时拿到的仍是对应目标控制时刻的观测，主要问题是重复编码中间未使用帧及间接的程序结构。
不能仅因使用packet接口就判定此前成功率无效。

PI0.5的真实异步buffer另有ready_timestamp/max_age过滤，不能把XVLA结论自动推广到PI0.5或真机。
本次直接路径仅支持online XVLA；未改动真机异步buffer及PI0.5推理接口。

## 新直接路径

在 `eval_libero_synthetic_delay.py` 或 `eval_xvla50_delay_pool.py` 显式设置：

```bash
--xvla-observation-delivery=direct
```

环境每步仍推进并更新状态，但只在query目标时刻编码一次当前观测。
callback直接返回`XVLAObservationPacket`，模型直接消费，不写入pending槽。
packet在这里仅是携带tokens、delay、domain、执行prefix的结构化值，不承担排队。
同一物理时刻重复查询不重新编码；继续用同一观测，old=new，与训练时钟一致。
兼容异步调用：callback返回None时，模型仍走原buffer读取逻辑。
默认仍为`buffered`以保证已启动正式评测不混用路径；后续受控XVLA仿真可显式使用direct。

## 验证

新增delay5/15/20/25测试，shared noise下direct与buffered：
每次查询old/new delay、输入particle以及最终动作逐元素完全相同。
相关unit suite：13 passed。Ruff与git diff --check通过。

实际checkpoint：修复dropout的Phase B30k；LIBERO Spatial task0、init0、seed1000、delay25、replan20、horizon280。
两种路径各运行一个完整episode，均成功于step75、reward1、queue underruns0。
实际安装执行的两个新chunk共18次assimilation，事件轨迹完全相同，切换时刻为45/70，索引均25。
初始action diagnostics完全相同。该回合不等于验证所有任务、设备或墙钟异步情形。

总assimilation计数为buffered20/direct19：差异发生在episode已成功的最后一个未安装generation。
buffered在推进到终止step75前已编码step74；direct在发现该query终止后不再编码。
两条路径均丢弃该未完成生成，不用于动作执行；不能用这个差异声称有效控制轨迹不同。

原始证据：`research/outputs/xvla_sim_direct_20261103/{buffered,direct}/summary.json`、`comparison.json`。
本次验证在GPU5进行，未停止0/6/7上进行中的正式评测。

## 额外日志修复

`chunk_switch.online_assimilations` 原用全局events中`step>=observation_step`计数，
会把上一个chunk在共享边界时刻的最后一次assimilation重复计入。
改成当前generation局部`len(chunk_assimilations)`，只修正诊断计数，不改变推理、动作或成功率。

## 限制

这是受控物理步数的延时实验，不是墙钟性能评测。
直接路径减少不必要的编码，不能由此报告真机端到端延时或实时吞吐提升。
现有正式15/20/25评测继续使用buffered；本次检查没有发现必须重跑它们的观测时刻错误。
