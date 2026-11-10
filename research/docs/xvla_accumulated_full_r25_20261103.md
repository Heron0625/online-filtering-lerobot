# 正式规模累计delta曲线：统一replan25（2026-11-03）

用户确认统一replan25，并补齐低延时曲线、提高每卡并发。
本轮4800回合已经启动；不延用replan20，也不混入此前40回合pilot。

## 固定设定

- 模型：baseline50 final30k；修复dropout的OF Phase B final30k，路径保存在manifest。
- 控制协议：[延时切入delta定义](xvla_delayed_delta_protocol_20261103.md)，相对轨迹增量累计于持续目标。
- 高层动作：观测时生成固定delta；等待时执行旧chunk；切换时跳过d项，以实际切换pose初始化一次base；后续只累计。
- 底层absolute OSC仅跟踪累计后的目标，不恢复原预测的全局坐标。
- OF已执行prefix记录实际累计目标。
- replan25，20Hz，H50，delay0/5/10/15/20/25。
- 各档实际触发周期均25；D25时队列覆盖R+D=50，无额外排队或第二个在途完整推理。
- 每模型每档40任务×10初始状态=400回合，两个模型六档共4800。
- 正式规模沿用此前metrics的10初始状态，不声称覆盖LIBERO全部50初始状态。
- suite horizons280/280/300/520，paired per-task/episode seed1000，OF每query直接编码当前sim观测。

## 并发与数据保存

- 仅使用用户指定GPU1/5。初始每卡16路；11月3日13:29（北京时间）因其他进程占用增加发生显存分配失败，现续跑配置为GPU1四路、GPU5十二路，共16路。原GPU6/7上的评测已停止。
- OF最坏delay25、LIBERO10上60控制步的资源探针：CUDA peak allocated3751.75MiB、reserved3898MiB。
  初始预算未涵盖后续外部进程的显存增长；恢复时按当前实际占用降低并发。
- 不保存全量演示视频，跳过额外render/annotation/ffmpeg。真实模型相机输入仍由simulator生成，每个物理控制步照常推进。
  不将该渲染配置下的wall-clock指标与此前带视频运行混为实时性能对比。
- 各work item负责同suite两个task、每task10回合；模型/delay交错、共享动态工作队列。
- 启动分批错开，避免全部模型同时加载的瞬时峰值。
- 完成任务校验后原子发布进度，支持复用完整且同协议的任务结果，也支持从逐回合保存的前缀继续。续跑会推进初始状态计数，已验证下一回合首步动作、成功与步数和原结果一致。资源分配失败不计为模型失败，工作项重新入队并退出该worker槽位。
- 子进程独立进程组，父进程中断/失败时统一终止，避免旧进程继续写同一路径。
- 聚合检查R25、delay、accumulated标记、actual tracker模式、unique task/episode、H50和无截断。

## 路径

- 脚本：`research/scripts/run_xvla_accumulated_full_eval.py`。
- tmux：`xvla_full_r25_1103`，progress窗口。
- 输出：`research/outputs/xvla_accumulated_full_r25_20261103`。
- 清单：manifest.json；进度pool_records.json与pool.log。
- 资源探针：artifacts/resource_probe；其截断summary不参与任何正式聚合。
- 最终comparison.json及各模型/delay目录下aggregate_summary.json；完成后自动更新metrics本轮章节。
- 相关执行回归和聚合校验测试37 passed。

## 表述边界

当前XVLA高层与PI0.5都生成一整块末端增量逐条执行，这是共同形式。
但PI/LIBERO原始相对命令针对当时实测pose，当前轨迹增量累计到持续目标；不把二者说成逐条控制信号严格相等。
本轮验证该固定执行协议下的baseline/OF延时曲线，不预设延时损害必须单调或在所有任务上都很大。


GPU迁移后仍复用同一结果目录与相同协议。已验证完整工作项可复用；未完成的20回合工作项重跑，避免跳过reset后初始状态错位。
新增episode_progress.json和逐回合wall_time_s日志用于状态和ETA，不参与控制或改变初始状态选择。
