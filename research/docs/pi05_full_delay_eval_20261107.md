# PI0.5全量延时评测（2026-11-07）

官方LIBERO PI0.5与双时刻Online Filtering Phase 2 final30k对照，delay0/5/10/15/20/25。两模型各40任务×10初始状态，每档400回合，共4800回合。

- 20Hz、H50、replan25，各档实际触发周期一致；10次去噪查询。
- 原生PI0.5 relative OSC，不使用XVLA差分转换。
- suite horizons：Spatial280 / Object280 / Goal300 / LIBERO-10 520。
- 配对种子1000，逐task/episode独立设种；初始chunk同步生成，后续delay期间旧chunk继续执行。
- 每query先推进到floor(kD/9)，直接编码当前多相机与proprio；保留已执行前缀。真实异步buffer路径保持可用。
- 原始官方checkpoint默认compile=true，本次两模型均禁用compile，使用eager推理。
- 关闭额外诊断视频，保留模型相机输入与正常物理仿真。
- 仅GPU0/1/4/5，按当前剩余显存分配3/3/5/4个worker，共15路；资源错误重新入队并退出相应槽位，不计策略失败。
- 每工作项两个task各10回合，共240项；支持完整任务复用及逐回合前缀续跑。

来源：

- baseline：`/data/yhr/pretrain/recovered_zhoukai/pretrain/pi05_libero_finetuned`。
- OF：`/data/yhr/output/online_filtering/ckpt/pi05_phase2_clean_67_2gpu_b16_30k_20261105/checkpoints/030000/pretrained_model`。
- launcher：`research/outputs/pi05_full_r25_20261107/artifacts/launch.sh`。
- tmux：`pi05_full_r25_1107`。
- 进度、逐回合输出与最终汇总：`research/outputs/pi05_full_r25_20261107/`。最终comparison.json通过完整性验证后自动更新metrics对应总表及四suite分表。

本轮单独记录R25结果，历史R20主表保留。最大delay25、LIBERO-10单回合资源探针位于artifacts/probe；不计入正式4800回合。


启动验证：直接交付、前缀与连续生成、PI05聚合校验、XVLA兼容性相关测试共62项通过，Ruff与diff检查通过。最大delay25、LIBERO-10完整520步资源探针peak allocated9220.36MiB、wall time195.87秒；178个观测packet全部在对应query被接收，queue underruns=0。探针成败不计正式评测。正式15个worker均已加载权重并进入仿真，无启动报错。
