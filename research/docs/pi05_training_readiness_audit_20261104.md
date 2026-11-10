# PI0.5 + Online Filtering 开训前检查（2026-11-04）

本次检查覆盖当前双时刻 online_flow 路径、canonical / EMA rollout 两阶段、实际 LIBERO 时序数据、预训练加载和现有训练脚本。已修复下列问题，真实模型的两阶段前向、反向与参数更新检查通过。

## 检查发现与修复

| 问题 | 影响 | 修复 |
|---|---|---|
| PI05.from_pretrained 在读文件或加载权重异常后返回模型 | 训练可能使用未加载或部分加载的权重继续运行 | 失败抛出带来源路径的异常；覆盖文件缺失、错误键、shape不匹配和正常加载 |
| EMA rollout 初始化复用 canonical 的补全逻辑，缺失在线模块时会重新初始化 | Phase 2可能未接续Phase A学到的双时刻分支 | Phase 2要求resampler、adapter、old/new role及delay-pair权重完整；canonical仍允许从backbone初始化新模块 |
| online_flow配置允许单次去噪查询 | canonical时钟要求至少两次查询，进入训练后才报错 | 配置阶段验证num_inference_steps≥2 |
| 零延时online_flow被旧posterior的stage_b gap检查拒绝 | 无法配置max_inference_delay=0对照 | gap约束仅应用于旧stage_b路径 |
| generic Phase 2脚本将dropout起点设为0，继承终点0.25 | 原意为退火，实际却逐步增强dropout | 显式设0.25→0，前6000步保持，随后18000步退火 |
| 四个PI05训练脚本保留/data/zhoukai路径 | 本机无法直接找到初始权重、tokenizer和数据 | 改为现有/data/yhr/pretrain路径 |

修复脚本：`run_online_flow_phase1_3xa100.sh`、`run_online_flow_phase2_2xa100.sh`、`run_online_flow_no_innovation_phase1_2xa100.sh`、`run_online_flow_no_innovation_phase2_2xa100.sh`。本次未启动正式训练。

## 已核对的计算路径

- Phase A使用 `training_stage=online_flow, online_flow_phase=canonical`，Phase 2使用 `online_flow_phase=ema_rollout`。
- old/new每时刻64视觉tokens加1个proprio token，合计130条件tokens；相机与状态选取相同物理索引。
- slow图像使用时间窗第0帧，加载的processor已将prompt state的temporal_index协调到0。
- PI05 slow视觉mask在VLM融合前应用；canonical与rollout student均使用该路径。
- 查询时钟为floor(kD/(N−1))；teacher预测前commit新增前缀，更新后再次保持前缀；不同rollout长度的样本互不污染。
- canonical目标z−A；rollout目标(x−A)/t；loss只覆盖有效后缀与真实动作维。
- EMA teacher不注册到student参数，不参与梯度更新；训练器初始化、EMA更新及teacher恢复路径已检查。
- DDP默认find_unused_parameters=true，兼容当前未参与online_flow计算的旧posterior模块。

## 真实模型与真实数据检查

使用现有LIBERO初始权重、真实DINOv3与本地tokenizer，在GPU0运行有限步诊断，结束后释放模型；未保存训练checkpoint。

输入：两路相机各 `[1,26,3,256,256]`，状态 `[1,26,8]`，动作 `[1,50,7]`。使用实际预训练processor与归一化统计，数据读取走正式delta_timestamps解析。metadata为10fps，按其时间戳读取连续行。

设置：BF16 backbone、gradient checkpointing开启、slow backbone可训练、DINO固定。为直接检验fast分支的梯度路径，诊断强制完全遮挡slow视觉。优化器使用SGD，仅执行canonical两步及EMA recovery一步。

| 阶段 | Loss | 结果 |
|---|---:|---|
| canonical第1步 | 1.3639574 | 输出投影梯度非零，其余新增模块梯度为零，符合零初始化残差设计 |
| canonical第2步 | 0.8663560 | adapter attention、resampler、old/new role、delay-pair梯度均非零 |
| EMA rollout | 1.1810083 | 前向、反向、student更新及teacher EMA更新通过，teacher无梯度 |

第2步梯度norm汇总：adapter输出投影7.0330，adapter attention0.003140，resampler特征投影0.00002722，role0.00002712，delay-pair0.0003631。所有已生成梯度和loss均有限。

诊断峰值allocated约28089 MiB，包含student与EMA teacher；正式Adam训练及更大batch另需预算优化器状态和激活显存。以上loss用于验证计算可运行，正式实验仍从初始权重独立启动。

产物：`research/outputs/pi05_pretrain_audit_20261104/{smoke.py,smoke.log,smoke_report.json}`。

## 验证记录

- 当前相关单元测试87项通过，覆盖条件构造、连续推理、时间采样、前缀、dropout、RTC、加载失败与Phase 2权重完整性。
- 原PI05/OpenPI processor一致性测试最初因远程gated tokenizer返回401而失败；改用已有本地PaliGemma tokenizer运行原测试，通过1项。运行器为 `research/outputs/pi05_pretrain_audit_20261104/run_local_tokenizer_parity.py`。
- 合计88项通过；修改的Python文件Ruff通过，四个脚本bash语法检查通过。

## 正式训练使用的资源

- baseline：`/data/yhr/pretrain/recovered_zhoukai/pretrain/pi05_libero_finetuned`。
- DINOv3：`/data/yhr/pretrain/dinov3-vits16-pretrain-lvd1689m`。
- tokenizer：`/data/yhr/pretrain/recovered_zhoukai/pretrain/paligemma-tokenizer`。
- dataset：`/data/yhr/pretrain/recovered_zhoukai/dataset/libero`。

当前正确顺序为：从baseline启动canonical Phase A，完成后以其最终双时刻权重启动EMA rollout Phase 2。实际启动时显式填写GPU、输出目录、batch和训练步数；如训练完整backbone，设置 `train_expert_only=false, freeze_vision_encoder=false`。训练脚本的默认GPU和输出目录沿用原配置，尚未为新一轮指定。


## 本轮正式训练已启动

- 2026-11-04北京时间19:49启动，仅GPU6/7；Phase A canonical，H50、max delay25、10次去噪。
- 每卡batch16，global32，梯度累计1；30k更新，每10k保存，无训练中评测。
- DINO固定，完整slow backbone与action expert可训练，gradient checkpointing开启，BF16；AdamW峰值学习率2.5e-5、1000步warmup。
- slow视觉dropout在step500后用2000步从0升到0.25，选中样本的slow视觉tokens完全mask（min=max=1）。
- 实际已连续完成30步，无OOM及数值错误；双卡各约51875MiB，稳定步速约2.18秒。预计北京时间11月5日14:00–15:00完成Phase A。
- 输出：`/data/yhr/output/online_filtering/ckpt/pi05_phase_a_clean_2gpu_b16_30k_20261104`。
- tmux：`pi05_phase_a_67_1104`，train/log窗口。
- 日志与manifest：`research/outputs/pi05_phase_a_67_20261104/`；复现脚本及源文件快照在其artifacts目录。
- 已完成初始稳定性确认，按用户要求停止持续监控。


## Phase 2接续训练（2026-11-05）

Phase A于北京时间11月5日13:56完成30k并保存最终权重。Phase 2先尝试GPU2/6；GPU2另有约25GiB占用，batch16首次优化器更新OOM。batch8初始化期间按用户新指令停止该尝试，改用GPU6/7重新启动。旧尝试保留在 `research/outputs/pi05_phase2_26_20261105`。

当前Phase 2：GPU6/7，每卡batch16、global32，30k steps、每10k保存，无训练中评测。使用Phase A最终030000权重，online_flow / ema_rollout，rollout1–9、EMA0.999、LR5e-6、warmup300。slow dropout前6000步0.25，随后18000步退火到0；每个选中样本的slow视觉全mask。max delay25、H50、10次去噪保持一致。

已连续完成20步以上，loss和梯度有限，无OOM；双卡约60GiB，稳定约6秒/步，预计北京时间11月7日傍晚完成（约50小时）。初始稳定性确认后停止持续监控。

- tmux：`pi05_phase2_67_1105`，train/log窗口。
- 输出：`/data/yhr/output/online_filtering/ckpt/pi05_phase2_clean_67_2gpu_b16_30k_20261105`。
- 复现脚本、日志、manifest：`research/outputs/pi05_phase2_67_20261105/`。
