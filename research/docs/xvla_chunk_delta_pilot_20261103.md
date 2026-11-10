# XVLA chunk内差分、相对OSC执行的小规模对照（2026-11-03）

> **已停止，不作为等价相对控制或延时损害的有效对照。** 同状态真实OSC检查确认缩放/旋转转换正确，但相邻控制器目标差分不等于原始相对命令，会消除持续跟踪误差补偿。详见末尾排查结果。

## 目的与结论边界

测试控制接口改变是否放大延时影响，以及Online Filtering在同一接口下是否仍有收益。
不是为了选择性降低baseline：baseline和修复版Phase B完全相同地转换，保留零延时对照，报告所有已预定delay。
原绝对接口也是OSC动力学控制，并不瞬移、也不保证一步到达目标。
本实验保持已训绝对位姿模型权重，只改变执行接口，是接口迁移压力测试，不等于重训了相对动作模型，
也不等于PI0.5原始相对动作标签。结果不能直接替换绝对控制主表。

## 转换定义

从一个绝对目标chunk `(p_i,R_i,g_i)` 生成归一化LIBERO相对命令：

- i>0：`delta_p_i=(p_i-p_(i-1))/0.05`。
- `delta_rot_i=Log(R_i R_(i-1)^T)/0.5`，世界坐标左乘，不能直接对6D或axis-angle坐标相减。
- 仅i=0使用本次推理触发时刻的实测controller-site位姿作为previous pose。
- gripper始终为原来的开合命令，不取时间差分。
- 控制器照常把命令裁剪到[-1,1]，记录发生裁剪的控制步数。

先对完整chunk差分，再按delay丢掉前缀。新块在d切入时仍使用`p_d-p_(d-1)`，
**不**改成`p_d-p_current_measured`；后者会重新产生同一绝对目标，不能测试目标间差分的影响。
这也意味着被跳过的预测前缀并未实际执行，不保证新块差分轨迹与旧块实际末端连续，必须作为接口迁移的限制报告。

整个chunk一致的绝对平移会在差分中抵消。因此这种接口可能同时消除模型依赖绝对整体位移表达的校准。
若本实验OF没有优势，不能单独据此否定其绝对目标校准；若要独立研究切块衔接，需另设边界转换规则，不能事后改本轮规则。

## 已执行prefix对齐

相对命令实际作用于当前实测位姿，不能继续把原绝对预测作为已执行prefix。
每步用实测controller-site pose、实际裁剪后的delta和上一orientation goal，重建实际OSC绝对目标：
`p_goal=p_measured+0.05*clip(delta_p)`，`R_goal=Exp(0.5*clip(delta_rot)) R_measured`。
零旋转命令保留上一goal（与OSC一致）。将该20D绝对目标提供给仍使用绝对模型坐标的Online Filtering prefix。
首次零旋转时若无先前命令记录，用实测site orientation初始化；不能声称恢复了不可观测的reset前controller内部goal。
baseline使用同样的相对执行与裁剪，gripper和相机/数据处理保持一致。

## 预先固定的pilot

- baseline：`xvla50_corrected_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model`。
- OF：`xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102/checkpoints/030000/pretrained_model`。
- delay0/15/25，每个模型每档40任务各1初始状态，共240回合。
- 20Hz、replan20、horizon50、suite horizons280/280/300/520。
- 同task/episode重置采样seed，使用`--diagnostic-paired-seed`；该pilot不混入原400回合正式协议。
- OF受控simulator使用query时刻直接编码当前观测，无观测buffer；baseline保持原生生成。
- GPU2/6/7每卡5worker、共15路；不停止其他训练。

先读零延时成绩与裁剪率：若零延时已大幅退化，优先解释为接口迁移/衔接损伤，不能说成纯延时导致。
40回合仅用于筛选后续实验，需要完整样本与适当对照才可形成主结论。

## 实现与验证

- `src/lerobot/policies/xvla/utils.py`：目标差分与实际相对命令目标重建。
- `research/scripts/eval_libero_synthetic_delay.py --xvla-chunk-delta`：显式实验开关；默认绝对路径不变。
- 单独对XVLA设置`--control-mode=relative`但未转换输出现在明确报错，避免把米/绝对角度直接当归一化delta。
- `research/scripts/run_xvla_chunk_delta_eval.py`：固定六组小规模对照。
- 新增测试验证理想跟踪时重建、非交换旋转、延时切入不重定位、裁剪/零旋转保留、gripper不差分。
- 相关回归测试26 passed，Ruff与git diff --check通过。

Tmux：`xvla_chunk_delta_1103`，progress窗口。
输出：`research/outputs/xvla_chunk_delta_20261103`，pool.log及最终comparison.json。
目前未报告pilot成功率；后续需同时看完成数量、queue underruns、delta_clipped_commands。


## 零延时异常排查与停止（2026-11-03）

已停止本轮pool及worker，保留日志和部分episode，仅用于审计，不发布为相对控制主结论。
前期只用理想跟踪条件测试代数关系不充分，未在批量启动前检验实际控制器语义。

真实原始专家数据30个parquet共20,258对连续步：原平移命令RMS0.33540，目标差分后0.08643，
误差RMS0.25927，相关系数0.9097。令 `G_i=S_i+0.05*u_i`，则严格有：
`(G_i-G_(i-1))/0.05 = u_i + (S_i-G_(i-1))/0.05`。
右边额外项就是上一步未消除的跟踪误差；只有当前实测位姿等于上一目标才是等价转换。
即使原始相对命令恒定，目标差分也会将其削弱，控制器不能维持原来的纠偏驱动力。

真实安装OSC在同一仿真状态、未执行偏差动作的set_goal检查：
- 配置确认平移scale0.05、旋转scale0.5。
- 目标减当前实测位姿的逆转换：与绝对目标position最大误差0，rotation matrix误差1.22e-6。
- 已执行prefix目标重建与controller position误差0。
- 减去相差1cm的未到达上一预测目标，实际controller目标也差1cm，直接复现语义差异。
- 部分零延时失败轨迹没有动作裁剪，不能把异常主要归因于clip。

证据：`expert_difference_check.json`、`check_real_controller.py`、`real_controller_check.json`。
新增回归例：真实每步前进2mm、持续命令目标偏移10mm时，原归一化相对命令为0.2，
目标差分在后续步变为0.04。不能将这个退化解释成延时导致。

若目标是保持同一物理目标，逐步减当前实测state能做正确逆转换，但会保留绝对目标反馈，
不应预期因此baseline显著变差。若目标是PI式开环相对动作，需要定义/训练相应动作语义，不能直接假设相邻goal差分等价。

## 执行接口修复与验证

原`--xvla-chunk-delta`已在入口禁用并明确报错，不再运行“目标差分等价PI相对命令”的实验。
新开关：`--xvla-relative-execution`。
队列保留绝对20D目标，执行每一条前以当时的实测pose生成7D归一化相对命令，不能在生成chunk时预先计算整个相对队列。
模型及target坐标未改变；已执行prefix按实际相对命令重建的绝对目标填写。
队列耗尽时保留最后一个绝对目标，经同一转换发送，不会把20D全零当有效旋转。

零旋转细节：OSC收到严格零旋转会保持上一次goal；当请求的目标恰好为当前姿态、但旧goal不同/未知时，
逆转换发送1e-7rad更新以触发controller重置姿态目标，误差有界于数值精度附近。
相对命令仍遵守OSC[-1,1]限制，超范围会记录裁剪，不能保证这类目标与无限制绝对命令等价。

真实模型完整回合（baseline、Spatial task0、init0、seed1000、delay0、replan20、horizon280）：
原绝对与修复相对均step73成功、reward1、underruns0、clipped0；初始预测和chunk切换事件完全一致。
它是一个回合的正向回归检查，不是完整成功率验证，也不证明所有控制状态等价。
证据：`research/outputs/xvla_relative_fix_20261103/comparison.json`及两组summary/video。
相关回归测试29 passed，Ruff/git diff --check通过。

修复的是相对执行接口；保持同一绝对目标的转换预期保留原有反馈能力。
若研究PI式模型直接预测相对命令的开环chunk，应使用相对命令标签训练，不能把此适配器当作已完成该模型迁移。

## 用户授权修复接口的并行对照

新脚本`research/scripts/run_xvla_relative_execution_eval.py`，不复用旧输出。
baseline/OF各delay0/15/25，每档40回合，GPU2/6/7各5模型、共15路。
输出`research/outputs/xvla_relative_execution_20261103/comparison.json`，tmux `xvla_relative_exec_1103`。
聚合前检查实际接口标记、relative mode、delay范围、任务唯一性及无smoke截断。
此轮只验证修复后的执行接口；原目标差分数据保持停止/审计状态。

## 最新用户执行定义修正

用户明确要求整块预计算delta、逐步执行，实时逆转换不符合该要求。
已新增并验证 `--xvla-precomputed-deltas`，见 [预计算差分执行说明](xvla_precomputed_delta_execution_20261103.md)。
该路径不重算发送命令，不附加幅度增益。它恢复了最初目标差分操作，不代表PI原始标签等价问题已经消失。
此前38/40等实时逆转换结果仍保持独立标签，不作为用户期望路径的结果。
