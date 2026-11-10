# 按用户定义恢复预计算chunk差分执行（2026-11-03）

本说明优先于前文将实时逆转换视为本实验修复的说法。用户要求：生成chunk后先算完整相对命令序列，再逐条执行，执行期间不按当前state重算。

## 三种模式明确分开

|接口|队列中存储|执行前是否用实时state重算命令|
|---|---|---|
|默认原生绝对控制|绝对目标|否，直接给绝对OSC|
|`--xvla-relative-execution`|绝对目标|是，逆转换到当时实测pose；不是本次用户要求|
|`--xvla-precomputed-deltas`|预先算好的归一化7D增量|**否；本次用户要求**|

两个相对开关不能同时启用。旧`--xvla-chunk-delta`仍报错，要求显式选择模式，防止旧脚本/旧输出名称继续混用。

## 预计算定义

- 第0个命令：目标0减本次生成触发时的实测pose。
- 之后命令：目标i减目标i-1；旋转使用`Log(R_i R_(i-1)^T)`，不直接减6D/axis-angle数值。
- 平移/旋转分别除以OSC已验证的0.05/0.5，只做单位转换，不乘拟合增益。
- gripper保留开合命令，不做差分。
- 每个新chunk只计算一次，delay d时执行预计算数组的`d:`，不在切入时重新定位。
- 真正发送的命令仅经过与LIBERO相同的[-1,1]裁剪。
- baseline执行分支不读取实时pose来生成命令。OF读取pose仅重建实际发出的绝对controller goal，用于已执行prefix；不修改队列或已计算命令。
- 低层OSC本身仍有反馈，这是相对控制器的正常行为；“不重算”指评测适配器不把排队命令改回绝对目标追踪。

## 实现核验

`research/scripts/eval_libero_synthetic_delay.py`新增明确开关及`--record-action-trace`。
每步记录queued_action、sent_command和execution_mode；默认不产生额外逐步trace。

真实baseline/修复版OF各delay0/15，共4个检查，每个50控制步（截断诊断，不是成功率评测）：
- 共200步，每步`sent_command == clip(queued_action)`，最大误差0。
- 初始queue第0个命令与initial diagnostics一致。
- delay0的新chunk索引为0；delay15新chunk索引为15。
- 所有summary均标记precomputed=true、measured-relative=false。
- 相关29项unit回归测试通过；Ruff/git diff --check通过。

证据：`research/outputs/xvla_precomputed_contract_20261103/contract_validation.json`，以及4组summary中的action_trace。

## 必须保留的解释边界

本次恢复的命令数学定义，与最初相邻目标差分相同；不是一种已证明能恢复PI原始命令幅度的新方法。
之前指出的controller goal差分与原相对命令不同仍成立：不能把上一目标当成已经到达的实测位置。
不能因为现在执行契约符合用户定义，就声称之前低成功率的原因已被数值修正，或声称与PI原生相对标签等价。

此前38/40、37/40等“修复后相对控制”成绩属于实时逆转换模式，不能用于本次预计算差分主张。
本次未自动重启240回合pilot，也不把50步检查中的成功/失败计入benchmark。

## 用户随后要求查看效果：启动完整pilot

此前仅验证200控制步，不曾自动启动完整回合。用户询问效果/完成时间后，已启动独立240回合pilot：
baseline/OF各delay0/15/25，每组40回合，GPU6/7各5进程。GPU2已被其他任务占用，没有抢占。
脚本`research/scripts/run_xvla_precomputed_delta_eval.py`，tmux `xvla_precomputed_eval_1103`。
输出`research/outputs/xvla_precomputed_eval_20261103/comparison.json`。
本轮命令定义与最初目标差分相同，没有暗中做幅度补偿；不能预设早先低成功率会因此消失。

## 后续真实执行器检查（优先阅读）

仅验证发送命令与预计算一致还不充分。已在真实OSC中隔离出目标参考点错配，详见
[真实执行器审计](xvla_delta_executor_audit_20261103.md)。固定10mm目标差分为[10mm,0,...]，
当前relative OSC只走约2.03mm，绝对/累计目标均约10.44mm。
当前预计算pilot已停止，不再用它判断纯延时效应。
