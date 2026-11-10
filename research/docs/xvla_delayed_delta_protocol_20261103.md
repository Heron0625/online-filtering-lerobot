# 延时切入的相对轨迹执行协议（2026-11-03，按最新用户定义）

## 本轮研究问题

在t0观测生成新chunk的期间，旧chunk仍在执行。新chunk在d步后才能使用，前d项没有实际执行。
评测要保留旧chunk执行状态与新chunk原先预期之间的偏差，不能在切换时直接跳回新chunk的全局绝对目标。
同一执行协议下比较baseline/OF，检验在线观测是否能帮助校准后续动作。
不能以“零延时恢复成功率”代替核对延时切换语义。

## 最终落地定义

1. t0生成完整绝对目标序列G，立即预计算delta序列。delta0参考t0实测pose，之后参考前一预测目标。
2. 推理等待期间旧chunk及其目标参考继续执行，环境推进d控制步。
3. 新chunk安装时丢弃delta[:d]，不执行、不虚拟积分这些未执行项。
4. 第一条实际执行的新命令delta[d]到来时，用当时的实际pose初始化一次目标base B。
5. 此后每步仅更新 `B_pos += .05*clip(delta_pos)`、`B_R = Exp(.5*clip(delta_rot)) @ B_R`。
   gripper为原开合命令。chunk内不再用实测pose重置B，也不使用原绝对G重新定位。
6. 下一个新chunk真正安装后，才允许初始化它自己的base。初始化发生在第一条执行命令，而不是新chunk尚在推理时。

累计目标B交给绝对OSC作低层追踪器；这里的绝对目标是**偏移后的累计目标B**，不是重新索引原预测G。
新模式解决了relative OSC把后续零delta解释为“以当前实测位置重新设目标”、取消未完成运动的问题。
它不把累计误差通过每步减当前state消除。

位置例：预测G_(d-1)=.02、G_d=.03，实际切换位置=.10。
第一条执行delta=.01，累计目标=.11，不是原全局目标.03。
之后零delta保持.11，不能改为当时尚未到达.11的实测位置。
这区分了“退回原绝对轨迹”与“取消未完成增量”两种错误。

## 明确的开关

```bash
--xvla-precomputed-deltas --xvla-delta-execution=accumulated
```

- 默认旧`current_pose`增量接口保留审计/复现，不用于本轮结果。
- `--xvla-relative-execution`是每步实时逆转换，不能与本轮混用。
- `control_mode=absolute`只描述底层OSC tracker；高层执行语义由上述两个字段明确标记。
- OF已执行prefix记录实际发出的累计目标B，不能记录原G，否则会给模型错误的执行历史。
- delta裁剪仍为LIBERO已核对的[-1,1]、.05/.5缩放，无额外拟合增益。

## 验证证据

- 32项相关unit测试通过，包含恒定目标后续零delta保持、无delay重建、delay切入保留偏移、非交换旋转。
- 真实baseline/OF各delay0/15、每组50控制步（共200步，仅执行诊断）：
  - base仅在初始以及chunk安装处重置，其他步精确沿用上一累计目标。
  - delay0重置发生step0/20/40，chunk索引均0。
  - delay15重置发生step0/35，第二次索引15，没有补执行被跳过项。
  - 累加位置公式最大误差约1.9e-9米，发送的低层位置目标等于累计目标。
- 前次真实控制器恒定10mm目标实验中，累计方式与绝对目标方式实际轨迹完全相同，均前进10.44mm；直接当前pose-relative方式只前进2.03mm。

原始trace：`research/outputs/xvla_accumulated_contract_20261103/validation.json`及四组summary。
Ruff/git diff --check通过。本轮50步检查不作为成功率结果。

## 已启动的独立评测

- baseline50 final30k、修复dropout Phase B final30k。
- delay0/15/25，各模型各40任务×1初始状态，共240回合。
- 20Hz、replan20、horizon50、suite limits280/280/300/520、paired per-episode seed1000。
- OF使用query时刻直接编码当前simulator观测。
- GPU6/7每卡5模型、共10进程；不占用其他训练。
- 启动脚本：`research/scripts/run_xvla_accumulated_delta_eval.py`。
- tmux：`xvla_accumulated_eval_1103`，progress窗口。
- 输出：`research/outputs/xvla_accumulated_eval_20261103/comparison.json`。
- 严格检查precomputed=true、delta_execution=accumulated、实测逆转换=false、底层absolute tracker、delay匹配、无截断及完整task覆盖。

此前直接relative-OSC失败、实时逆转换95%等结果都不混入本轮。
这是对既有绝对输出模型采用新的相对轨迹执行协议，不能声称模型已在PI原始相对标签上重新训练。


## 本轮完成与待核对问题

240回合完成，完整数字以metrics为准。baseline在delay0/15/25为37/12/16成功，OF为38/33/27成功（各40回合）。
所有queue underruns为0。大delay差距明显，但不能忽略零delay等价性尚未完全验证：
Spatial task5 init0、matched seed下，原绝对baseline132步成功，累计delta baseline280步失败，均无裁剪。
证据在`research/outputs/xvla_accumulated_zero_parity_20261103`。此差异尚需执行trace定位，不能先用延时解释或直接归因数值误差。

## replan与实际触发周期核对

已直接检查runner和本轮episode事件。安装新chunk后，`countdown=max(0,replan_interval-delay)`；
由于只有一个完整推理在途，固定delay D时稳态触发周期为`max(R,D)`。
R20下：D0触发20/40/60；D15触发20/40/60、切换35/55/75；D25触发20/45/70、切换45/70/95。
因此当前D15与D25不只观测年龄不同，还涉及实际更新周期20对25的差异。
若下一轮统一R25，则D0/15/25的触发周期均25；H50能在固定D25仿真中正好覆盖R+D=50。
这提高延时比较的可解释性，不代表一定提高成功率。对于D25，R20和R25的稳态周期都已经是25，
变化主要包括首次触发由20改25，以及后续闭环轨迹的相位；不能声称R25会自动修复D25性能。
