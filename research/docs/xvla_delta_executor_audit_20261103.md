# 预计算目标差分：真实执行器参考点审计（2026-11-03）

## 问题定位

当前预计算、出队、单位换算与旋转公式按定义执行，200步trace已验证发送命令未被实时重算。
但接口接法存在语义不匹配：

- 差分数组表达“相对上一**计划目标**的增量”。
- LIBERO OSC的`set_goal_position(delta,current_position)`执行“当前**实测位置**加增量”。
- 它不把命令累计到上一控制目标。

源码位置：`research/scripts/eval_libero_synthetic_delay.py::step_env`将预计算命令直接发入relative OSC；
安装的`robosuite/controllers/osc.py::set_goal`调用`set_goal_position(scaled_delta[:3],self.ee_pos,...)`；
`robosuite/utils/control_utils.py::set_goal_position`返回`current_position+delta`。
因此不能从“发送的delta没被修改”推出“低层执行了预期的目标轨迹”。

## 真实控制器的最小物理实验

不加载策略、不采集恢复训练数据；三个模式均从同一LIBERO Spatial task0/init0/seed1000重置。
初始状态完全相同，仅测试x方向固定10mm目标，执行20个控制步。
原预计算差分序列为`[10mm,0,0,...]`，不涉及模型误差、分块、延时、裁剪或OF。

| 执行方式 | 最终x位移 | 最终目标距离 |
|---|---:|---:|
| 每步发送原绝对目标 | 10.437mm | 0.569mm |
| 预计算差分送入当前实测pose-relative OSC | 2.027mm | 7.992mm |
| 相同差分累计为目标，再由OSC追踪 | 10.437mm | 0.569mm |

第二个控制步，relative OSC的位置goal精确等于当时实测位置，距原目标约8.916mm。
即第一步没走完的目标被后续零命令取消，而非继续追踪。
绝对与累计目标两模式的逐步controller goals和实测位置逐元素完全一致。
这把接口差异从模型、延时和chunk切换等因素中分离出来。

证据：`research/outputs/xvla_delta_executor_audit_20261103/check_executor.py`、`results.json`、`validation.json`。

## 为什么PI0.5的相对动作不能从相邻goal差分直接取得

LIBERO原始相对命令u在每个真实状态S上生成controller goal：`G=S+0.05*u`。
它本身含有维持低层驱动力所需的相对偏移。
相邻G差分会额外减掉上一目标未完成的跟踪量，只有理想跟踪条件下才还原原命令。
此前20,258条专家数据检查中，原平移命令RMS0.3354、goal差分后0.0864，与物理实验一致。
PI相对接口也有低层反馈，不是每个delta都保证瞬间实现。

## 处理状态和后续边界

已停止当前240回合预计算delta评测，保留部分记录，不作为延时-only或PI接口等价结论。
不再通过继续批量评测掩盖这个已确定的问题。

1. 若要执行“相邻目标差分所描述的目标轨迹”，执行器应累计这些目标增量。
   这在同一chunk内重建原绝对目标，因此不能预设baseline会变差；跨chunk的累计参考点仍需明确定义。
2. 若要真正采用LIBERO PI式相对命令，应在原始相对命令标签上训练XVLA/OF。
   这可以使用现有原始LeRobot数据，不需要人为扰动执行或额外sim replay；但它是新的训练配置，而非已有权重的等价后处理。

本次只定位并验证执行器差异，没有擅自启动新的相对标签训练或恢复旧批量实验。
