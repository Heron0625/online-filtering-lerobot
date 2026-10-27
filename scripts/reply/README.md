# RB75 双臂逐段回放

先修改 `scripts/reply/config.yaml` 的 `dataset.path`，指定一次采集作业：

```yaml
dataset:
  path: /media/rpp/2955b89d-64af-4f7f-bd14-ae4d766151b2/home/robuster/wsvla/nas_data/lerobot_rb75/10261644
    path保存在另一个系统中
  joint_source: observation.state
  speed_multiplier: 1.0
```

在本仓库根目录执行：

```bash
bash scripts/reply/reply.sh
```

也可以指定另一份配置：

```bash
bash scripts/reply/reply.sh /路径/另一份config.yaml
```

## 按键流程

1. 启动时预检所有 episode，不发送控制指令。若作业内有已转换完成的 `lerobot/`，优先读取；否则读取按 n 接受的 SQLite 暂存段，忽略 `.partial.sqlite3`。也可以直接把 `dataset.path` 指向 `lerobot/`。
2. 终端先列出可用 episode，输入编号并回车，选择从哪段开始。例如输入 `2` 选择 `episode_000002`，编号从 0 开始；编号有空缺时按实际列表选择。直接回车选择列表第一段，输入 `f` 回车退出。选择完成后启动所需驱动，并显示“按右方向键 →：先对齐到第一帧，然后播放本段；按 f：结束本次播放”。未按右方向键前不发送机械臂关节运动指令；自动启动的夹爪驱动仍可能执行初始化动作。
3. 按 → 后检查从臂反馈和控制话题，以默认最大关节速度 **3°/秒** 平稳对齐到本段第一帧。使用平滑起停的三次插值，至少用 5 秒；移动幅度大时自动延长。必须确认关节反馈到位后才播放本段。
4. 播放结束显示本段完成，再次提示按 → 播放下一段或按 f 结束。从所选段开始，按编号顺序播放后续各段，前面的段跳过；下一段也会先对齐到它自己的第一帧。
5. 所有 episode 播放完成后自动退出。等待、对齐、播放中均可按 f 结束；Ctrl+C 也可中止。不自动回到第一段起点。

结束或异常时停止发送后续轨迹；已开始运动时，在反馈仍有效且无其他控制发布者的情况下发送一次当前位置保持目标。它是软件停止，不代替硬件急停，不控制其他正在运行的程序。

## 机械臂及夹爪接入

回放使用现有 ROS2 servo adapter 控制链路，不在 50 Hz 循环里反复发送旧示例中的 TCP `movej` 规划命令。参考了 InterACT-LeRobot 的回放数据校验与单位约定，同时沿用 rpp_ws 中现有推理执行器的控制话题：

| 用途 | 左右臂话题（`{side}` 为 left/right） |
| --- | --- |
| 关节反馈 | `/vla_deviceea100/slave_arm/{side}/Joint_angle` |
| 关节指令 | `/vla_deviceea100/master_arm/{side}/Joint_angle` |
| 夹爪指令 | `/vla_deviceea100/arm/{side}/gripper_control` |

关节消息为 `trace_msgs/msg/JointAngle`，夹爪为 `trace_msgs/msg/GripperStatus`。从臂驱动和夹爪控制程序需要运行；**暂停主臂遥操作和其他推理/回放控制程序的指令发布**，否则程序会检测到冲突并拒绝运动。不要只停止移动主臂而仍让它持续发布控制话题。

默认自动启动缺失的 **Zenoh 桥接、左右从臂驱动和左右夹爪控制**，不启动主臂和相机。启动方式参考桌面《数据采集及模型推理使用手册.odt》及 collect_ws 的 bringup：手册的数据回放章节要求先启动推理相关驱动，回放工具现在直接管理所需部分，无需启动相机或推理模型。

- `runtime.start_bridge: true`：执行 `bridge_command`，默认 `zenoh-bridge-ros2dds`，沿用采集 bringup 的本地桥接方式。若现场需要指定监听/远端地址，在 YAML 命令列表中修改；推理 bringup 的远端主臂 IP 不在这里默认启用。
- `runtime.start_slave_arms: true`：使用 `robot.left/right` 的 IP、TCP/UDP 端口启动 `slave_arm`，参数与 `/home/rpp/scripts/start_slave_arm.sh arms` 一致。
- `runtime.start_grippers: true`：使用 `gripper_executable` 和 `gripper_ports` 启动左右夹爪，默认左 `/dev/ttyUSB0`、右 `/dev/ttyUSB1`，来自旧 `start_gripper.sh`。修改串口映射时以现场接线为准；驱动初始化可能使夹爪动作。
- `runtime.driver_log_dir`：每次启动创建独立日志子目录，终端打印具体路径。驱动退出时明确指出哪一路和对应日志，避免只提示反馈超时。

已运行的同名驱动按左右侧复用，不重复启动；若已运行从臂的 IP/端口或夹爪串口与 YAML 不符，则报错。复用外部驱动时不会修改其连接参数；外部驱动不会随本次回放退出而关闭，仅清理本次启动的进程。若已经通过旧 bringup 启动全部驱动，可将对应开关设为 false；不同参数的桥接复用情况需核对现有配置。

ROS_DOMAIN_ID、RMW_IMPLEMENTATION、CYCLONEDDS_URI 沿用启动终端环境，须与桥接一致。手册示例为 Domain 4 和 CycloneDDS；网络接口名以本机实际配置为准。

播放路径上各姿态应适合当前场景；对齐使用关节插值，不包含环境感知或避障规划。

## 数据和速度设置

数据向量必须符合 `scripts/collect/` 的定义：左 7 关节、左夹爪、右 7 关节、右夹爪。LeRobot 中关节为弧度，发布时转换为现有驱动所需的度，并将关节消息第 8 个未使用槽位补零。

默认回放 `observation.state` 中记录的从臂话题关节轨迹；`joint_source: action` 可改为主臂关节指令轨迹。每段对齐到当前所选轨迹的第一帧。**夹爪始终使用 action 第 8/16 维的命令值**，不把 state 中可能是 0～255 等范围的位置反馈直接下发。

夹爪命令默认接受 0～1，沿用已有推理执行器的二值控制约定。若现场原始命令范围不同，调整 `gripper_scale`、`gripper_offset` 和命令上下限；程序拒绝超范围值，不自动猜测或截断。`gripper_enabled: false` 可以关闭夹爪控制。

关节目标持续按 `command_hz` 发布；夹爪在首次播放及命令值变化时发布，命令不变时不重复发送，沿用现有推理执行器的做法。对齐阶段只发关节目标。偏差保护报错会显示具体左右臂、关节编号、反馈角度和目标角度。

- `dataset.speed_multiplier`：1.0 原速、0.5 半速。根据源数据 FPS 计算持续时间，以 `playback.command_hz` 的节奏对关节轨迹线性插值；夹爪使用当前帧命令，不对二值命令插值。
- `alignment.speed_deg_s`：仅影响每段播放前的对齐速度。
- `alignment.tolerance_deg`：实际关节到位容差。
- `playback.max_joint_speed_deg_s`：预检整段关节目标速度上限。
- `playback.max_tracking_error_deg`：实际关节与当前目标允许的最大偏差。
- `playback.feedback_timeout_s`：反馈过期则停止回放。
- `playback.max_tick_lateness_s`：调度延迟阈值，须小于一个控制周期；超出后停止，不突发追赶多条指令。

只预检数据，不连接机器人：

```bash
bash scripts/reply/reply.sh scripts/reply/config.yaml --check
```

此检查需要先填实际数据路径，检查内容包括维度、数值、帧连续性、夹爪命令范围和轨迹速度。它不证明 ROS 话题可用或真机跟踪正常。首次运行建议用短 episode 验证。

## 环境与测试

Bash 自动读取 YAML：LeRobot/Parquet 预处理用 `runtime.dataset_python`，ROS2 控制用 `runtime.ros_python`，并加载配置中的 ROS 和 workspace setup。预处理只读取数值列，不解码四路视频，也不修改原始数据；临时数值文件在退出后删除。

无硬件测试：

```bash
PYTHONPATH=src uv run --no-project \
  --python /home/rpp/miniconda3/envs/lerobot/bin/python \
  python scripts/reply/test_reply.py -v
```

测试覆盖 SQLite/真实 LeRobot Parquet 读取、轨迹顺序和单位、夹爪命令来源、速度与异常数据拒绝、平滑对齐、实际到位确认、播放/对齐期间取消，以及每段均需按键触发的流程。测试使用模拟机械臂，不验证真机运动。

手册另有旧版本备注称观测值由动作值代替；因此从臂话题值是否为实测关节反馈，还需结合当前 SDK/驱动确认，不能仅凭话题名或历史手册断定。
