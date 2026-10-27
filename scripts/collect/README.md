# RB75 双臂 LeRobot 数采

在本仓库根目录启动（不是 InterACT-LeRobot）：

```bash
bash scripts/collect/record.sh
```

配置集中在 `scripts/collect/config.yaml`。当前为 50 Hz 数据同步采集、16 维状态/动作、四路 640×480 RGB 相机（源频率 60 FPS），数值输入的最低检查目标为 50Hz，允许更快输入。采集前改 `dataset.task` 为实际任务描述，并检查 `save_dir`。采集 YAML 不改变外部驱动的真实发布频率。

脚本自动加载 ROS Humble 和 collect_ws 环境。系统 Python 3.10 负责订阅 ROS；按 s 后使用 YAML 中 `converter_python` 指向的 LeRobot 环境转换，无需在训练环境安装 ROS。

只检查配置和依赖，不启动硬件：

```bash
bash scripts/collect/record.sh scripts/collect/config.yaml --check
```

## 现场启动

默认沿用**已运行的主从遥操作、夹爪和相机 ROS 节点**，采集器本身不发布机器人控制指令。先按你们现有流程启动主从遥操作和夹爪，再运行上面的 Bash；所有 12 路输入到齐后，终端提示“输入就绪”。缺哪一路会显示名称，不会开始录制缺失数据。

如果相机尚未启动，将 YAML 的 `runtime.start_cameras` 改为 `true`，Bash 启动后会根据同一 YAML 的序列号、分辨率、源 FPS 启动四个 RealSense 节点。如果从臂驱动尚未启动，也可以将 `start_slave_arms` 改为 `true`，使用 YAML 的左右 IP、TCP/UDP 端口启动本地 `slave_arm`。不要重复启动已经运行的驱动。

主臂的远端 SSH、夹爪串口仍由现有遥操作程序管理；采集器不复制这套控制链路。若沿用现有相机节点，YAML 相机尺寸必须与该节点实际输出一致；源 FPS 和序列号也需要在现有驱动配置中对应，采集器不会修改外部节点参数。

| 按键 | 行为 |
| --- | --- |
| 右方向键 → | 输入就绪且当前空闲时，开始新 episode |
| f | 放弃当前 episode，回到等待状态 |
| n | 完成当前 episode，等待后台暂存写完，回到等待状态 |
| s | 空闲时结束作业，统一转换所有已经按 n 接受的 episode |
| Ctrl+C | 中断作业，不自动转换；已接受的段保留，当前段放弃 |

采集中按 s 会提示先按 n 或 f。n 不会触发 MP4 编码或 LeRobot 转换，但需要将当前段 JPEG 暂存队列写完，终端确认后再开始下一段。不要同时用旧采集器往本作业目录写数据。

## 数据定义

两个向量的排列固定为：

```text
[left_joint_1 ... left_joint_7, left_gripper,
 right_joint_1 ... right_joint_7, right_gripper]
```

- `observation.state`：从臂关节反馈 + `slave_arm/{side}/Gripper_position` 夹爪反馈。
- `action`：主臂关节指令 + `slave_arm/{side}/Gripper_status` 夹爪命令。夹爪命令话题沿用 collect_ws 的旧采集配置；若现场改为从 `master_arm/...` 发布，在 YAML 修改对应 topic 即可。
- ROS `JointAngle.angle` 虽然有 8 个槽位，只读取前 7 个关节，夹爪独立订阅；关节按现有配置由度转换为弧度。
- 夹爪默认保留消息原始单位，不猜测其归一化范围。每一路的 `scale`、可选 `offset` 可以单独配置，计算顺序为 `raw * scale + offset`。
- 四路图像：`observation.images.head/chest/right/left`，最终为 RGB MP4 视频。
- 最终格式由**当前仓库的 LeRobotDataset API**生成（Parquet、视频及 meta），不复用旧示例的 14 维 HDF5 或“下一帧状态作为 action”的近似定义。

首次短采后，应核对主臂角度、从臂角度以及夹爪命令/反馈的方向与数值范围。程序只验证消息、维度、有限值和时序，不能从消息类型判定实际夹爪机械单位。

## 时间采样与暂存

50Hz 每 20ms 保存一组完整数据。优先在 `recording.sync.max_offset_ms: 15` 和 `max_span_ms: 20` 的窗口中配对全新消息，等待 `lookahead_ms: 25` 后处理；无法配对时，各路选截至目标时刻最近收到的有效消息，允许沿用上一条，不插值、不丢整组、不切 episode。25ms 是流水线等待延迟，不改变 50Hz 保存频率。

这是本机单调时钟接收时间对齐，不是设备硬件曝光/测量同步。源 header 单独保存；沿用旧值时，组内实际时间跨度可超过优先配对的 20ms，终端和质量统计会显示实际跨度。非零源时间戳重复或倒退的消息不会刷新已选消息的年龄。

`recording.sync.repeat_warning_steps: 3` 表示某一路连续 3 步沿用旧值时严重告警，但继续录制。`recording.max_age_ms: 100` 是硬上限：在采样执行时任何一路超过这个年龄则停止当前段，异常数据单独保留、不自动转换。采样线程超过 `max_tick_lateness_ms` 时同样停止，避免补造或压缩时间轴；磁盘/队列错误仍终止当前段。已暂存段不受影响。

源频率开始前需要积累 `recording.frequency.window_s: 1` 秒并通过检查。采集中按 `report_interval_s: 2` 秒诊断频率、打印重复次数/比例、最大数据年龄、最大已选消息更新间隔和组内时间差；按 n/f 结束一段时也显示统计。`allow_faster: true` 允许高速输入降采样，低频/断流诊断只告警。当前数值源目标 50Hz（下限 47.5Hz），相机源 60Hz（下限 57Hz）；这些是输入诊断目标，不改变外部驱动。

SQLite 每帧保留目标时刻、每路接收时间、源时间戳、采样时的数据年龄和是否重复使用。接受的段另存 `episode_XXXXXX.quality.json` 汇总统计。新录制数据保持连续 50Hz，一次手动采集对应一个 LeRobot episode。转换不会在缺口处切段；如旧版 SQLite 已有丢步缺口，则明确拒绝转换，避免悄悄压缩真实时间。已有旧数据不自动修改。

只读检查本机实际输入（不启动驱动、不发布指令，使用与采集器相同的订阅 QoS）：

```bash
source /opt/ros/humble/setup.bash
source /home/rpp/collect_ws/install/setup.bash
/usr/bin/python3 scripts/collect/measure_hz.py --seconds 15
```

2026-10-26 同时订阅全部 12 路的 15 秒实测：相机约 59.5～59.7Hz，从臂关节、夹爪反馈及夹爪命令约 49.8～49.9Hz，主臂关节约 101.7～102.0Hz。因此现有驱动输入不满足统一 60Hz：从臂/夹爪不能提供 60 条每秒的新消息。设置目标 60 后采集器将拒绝开始，不会补出不存在的数据。需先修改实际驱动的发布机制并复测；其控制定时器参数不能未经验证就视为反馈发布频率。该实测没有同时进行四路 JPEG/SQLite 写入，不能证明完整采集负载下的频率稳定性。

进一步只读检查了当前安装的 `/opt/arm-servo-adapter/bin/slave_arm`：`statusPublishThread` 和 `posPublishThread` 内部有固定 `sleep_for(20ms)`，说明存在驱动软件层的发布限速。未找到与当前二进制对应的完整 `slave_arm.cpp` 源码。要提升输入频率，应先取得源码、把反馈发布/实际读取周期分别配置化，再编译和实测；不能靠采集 YAML、控制定时器参数或重复旧读数实现更高的真实更新频率。夹爪还需核查串口/SDK 实际读取周期和响应耗时。当前未替换二进制或重启现场驱动。

数值相同但时间戳更新属于有效的新消息，例如静止机械臂；不通过图像内容比较来判断是否重复。JPEG/SQLite 写入使用有界后台队列，最终 MP4 编码仅在作业结束后执行。

输出示例：

```text
<save_dir>/<月日时分>/             # 例如 10261405；同名时追加 _01、_02，避免覆盖
  config.yaml                    # 本次配置快照，转换以它为准
  episode_000000.sqlite3          # n 接受的原始暂存（图像 JPEG + 数值 + 时间戳）
  episode_000002.sqlite3          # f 放弃的编号可造成空缺，最终编号连续
  episode_000003.partial.sqlite3  # 异常残留，不参与转换
  camera_head.log                # 仅自动启动对应驱动时生成
  lerobot/                       # s 之后成功转换的最终数据集
    meta/
    data/
    videos/
```

转换成功保留暂存数据，不自动上传 Hub、不覆盖已有数据集。失败的转换放在 `.lerobot_*.incomplete`，不会冒充完成的数据集。中断或转换失败后，可以用原始作业目录重试：

```bash
bash scripts/collect/record.sh scripts/collect/config.yaml --convert /绝对路径/作业目录
```

手动转换使用作业内的 `config.yaml` 作为数据定义，启动解释器仍取命令所给 YAML 的 `runtime.converter_python`。已经存在 `lerobot/` 时拒绝覆盖。异常产生的 `.partial.sqlite3` 需要人工检查，不会自动当作有效段。

## 不连接硬件的测试

```bash
PYTHONPATH=src uv run --no-project \
  --python /home/rpp/miniconda3/envs/lerobot/bin/python \
  python scripts/collect/test_record.py -v
```

覆盖：16 维排列及真实动作来源、角度单位、图像 RGB/BGR 和行补齐、缺失/超时/时序跨度、输入频率与断帧/重复帧、按键流程、暂存/放弃、写入异常与队列溢出，并实际生成和重新读取两段四相机的 LeRobot 数据集。模拟测试不能证明真机在四相机完整分辨率下能稳定达到 50Hz，需先让驱动满足要求再现场短采确认。

## 回放这次作业

采集工具目录现为 `scripts/collect/`，默认配置文件现为 `config.yaml`。
另有独立的 `scripts/reply/` 回放工具，将其 `config.yaml` 中的 `dataset.path`
设置为本次作业目录后运行 `bash scripts/reply/reply.sh`，详见 `../reply/README.md`。


## 并行转换

`conversion.parallel_encoding: true` 使用 LeRobot 原生多进程接口同时编码四路相机视频；`encoder_threads: 2` 控制每路编码线程，`image_writer_threads: 4` 控制中间图像写入线程。episode 顺序提交，避免多个任务同时改写数据集索引。默认四路并行，不将每路线程数都设成全部 CPU 核数。旧作业 YAML 缺少该配置时采用同样默认值。
