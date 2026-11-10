# RM75 在线过滤推理（独立入口）

本目录提供 `online_filtering_pi05` 常驻模型 server 和独立机器人 client。
`/home/rpp/project_my/lerobot/scripts/infer/` 的文件和官方 PI05 配置保持原样。
硬件驱动、输入检查、初始化和下发函数以只读 import 复用该目录，在线执行循环独立实现。

当前配置使用下载的 `rm75_cube_bowls_pi05_ema_rollout_10k_25hz_20261109_gpu67/full8/checkpoints/010000/pretrained_model`：
state14、action16、head/chest/right/left 四相机、25 Hz、50 点动作块、10 步去噪。
任务文本已与训练集 `meta/tasks.parquet` 核对。

## 启动

终端 A，在 online-filtering-lerobot 根目录启动常驻服务：

```bash
bash scripts/online_infer/run.sh --server
```

等待 `在线模型服务就绪`。服务常驻 GPU，client 退出后可以重新连接；Ctrl+C 结束服务。

终端 B，在原机器人仓库目录，先检查新配置，再只预测：

```bash
cd /home/rpp/project_my/lerobot
bash scripts/online_infer/infer.sh --check-config
bash scripts/online_infer/infer.sh --dry-run
```

只预测需要已有四相机和关节反馈，不启动控制驱动、不做 init、不发关节或夹爪命令。
此模式的执行前缀是虚拟命令，仅用于接口联调，不能用它的结果评估闭环表现。

真机执行使用独立的显式入口：

```bash
bash scripts/online_infer/infer.sh --execute
```

保持原来的按键流程：→ 初始化，→ 开始，f 结束。执行时复用原有初始化和动作检查。
新配置默认只预测；即使把 YAML 的 `control.enabled` 改为 true，也必须传 `--execute` 才发送模型命令。

服务读取本仓库 `scripts/online_infer/config.yaml`；机器人入口读取原机器人仓库新增的
`scripts/online_infer/config.yaml`，两份配置独立。修改任务或执行频率时必须保持两侧一致；模型或任务改变后重启 server。
可选第二个参数指定配置：`bash run.sh --server /path/config.yaml`。

## 并行和时间约定

- 通信接收与计算分离；一个 action expert 线程、一个 DINO fast 编码线程。
- DINO 在独立 CUDA stream 编码，等待该 stream 的完成 event 后才发布特征。
  初始观测仍由 expert 路径编码一次；推理期间的 fast 更新由 DINO 线程处理。
- client 发送线程和 server DINO 队列都只保留最新待处理观测，不堆积旧图。
- expert 每个去噪步查询最新 ready packet，没有新特征就继续计算，不等待 DINO。
- 一个连接只允许一个在途 chunk，`request_id`/`generation` 拒绝旧包和旧结果。
- `origin` 是初始观测对应的客户端已下发命令序号。动作第 0 项是 origin 后第一个控制点。
- `capture_step` 根据观测时间在客户端命令历史中定位，为该观测相对 origin 的已执行步数。
  它不是 DINO 调用次数，也不是网络包序号。
- `executed_prefix` 来自实际下发的完整 16 维命令，含过渡混合和 hold 命令；server 用 checkpoint
  的 action normalizer 转换后硬锁前缀。只支持原生左右顺序、连续夹爪和绝对动作。
- 返回块按**切换时**客户端已下发步数裁掉过期部分，对新旧轨迹的全部重叠区间进行线性混合，与原异步推理一致。
- 观测年龄包含传输、排队、DINO 编码与等待；初始观测年龄也从输入时间计算，不从 slow prefix 完成时开始。
- 当前沿用采集模块的 host monotonic **接收时间**，四相机/状态取最老接收时间作为保守的 bundle 原点；
  同时传送各路 arrival 和原始 ROS header 供审计。ROS header 目前不用于曝光时间同步。
  这不能消除相机到 ROS 的隐含延迟或四路时间跨度；输入仍受原配置的 max_age/max_span 约束。
- 每次连接做 7 次时钟交换，选择最小 RTT 样本，将 client monotonic 映射到 server。
  使用偏移下界保守计算 age，并检查时钟误差界。长时运行需要重新连接校准；默认每轮最多 60 秒。
- 推理期间客户端累计 delay 超过训练范围 30 步（25 Hz 下约 1.2 秒）、结果整块过期、输入丢失、
  通信失败或调度超时都会结束当前连接。恢复输入后重新运行 client，从最新观测开始。
  不突发补发过期动作。25 Hz 是动作控制频率，模型按块计算，不能解释为每秒 25 次完整推理。

同机两个仓库默认通过 `unix:/tmp/lerobot_online_rpp/policy.sock` 通信。
跨机可将 server 地址设为 `tcp:0.0.0.0:8765`，client 地址设为 `tcp:SERVER_IP:8765`；
TCP 模式用于受控内网，不提供认证或传输加密。图像为原始 RGB，每包约 3.7 MB，25 Hz 约 92 MB/s，
需实测网络和传输延迟；当前同机使用 Unix socket 避免网络瓶颈。

## 离线验证

零输入权重检查，不导入机器人控制：

```bash
bash scripts/online_infer/run.sh --check
```

server 启动后，模拟 fast 观测双向流，无机器人连接：

```bash
uv run --no-project --python /path/to/model/python python scripts/online_infer/smoke_client.py
```

`smoke_client.py` 使用虚拟 25 Hz 执行前缀和零图像/状态，只检验软件调度，不检验任务效果。
日志 `dino_ready` 表示编码完成；`assimilated` 才表示 expert 在该 flow step 实际吸收新观测。
client 每块打印耗时、过期裁剪步数和吸收记录。

本机 RTX 3060 12 GB 的两块零输入联调：约 1.080 s / 0.897 s，分别吸收 8 / 9 个 fast packet；
这是一次软件测量，包含首次计算开销，不代表真机观测下的稳定延迟或成功率。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project --python /path/to/model/python python -m pytest \
  -q --confcutdir=scripts/online_infer/tests scripts/online_infer/tests
```

关闭 pytest 自动插件用于避开本机 ROS launch_testing 与 pytest 的版本不兼容。

## 下发动作平滑

独立 client 配置的 `control.action_smoothing.enabled` 控制最终机械臂发送层的平滑，默认 true。
所有 merge、裁剪后的目标在 `checked_publish` 前逐 tick 经过同一个滤波器，模型输出和动作块不被滤波器改写。
`alpha: 0.5` 表示 `候选关节目标 = alpha × 当前目标 + (1-alpha) × 上次实际下发目标`。
随后通过 `max_step_deg: 1.0` 限制每关节相邻两条下发指令的最大变化；25 Hz 下对应最高 25 度/秒。
alpha 越小越平滑、滞后越大；max_step_deg 越小响应越慢。夹爪不参与 EMA 或单步限幅。

```yaml
control:
  action_smoothing:
    enabled: true
    alpha: 0.5
    max_step_deg: 1.0
```

`enabled: false` 完全关闭此层平滑；`alpha: 1.0` 只关闭 EMA，`max_step_deg: null` 只关闭单步限幅。
首步从实测关节姿态初始化，后续以最终发送成功的目标为准，跨 chunk 不重置。
队列耗尽时保持最后已发送目标，不切换到滞后的反馈位置，也不重置滤波器。
取消或异常退出仍使用原有立即 hold 路径，避免结束后继续追赶旧目标。

原始目标和平滑后的最终命令都经过动作检查；发送成功后才推进滤波状态。
发送函数返回实际下发的完整 action16，客户端将它记录为 online filtering 的执行前缀。
测试覆盖 chunk 大跳变、反馈滞后、等待下一块、发送失败以及最终 wire 关节指令变化上限。

两份独立配置都已加入该选项；实际 client 默认读取 `/home/rpp/project_my/lerobot/scripts/online_infer/config.yaml`。
启动时会打印实际配置路径和最终下发平滑参数，便于确认修改的是当前使用的配置。
修改参数后重启 client 即可，server 无需重启。

## 判断小停顿是否来自队列耗尽

client 每次换块记录：

- `latency`：server 模型处理耗时；`client_elapsed_s`：client 从请求入队到应用结果的耗时。
- `trim`：新块已过期、被裁掉的动作点数。
- `old_remaining` / `old_remaining_ms`：换块前旧队列还剩多少点、对应多少毫秒。
- `new_remaining`：新块裁剪后的可执行点数；`overlap`：本次线性 merge 的重叠点数。
- `hold_count` / `hold_total_s`：本轮因队列耗尽而进入等待的累计次数、已结束等待段的累计时长。

实际发出第一条队列耗尽保持指令时打印 `queue_hold_start`，期间不逐 tick 刷屏。
新块恢复时打印 `queue_hold_end reason=new_chunk duration_s=... ticks=...`。
退出时打印 `queue_summary`，包含所有等待次数、保持指令条数、累计时长；如果退出时仍在等待，也计入这一段。
首次静止等待首块不算队列耗尽。`old_remaining=0` 本身不说明已停顿：新块可能恰好及时到达；
应以 `queue_hold_start` 和计数为准。hold 时长按 client monotonic 时钟统计，不能代表物理臂精确停止时间。
如果小停顿时 `hold_count` 始终为 0，说明当时没有进入队列耗尽保持分支，可继续排查最终平滑和轨迹速度。

`assimilated_count` 默认只显示 expert 实际吸收新观测的次数；初始首块没有新观测、计数为 0 是正常情况。
需要逐步核查时设置：

```yaml
online:
  log_assimilated_details: true
```

详细 `assimilated=[[0,9],[1,12],...]` 每项为 `[去噪步序号, 新观测相对 chunk 原点的执行步数]`。
例如 `[0,9]` 表示第 0 个去噪步使用了原点后第 9 个控制步对应的新观测；25 Hz 下为原点后约 360 ms。
第二个数不是观测年龄，也不是 DINO 编码次数；列表长度也不等于编码完成数量，因为旧包可能被更新包替代。
这些记录用于确认在线特征实际进入 expert，不是动作点、任务成功率或模型质量指标。
只需日常运行时保留次数即可，关闭详细列表不影响模型计算。此选项只控制 client 列表打印，server 的逐事件诊断仍保留。
