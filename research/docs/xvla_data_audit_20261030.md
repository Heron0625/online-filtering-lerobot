# XVLA 数据审计（2026-10-30）

> 最新状态见末尾“修复落地与再次比对”：修复已落地，新数据已生成，尚未训练；前面的检查记录保留用于追踪。

## 结论

原 LeRobot 数据没有出现大范围 relative action 损坏或隔行降采样的证据；但当前 XVLA 训练路径至少有两项实质性的坐标/预处理错误：

1. 将 `observation.state[3:6]` 的 Panda **末端 body** 姿态当成了 OSC controller **grip_site** 姿态。两者相差约绕局部 Z 轴 -90° 的固定旋转。这同时影响绝对 action target 与训练 proprio；推理的 `LiberoProcessorStep` 却使用 controller site 姿态。
2. 现有 LeRobot 两个相机都相对 simulator 原始图像旋转了 180°。XVLA 推理约定是 agent view 旋转 180°、wrist view 不旋转。训练 processor 没有对 LeRobot wrist 做补偿，形成腕部图像训练/推理方向差异。

这两项足以解释微调 loss 可以下降、闭环性能却变差。尚未做修复后的训练消融，不能声称已经证明它们是全部原因。此次只新增诊断及记录，没有修改训练实现、数据或启动训练。

## 数据与检查范围

- 原训练数据：`/data/yhr/pretrain/recovered_zhoukai/dataset/libero`
- 原始 HDF5：`/data/yhr/pretrain/xvla_official/libero_hdf5_raw`
- 已 replay HDF5：`/data/yhr/pretrain/xvla_official/libero_xvla_preprocessed`
- 完整 raw HDF5 覆盖 18 个任务；其对应的 720 条 LeRobot episode（144,304 行），均能唯一匹配一条 raw action 序列：去除 no-op 后长度相同、逐元素最大误差 < 1e-6。
- replay 对比覆盖两个完整任务文件：stove 48 条（4,292 行）、microwave 28 条（8,168 行）可匹配训练轨迹。由于 replay 文件重排 demo ID 且未保存来源 ID，按长度、完整 gripper 序列及最小位置目标误差匹配；此项不是独立的严格逐 demo parity 证明。
- 另独立 replay 已由完整 action 序列唯一确认的 `turn_on_the_stove/demo_25`，与 LeRobot episode 388（90 行）直接核对，排除 replay 文件 demo 编号错配的影响。

## 旋转坐标错误的直接证据

安装的 robosuite 源码中：

- `robots/single_arm.py: eef_quat()` 返回 `robot_model.eef_name` 的 **body quaternion**；LeRobot 的 axis-angle state 对应这个量。
- `controllers/base_controller.py: update()` 从 `eef_name` 的 **site_xmat** 取 controller 姿态。
- `single_arm.py` 把 controller `eef_name` 设置为 gripper 的 `grip_site`。
- `models/assets/grippers/panda_gripper.xml` 的 gripper body 带固定旋转 `quat="0.707107 0 0 -0.707107"`。

当前 `src/lerobot/policies/xvla/utils.py` 使用：

```text
R_goal = Exp(0.5 * clipped_relative_rotation) @ R_body
```

对于这套 Panda 数据，应先转换 body 到 site：

```text
R_site = R_body @ Rz(-90 degrees)
R_goal = Exp(0.5 * clipped_relative_rotation) @ R_site
```

独立 replay 的 90 行结果：

| 检查 | 平均误差 |
|---|---:|
| LeRobot state 旋转 vs replay body 旋转 | 0.168° |
| 当前公式 target vs replay controller goal | 89.975° |
| 加上 body→site 固定旋转后 vs replay controller goal | 0.164° |
| 使用 replay site 姿态计算 target vs replay controller goal | 0.014° |

补偿后的残差包含重放差异、观测缓存与 controller update 时间差，不能把近似离线换算宣称为逐元素 exact replay。位置 state 对直接 replay 的平均绝对误差为 0.001505 m。

两个 replay 文件的加权 rotation6D MAE：

| 任务 | 当前公式 | 补偿 body→site 后 |
|---|---:|---:|
| turn on the stove | 0.68998 | 0.00820 |
| mug in microwave | 0.74001 | 0.00728 |

对应位置目标 MAE 分别为 0.00165 m、0.00529 m；不同重放轨迹、接触和随机性可能产生位置偏差，不应直接归为 label 计算错误。

## 图像方向与时间对齐

直接 replay 的 episode 388，抽样帧 0/20/40/60；分别比较 LeRobot 解码帧与 simulator 原图、180° 旋转原图，并检查相邻帧偏移。

- Agent view：旋转 simulator 原图后像素 MAE 约 6.94–7.90 / 255；不旋转约 73.80–75.36。符合 XVLA 的 agent view 方向。
- Wrist view：旋转 simulator 原图后像素 MAE 约 8.35–19.67 / 255；不旋转约 51.55–84.89。LeRobot wrist 与 XVLA 推理方向相反。
- Agent view 所有抽样帧，offset=0 都优于 ±1；wrist 4 帧中 3 帧 offset=0 最佳。没有发现应对所有标签盲目平移一帧的证据；wrist 的单帧例外可受独立 replay 轨迹差异影响。
- `make_xvla_pre_post_processors` 和 pretrained reconcile 都只增加 float/ImageNet normalize，没有训练 wrist 方向转换。
- `LiberoProcessorStep` 推理仅 flip agent view，不 flip wrist；与本地官方 client 一致。

对比图：`research/outputs/xvla_data_audit/image_orientation_comparison.jpg`。上下两行分别为 agent/wrist；每行从左至右为 LeRobot 帧、simulator 原图、simulator 图旋转 180°。

## 官方训练契约的额外差异

本地官方 loader 使用 `abs_trajectory[0]` 作为 proprio，`abs_trajectory[1:]` 作为 targets；官方 client 在后续 replan 也使用上一 chunk 的末尾预测 goal 作为 proprio。当前 LeRobot 用实测 pose，语义并不完全相同。必须先修正 body/site 坐标后再评估此差异；不能只根据 action[1:] 就给现有 LeRobot targets 加一帧 offset。

原始 HDF5 的 state 与 LeRobot state 不完全相同，但 action 序列完全一致，且 LeRobot body 姿态和重新 replay 高度一致；这是重生成数据的特征，不能据此断言原 LeRobot state 已损坏。

## 后续修复与验证建议

1. 仅在明确的 LIBERO Panda 原始 8D state 路径补上 body→site 旋转，同时修正 action target 与训练 proprio；避免对已处于 controller 坐标的 20D 数据或其他机器人重复补偿。
2. 在明确的原 LeRobot LIBERO 数据训练路径把 wrist 旋转 180°；不要对 simulator 推理或已 replay、按官方方向保存的 HDF5 重复 flip。
3. 用真实 replay reference 加回归检查。当前 synthetic target 单测与实现共享同一 body=site 假设，所以此前 5 个测试通过仍未发现这个错误。
4. 先做 30-chunk 短微调对照，再做 50-chunk；无需因此重新下载全部 HDF5 或重跑全量 replay。

## 复现产物

- 数值匹配脚本：`research/scripts/diagnose_xvla_data_alignment.py`（在仓库根目录以 `PYTHONPATH=src` 运行；路径按本机交接配置）。
- `research/outputs/xvla_data_audit/raw_summary.json`、`raw_matches.json`
- 两个 task 的 replay 误差 JSON
- `direct_replay_comparison.json`、`stove_demo25_replay.npz`
- `reproduce_stove_replay.py`、`reproduce_stove_comparison.py`

outputs 目录遵循仓库现有忽略规则，仅保存在本机。环境未提供 uv；本次使用交接文档指定 conda 环境的 Python。

## 第二轮复核：实际 processor 与新增图像样本

在第一次诊断基础上，再从两个已有 replay 文件各抽 4 条匹配轨迹，每条取 2 帧，合计每相机 16 帧。对齐时明确使用 replay JPEG 的 post-action i-1 对 LeRobot 的 pre-action i。

| 相机 | LeRobot 与已有 replay 同方向更接近 | 同方向像素 MAE | 旋转 replay 180° 后 MAE |
|---|---:|---:|---:|
| agent | 16/16 | 7.314 | 90.714 |
| wrist | 0/16 | 83.374 | 9.892 |

其中 replay agent 已按官方约定旋转，replay wrist 保持 simulator 原方向。进一步调用当前 `make_pre_post_processors` 加载 long50-fixed-v2-init 的实际 processor（仅 device/tokenizer 改成本机 CPU/离线目录），并调用 `make_env_pre_post_processors` 建立推理环境 processor，以同一张相机图构造数据集与 simulator 输入，隔离轨迹重放差异：

- 主相机训练/推理最终 tensor 的最大绝对差为 **0**。
- 腕部相机的最大绝对差为 **3.2537**（ImageNet 标准化后）；将训练腕部图旋转 180° 后差为 **0**。

因此该差异不是只由代码推断，也不是视频解码丢帧/压缩造成的。

为什么官方 checkpoint 推理正常仍可能出现这个问题：

1. 官方权重的 simulator 推理不读取本地 LeRobot 视频；环境 factory 为 XVLA 选择专用 processor，只翻转 agent。正常推理不能验证 LeRobot 微调的数据适配。
2. LeRobot 微调读取视频后走 policy processor，没有先走推理专用 env processor。其 wrist 保持了数据集的 180° 方向。
3. `src/lerobot/envs/libero.py:render()` 只显示 `pixels` 中第一个相机；常规 rollout 视频没有展示模型实际接收到的腕部图像，不能据其判断腕部方向是否一致。
4. 已微调模型受该差异影响的实际成功率，需要修复后的消融验证；不能根据图像差异直接推导下降了多少。

新增证据：`additional_image_comparison.json`、`processor_path_comparison.json`；复现脚本 `check_image_paths.py` 均在本机 `research/outputs/xvla_data_audit/`。

## 第二轮复核：同一 simulator state 的几何与 controller 目标

对 stove demo 0、25 与 microwave demo 10，合计 455 个有效控制步，直接从同一时刻的 sim body matrix 与 controller site matrix 取值，避免使用缓存的 observation quaternion；在执行动作前调用实际 controller `set_goal(action[:6])` 对比。

- `R_body @ Rz(-90°)` 与实际 site 姿态的最大角误差约 2.5e-14°，固定变换得到直接验证。
- 位置目标与 controller goal_pos 的最大绝对误差均为 0 m（使用同状态 controller 位置）。
- 437 个非零旋转 command：补偿后的公式与实际 goal 的最大角误差 < 2e-6°。
- 18 个全零旋转 command：实际 OSC controller **沿用已有 goal_ori**，并非重新令目标等于当前实测 site 姿态。简单的逐行公式在这些点仍有残差，两个含零旋转的 demo 分别最大约 0.469°、0.660°。

这补充修正了先前对 0.16° 残差的解释：跨 replay 轨迹/观测缓存确实是误差来源，但不能把所有剩余误差都归因于 replay；还存在可明确建模的 controller 历史状态。若追求忠实的 absolute target，应按 episode 顺序重建 goal，并在全零旋转输入时保留上一个 goal，而不是仅对任意 batch 内的单行做无状态转换。episode 起点需处理初始 goal。

不需要因此恢复全量 simulator replay：可先在现有完整 episode 上做顺序目标转换，并以小规模同状态 replay 验证。新增证据 `same_state_geometry.json`，复现脚本 `check_same_state.py` 位于本机 outputs 目录。此轮仍未修改训练实现。

## 修复落地与再次比对（本轮最新结果）

此前“尚未修改实现”的表述仅指前两轮检查。本轮已修复并生成可供训练的新数据，尚未启动训练。

### 已实现

- 新转换脚本 `research/scripts/convert_libero_lerobot_to_xvla.py`，按完整 episode 的 frame_index 顺序转换。Panda body 姿态右乘固定 body→site 旋转；零旋转 command 保留前一个 goal，不在随机训练窗口里重建历史。
- 输出 action 和 measured-site proprio 均为 20D。proprio 仍采用现有推理路径的实测 pose 语义，不擅自替换成官方训练的上一 goal。
- 原始视频以目录符号链接复用；加入显式 `observation.xvla_wrist_flip` 标记，新 processor 消费一次标记后翻转腕部图像。没有标记的 simulator 输入与已有 canonical replay 图像不变。
- `XVLAPolicy` 遇到原始 7D relative action 训练窗口时明确报错，防止继续走缺少历史的旧转换路径。
- 原 replay→LeRobot 脚本改为直接保存 controller-frame 20D proprio，避免把已经 canonical 的 goal 编码成 8D 后再次误当成 body 姿态。
- XVLA long-horizon 与 Stage A/B 训练脚本的默认 dataset root 更新为新数据；可用 `XVLA_DATASET` 覆盖。

新数据：`/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v1`

共 **1,693 episodes / 273,465 frames**；12,860 个零旋转 command。没有改动原始数据、原视频或 metadata 的存储时间轴。控制样本仍是连续 20 Hz，存储 timestamp/fps 继续为原始 10 fps。

复现转换：

```bash
PYTHONPATH=src /data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python \
  research/scripts/convert_libero_lerobot_to_xvla.py \
  --source /data/yhr/pretrain/recovered_zhoukai/dataset/libero \
  --output /path/to/new_output_directory
```

### 与实际 controller 的同状态验证

三条轨迹共 455 步、含 18 个零旋转 command。使用真实 controller 初始 goal、同一 simulator state 的 body pose 和 controller position：

- 位置目标最大绝对误差：**0 m**。
- 旋转目标最大角误差：**1.88e-6°**。
- 20D target 最大元素误差：**3.39e-8**。

另针对两条 outlier 轨迹验证了 402 步，同状态公式仍满足同级精度。新增了由真实 controller.set_goal 录制的 90 步 regression fixture，包含 5 个零旋转指令，测试不依赖安装 simulator。

### 与已有独立 replay 文件的比对

仍使用先前固定的 76 条对应关系（12,460 帧），没有为了缩小误差重新挑选匹配：

| 任务 | episodes | 位置 L2 均值 | 位置 L2 P95 | 位置 L2 最大 | 旋转角均值 | 旋转角 P95 | 旋转角最大 |
|---|---:|---:|---:|---:|---:|---:|---:|
| turn on the stove | 48 | 3.59 mm | 14.94 mm | 40.64 mm | 1.275° | 6.885° | 55.31° |
| mug in microwave | 28 | 10.65 mm | 42.56 mm | 165.55 mm | 1.150° | 5.816° | 11.27° |

两任务 gripper mismatch 均为 **0**。这些是不同 rollout 之间的差异，不能当成同状态转换公式误差，也不能称为逐帧无误差。对应关系本身仍受旧 replay 未保存原始 demo ID 的限制。

为排查最大值，重新 replay 原始 stove demo_26 和 microwave demo_8：

- stove：新 replay 对修正后 LeRobot 的旋转平均/最大差为 **0.678° / 1.87°**；对旧 replay 则为 **8.164° / 56.82°**。
- microwave：新 replay 对修正后 LeRobot 的位置 L2 平均/最大差为 **11.37 mm / 44.61 mm**；对旧 replay 则为 **38.51 mm / 196.21 mm**。

这说明旧 replay 自身不能作为无误差的逐帧真值；差异可能来自重放轨迹、环境状态/动力学、缓存时刻，以及旧文件的对应关系，尚未把这些因素逐项分离。不应通过把旧 replay 的目标直接塞给原 LeRobot 图像来“消除”残差。

复现比对：

```bash
PYTHONPATH=src /data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python \
  research/scripts/compare_xvla_converted_replay.py \
  --converted /data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v1 \
  --replay /data/yhr/pretrain/xvla_official/libero_xvla_preprocessed
```

### 图像与数据加载验证

- 新 processor 下，同一组原图构造的训练/推理输入，主相机和腕部相机最终 tensor 最大差均为 **0**。
- 新增的每相机 16 帧与已有 replay 比对中，两相机均 16/16 方向一致。像素 MAE 为主相机 7.31/255、腕部 9.89/255；仍包含 AV1/JPEG 压缩和不同 rollout 差异。
- 真实 `LeRobotDataset` 加载 episode 388，通过 50 步 action/state sampling 与实际 pretrained processor，得到 `[1,50,20]` action/state，方向标记正确消费；第 k+1 行 target 与下一个样本第 k 行 target 一致，确认没有因 10fps metadata 重复/跳行。
- 修复后的脚本从头重新生成第二份完整数据，逐 parquet、info、stats 验证与已加载验证的版本一致；校验副本已清理。
- 相关测试 **13 passed**；修改文件 Ruff 检查、训练脚本 bash syntax 与 git diff whitespace 检查通过。

### 仍需明确的边界

原 LeRobot 数据没有保存 episode 开始前 controller.goal_ori。273 条 episode 的开头含零旋转，合计 **894 行（0.327%）**；本次转换以首帧实测 site 姿态初始化该历史 goal，在第一次非零旋转后就使用可重建的目标。这部分是显式的初值近似，记录在 `meta/xvla_conversion.json` 中，不能宣称逐元素恢复了当时的真实初始 goal。上面的同状态数值精度使用了实际初始 goal，因此不能直接作为这 894 行的误差界限。

当前建议仍是用修正数据先做 30-chunk 短微调对照，不立即恢复全量下载/replay；若要完全去掉首段初值近似，需额外获取对应 episode 的 controller 初始 goal，或在训练中排除这部分前导目标。

本机 outputs 新产物：`corrected_same_state_comparison.json`、`corrected_replay_comparison.json`（含逐 episode 与最大误差帧）、`outlier_targeted_replay_comparison.json`、`corrected_image_comparison.json`、`corrected_processor_path_comparison.json`、`converted_loader.log`。

## 30-chunk 训练前的 validity mask 补充

已生成 `/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked`，在 v1 相同的 targets/state/video 基础上增加逐 action validity。273 条 episode 开头的 894 个初值不确定目标均不参与 loss。dataset factory 按 action delta 采样该标记，policy 将其与 episode-tail padding 合并；Online Filtering 继承同一 mask。真实 30-step dataset loader 检查通过（episode 0 前 10 步正确排除）。使用 v2 训练时不再对这 894 行施加监督。

30-step 对照启动脚本：`research/scripts/run_xvla30_corrected_2gpu.sh`。配置为官方30 checkpoint、GPU4/5、per-GPU batch32（global64）、30k steps、float32全参微调、每10k保存（按用户最新要求，早期不足200步的试启动已重跑）。训练过程与实际启动状态记录在对应日志，不在本审计文档中预填成功率。

## 用户纠正训练目标：已切换为真正的 50-chunk（2026-10-31）

用户明确要求停止 30-chunk 对照，直接训练 50-chunk。原 30-chunk 训练已停止，已有 10k/20k checkpoints 保留。

当前训练：

- 官方 `/data/yhr/pretrain/online_filtering_baselines/xvla-libero` 30-step 权重，经 action/condition positional remap 独立初始化到50；**没有从本次30-chunk微调权重初始化**。
- init：`/data/yhr/pretrain/online_filtering_baselines/xvla-libero-long50-corrected-v2-init`
- 修正后的 v2_masked 数据；真实 batch 的 action/state/validity 时间维均已验证为50。
- GPU4/5，per-GPU batch32 / global64，float32全参微调，30,000 steps，每10,000保存。
- 启动脚本：`research/scripts/run_xvla50_corrected_2gpu.sh`
- tmux：`xvla50_corr_b32_1031`
- log：`/data/yhr/output/online_filtering/logs/xvla50_corrected_v2_2gpu_b32_30k_20261031.log`
- output：`/data/yhr/output/online_filtering/ckpt/xvla50_corrected_v2_2gpu_b32_30k_20261031`

已核对运行时 config 的 chunk_size/n_action_steps 均为50，并完成实际优化步骤。还没有生成50-chunk训练 checkpoint或相关成功率结果。

## 50-chunk 训练完成与评测启动（2026-10-31）

真正50-chunk训练已完成30k steps并保存030000，耗时8:15:21；最终总loss约0.005。
无延迟、20Hz、replan20的40-episode快检为38/40（95%），queue underrun为0。
按用户要求，delay15/20/25均评测400episodes、与metrics相同suite horizons。
25档已有独立并行worker；15/20追加GPU3–7每卡5个模型worker、每work item两个任务的动态队列。
完整数值统一写 `research/docs/metrics.md`，这里不重复建立主结果表。

## Cleanup and Phase A launch (2026-10-31, latest)

Authorized obsolete XVLA artifacts were removed: approximately52.53GiB across28
entries, with small configs/logs and an explicit deletion manifest preserved at
`research/outputs/cleanup_20261031/manifest.json`. This includes stopped30-chunk
weights,50-chunk10k/20k intermediates, obsolete initializations, v1 data, smoke
replay files, invalid/partial evaluations, and stale transfer fragments. Final50
030000, official30/recovered weights, v2_masked data,40-task raw/replay data,
formal evaluations, and PI0.5 experiment checkpoints were retained. Broken-link
scan passed after cleanup. Available storage increased from525GiB to578GiB.

Phase A is running in tmux `xvla50_phase_a_45_1031` on CUDA_VISIBLE_DEVICES=4,5
(verified on both ranks). Source is the trained50-chunk030000 baseline through
`xvla50-online-filtering-stage-a-corrected-v2-init`; no positional remap is repeated.
Configuration:50chunk, max delay25, per-GPU batch32/global64,30k steps,save10k,
float32 weights with bf16 AMP,frozen DINO,full Florence/action expert fine-tuning,
no environment evaluation and no held-out evaluation. Phase-A training now hard
commits the executed prefix before prediction, matching deployment;14 relevant
regression tests passed. Actual temporal preprocessing was checked for26 image
times and50 action/state/validity times. Optimizer steps and finite loss observed.

Log: `/data/yhr/output/online_filtering/logs/xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031.log`
Output: `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031`
Manifest: `/data/yhr/output/online_filtering/logs/xvla50_phase_a_v2_artifacts/run_manifest.json`
