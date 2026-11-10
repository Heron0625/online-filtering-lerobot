# 当前 Online Filtering 方案与 Agent 接手说明（2026-11-01）

本文按当前工作区代码、XVLA 启动配置和实际 checkpoint 核对。用于新训练和跨对话交接；旧版 flow/v2/v3 文档及旧 Stage A/B 脚本记录了不同方法，不能直接作为当前方案启动依据。实验数字统一放在 [metrics.md](metrics.md)，本文不复制结果表。

## 1. 先读：两个 backbone 的阶段名不一样

| 研究阶段 | PI0.5 当前正确配置 | XVLA 当前正确配置 | fast observation |
|---|---|---|---|
| Phase A：canonical 初始化 | `training_stage=online_flow`, `online_flow_phase=canonical` | `training_stage=stage_a` | old + new 两时刻 |
| Phase 2 / Stage B：EMA rollout recovery | `training_stage=online_flow`, `online_flow_phase=ema_rollout` | `training_stage=stage_b` | old + new 两时刻 |
| PI0.5 历史 Stage A | `training_stage=stage_a` | 不适用 | **只有单时刻** |
| PI0.5 历史 Stage B | `training_stage=stage_b` | 不适用 | 独立 posterior updater，非当前统一 flow |

**不要把 PI0.5 和 XVLA 的 `stage_a` 字符串当成同一方法。**
PI0.5 `forward()` 先判断 `training_stage`。只设 `online_flow_phase=canonical`，但保留
`training_stage=stage_a`，仍然进入 `_forward_stage_a()`，不会创建或使用新版 old/new role/delay 模块。
同理，PI0.5 新 Phase 2 不能填 `training_stage=stage_b`。

已核实真机历史产物：
`/data/yhr/output/rm75_task1_pi05_stage_a_25hz_20261031/full8/checkpoints/020000/pretrained_model/{config.json,train_config.json}`
均为 `type=online_filtering_pi05, training_stage=stage_a, online_flow_phase=canonical`。
它是旧单时刻训练，不能标记为新版双时刻 Phase A。该产物及其衍生的错误 Phase 2 已按用户要求停止并删除；正确重训从 task1 baseline 初始化，不复用错误单时刻权重。
仅修改旧 checkpoint 的配置字段不能补出已训练的双时刻模块；若迁移旧权重，必须显式初始化新模块、核对 missing/unexpected keys，并作为新的 canonical 训练记录。

### 真机 task1 正确重训记录

- 初始化：有效 task1 π0.5 baseline 最终 20k checkpoint，未复用错误单时刻 fast 权重。
- 新 Phase A：GPU3、batch8、20k steps，`training_stage=online_flow`、`online_flow_phase=canonical`。
- 输出：`/data/yhr/output/rm75_task1_pi05_canonical_25hz_20261101`；tmux：`rm75_task1_pi05_canonical_gpu3`。
- 10-step smoke 已核实：old/new 各65 tokens、拼接130 tokens，role/pair/adapter 均有梯度；固定 RNG 的 fast 条件置零对照改变了 loss。这仅说明计算路径与局部条件依赖，不证明闭环效果。
- Phase A 完成并核实实际 role/pair 权重后，自动测试并接续正确 Phase 2：GPU3、batch8、30k steps、rollout 1–4。
- Phase 2 输出：`/data/yhr/output/rm75_task1_pi05_ema_rollout_25hz_20261101`。两阶段各只保存最终 checkpoint。
- 时序沿用真机文档：25Hz、chunk50、replan10、max delay30，四段延迟分布不变。
- 清理范围与详细启动配置：`outputs/rm75_task1_pi05_canonical_25hz_20261101_setup/`。

## 2. 研究目标与物理时间

目标是：slow VLA 推理尚未完成时，继续接收新视觉/状态观测，修正尚未执行的动作 suffix。
需要分别验证 baseline 随物理延时增大而退化，以及相同 backbone/控制协议下 Online Filtering 的改善。
不预设每个 task 在有限样本下都严格单调，也不把训练 loss 当作闭环成功率。

- 在触发时刻 `t0` 收集 slow context，生成 horizon 为 `H` 的动作块。
- 推理期间旧 chunk 继续执行。累计过去 `d` 个控制步后，新块从索引 `d` 开始使用。
- `A[:d]` 是已执行前缀，`A[d:]` 才是可修改/执行后缀。切换时不能重新从索引 0 执行。
- 物理延时为 `d / control_hz`；LIBERO 20 Hz 每步 50 ms，真机 25 Hz 每步 40 ms。
- replan interval 与 inference delay 是不同量，采样 delay 不额外加 replan interval。
- 保证旧队列能覆盖等待期，通常要求 `replan + max_delay <= H`，真机宜留裕量。

## 3. 共用架构：一个 fast 分支，两个时刻的条件

slow context 固定在触发时刻，包含语言、初始视觉和 backbone 原有的状态条件。
每次生成期间缓存 slow 特征/KV；fast packets 不重新跑完整 slow VLM。

一个 fast packet 是**一个时刻的多相机图像与同一时刻 proprio**，不是一个相机。
当前 LIBERO 配置为外部、腕部两路相机。两时刻分别记作 `r`（old）、`d`（new）：

```text
(I_r, q_r) → frozen DINOv3 + trainable spatial resampler → F_r
(I_d, q_d) → frozen DINOv3 + trainable spatial resampler → F_d
条件 = concat(F_r + old_role + delay_pair_embedding,
              F_d + new_role + delay_pair_embedding)
delay_pair_embedding = MLP([r, d, d-r] / max_delay)
```

当前两相机 `fast_num_tokens=64` 表示 **64 个视觉 tokens**，另加 1 个 proprio token。
因此每个时刻 65 tokens，old/new 拼接为 130 tokens。模型维度随 backbone 变化。
PI0.5 fast resampler 还对单包编码 delay；XVLA 在 pair 层编码 delay。

通过若干 action-transformer 层的 **zero-init residual cross-attention adapters** 注入条件，
只更新 action tokens。PI0.5 默认层 2/5/8/11/14/17，XVLA 为 3/7/11/15/19/23。
没有显式 `new-old` innovation token、独立 posterior transport 网络或 learned residual gate。
旧类仍可能存在于源码中，不代表当前路径使用它们。

这里的“两时刻”是相邻**去噪查询**前后的条件版本，不要求相邻控制帧。
新包未到或离散 delay 重复时允许 old=new；这不是退回旧单时刻架构。
它也不是把最近两帧当成固定滑窗，或分别运行两个 fast 模型。

## 4. 部署时钟、因果约束与前缀

受控延时评测使用 `N=10` 次去噪查询，总 delay 为 `D`：

```text
d_k = floor(k * D / (N-1))
t_k = 1 - k/N
old = d_max(k-1,0), new = d_k
```

例如 D=25，查询时刻为 `0,2,5,8,11,13,16,19,22,25`。
开头 old=new=0；之后可能是 (0,2)、(2,5)，不是始终相差一个控制步。
在线缓冲区只消费当前 generation 的有效最新包；不能用错 generation 或未来帧。

每次查询前，应把 `particle[:d]` hard commit 成已执行动作；更新后也保持前缀。
训练用数据动作作为执行前缀代理，部署用实际发送命令，且二者须处于相同模型动作坐标/归一化空间。
fast adapter 的 action-position mask 阻止新包改写 `<d` 位置；loss 仅计算有效、未 padding 的 suffix。
这三项约束不能互相替代。数据末尾 observation padding 也不能当作有效未来观测。

合成 delay runner 根据上述控制步时钟推进仿真，**不是实测墙钟异步延迟**。
真机要使用 packet timestamp、latest-ready 逻辑和真实执行计数；不照搬合成时钟。

## 5. Phase A：canonical flow

采样数据片段、总 delay D 和一个去噪查询 k。slow 取 t0，fast 取 old/new，目标始终是同一数据动作块 A。
在有效 suffix 上构造 `x_t = t*z + (1-t)*A`，前缀直接设为 A。

| 项目 | PI0.5 | XVLA |
|---|---|---|
| 训练函数 | `_forward_online_flow_canonical` | `_stage_a_forward` |
| 模型输出 | velocity | action endpoint |
| canonical target | `z-A` | `A` |
| objective | suffix velocity MSE（真实动作维） | 原生 position/rotation6D/gripper loss，仅有效 suffix |
| slow 结构 | PI0.5 prefix/KV + action expert | Florence 特征 + action transformer |

slow visual dropout 用于减轻完全依赖旧视觉的捷径。当前 LIBERO Phase A 初始为 0，
从 step 500 开始在 2000 steps 内升到 0.25。两 backbone 具体 masking 位置不同，
不能把同一概率解释为完全相同的逐像素遮挡。

## 6. Phase 2：EMA rollout recovery

从正确的双时刻 Phase A 权重初始化同结构 student；EMA teacher 衰减 0.999。
teacher 仅提供部署相关的中间 particle，**监督终点仍然是数据动作 A**。

1. 从噪声开始，沿相同的 old/new 控制时钟 rollout 若干查询。
2. 各步按已执行 prefix 约束 particle；student 接收 stop-gradient rollout particle。
3. student 使用 recovery 时刻的 old/new 条件，学习恢复数据 suffix。
4. 本阶段只优化 rollout recovery，不额外混 canonical loss。

PI0.5 当前默认 rollout 长度 1–4，XVLA 当前训练 1–9；这不是两套不同的观测输入格式。
PI0.5 teacher 使用 Euler 更新 `x ← x-v/N`，recovery velocity target 为 `(x-A)/t`。
XVLA teacher 使用 endpoint 重构 `x_next = t_next*z + (1-t_next)*A_hat`，student 监督 endpoint A。
当前 XVLA Stage B 的 slow dropout 在前 6k steps 保持 0.25，再用 18k steps 退火到 0。
EMA teacher 单独保存用于 resume；新阶段从 Phase A 初始化，不等于沿用 Phase A 已衰减到底的 optimizer/scheduler。

### 实现核对边界

XVLA 当前 Stage A、Stage B teacher/student 均在预测前 hard commit；Stage B 修复在本次训练启动前完成。
PI0.5 `_forward_online_flow_ema_rollout` 的 teacher 现已修复：每步预测前按 new delay 补齐已执行 prefix，proposal 后再次 commit；student recovery 前也会 commit。
新增回归测试 `tests/policies/pi0_pi05/test_online_flow_prefix_commit.py`，先复现了原问题，再验证修复。
测试同时覆盖不同 rollout 长度的 batch：已经停止的样本保留自己的 particle，不能因其他样本继续 rollout 而泄漏后续 GT prefix。
该修复用于此次正确双时刻重训；原错误 Phase 2 已停止，不作为当前方案结果。

## 7. 训练前如何确认没有跑到旧入口

先检查实际解析后的 config 和 forward 路径，不能只看 tmux 名称、脚本文件名或 job_name。

PI0.5 新 Phase A 最小关键参数：

```bash
--policy.type=online_filtering_pi05 \
--policy.training_stage=online_flow \
--policy.online_flow_phase=canonical
```

Phase 2 保留 `training_stage=online_flow`，改 `online_flow_phase=ema_rollout`。
如果通过 `--policy.path` 加载 checkpoint，仍显式覆盖两个字段；核对新模块权重是否初始化/加载正确。
`--resume` 用于同一实验精确续训，不用于把旧架构改名成新方法。

检查保存的模型配置：

```python
import json
from pathlib import Path
p = Path('/absolute/path/to/pretrained_model')
c = json.loads((p / 'config.json').read_text())
assert c['type'] == 'online_filtering_pi05'
assert c['training_stage'] == 'online_flow'
assert c['online_flow_phase'] in ('canonical', 'ema_rollout')
```

再核对 `train_config.json`、启动 log、数据时间窗口及一次实际 forward：

- 应创建 `online_condition_roles` 和 `online_delay_pair_proj`；旧 stage_a 不会创建它们。
- canonical 应记录 `canonical_flow_loss, mean_old_delay, mean_new_delay`；旧 stage_a 主要记录 `clean_flow_loss, mean_delay`。
- EMA rollout 应记录 `online_rollout_loss` 和 rollout old/new delay。
- 日志不能独立证明条件被有效使用；应结合源码调用和条件消融。
- 新 Phase A 从 backbone checkpoint 开始时 adapter 输出零初始化合理；把已训 checkpoint 当作 init 时不能再次把已学 adapter 清零。

旧入口示例：`run_stage_a_2xa100.sh`, `run_stage_a_then_b_2xa100.sh`, `run_stage_b_v3_1xa100.sh`。
`run_online_flow_phase1_3xa100.sh` / `run_online_flow_phase2_2xa100.sh` 可帮助定位正确字段，
但其中存在历史机器路径，不能不检查就执行。所有新任务先确认数据、输出、设备和配置。

## 8. 当前 XVLA 实验与数据约束

- Phase A：50 chunk，30k steps 完成；最终目录：
  `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model`。
- Phase B：从上述权重初始化，仅 GPU 4,5；每卡 batch32、global64，30k steps，每 10k 保存，无训练中评测。
  输出 `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_b_v2_2gpu_b32_30k_20261031`。
  tmux `xvla50_phase_b_45_1031`。
- Phase A 独立评测在 GPU2，delay15/20/25 各400回合，20Hz、replan20、seed1000，suite horizon280/280/300/520。
  tmux `xvla50_phase_a_eval_gpu2`；完成后结果写 metrics，不在本文推测是否完成。
- XVLA 50 chunk baseline 是实际训练50长度，不能换成官方30长度在大延时下排空队列的结果。
- 当前数据 `/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked`。
  使用离线完整 episode 转换后的20D绝对控制目标、body→grip-site旋转修正、wrist方向标记、894条不可重建初始目标mask。
- LeRobot metadata 的10fps与LIBERO物理控制20Hz不能直接混同；原始行仍是连续控制步，不能随意改metadata或隔行取样。
- 真机25Hz不照搬LIBERO控制坐标、相机旋转、时间语义或状态维度。
  真机参数另见 [real_robot_25hz_training_config.md](real_robot_25hz_training_config.md)。

## 9. 方法合理性与结论范围

old/new 双条件在 canonical 与 rollout 两阶段保持同结构，允许模型看到条件演变，设计上自洽。
不过 old/new 不是充分统计量的证明，fast 模块是否真正使用视觉、是否提高闭环成功率，需要实验回答。
干净专家轨迹中不同观测仍对应同一目标，忽略 fast 的捷径仍可能存在；dropout 只是缓解。
EMA rollout 只扩展模型 particle 分布，不等于采集了错误动作执行后的反事实物理恢复轨迹。
因此不要把双时刻架构、较低 loss 或单次成功称为方法有效的充分证据。
历史 PI0.5 成功结果主要属于此前 no-innovation/annealed 实验，不能自动归属当前 clean cross-attention 实现。

## 10. 代码索引

- PI0.5 配置/分流/训练/推理：`src/lerobot/policies/online_filtering_pi05/{configuration,modeling}_online_filtering_pi05.py`。
- PI0.5 fast encoder、resampler、adapter、历史 updater 和 runtime：`src/lerobot/policies/pi05/online_filtering.py`。
- XVLA 配置/训练/推理：`src/lerobot/policies/online_filtering_xvla/{configuration,modeling}_online_filtering_xvla.py`。
- XVLA 当前启动：`research/scripts/run_online_filtering_xvla_stage_{a,b}.sh`。
- 连续仿真评测：`research/scripts/eval_libero_synthetic_delay.py`。
  XVLA初始同步chunk也需进入online路径；PI0.5和XVLA packet字段不同，不能复用timestamp假设。
- 时间窗口：`src/lerobot/datasets/factory.py` 和两种 policy config 的 delta indices。
- 历史上下文：[project_handoff_20261030.md](project_handoff_20261030.md)、[XVLA数据审计](xvla_data_audit_20261030.md)。

工作区含其他任务改动。接手时读 `git status`/实际diff与产物配置，不覆盖现有改动，不擅自停止其他GPU任务。

## 11. 本次核对证据

除配置外，直接读取 safetensors header：XVLA Phase A 最终权重包含 `model.condition_roles`
和 `model.delay_pair_proj.{0,2}.{weight,bias}`；上述真机20k权重不包含 old/new roles 或 delay-pair 参数。
因此不是仅根据文件名推断单/双时刻。

最初核对运行以下现有回归测试，35 passed；当时未改动训练代码或重启训练：

```bash
PYTHONPATH=src /data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python -m pytest \
  tests/unit/policies/test_online_filtering_xvla.py \
  tests/policies/pi0_pi05/test_online_filtering_pi05_stage_a.py \
  tests/policies/pi0_pi05/test_continuous_online_generation.py -q
```

后续修复第6节问题并增加 prefix/不同 rollout 长度回归测试，连同延迟分布测试共 42 项通过。
这些测试不能代替闭环成功率或真机验证。


## 12. 后续实现审计更新

见 [2026-11-01实现审计与诊断](xvla_implementation_audit_20261101.md)。
上文第6节记录的PI0.5 teacher预测前新增prefix问题现已修复并加测试；旧实验权重未改变。
另已修复XVLA融合后slow dropout的视觉信息泄漏，以及无callback时绕过fast分支的标准API路径。
本次已训XVLA A/B使用旧dropout，历史成绩不能当作修复后训练结果。

## 13. 受控 simulator 与真机观测交付

见 [XVLA simulator直接取观测核对](xvla_simulator_direct_observation_20261103.md)。
新增 `--xvla-observation-delivery=direct`：query前按控制时钟推进环境，编码当前观测直接传入，
不经过观测buffer；真机继续异步latest-ready路径。默认buffered保留给已启动实验的一致性。
13项unit检查及一个实模型完整回合验证未发现XVLA原buffer路径取错观测时刻。

## 14. 最新 delta 执行评测协议

用户要求保留推理期间旧chunk导致的状态偏差，不能在切换时恢复新chunk的原全局绝对轨迹。
现已实现“预计算delta + 切换实际pose初始化一次 + 持续目标累计”的独立评测。
见 [最终协议与验证](xvla_delayed_delta_protocol_20261103.md)，tmux `xvla_accumulated_eval_1103`。
请勿将此前直接relative OSC失败或实时逆转换结果混作此协议。
