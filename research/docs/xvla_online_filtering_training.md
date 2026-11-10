# X-VLA + Online Filtering

> 当前双时刻方案与阶段配置以 [Online Filtering 当前方案（2026-11-01）](online_filtering_current_method_20261101.md) 为准。PI0.5 新 Phase A 必须使用 `training_stage=online_flow, online_flow_phase=canonical`；`training_stage=stage_a` 是旧单时刻路径。XVLA 的 `stage_a` 则是当前双时刻路径。

## Architecture

- The stock Florence condition at control step zero is the fixed slow condition.
- A frozen DINOv3 ViT-S/16 encodes the two real LIBERO cameras for every fast packet.
- A spatial resampler produces 32 tokens per camera plus one proprio token: 65 tokens per packet.
- The previous and latest packets are role/delay tagged and concatenated into 130 condition tokens.
- Zero-initialized residual cross-attention adapters update only action tokens at X-VLA layers
  3, 7, 11, 15, 19 and 23. There is no learned residual gate and no observation-difference token.
- The causal hard mask prevents a packet captured at delay `d` from changing action positions `< d`.

$$
C_{r,d}=[F_r+R_{old}+E(r,d,d-r);F_d+R_{new}+E(r,d,d-r)].
$$

## Stage A

The total delay mixture assigns 50% probability to 1--10, 25% to 11--20 and 25% to 21--25.
For ten denoising queries, training uses the same linear clock as evaluation:

$$
d_k=\left\lfloor\frac{kD}{9}\right\rfloor,
\qquad
t_k=1-\frac{k}{10}.
$$

X-VLA retains its endpoint parameterization and native position/rotation/gripper objective. Loss is
computed only on valid, unexecuted actions. Slow visual masking is zero for the first 500 steps, then
ramps linearly to 25% over 2k steps. This preserves exact stock-X-VLA initialization before gradually
forcing the zero-init fast adapters to carry visual information. On selected rows all slow visual keys
are masked while language and fast tokens remain available.

## Stage B

An EMA teacher rolls out one to nine deployment-clock queries, hard-committing the executed prefix
before every teacher/student query and after every teacher update. The student receives the resulting rollout particle and regresses the dataset
endpoint on the valid suffix only. Stage B uses rollout loss only. Slow visual masking stays at 25%
through step 6k, then anneals to zero over 18k steps. EMA decay is 0.999, and teacher state is saved
separately for exact resume.

## Default run

- `max_inference_delay=25`, action horizon 50, denoising queries 10;
- frozen DINOv3, full Florence/action-transformer fine-tuning;
- Stage A 30k steps, Stage B 30k steps;
- checkpoints every 10k steps;
- migrated dataset root: `/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked`.

## 2026-10-31 Phase A → Phase B

Phase A completed 30,000 updates in 8h38m, ending 2026-10-31 21:30 Beijing time.
Final checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model`.
Final logged loss 0.009 is a training loss, not a rollout success rate.

Phase B starts from that checkpoint on GPUs 4,5 only, batch 32 per GPU, global batch 64,
30k updates, float32 weights with bf16 AMP. Saves every 10k, no evaluation.
EMA teacher decay 0.999, rollout queries 1–9; visual dropout anneals 0.25 → 0 over steps 6k–24k.
The executed prefix is committed before teacher and student predictions; teacher slow inputs are
computed under no_grad. Regression suite: 15 passed.

- tmux: `xvla50_phase_b_45_1031`, window `progress`.
- output: `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_b_v2_2gpu_b32_30k_20261031`
- log: `/data/yhr/output/online_filtering/logs/xvla50_online_filtering_stage_b_v2_2gpu_b32_30k_20261031.log`
- launch and code snapshot: `/data/yhr/output/online_filtering/logs/xvla50_phase_b_v2_artifacts`

## 2026-11-01 修复 dropout 的 10k Phase A 对照

用户授权本轮只训10k；按最新指示使用GPU0、2，不等待GPU4、5，不中断其他任务。
从与此前Phase A相同的XVLA50 baseline init重新训练，不续训旧Phase A/B，不添加邻域采样。
保持每卡batch32、global64、seed1000、30k学习率调度、dropout 500–2500步升至0.25；
仅训练停止点改为10k，保存间隔10k，无训练中评测。
本轮使用Florence图文融合前visual dropout修复；启动前相关测试18 passed。

- tmux：`xvla50_dropoutfix_10k_02`，窗口`progress`。
- log：`/data/yhr/output/online_filtering/logs/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101.log`
- output：`/data/yhr/output/online_filtering/ckpt/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101`
- 运行配置、代码快照、启动脚本：`/data/yhr/output/online_filtering/logs/xvla50_phase_a_dropoutfix_10k_20261101_artifacts`。

这不是旧Phase A30k与修复版10k的等步数正式对照；后续比较需要标注训练预算，
旧Phase A10k权重保留，可以用于同预算比较。

启动时发现Cosine scheduler默认把30k调度自动压到10k，已停止该启动尝试并归档日志，未作为对照结果。
新增向后兼容开关 `policy.scheduler_scale_to_training_steps=false`，重新从同一baseline初始化启动。
默认值仍为true，其他实验行为不变；针对短程保持长程调度的回归测试通过。
最终启动脚本在本轮artifacts中，包含该显式开关；学习率warmup1000、decay30000。

## 2026-11-02 修复 dropout 的 Phase A 扩展到总30k

用户授权完整Phase A；在GPU0、2从修复版010000精确resume，保留optimizer、scheduler和RNG状态。
总steps30000（再训20000），每卡batch32/global64，仍每10k保存、无训练中评测、不加入邻域采样。
10k对照当初关闭了scheduler自动压缩，继续原30k学习率曲线。
为保持resume与引用路径，输出目录沿用带`10k`的历史目录名：
`/data/yhr/output/online_filtering/ckpt/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101`；
后续020000/030000也在此目录，实际训练预算看各checkpoint的train_config，不看父目录名称。

- tmux：`xvla50_dropoutfix_30k_02`，`progress`窗口。
- log：`/data/yhr/output/online_filtering/logs/xvla50_phase_a_dropoutfix_resume30k_20261102.log`
- launch/code diff：`/data/yhr/output/online_filtering/logs/xvla50_phase_a_dropoutfix_resume30k_20261102_artifacts`。

## 2026-11-02 修复版 Phase A 完成，启动 Phase B

修复版Phase A累计30k于北京时间2026-11-02 05:34完成，后20k耗时5h31m。
用户授权接续Phase B，从修复版Phase A的030000模型初始化新optimizer/teacher，不resume旧Phase B。

- source：`/data/yhr/output/online_filtering/ckpt/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101/checkpoints/030000/pretrained_model`
- output：`/data/yhr/output/online_filtering/ckpt/xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102`
- log：`/data/yhr/output/online_filtering/logs/xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102.log`
- tmux：`xvla50_phase_b_dropoutfix_02`，窗口`progress`。
- launch/diff：`/data/yhr/output/online_filtering/logs/xvla50_phase_b_dropoutfix_20261102_artifacts`。

GPU0、2，每卡batch32/global64，50chunk/maxdelay25，30ksteps，每10k保存，无训练中评测。
EMA0.999，rollout1–9；slow dropout前6k为0.25，接着18k退火到0，使用融合前mask修复。
不加入邻域采样。本轮结果尚待训练完成与独立评测。
