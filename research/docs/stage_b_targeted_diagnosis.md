# Stage B 关键阶段诊断

诊断脚本：`research/scripts/diagnose_online_gripper_transitions.py`。
结果目录：`/data/zhoukai/robotmine/eval/stage_b_validation/`。

## 选样与控制

从真实 LIBERO 数据的夹爪指令符号切换中，选择 task 分散的 8 个 rising 和 8 个 falling 事件。它们只是抓取／释放阶段的代理，不能当作已确认物理接触。每个事件测试 flow grid step 3 和 8，同一事件共享同一噪声。共 16 个事件、32 个案例。

固定 slow 观测；旧观测相对控制步为 0，新观测为 8。半数事件在相对步 4 切换夹爪指令，半数在步 10 切换，因此覆盖跨越切换与切换前两种情形。

固定新 delay、commitment 和同一个旧 particle 后，比较四种 condition：

1. stale：旧图像 + 旧 proprio；
2. image_only：新图像 + 旧 proprio；
3. proprio_only：旧图像 + 新 proprio；
4. fresh：新图像 + 新 proprio。

记录同一 particle 上的速度差异及继续积分后的动作差异，按真实动作维度分别保存；排除 padding 和已执行前缀。这些是敏感性干预，混搭 condition 不一定符合物理状态，不用于直接判断策略成功率。

## Teacher 与 updater

同时比较：

- 原训练配对：delay、观测和 committed prefix 一起变化；
- observation-only：delay／prefix 固定，只改变当前图像与 proprio；
- delay/commitment-only：图像与 proprio 固定，只改变 delay 嵌入、attention mask 和 committed prefix；这三个控制因素仍是联合干预，不是单独时间嵌入消融。

三种 teacher delta 不构成可直接相加的分解，因为模型与积分是非线性的。

从真实训练配对的旧 particle 出发，比较 updater 输出是否更接近新 teacher particle，再在同一新 condition 下继续积分，比较有／无 updater 的最终动作与 teacher reference 的差异。

另外记录各 condition 的最终动作对 GT 的误差。随机噪声采样的动作不必逐点等于示范，因此 GT MSE 只是辅助配对指标，不能替代闭环任务成功率。

## 复现

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -u research/scripts/diagnose_online_gripper_transitions.py \
  --checkpoint /data/zhoukai/robotmine/ckpt/online_filtering_pi05_clean_causal/stage_b_online_teacher/checkpoints/003000/pretrained_model \
  --output /data/zhoukai/robotmine/eval/stage_b_validation/gripper_transition_report.json
```

脚本仅推理，不执行 optimizer update。输出逐批保存，完整运行应有 32 个 samples；不完整报告不能当作完整诊断。数据可能参与过训练，不是 held-out 泛化评测。
