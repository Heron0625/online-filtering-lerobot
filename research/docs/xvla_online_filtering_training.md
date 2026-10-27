# X-VLA + Online Filtering 训练框架

## 模型

- policy type：`online_filtering_xvla`；
- old/new observation 共用 Florence-2 编码器；
- delay-conditioned gate 融合两组特征，不额外构造 `new-old` token；
- 默认全参微调 Florence vision/language encoder、action transformer 和 soft prompt；冻结仅保留为显存消融开关。

融合写成：

$$
C=(1-g)C_{old}+gC_{new},\qquad g=\sigma(\operatorname{MLP}(d_{old},d_{new},s)).
$$

## Stage A

从同一轨迹窗口采样：

$$
0\le d_{old}\le d_{new}\le 10.
$$

按 X-VLA 原始 flow-matching 目标训练 two-condition policy。新增 gate 初始化为 $g=0.5$；当 old/new 相同时与单条件输入等价。

## Stage B

- 从 Stage A checkpoint 初始化 student 和 EMA teacher；
- teacher 按部署一致的线性 delay schedule rollout 1--9 个去噪查询；
- 已执行 action prefix hard commit；
- student 只在未执行 suffix 上回归数据集 action；
- suffix 复用 X-VLA 原始 position/rotation/gripper action-space loss；
- 每个 optimizer step 后以 decay `0.999` 更新 EMA。

$$
\theta_{EMA}\leftarrow0.999\theta_{EMA}+0.001\theta.
$$

## 建议首轮配置

- LIBERO：`max_inference_delay=10`，chunk 30，denoising steps 10；
- Stage A：30k steps；Stage B：30k steps；
- 全局 batch 先按现有显存实测，OOM 时优先用 gradient accumulation 保持全局 batch；
- 每 10k 保存一次；先做 20-step smoke，再启动正式训练。
- 迁移后的 LIBERO root：`/data/yhr/pretrain/recovered_zhoukai/dataset/libero`。

