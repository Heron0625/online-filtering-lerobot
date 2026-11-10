# Real-Robot 25 Hz Training Configuration

> 当前双时刻方案与阶段配置以 [Online Filtering 当前方案（2026-11-01）](online_filtering_current_method_20261101.md) 为准。PI0.5 新 Phase A 必须使用 `training_stage=online_flow, online_flow_phase=canonical`；`training_stage=stage_a` 是旧单时刻路径。XVLA 的 `stage_a` 则是当前双时刻路径。

Backbone: **π0.5 (`pi05`)** for both the standard baseline and Online Filtering.

## Deployment assumptions

- Camera, state recording, and action execution frequency: `25 Hz`.
- One action step represents `40 ms`.
- The VLA runs asynchronously; one inference takes approximately `700–1000 ms`.
- “25 Hz inference” means that the predicted action chunk is executed at `25 Hz`. It does not mean running the full model 25 times per second.
- Allow at most one full VLA inference in flight. Do not queue inference requests faster than the model can finish them.

## Required parameters

```text
data_hz = 25
control_hz = 25
chunk_size = 50
replan_step = 10
max_inference_delay = 30
```

At `25 Hz`, a 50-step chunk covers:

$$
50 / 25 = 2.0\ \mathrm{s}.
$$

`replan_step=10` means that the next asynchronous observation and inference start after the current chunk has executed for:

$$
10 / 25 = 0.4\ \mathrm{s}.
$$

## Training delay

Convert end-to-end inference latency to action steps using:

$$
d = \operatorname{round}(25t),
$$

where $$t$$ is latency in seconds. The measured latency corresponds to:

$$
700\ \mathrm{ms} \rightarrow 18\ \text{steps},
$$

$$
1000\ \mathrm{ms} \rightarrow 25\ \text{steps}.
$$

Use `max_inference_delay=30` to cover latency up to:

$$
30 / 25 = 1.2\ \mathrm{s}.
$$

Recommended delay sampling distribution:

| Delay range | Probability | Purpose |
|---|---:|---|
| `1–10` | 15% | Low-latency robustness |
| `11–17` | 20% | Transition region |
| `18–25` | 50% | Main measured deployment range |
| `26–30` | 15% | Tail latency and system jitter |

The delay is the number of 25 Hz control steps elapsed between observation capture and inference completion. Do not include `replan_step` in the sampled delay.

## Queue coverage constraint

The configuration must satisfy:

$$
\text{replan\_step} + \text{max\_inference\_delay} < \text{chunk\_size}.
$$

For the selected values:

$$
10 + 30 = 40 < 50.
$$

The worst-case remaining overlap is therefore:

$$
50 - 10 - 30 = 10\ \text{steps} = 0.4\ \mathrm{s}.
$$

Use these remaining steps for old/new chunk transition. The default blending window should be `5–10` steps. Do not linearly blend discrete gripper commands; use a thresholded switch or retain the previous gripper command until the switch point.

## Data and training consistency

- Record images, robot state, and actions with synchronized timestamps at `25 Hz`.
- Construct 50-step training targets from this 25 Hz timeline; do not resample them to the LIBERO 20 Hz convention.
- Train both the standard π0.5 baseline and π0.5 + Online Filtering with `chunk_size=50` and the same 25 Hz action semantics.
- Stage A and Phase 2 must both use `max_inference_delay=30` and the delay distribution above.
- Report real-device latency using end-to-end observation-to-action-ready time, preferably including p50, p95, and p99.
- If p99 exceeds `1.2 s`, reconsider `chunk_size` or `replan_step`; do not silently clamp the measured delay to 30.
