# X-VLA + Online Filtering

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
after every query. The student receives the resulting rollout particle and regresses the dataset
endpoint on the valid suffix only. Stage B uses rollout loss only. Slow visual masking stays at 25%
through step 6k, then anneals to zero over 18k steps. EMA decay is 0.999, and teacher state is saved
separately for exact resume.

## Default run

- `max_inference_delay=25`, action horizon 30, denoising queries 10;
- frozen DINOv3, full Florence/action-transformer fine-tuning;
- Stage A 30k steps, Stage B 30k steps;
- checkpoints every 10k steps;
- migrated dataset root: `/data/yhr/pretrain/recovered_zhoukai/dataset/libero`.
