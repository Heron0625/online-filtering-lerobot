# Metrics

## Dynamic delay 1–10, 20 Hz

| Model | Replan interval | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase A, 30k | 20 | 2000 | 88.0% | 95.6% | 94.8% | 88.4% | 91.70% |
| Phase 2, mixed 30k | 20 | 2000 | 84.8% | 95.4% | 95.8% | 83.2% | 89.80% |
| Phase 2, no-innovation annealed 30k | 20 | 2000 | 89.4% | 95.8% | 94.2% | 88.4% | 91.95% |
| Phase 2, no-innovation annealed 60k | 20 | 2000 | 88.4% | 96.4% | 95.4% | 88.0% | 92.05% |
| X-VLA, PI0.5-matched horizon | 20 | 2000 | 94.0% | 98.4% | 97.2% | 95.0% | 96.15% |
| π0-FAST | 0 | 2000 | 61.2% | 54.4% | 66.8% | 47.2% | 57.40% |
| OpenVLA-OFT | 0 | 2000 | 67.8% | 42.2% | 63.0% | 40.6% | 53.40% |

## Fixed delay, replan interval 20, 20 Hz

| Model | Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official PI0.5 | 1 | 400 | 95.0% | 98.0% | 96.0% | 95.0% | 96.00% |
| Phase A | 1 | 400 | 93.0% | 100.0% | 98.0% | 91.0% | 95.50% |
| Phase 2 | 1 | 400 | 98.0% | 99.0% | 95.0% | 98.0% | 97.50% |
| Official PI0.5 | 5 | 400 | 89.0% | 96.0% | 92.0% | 84.0% | 90.25% |
| Phase A | 5 | 400 | 85.0% | 96.0% | 93.0% | 88.0% | 90.50% |
| Phase 2 | 5 | 400 | 89.0% | 97.0% | 95.0% | 89.0% | 92.50% |
| Official PI0.5 | 10 | 400 | 73.0% | 95.0% | 92.0% | 66.0% | 81.50% |
| Phase A | 10 | 400 | 79.0% | 87.0% | 89.0% | 74.0% | 82.25% |
| Phase 2 | 10 | 400 | 82.0% | 94.0% | 94.0% | 78.0% | 87.00% |

## Large fixed delay, replan interval 20, 20 Hz

| Model | Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official PI0.5 | 15 | 400 | 66.0% | 81.0% | 90.0% | 54.0% | 72.75% | 0 |
| Official PI0.5 | 20 | 400 | 65.0% | 78.0% | 84.0% | 46.0% | 68.25% | 0 |
| Official PI0.5 | 25 | 400 | 60.0% | 71.0% | 75.0% | 39.0% | 61.25% | 0 |
| X-VLA | 15 | 400 | 87.0% | 97.0% | 94.0% | 92.0% | 92.50% | 18,281 |
| X-VLA | 20 | 400 | 86.0% | 97.0% | 76.0% | 82.0% | 85.25% | 48,181 |
| X-VLA | 25 | 400 | 69.0% | 76.0% | 61.0% | 61.0% | 66.75% | 120,081 |

## X-VLA corrected-data 30-chunk quick check (2026-10-31)

Diagnostic only: 40 tasks × 1 initial state per model, seed 1000 / episode index 0.
Both models evaluated with the current identical processors, delay=0, 20 Hz,
replan interval=30, chunk size=30, 10 denoising steps. Full suite horizons:
Spatial/Object/Goal/LIBERO-10 = 280/280/300/520 control steps; no rollout caps.
Software llvmpipe rendering was used for both models; inference on GPU3.
These small-sample results are not a replacement for the main benchmark table.

| Model | Steps | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official X-VLA-30, current eval pipeline | 0 | 40 | 9/10 | 10/10 | 10/10 | 7/10 | 36/40 (90.0%) | 0 |
| X-VLA-30, corrected LeRobot data v2 | 10,000 | 40 | 9/10 | 10/10 | 9/10 | 9/10 | 37/40 (92.5%) | 0 |

The fine-tuned checkpoint retains usable baseline performance in this diagnostic;
+2.5 pp is one additional successful episode and is not evidence of a significant
improvement. This test does not establish 50-chunk quality or delay robustness.
Training uses GPU4/5, batch32 per GPU (global64), float32, and masks the 894
leading targets whose initial controller goal was not stored in the source data.

Checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla30_corrected_v2_2gpu_b32_30k_20261030/checkpoints/010000/pretrained_model`

Raw results and videos: `/data/yhr/output/online_filtering/eval/xvla30_corrected_10k_vs_official_d0_r30_quick40`

Aggregate: `/data/yhr/output/online_filtering/eval/xvla30_corrected_10k_vs_official_d0_r30_quick40/comparison_summary.json`

Reproducer: `research/scripts/run_xvla30_corrected_quick_eval.sh`.

## X-VLA-50 corrected data, 30k checkpoint (2026-10-31)

True 50-action-horizon fine-tune from the official30 positional-remapped initialization,
not a 30-step fine-tune extended only at inference. Training completed 30,000 updates.
Evaluation uses 20 Hz, replan interval20 and full Spatial/Object/Goal/LIBERO-10
horizons 280/280/300/520. Delay 15/20/25 corresponds to 750/1000/1250 ms.

| Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (quick check) | 40 | 9/10 | 10/10 | 10/10 | 9/10 | 38/40 (95.0%) | 0 |
| 15 | 400 | 91.0% | 96.0% | 100.0% | 90.0% | 94.25% | 0 |
| 20 | 400 | 87.0% | 91.0% | 96.0% | 87.0% | 90.25% | 0 |
| 25 | 400 | 88.0% | 88.0% | 98.0% | 84.0% | 89.50% | 0 |

Delays15/20/25 completed: 400 episodes per delay (40 tasks × 10 initial states).
The no-delay quick check has fewer initial states and is not a matched400-episode
control. Legacy X-VLA large-delay rows above contain queue underruns and should
not be treated as clean latency-only reference points.

Checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla50_corrected_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model`

Results: `/data/yhr/output/online_filtering/eval/xvla50_corrected_30k_{zero40,delay15_400,delay20_400,delay25_400}_r20`

Launchers: `research/scripts/run_xvla50_corrected_eval.sh` (zero/25),
`research/scripts/eval_xvla50_delay_pool.py` (15/20; five model workers per GPU,
two tasks per work item, dynamically scheduled on GPUs3–7).

## Storage maintenance (2026-10-31)

Obsolete stopped XVLA-30 diagnostic weights and XVLA-50 10k/20k intermediate
checkpoints were removed after the final50 baseline was validated. Their small
configs/logs are archived under `research/outputs/cleanup_20261031/metadata`.
Historical numerical results above remain valid records; their old intermediate
weight paths are no longer loadable. The final50 030000 full checkpoint, formal
rollouts/summaries, official initialization, v2_masked training data, raw/replay
HDF5, and successful PI0.5 checkpoints remain available.

Phase A has now started from the final50 30k checkpoint on GPUs4/5 only:
chunk50, batch32 per GPU / global64, 30k steps, save every10k, no environment
or held-out evaluation. No Phase-A success-rate results are available yet.


## X-VLA-50 Online Filtering Phase A, 30k checkpoint (evaluation completed)

Started on GPU 2 only; Phase B continues independently on GPUs 4,5.
Checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model`.
Same baseline protocol: delay 15/20/25, 400 episodes each (40 tasks × 10 initial states),
20 Hz, replan 20, seed 1000, suite horizons 280/280/300/520, online fast observations enabled.
Five concurrent model processes on GPU 2, two tasks per work item.
| Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 400 | 89.0% | 94.0% | 98.0% | 90.0% | 92.75% | 0 |
| 20 | 400 | 87.0% | 89.0% | 93.0% | 84.0% | 88.25% | 0 |
| 25 | 400 | 89.0% | 90.0% | 95.0% | 81.0% | 88.75% | 0 |
Outputs: `/data/yhr/output/online_filtering/eval/xvla50_phase_a_30k_delay{15,20,25}_400_r20`;
automatic final aggregation: `xvla50_phase_a_30k_latency_comparison.json` in the same eval root.
Tmux: `xvla50_phase_a_eval_gpu2`, window `progress`.


## X-VLA-50 Online Filtering Phase B, 30k checkpoint (evaluation completed)

Phase B completed 30k steps on 2026-11-01 13:13 Beijing time, elapsed 13h56m.
Checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla50_online_filtering_stage_b_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model`.
Evaluation: GPU 1,2,6,7, five workers per GPU, 20 concurrent models. Delay 15/20/25,
400 episodes each, 20Hz, replan20, seed1000, suite horizons 280/280/300/520.
| Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 400 | 89.0% | 96.0% | 99.0% | 85.0% | 92.25% | 0 |
| 20 | 400 | 85.0% | 92.0% | 98.0% | 83.0% | 89.50% | 0 |
| 25 | 400 | 91.0% | 95.0% | 98.0% | 78.0% | 90.50% | 0 |
Outputs: `/data/yhr/output/online_filtering/eval/xvla50_phase_b_30k_delay{15,20,25}_400_r20`.
Tmux: `xvla50_phase_b_eval_parallel`, window `progress`.

## X-VLA-50 Online Filtering Phase B dropoutfix, 30k checkpoint (evaluation completed)

Checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102/checkpoints/030000/pretrained_model`.
Uses the repaired prefusion slow visual dropout training, initialized from the repaired Phase A final30k.
Evaluation: GPU0/6/7, five models per GPU, 15 workers; delay15/20/25, 400 episodes each,
20Hz, replan20, seed1000, suite horizons280/280/300/520. Normal online observations; no ablation or diagnostic seed override.
| Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 400 | 89.0% | 96.0% | 98.0% | 95.0% | 94.50% | 0 |
| 20 | 400 | 90.0% | 94.0% | 93.0% | 88.0% | 91.25% | 0 |
| 25 | 400 | 89.0% | 94.0% | 97.0% | 80.0% | 90.00% | 0 |
Outputs: `/data/yhr/output/online_filtering/eval/xvla50_phase_b_dropoutfix_30k_delay{15,20,25}_400_r20`.
Tmux: `xvla50_dropoutfix_eval_parallel`, window `progress`.

## XVLA target-difference relative-control pilot (2026-11-03, stopped after semantic audit)

Separate experimental execution interface, not the original absolute-control protocol.
Both baseline50 final30k and repaired Phase B final30k: delay0/15/25, 40 episodes each, 240 total.
20Hz/replan20, paired per-episode seed. Relative commands difference consecutive predicted absolute targets;
no reanchor at delayed installation. Report zero-delay regression and clipping along with success.
Definition and limitations: [relative-control pilot](xvla_chunk_delta_pilot_20261103.md).
Stopped after zero-delay regression: goal-to-goal differences are not equivalent to original relative commands under imperfect tracking. Partial episodes are audit artifacts, not a valid latency-only comparison. See the linked audit for real-controller and expert-data checks.

## XVLA execution-time relative target conversion pilot (2026-11-03, completed)

Separate from the stopped target-to-target difference experiment. Keeps absolute targets in the queue;
at each action execution converts against the then-measured pose using `--xvla-relative-execution`.
Baseline50 final30k and repaired Phase B final30k, each delay0/15/25 × 40 tasks × 1 initial state: 240 episodes total.
20Hz, replan20, paired per-episode seed, suite horizons280/280/300/520; OF uses direct simulator observations.
GPU2/6/7, five model processes per GPU, 15 workers. No failed old-run summaries are included.
This checks the corrected execution interface and clipping; it is not a newly trained PI-style delta policy.
Final results: `research/outputs/xvla_relative_execution_20261103/comparison.json`.
Tmux: `xvla_relative_exec_1103`, progress window.

| Delay | Baseline | Repaired Phase B | Difference |
|---|---:|---:|---:|
| 0 | 38/40 (95.0%) | 37/40 (92.5%) | -2.5 pp |
| 15 | 36/40 (90.0%) | 37/40 (92.5%) | +2.5 pp |
| 25 | 36/40 (90.0%) | 34/40 (85.0%) | -5.0 pp |

All 240 episodes completed; queue underruns0. Clipped-command counts (baseline/OF) are 62/334 at delay0, 281/247 at15, 418/488 at25. These are controller command counts, not episode counts.
No consistent OF gain in this 40-episode-per-arm pilot. The execution-time inverse conversion retains absolute-target feedback where commands are representable; it does not test a native relative-action-trained policy.

## XVLA precomputed target-delta pilot (2026-11-03, stopped for executor semantics audit)

User-requested execution: compute all relative commands once at chunk creation, then send sequentially;
no execution-time state subtraction and no extra gain. Uses `--xvla-precomputed-deltas`.
Baseline50/OF repaired Phase B30k, delay0/15/25 × 40 episodes per model: 240 total.
GPU6/7, five model processes per GPU (10 workers). Distinct from both the stopped initial attempt and
completed measured-target inverse-conversion test; neither earlier result is merged into this output.
Final results: `research/outputs/xvla_precomputed_eval_20261103/comparison.json`.
Tmux: `xvla_precomputed_eval_1103`, progress window.


The precomputed-delta run was stopped after abnormal zero-delay failures. Real OSC audit shows
consecutive-target increments are interpreted relative to the current measured pose, not accumulated
on the previous commanded target. A fixed10mm target yields only2.03mm displacement with this
interface versus10.44mm for absolute/target-accumulating execution over20 control steps.
See [executor audit](xvla_delta_executor_audit_20261103.md). Do not treat this pilot as latency-only evidence.

## XVLA delayed delta trajectories with persistent targets (2026-11-03, completed; zero-delay parity caveat)

Latest user-defined protocol: precompute deltas at trigger time, discard expired prefix, initialize a persistent target once at the actual switch pose, then integrate suffix deltas without per-step reanchoring or restoring original global targets.
Baseline50/OF repaired Phase B30k, delay0/15/25 × 40 episodes per model, 240 total; GPU6/7 × 5 models.
[Protocol and validation](xvla_delayed_delta_protocol_20261103.md).
Results pending: `research/outputs/xvla_accumulated_eval_20261103/comparison.json`.
Tmux: `xvla_accumulated_eval_1103`.

| Delay | Baseline | Repaired Phase B | OF minus baseline |
|---|---:|---:|---:|
| 0 | 37/40 (92.5%) | 38/40 (95.0%) | +2.5 pp |
| 15 | 12/40 (30.0%) | 33/40 (82.5%) | +52.5 pp |
| 25 | 16/40 (40.0%) | 27/40 (67.5%) | +27.5 pp |

All240 episodes completed; queue underruns0. Clipped-command counts baseline/OF: 2/7 at delay0, 0/52 at15, 0/112 at25.
Important unresolved parity check: LIBERO Spatial task5 init0, matched per-episode seed, delay0: native absolute baseline succeeds at132 steps, accumulated baseline fails at280, neither clipped. Initial trajectory reconstruction tests do not establish full closed-loop equivalence. Treat the large-delay gains as preliminary until this discrepancy is audited.

## XVLA accumulated-delta full curves, replan25 (2026-11-03)

| Delay | Baseline | OF | OF minus baseline |
|---|---:|---:|---:|
| 0 | 390/400 (97.50%) | 381/400 (95.25%) | -2.25 pp |
| 5 | 214/400 (53.50%) | 320/400 (80.00%) | +26.50 pp |
| 10 | 181/400 (45.25%) | 286/400 (71.50%) | +26.25 pp |
| 15 | 184/400 (46.00%) | 266/400 (66.50%) | +20.50 pp |
| 20 | 185/400 (46.25%) | 268/400 (67.00%) | +20.75 pp |
| 25 | 185/400 (46.25%) | 295/400 (73.75%) | +27.50 pp |

- Protocol: precompute target deltas at observation, skip expired prefix, initialize a persistent target once at actual chunk installation, accumulate suffix without live-state reanchoring.
- Fixed replan25, 20Hz, H50; delays0/5/10/15/20/25, actual trigger period25 at every point.
- Baseline50 final30k and repaired OF Phase B final30k, 40 tasks × 10 initial states =400 episodes per model/delay, 4800 total.
- Suite horizons280/280/300/520; paired per-task/episode seed1000. OF uses direct current-simulator observation per denoising query.
- Completed all 4800 episodes on GPUs1/5. Initially 16 workers per GPU; after a CUDA allocation failure due to increased external GPU occupancy, resumed with GPU1:4 and GPU5:12. Saved episode prefixes were reused; interrupted attempts are not counted as policy failures. Disable only diagnostic video rendering/encoding; model camera observations and physics unchanged.
- This is independent of the earlier replan20 40-episode pilot; do not merge them.
- Results: `research/outputs/xvla_accumulated_full_r25_20261103/comparison.json`.
- Progress/validated jobs: same root `pool.log`, `pool_records.json`, `manifest.json`.
- Tmux: `xvla_full_r25_1103`, progress window.


## PI0.5 and XVLA delay curves, 0–20 steps (2026-11-03)

These are the exact data used in the combined plot. All rates are overall LIBERO success rates, with 400 episodes per model/delay (40 tasks × 10 initial states), at 20 Hz. PI0.5 now uses the completed matched R25 evaluation of official PI0.5 and the clean Phase 2 checkpoint; XVLA uses its completed R25 evaluation. The historical R20 PI0.5 points are retained in their original source records and are not mixed into this plot.

| Delay (steps) | Delay (ms) | PI0.5 baseline | PI0.5 + OF | XVLA baseline | XVLA + OF |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 93.50% | 94.25% | 97.50% | 95.25% |
| 5 | 250 | 88.50% | 90.00% | 53.50% | 80.00% |
| 10 | 500 | 83.00% | 85.00% | 45.25% | 71.50% |
| 15 | 750 | 75.75% | 81.75% | 46.00% | 66.50% |
| 20 | 1000 | 70.50% | 75.75% | 46.25% | 67.00% |

- PI0.5: replan25, native relative OSC, clean two-condition OF Phase 2 final30k. Source: `research/outputs/pi05_full_r25_20261107/comparison.json`; complete results and per-suite breakdown are recorded below.
- XVLA: replan25, H50, accumulated-delta execution, final baseline30k vs repaired OF Phase B30k; source is the full R25 evaluation above. All 4800 episodes have zero queue underruns.
- Both backbones use replan25 and 400 episodes per point. Their action execution semantics and OF model implementations differ, so the plot supports within-backbone baseline/OF comparisons rather than a direct ranking between backbones.
- Delay25 is retained in both complete results tables, but excluded from this 0–20 figure as requested.
- Figure: [PNG](../figures/libero_delay_0_20.png), [PDF](../figures/libero_delay_0_20.pdf), [SVG](../figures/libero_delay_0_20.svg). Data: [CSV](../figures/libero_delay_0_20.csv). Reproducer: `research/figures/plot_delay_curves.py`.

### XVLA full R25 evaluation: per-suite breakdown

Each suite has 100 episodes per row. Rates below are percentages; queue underruns are episode-total counts.

| Model | Delay | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 94.00% | 100.00% | 99.00% | 97.00% | 97.50% | 0 |
| OF Phase B | 0 | 89.00% | 98.00% | 99.00% | 95.00% | 95.25% | 0 |
| Baseline | 5 | 35.00% | 71.00% | 49.00% | 59.00% | 53.50% | 0 |
| OF Phase B | 5 | 70.00% | 92.00% | 68.00% | 90.00% | 80.00% | 0 |
| Baseline | 10 | 28.00% | 69.00% | 43.00% | 41.00% | 45.25% | 0 |
| OF Phase B | 10 | 63.00% | 80.00% | 69.00% | 74.00% | 71.50% | 0 |
| Baseline | 15 | 19.00% | 75.00% | 49.00% | 41.00% | 46.00% | 0 |
| OF Phase B | 15 | 50.00% | 73.00% | 69.00% | 74.00% | 66.50% | 0 |
| Baseline | 20 | 32.00% | 78.00% | 52.00% | 23.00% | 46.25% | 0 |
| OF Phase B | 20 | 51.00% | 75.00% | 71.00% | 71.00% | 67.00% | 0 |
| Baseline | 25 | 42.00% | 69.00% | 61.00% | 13.00% | 46.25% | 0 |
| OF Phase B | 25 | 75.00% | 85.00% | 78.00% | 57.00% | 73.75% | 0 |


## Obsolete XVLA A/B checkpoint cleanup (2026-11-05)

Removed the pre-dropoutfix checkpoints `xvla50_online_filtering_stage_a_v2_2gpu_b32_30k_20261031` and `xvla50_online_filtering_stage_b_v2_2gpu_b32_30k_20261031`, freeing approximately72.31GiB. Historical results above remain recorded; their old model paths are no longer loadable. Configurations and small text metadata are archived in `research/outputs/cleanup_xvla_old_ab_20261105/`. Retained the final H50 baseline and repaired dropoutfix Phase A/B checkpoints used for current experiments.


## PI0.5 clean Online Filtering full delay curves, replan25 (2026-11-07)

Completed evaluation: official PI0.5 versus clean two-condition OF Phase 2 final30k. Each model/delay covers 40 tasks × 10 initial states = 400 episodes; delays 0/5/10/15/20/25, 4800 episodes total. Both use native PI0.5 relative OSC, 20Hz, H50, fixed replan25, 10 denoising queries, paired per-task/episode seed1000 and suite horizons280/280/300/520. OF encodes the current simulator observation directly at each query. Both run eager inference; diagnostic videos are disabled. This is a new matched R25 evaluation, separate from the historical R20 results.

| Delay | Official PI0.5 | PI0.5 + OF Phase 2 | OF minus baseline |
|---|---:|---:|---:|
| 0 | 374/400 (93.50%) | 377/400 (94.25%) | +0.75 pp |
| 5 | 354/400 (88.50%) | 360/400 (90.00%) | +1.50 pp |
| 10 | 332/400 (83.00%) | 340/400 (85.00%) | +2.00 pp |
| 15 | 303/400 (75.75%) | 327/400 (81.75%) | +6.00 pp |
| 20 | 282/400 (70.50%) | 303/400 (75.75%) | +5.25 pp |
| 25 | 251/400 (62.75%) | 283/400 (70.75%) | +8.00 pp |

### Per-suite success rates

| Model | Delay | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official PI0.5 | 0 | 89.00% | 98.00% | 91.00% | 96.00% | 93.50% | 0 |
| OF Phase 2 | 0 | 91.00% | 100.00% | 91.00% | 95.00% | 94.25% | 0 |
| Official PI0.5 | 5 | 87.00% | 96.00% | 92.00% | 79.00% | 88.50% | 0 |
| OF Phase 2 | 5 | 87.00% | 95.00% | 94.00% | 84.00% | 90.00% | 0 |
| Official PI0.5 | 10 | 80.00% | 92.00% | 90.00% | 70.00% | 83.00% | 0 |
| OF Phase 2 | 10 | 78.00% | 95.00% | 91.00% | 76.00% | 85.00% | 0 |
| Official PI0.5 | 15 | 72.00% | 85.00% | 88.00% | 58.00% | 75.75% | 0 |
| OF Phase 2 | 15 | 74.00% | 91.00% | 90.00% | 72.00% | 81.75% | 0 |
| Official PI0.5 | 20 | 75.00% | 81.00% | 82.00% | 44.00% | 70.50% | 0 |
| OF Phase 2 | 20 | 70.00% | 89.00% | 88.00% | 56.00% | 75.75% | 0 |
| Official PI0.5 | 25 | 66.00% | 74.00% | 78.00% | 33.00% | 62.75% | 0 |
| OF Phase 2 | 25 | 65.00% | 85.00% | 85.00% | 48.00% | 70.75% | 0 |

- Baseline: `/data/yhr/pretrain/recovered_zhoukai/pretrain/pi05_libero_finetuned`.
- OF: `/data/yhr/output/online_filtering/ckpt/pi05_phase2_clean_67_2gpu_b16_30k_20261105/checkpoints/030000/pretrained_model`.
- Results: `research/outputs/pi05_full_r25_20261107/comparison.json` (published after all validated episodes finish).
- Pool: `research/scripts/run_pi05_full_delay_eval.py`; output root containsmanifest.json, pool_records.json, pool.log and per-episode progress.

## PI0.5 and XVLA Phase A full delay evaluation, replan25 (2026-11-08)

Completed matched evaluation of the Phase A checkpoints on both backbones. Each
method/delay has 400 episodes (40 tasks × 10 initial states), with delays
0/5/10/15/20/25, for 4,800 episodes per backbone. Evaluation uses 20 Hz,
replan25, paired task/episode seeds, and suite horizons 280/280/300/520.
PI0.5 uses native relative OSC; XVLA uses precomputed target deltas accumulated
from the switch pose. Queue underruns were zero in all conditions.

| Delay (steps) | PI0.5 baseline | PI0.5 Phase A | XVLA baseline | XVLA Phase A |
|---:|---:|---:|---:|---:|
| 0 | 374/400 (93.50%) | 380/400 (95.00%) | 390/400 (97.50%) | 385/400 (96.25%) |
| 5 | 354/400 (88.50%) | 372/400 (93.00%) | 214/400 (53.50%) | 304/400 (76.00%) |
| 10 | 332/400 (83.00%) | 341/400 (85.25%) | 181/400 (45.25%) | 296/400 (74.00%) |
| 15 | 303/400 (75.75%) | 316/400 (79.00%) | 184/400 (46.00%) | 274/400 (68.50%) |
| 20 | 282/400 (70.50%) | 300/400 (75.00%) | 185/400 (46.25%) | 285/400 (71.25%) |
| 25 | 251/400 (62.75%) | 294/400 (73.50%) | 185/400 (46.25%) | 302/400 (75.50%) |

### Per-suite success rates

Each suite has 100 episodes per method/delay. Rates are percentages.

| Model | Delay | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| PI0.5 baseline | 0 | 89% | 98% | 91% | 96% | 93.50% |
| PI0.5 Phase A | 0 | 90% | 99% | 94% | 97% | 95.00% |
| PI0.5 baseline | 5 | 87% | 96% | 92% | 79% | 88.50% |
| PI0.5 Phase A | 5 | 89% | 97% | 96% | 90% | 93.00% |
| PI0.5 baseline | 10 | 80% | 92% | 90% | 70% | 83.00% |
| PI0.5 Phase A | 10 | 81% | 93% | 92% | 75% | 85.25% |
| PI0.5 baseline | 15 | 72% | 85% | 88% | 58% | 75.75% |
| PI0.5 Phase A | 15 | 74% | 87% | 89% | 66% | 79.00% |
| PI0.5 baseline | 20 | 75% | 81% | 82% | 44% | 70.50% |
| PI0.5 Phase A | 20 | 70% | 86% | 88% | 56% | 75.00% |
| PI0.5 baseline | 25 | 66% | 74% | 78% | 33% | 62.75% |
| PI0.5 Phase A | 25 | 70% | 83% | 86% | 55% | 73.50% |
| XVLA baseline | 0 | 94% | 100% | 99% | 97% | 97.50% |
| XVLA Phase A | 0 | 93% | 100% | 98% | 94% | 96.25% |
| XVLA baseline | 5 | 35% | 71% | 49% | 59% | 53.50% |
| XVLA Phase A | 5 | 66% | 90% | 61% | 87% | 76.00% |
| XVLA baseline | 10 | 28% | 69% | 43% | 41% | 45.25% |
| XVLA Phase A | 10 | 61% | 83% | 74% | 78% | 74.00% |
| XVLA baseline | 15 | 19% | 75% | 49% | 41% | 46.00% |
| XVLA Phase A | 15 | 53% | 76% | 70% | 75% | 68.50% |
| XVLA baseline | 20 | 32% | 78% | 52% | 23% | 46.25% |
| XVLA Phase A | 20 | 53% | 81% | 77% | 74% | 71.25% |
| XVLA baseline | 25 | 42% | 69% | 61% | 13% | 46.25% |
| XVLA Phase A | 25 | 76% | 92% | 79% | 55% | 75.50% |

The baseline rows are official PI0.5 and corrected-data XVLA-50, respectively.
Within-backbone comparisons show the Phase A result against its corresponding
baseline; PI0.5 and XVLA action execution semantics differ, so cross-backbone
rates should not be read as a direct model ranking.

- PI0.5 Phase A checkpoint: `/data/yhr/output/online_filtering/ckpt/pi05_phase_a_clean_2gpu_b16_30k_20261104/checkpoints/030000/pretrained_model`.
- XVLA Phase A checkpoint: `/data/yhr/output/online_filtering/ckpt/xvla50_phase_a_dropoutfix_2gpu_b32_10k_20261101/checkpoints/030000/pretrained_model`.
- Results: `research/outputs/pi05_phase_a_full_r25_20261108/comparison.json` and `research/outputs/xvla_phase_a_full_r25_20261108/comparison.json`.
