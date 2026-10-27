# Metrics

## Dynamic delay 1–10, replan interval 20, 20 Hz

| Model | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Phase A, 30k | 2000 | 88.0% | 95.6% | 94.8% | 88.4% | 91.70% |
| Phase 2, mixed 30k | 2000 | 84.8% | 95.4% | 95.8% | 83.2% | 89.80% |
| Phase 2, no-innovation annealed 30k | 2000 | 89.4% | 95.8% | 94.2% | 88.4% | 91.95% |
| Phase 2, no-innovation annealed 60k | 2000 | 88.4% | 96.4% | 95.4% | 88.0% | 92.05% |
| X-VLA, PI0.5-matched horizon | 2000 | 94.0% | 98.4% | 97.2% | 95.0% | 96.15% |
| π0-FAST | 2000 | 61.2% | 54.4% | 66.8% | 47.2% | 57.40% |

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
| X-VLA | 15 | 400 | 87.0% | 97.0% | 94.0% | 92.0% | 92.50% | 18,281 |
| X-VLA | 20 | 400 | 86.0% | 97.0% | 76.0% | 82.0% | 85.25% | 48,181 |
