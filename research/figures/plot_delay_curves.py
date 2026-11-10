"""Plot recorded LIBERO results; missing PI05 measurements are not imputed."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
raw_xvla = json.loads((ROOT / 'research/outputs/xvla_accumulated_full_r25_20261103/comparison.json').read_text())
raw_pi05 = json.loads((ROOT / 'research/outputs/pi05_full_r25_20261107/comparison.json').read_text())
delays = [0, 5, 10, 15, 20]
curves = [
    ('PI0.5 baseline (R25)', delays, [100 * raw_pi05[f'baseline_d{d}']['success_rate'] for d in delays], '#3478b8', '--', 's'),
    ('PI0.5 + Online Filtering (R25)', delays, [100 * raw_pi05[f'online_d{d}']['success_rate'] for d in delays], '#3478b8', '-', 'o'),
    ('XVLA baseline (R25)', delays, [100 * raw_xvla[f'baseline_d{d}']['success_rate'] for d in delays], '#df7d29', '--', 's'),
    ('XVLA + Online Filtering (R25)', delays, [100 * raw_xvla[f'online_d{d}']['success_rate'] for d in delays], '#df7d29', '-', 'o'),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(8.2, 5.6))
for name, x, y, color, style, marker in curves:
    ax.plot(x, y, label=name, color=color, linestyle=style, marker=marker, linewidth=2.2, markersize=6)
ax.set(xlim=(-.5, 20.5), ylim=(35, 102), xlabel='Inference delay (control steps; 20 Hz)', ylabel='LIBERO success rate (%)', title='Online Filtering under inference delay')
ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks(range(40, 101, 10))
ax.grid(axis='both', alpha=.18);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower left', frameon=False, fontsize=10)
fig.text(.12, .045, '400 episodes per point. Both evaluations use replan 25. XVLA uses accumulated-delta execution.\nPI0.5 uses native relative OSC; PI0.5 OF is the clean Phase 2 checkpoint.', fontsize=9, color='#555555', va='bottom')
fig.subplots_adjust(left=.12, right=.97, top=.90, bottom=.23)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT / f'libero_delay_0_20.{ext}', dpi=240)
with (OUT / 'libero_delay_0_20.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['method','delay_steps','success_rate_percent','episodes'])
    for name,x,y,*_ in curves:
        for delay,rate in zip(x,y,strict=True): w.writerow([name,delay,rate,400])
print(OUT / 'libero_delay_0_20.png')
