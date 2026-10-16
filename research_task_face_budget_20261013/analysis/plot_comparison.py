"""Static scientific figure from all five columns and frozen paired statistics."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'analysis'/'SUMMARY_STATISTICS.json'
data = json.loads(path.read_text(encoding='utf-8'))
methods = ['ma_rank1', 'ma_rank1_structured', 'zhao1_rank1',
           'zhao2t3_rank1', 'zhao2t3_rank1_structured']
labels = ['Ma\nrank1', 'Ma\n+prune', 'Zhao\norder1', 'Zhao\norder2+T3', 'Zhao 2+T3\n+prune']
rows = {r['method']:r for r in data['method_groups'] if r['group']=='all'}
colors = ['#707b87', '#276ead', '#919ba6', '#707b87', '#276ead']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'pdf.fonttype':42, 'ps.fonttype':42})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={'width_ratios':[1.1, 1.1, 1.7]})
x = np.arange(5)
for ax, field, title, ylabel in (
    (axes[0], 'successful_cases', '(a) Stable resolution within budget', 'Resolved cases / 32'),
    (axes[1], 'total_observed_elapsed_seconds', '(b) Cost of all scheduled runs', 'Total elapsed time (s) / 96 runs')):
    values = [rows[m][field] for m in methods]
    ax.bar(x, values, color=colors, width=.72)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.16)
    ax.set_ylim(0, 34 if field=='successful_cases' else max(values)*1.17)
    for j, v in enumerate(values):
        ax.text(j, v + (.7 if field=='successful_cases' else 2), str(v) if field=='successful_cases' else f'{v:.1f}',
                ha='center', va='bottom', fontsize=9)

ax=axes[2]
forest=[]
for baseline, prefix in [('ma_rank1','Ma'), ('zhao2t3_rank1','Zhao 2 + T3')]:
    for group, short in [('all','all'), ('population=ordinary','ordinary'), ('population=structural','structural')]:
        row=next(r for r in data['paired_groups'] if r['baseline']==baseline and r['group']==group)
        forest.append((f"{prefix} / {short} (n={row['eligible_cases']})", row))
for index, (label,row) in enumerate(forest):
    value, low, high = (row[k] for k in ('paired_log_median_speedup','ci95_low','ci95_high'))
    ax.errorbar(value, index, xerr=[[value-low],[high-value]], fmt='o', capsize=3,
                color='#276ead' if row['baseline'].startswith('zhao') else '#707b87', markersize=5)
ax.set_yticks(range(len(forest)), [r[0] for r in forest])
ax.invert_yaxis()
ax.set_xscale('log')
ax.axvline(1, color='#ad413b', linestyle='--', linewidth=1)
ax.set_xlim(.55, 7)
ax.set_xticks([.75, 1, 2, 4, 6], ['0.75', '1', '2', '4', '6'])
ax.set_xlabel('Baseline / pruning median time ratio (> 1 is faster)')
ax.set_title('(c) Same-conclusion pairs, excluding common quick', loc='left', fontsize=11)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', alpha=.16)
fig.suptitle('Shared-response continuous-position certification', fontsize=14)
fig.text(.5, .044, '32 cases from 16 paired templates; 3 repeats; 5 s per run. Error bars: 95% template-cluster bootstrap intervals.',
         ha='center', fontsize=9)
fig.text(.5, .015, 'Structural-group intervals condition on nonempty bootstrap samples (35 / 10,000 empty). Total cost includes UNKNOWN.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .10, 1, .92))
out=ROOT/'figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'comparison_overview.png', dpi=180)
fig.savefig(out/'comparison_overview.pdf')
(out/'FIGURE_SOURCE.json').write_text(json.dumps(dict(
    statistics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    all_methods_plotted=True, frozen_group_statistics=True,
    excluded_common_quick_from_pairs=True, cost_includes_unknown=True,
    structural_ci_conditioned_on_nonempty_bootstrap=True,
    physical_validation_claimed=False), indent=2), encoding='utf-8')
print(str(out/'comparison_overview.png'))
