"""Plot existing data only; never rerun estimators or simulators.

The original analysis script, its manifest and its figures remain untouched.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT/'analysis'/'closed_loop_summary'/'SUMMARY.json'
CASES = ROOT/'analysis'/'closed_loop_summary'/'cases.csv'
REPLAY = ROOT/'analysis'/'paired_replay'/'SUMMARY.json'
RUNS = ROOT/'analysis'/'paired_replay'/'runs.jsonl'
OUT = ROOT/'figures'
METHODS = ('ma', 'ma_structured', 'zhao2t3', 'zhao2t3_structured', 'exact_range')
LABELS = ('Ma', 'Ma + structure', 'Zhao2T3', 'Zhao2T3 + structure', 'Exact range')
COLORS = ('#3569a8', '#65a66b', '#bb7945', '#a0599c', '#555555')
SCENES = ('normal_v8', 'normal_v10', 'drop_v8', 'timeout_v8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    targets = [OUT/'comparison_v2.png', OUT/'comparison_v2.pdf', OUT/'COMPARISON_V2_VERIFY.json']
    if any(path.exists() for path in targets):
        raise FileExistsError('Preserve previous V2 plot/verification; no overwrite')
    summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    replay = json.loads(REPLAY.read_text(encoding='utf-8'))
    with CASES.open(encoding='utf-8', newline='') as file:
        cases = list(csv.DictReader(file))
    with RUNS.open(encoding='utf-8') as file:
        runs = [json.loads(line) for line in file if line.strip()]
    costs = {method: sum(r['measured_estimate_s'] for r in runs if r['method'] == method)
             for method in METHODS}
    # Three independent lightweight checks of the existing exported arithmetic.
    assert all(math.isclose(costs[m], replay['methods'][m]['estimate_total_s'], rel_tol=1e-12, abs_tol=1e-15)
               for m in METHODS), 'Recorded timing sums differ from original summary'
    assert all(math.isclose(costs[p['baseline']]/costs[p['structured']], p['total_cost_speed_ratio'], rel_tol=1e-12)
               and math.isclose(1-costs[p['structured']]/costs[p['baseline']], p['full_workload_cost_reduction_fraction'], rel_tol=1e-12)
               for p in replay['pairs']), 'Recorded workload comparison arithmetic differs'
    assert (len(cases) == summary['planned_cases'] == 20 and len(runs) == replay['all_runs'] == 90
            and all(sum(r['method'] == m for r in runs) == replay['methods'][m]['runs'] == 18
                    and sum(r['method'] == m and r['estimate']['status'] == 'UNKNOWN' for r in runs)
                    == replay['methods'][m]['unknown'] for m in METHODS)), 'Original denominator/outcome counts differ'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    lookup = {(row['scene_id'], row['method']): row for row in cases}
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.9))
    fig.subplots_adjust(left=.055, right=.985, bottom=.245, top=.76, wspace=.33)
    fig.suptitle('CARLA + CarSim: digital observation adapters with a shared braking backup',
                 x=.51, y=.97, fontsize=13)
    fig.legend([Patch(facecolor=color) for color in COLORS], LABELS,
               loc='upper center', bbox_to_anchor=(.52, .9), ncol=5,
               frameon=False, fontsize=10, columnspacing=1.8, handlelength=1.4)

    width = .145
    x = np.arange(len(SCENES))
    for index, method in enumerate(METHODS):
        values = [float(lookup[(scene, method)]['minimum_native_conservative_body_gap_m'])
                  for scene in SCENES]
        axes[0].bar(x+(index-2)*width, values, width, color=COLORS[index])
    axes[0].axhline(0., color='black', lw=.8)
    axes[0].set_xticks(x, ['Normal\n8 m/s', 'Normal\n10 m/s', 'Drop\n8 m/s', 'Late\n8 m/s'])
    axes[0].set_ylim(0., max(float(row['minimum_native_conservative_body_gap_m']) for row in cases)*1.15)
    axes[0].set_ylabel('Minimum conservative body gap (m)')
    axes[0].set_title('(a) All 20 closed-loop trajectories', pad=12)

    normal = ('normal_v8', 'normal_v10')
    xn = np.arange(len(normal))
    for index, method in enumerate(METHODS):
        values = [float(lookup[(scene, method)]['brake_applied_s']) for scene in normal]
        axes[1].bar(xn+(index-2)*width, values, width, color=COLORS[index])
    axes[1].set_xticks(xn, ['Normal\n8 m/s', 'Normal\n10 m/s'])
    axes[1].set_ylim(0., max(float(lookup[(scene, method)]['brake_applied_s'])
                           for scene in normal for method in METHODS)*1.2)
    axes[1].set_ylabel('Emergency brake application time (s)')
    axes[1].set_title('(b) Brake timing in normal scenarios', pad=12)

    millisecond_costs = [costs[method]*1000 for method in METHODS]
    axes[2].bar(np.arange(5), millisecond_costs, color=COLORS)
    axes[2].set_yscale('log')
    axes[2].set_ylim(1., 8000.)
    axes[2].set_yticks([1., 10., 100., 1000.], ['1', '10', '100', '1000'])
    axes[2].set_xticks(np.arange(5), ['Ma', 'Ma +\nstructure', 'Zhao2T3', 'Zhao2T3 +\nstructure', 'Exact\nrange'])
    axes[2].set_ylabel('Total estimate cost (ms / 18 calls; log scale)')
    axes[2].set_title('(c) Offline identical-query replay', pad=12)
    for index, method in enumerate(METHODS):
        record = replay['methods'][method]
        text = f"{millisecond_costs[index]:.2f} ms\nOK {record['ok']}; U {record['unknown']}"
        axes[2].annotate(text, (index, millisecond_costs[index]), xytext=(0, 5),
                         textcoords='offset points', ha='center', va='bottom', fontsize=7.5)
    for axis in axes:
        axis.grid(axis='y', alpha=.20)
        axis.set_axisbelow(True)

    fig.text(.055, .135,
             'Replay: 6 fixed queries x 3 repetitions per method; all timed-out and UNKNOWN work is included. U = UNKNOWN.',
             ha='left', va='center', fontsize=9)
    fig.text(.055, .095,
             'Zhao2T3 completed 0/18 calls; its cost ratio measures the full deadline-limited workload, not an equal-completion speedup.',
             ha='left', va='center', fontsize=9)
    fig.text(.055, .055,
             'Safety: 20/20 sampled stopline checks passed. These results establish neither a continuous-domain guarantee nor wall-clock real-time operation.',
             ha='left', va='center', fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[0], dpi=240)
    fig.savefig(targets[1])
    plt.close(fig)
    verification = dict(status='PASS_EXISTING_DATA_V2_PLOT_ARITHMETIC', checks=3,
        method_reruns=0, simulator_runs=0, original_files_modified=False,
        replay_queries=6, calls_per_method=18, retained_total_calls=90,
        costs_ms={method: costs[method]*1000 for method in METHODS},
        source_sha256={str(path): sha(path) for path in (SUMMARY, CASES, REPLAY, RUNS, Path(__file__))},
        figure_sha256={str(path): sha(path) for path in targets[:2]})
    targets[2].write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=verification['status'], png=str(targets[0]), pdf=str(targets[1]))))


if __name__ == '__main__':
    main()
