"""Post-execution summaries and preregistered identical-query replay.

This program never starts CARLA or CarSim and never alters frozen sources.
--replay and --analyze write distinct, exclusively created output directories.
Both require the whole planned stage to finish and hold the simulation lock, so
postprocessing cannot compete with an active measured simulation controller.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'
RESULTS = ROOT/'cosim_results'
ANALYSIS = ROOT/'analysis'
BASE = Path(r'E:\codex_project\CarlaCarSimValidation')
BACKUP = ROOT.parent/'vehicle_position_contract_20261013'/'source'/'position_backup.py'
METHOD_LABELS = {'ma': 'Ma', 'ma_structured': 'Ma + structure',
                 'zhao2t3': 'Zhao2T3', 'zhao2t3_structured': 'Zhao2T3 + structure',
                 'exact_range': 'Exact range'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                     allow_nan=False)+'\n', encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def write_csv(path, records):
    with Path(path).open('x', encoding='utf-8', newline='') as file:
        if not records:
            return
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def verified_inputs():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source drift '+path)
    execution = json.loads((RESULTS/'EXECUTION.json').read_text(encoding='utf-8'))
    if execution['planned'] != len(plan['cases']) or execution['executed'] != len(plan['cases']):
        raise RuntimeError('Postprocessing requires the complete planned denominator')
    by_id = {record['id']: record for record in execution['records']}
    if len(by_id) != len(plan['cases']) or set(by_id) != {case['id'] for case in plan['cases']}:
        raise RuntimeError('Execution identities differ from frozen case denominator')
    for case in plan['cases']:
        out = RESULTS/case['id']
        result = json.loads((out/'result.json').read_text(encoding='utf-8'))
        if result['plan_sha256'] != sha(PLAN) or result != by_id[case['id']]:
            raise RuntimeError('Final result identity differs '+case['id'])
        for name in ('native', 'policy', 'sensor', 'timing'):
            if sha(out/(name+'.csv')) != result[name+'_sha256']:
                raise RuntimeError('Retained log drift '+case['id']+'/'+name)
        if sha(out/'decision.jsonl') != result['decision_sha256']:
            raise RuntimeError('Retained decision drift '+case['id'])
    return plan, execution


def select_replay_queries(plan):
    registration = plan['paired_replay']
    if registration['source_methods'] != ['exact_range'] or registration['repetitions'] != 3:
        raise RuntimeError('Replay registration has a different source/repetition rule')
    queries, sources = [], []
    for scene_id in registration['source_scenes']:
        matches = [case for case in plan['cases']
                   if case['scene_id'] == scene_id and case['method'] == 'exact_range']
        if len(matches) != 1:
            raise RuntimeError('Replay source case not unique '+scene_id)
        case = matches[0]
        path = RESULTS/case['id']/'decision.jsonl'
        all_records = read_jsonl(path)
        eligible = sorted([record for record in all_records if record['estimate'] is not None],
                          key=lambda record: record['time'])
        selected = eligible[:3]
        sources.append(dict(scene_id=scene_id, case_id=case['id'], decision_path=str(path),
                            decision_sha256=sha(path), eligible_decisions=len(eligible),
                            selected_decisions=len(selected), selection='first at most three in increasing time'))
        source_result = json.loads((RESULTS/case['id']/'result.json').read_text(encoding='utf-8'))
        for index, record in enumerate(selected):
            queries.append(dict(query_id=f'{scene_id}__q{index:03d}', scene_id=scene_id,
                                source_case_id=case['id'], simulated_time_s=record['time'],
                                selected_rows=record['selected_rows'], current_packet=record['packet'],
                                prior=record['prior'], source_estimate_status=record['estimate']['status'],
                                invariant_body_radius_m=source_result['invariant_body_radius_m'],
                                stopline_s_m=source_result['stopline_s_m']))
    return queries, sources


def run_replay(plan):
    out = ANALYSIS/'paired_replay'
    out.mkdir(parents=True, exist_ok=False)
    estimators = load('offline_replay_frozen_estimators', ROOT/'source'/'sensor_estimators.py')
    backup = load('offline_replay_frozen_backup', BACKUP)
    methods = tuple(plan['methods'])
    if methods != tuple(estimators.METHODS):
        raise RuntimeError('Replay method inventory differs')
    queries, sources = select_replay_queries(plan)
    save(out/'QUERIES.json', dict(status='SELECTED_BY_FROZEN_RULE', plan_sha256=sha(PLAN),
                                 registration=plan['paired_replay'], sources=sources, queries=queries))
    execution = dict(status='STARTED', started_utc=utc(), plan_sha256=sha(PLAN),
                     analysis_source_sha256=sha(Path(__file__)), queries=len(queries),
                     planned_runs=len(queries)*len(methods)*plan['paired_replay']['repetitions'],
                     completed_runs=0, original_closed_loop_trajectories=False,
                     populations_or_confidence_inference=False)
    save(out/'EXECUTION.json', execution)
    records = []
    log = (out/'runs.jsonl').open('x', encoding='utf-8')
    try:
        for query in queries:
          for repetition in range(plan['paired_replay']['repetitions']):
            shift = repetition % len(methods)
            order = methods[shift:]+methods[:shift]
            for order_index, method in enumerate(order):
                prep_start = time.perf_counter()
                # Preparation is outside the frozen pure-estimate deadline.
                rows = json.loads(json.dumps(query['selected_rows']))
                packet = json.loads(json.dumps(query['current_packet']))
                prior = list(query['prior'])
                prepared = time.perf_counter()
                start = time.perf_counter()
                deadline = start+plan['paired_replay']['deadline_s']
                error = None
                try:
                    estimate = estimators.estimate(method, rows, prior, deadline)
                except Exception as exc:
                    estimate = dict(status='ERROR', b_interval=None, elapsed_s=time.perf_counter()-start,
                                    frontend_calls=0, cells_completed=0, deleted=0, error=repr(exc))
                    error = traceback.format_exc()
                finished = time.perf_counter()
                measured = finished-start
                guard_start = time.perf_counter()
                bounds = None
                margin = None
                guard_status = 'NO_OK_ESTIMATE'
                if estimate['status'] == 'OK' and estimate.get('b_interval') is not None:
                    try:
                        bounds = estimators.guard_bounds(packet, estimate['b_interval'])
                        if bounds.get('valid', False):
                            required = backup.next_period_required(
                                bounds['v_upper'], plan['decision_period_s'],
                                measurement_pad=plan['measurement_pad_m'],
                                deadline_delay=plan['response_delay_s'])+query['invariant_body_radius_m']
                            margin = query['stopline_s_m']-bounds['s_upper']-required
                            guard_status = 'ACCEPT_NORMAL' if margin >= 0 else 'BACKUP_MARGIN'
                        else:
                            guard_status = 'INVALID_STATE_BOUNDS'
                    except Exception as exc:
                        guard_status = 'GUARD_ERROR'
                        error = (error or '')+traceback.format_exc()
                guard_finished = time.perf_counter()
                record = dict(query_id=query['query_id'], scene_id=query['scene_id'],
                              simulated_time_s=query['simulated_time_s'], repetition=repetition,
                              order_index=order_index, method=method, estimate=estimate,
                              preparation_s=prepared-prep_start, measured_estimate_s=measured,
                              reported_estimate_s=estimate.get('elapsed_s'),
                              estimate_deadline_s=plan['paired_replay']['deadline_s'],
                              measured_estimate_late=measured > plan['paired_replay']['deadline_s'],
                              guard_s=guard_finished-guard_start, guard_status=guard_status,
                              bounds=bounds, guard_margin_m=margin,
                              estimate_plus_guard_s=measured+guard_finished-guard_start,
                              estimate_started_monotonic_s=start, estimate_finished_monotonic_s=finished,
                              error=error)
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()
                records.append(record)
                execution['completed_runs'] = len(records)
                save(out/'EXECUTION.json', execution)
        execution.update(status='COMPLETED_ALL_PREREGISTERED_REPLAY_RUNS', finished_utc=utc(),
                         runs_sha256=sha(out/'runs.jsonl'), query_source_sha256=sha(out/'QUERIES.json'))
        save(out/'EXECUTION.json', execution)
        summary = replay_summary(records, methods, len(queries))
        save(out/'SUMMARY.json', summary)
        write_csv(out/'runs_summary.csv', [dict(query_id=r['query_id'], scene_id=r['scene_id'],
                  time_s=r['simulated_time_s'], repetition=r['repetition'], order_index=r['order_index'],
                  method=r['method'], status=r['estimate']['status'], measured_estimate_s=r['measured_estimate_s'],
                  guard_s=r['guard_s'], preparation_s=r['preparation_s'],
                  late=r['measured_estimate_late'], cells_completed=r['estimate'].get('cells_completed', 0),
                  frontend_calls=r['estimate'].get('frontend_calls', 0), deleted=r['estimate'].get('deleted', 0))
                  for r in records])
        print(json.dumps(dict(status=execution['status'], queries=len(queries), runs=len(records))), flush=True)
    except Exception as exc:
        execution.update(status='FAILED_RETAIN_PARTIAL_REPLAY_NO_RETRY', error=repr(exc),
                         traceback=traceback.format_exc(), finished_utc=utc())
        save(out/'EXECUTION.json', execution)
        raise
    finally:
        log.close()


def replay_summary(records, methods, query_count):
    by_method = {}
    for method in methods:
        runs = [r for r in records if r['method'] == method]
        costs = [r['measured_estimate_s'] for r in runs]
        by_method[method] = dict(runs=len(runs), estimate_total_s=sum(costs),
            estimate_median_s=statistics.median(costs) if costs else None,
            guard_total_s=sum(r['guard_s'] for r in runs),
            preparation_total_s=sum(r['preparation_s'] for r in runs),
            ok=sum(r['estimate']['status'] == 'OK' for r in runs),
            unknown=sum(r['estimate']['status'] == 'UNKNOWN' for r in runs),
            errors=sum(r['estimate']['status'] == 'ERROR' for r in runs),
            measured_late=sum(r['measured_estimate_late'] for r in runs),
            frontend_calls=sum(r['estimate'].get('frontend_calls', 0) for r in runs),
            deleted=sum(r['estimate'].get('deleted', 0) for r in runs))
    pairs = []
    lookup = {(r['query_id'], r['repetition'], r['method']): r for r in records}
    for baseline, structured in (('ma', 'ma_structured'), ('zhao2t3', 'zhao2t3_structured')):
        values, qualified = [], []
        for key, left in lookup.items():
            if key[2] != baseline:
                continue
            right = lookup.get((key[0], key[1], structured))
            if right is None:
                continue
            ratio = left['measured_estimate_s']/right['measured_estimate_s']
            values.append(ratio)
            if left['estimate']['status'] == right['estimate']['status'] == 'OK' and not (
                    left['measured_estimate_late'] or right['measured_estimate_late']):
                qualified.append(ratio)
        lhs, rhs = by_method[baseline]['estimate_total_s'], by_method[structured]['estimate_total_s']
        pairs.append(dict(baseline=baseline, structured=structured, complete_pair_runs=len(values),
                          total_cost_speed_ratio=lhs/rhs if rhs else None,
                          full_workload_cost_reduction_fraction=1-rhs/lhs if lhs else None,
                          median_all_pair_speed_ratio=statistics.median(values) if values else None,
                          structured_slower_pairs=sum(v < 1 for v in values),
                          qualified_pairs=len(qualified),
                          median_qualified_speed_ratio=statistics.median(qualified) if qualified else None))
    return dict(status='DESCRIPTIVE_IDENTICAL_QUERY_REPLAY', queries=query_count,
                all_runs=len(records), methods=by_method, pairs=pairs,
                timing_role='Offline identical rows/current packet; not another closed-loop safety experiment',
                no_population_significance_claim=True, timed_out_work_in_totals=True)


def summarize_case(case, record):
    out = RESULTS/case['id']
    native_path = out/'native.csv'
    first = last = None
    native_min_gap = math.inf
    rows = 0
    with native_path.open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            x = float(row['x'])
            if first is None:
                first = x
            last = x
            rows += 1
            if 'stopline_s_m' in record:
                gap = record['stopline_s_m']-record['calibrated_initial_s_m']-(x-first)-record['invariant_body_radius_m']
                native_min_gap = min(native_min_gap, gap)
    decisions = read_jsonl(out/'decision.jsonl')
    estimates = [r['estimate'] for r in decisions if r['estimate'] is not None]
    active = [r for r in decisions if r['status'] != 'BACKUP_ALREADY_LATCHED']
    timing_rows = []
    with (out/'timing.csv').open(encoding='utf-8', newline='') as file:
        timing_rows = list(csv.DictReader(file))
    return dict(scene_id=case['scene_id'], case_id=case['id'], method=case['method'],
                status=record['status'], completed=record['status'] != 'FAILED',
                brake_applied_s=record.get('emergency_applied_s'),
                minimum_world_clearance_m=record.get('minimum_world_clearance_m'),
                minimum_native_conservative_body_gap_m=native_min_gap if math.isfinite(native_min_gap) else None,
                sampled_safe=record['status'] == 'SAMPLED_NO_STOPLINE_CROSSING' and native_min_gap >= -1e-8,
                progress_m=last-first if first is not None else None, native_rows=rows,
                active_decision_count=len(active), quick_prior_accept_count=sum(r['status'] == 'QUICK_PRIOR_ACCEPT' for r in decisions),
                estimator_calls=len(estimates), unknown_count=sum(r['status'] == 'UNKNOWN' for r in estimates),
                estimator_error_count=sum(r['status'] == 'ERROR' for r in estimates),
                timeout_count=sum('TIMEOUT' in r['status'] for r in decisions),
                frontend_calls=sum(r.get('frontend_calls', 0) for r in estimates),
                deleted=sum(r.get('deleted', 0) for r in estimates),
                cells_completed=sum(r.get('cells_completed', 0) for r in estimates),
                complete_guard_measured_total_s=sum(r['measured_compute_s'] for r in active),
                estimator_reported_total_s=sum(r.get('elapsed_s', 0.) for r in estimates),
                injected_time_report_excess_s=sum(max(0., r['reported_compute_s']-r['measured_compute_s']) for r in active),
                full_loop_wall_total_s=sum(float(r['full_cycle_wall_s']) for r in timing_rows),
                full_loop_wall_max_s=max((float(r['full_cycle_wall_s']) for r in timing_rows), default=None),
                full_loop_wall_over10ms_count=sum(float(r['full_cycle_wall_s']) > .01 for r in timing_rows),
                wall_stage_s=record['wall_s'], sampled_backup_membership=record.get('sampled_backup_membership'),
                final_stopped_tail_s=record.get('stopped_final_tail_s'),
                audit_true_bias_exclusion_count=record.get('audit_true_bias_exclusion_count'),
                error=record.get('error', ''))


def render_figure(plan, summaries, replay, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    methods = list(plan['methods'])
    scenes = list(dict.fromkeys(case['scene_id'] for case in plan['cases']))
    colors = ['#3569a8', '#65a66b', '#bb7945', '#a0599c', '#555555']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    lookup = {(r['scene_id'], r['method']): r for r in summaries}
    width = .14
    x = np.arange(len(scenes))
    for index, method in enumerate(methods):
        gaps = [lookup[(scene, method)]['minimum_native_conservative_body_gap_m'] for scene in scenes]
        values = [float(v) if v is not None else np.nan for v in gaps]
        axes[0].bar(x+(index-2)*width, values, width, color=colors[index], label=METHOD_LABELS[method])
        for scene_index, value in enumerate(values):
            if not math.isfinite(value):
                axes[0].text(scene_index+(index-2)*width, 0., 'F', color='red', ha='center')
    axes[0].axhline(0., color='black', lw=.8)
    axes[0].set_xticks(x, [s.replace('_', '\n') for s in scenes])
    axes[0].set_ylabel('Minimum conservative body gap (m)')
    axes[0].set_title('(a) All planned closed-loop cases')
    axes[0].legend(fontsize=7, loc='upper right')
    normal_scenes = list(plan['paired_replay']['source_scenes'])
    xnormal = np.arange(len(normal_scenes))
    for index, method in enumerate(methods):
        values = [lookup[(scene, method)]['brake_applied_s'] for scene in normal_scenes]
        axes[1].bar(xnormal+(index-2)*width, [v if v is not None else np.nan for v in values],
                    width, color=colors[index])
    axes[1].set_xticks(xnormal, [s.replace('_', '\n') for s in normal_scenes])
    axes[1].set_ylabel('Emergency brake application time (s)')
    axes[1].set_title('(b) Normal scenarios')
    if replay is None:
        axes[2].text(.5, .5, 'Identical-query replay not available', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_xticks([])
    else:
        costs = [replay['methods'][method]['estimate_total_s']*1000 for method in methods]
        axes[2].bar(np.arange(len(methods)), costs, color=colors)
        axes[2].set_xticks(np.arange(len(methods)), ['Ma', 'Ma +\nstructure', 'Zhao2T3', 'Zhao2T3 +\nstructure', 'Exact\nrange'])
        axes[2].set_ylabel('Total estimate call cost (ms)')
        for i, value in enumerate(costs):
            unknown = replay['methods'][methods[i]]['unknown']
            axes[2].annotate(f'{value:.1f}\nU={unknown}', (i, value), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)
        axes[2].set_ylim(0., max(costs)*1.22 if costs else 1.)
    axes[2].set_title('(c) Identical-query offline timing')
    fig.suptitle('CARLA + CarSim: digital observation adapters and shared braking backup\n'
                 'Sampled safety evidence; no continuous-domain or wall-clock real-time guarantee', fontsize=11)
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.savefig(out/'closed_loop_and_replay.png', dpi=220)
    fig.savefig(out/'closed_loop_and_replay.pdf')
    plt.close(fig)


def run_analysis(plan, execution):
    out = ANALYSIS/'closed_loop_summary'
    out.mkdir(parents=True, exist_ok=False)
    by_id = {record['id']: record for record in execution['records']}
    summaries = [summarize_case(case, by_id[case['id']]) for case in plan['cases']]
    replay_path = ANALYSIS/'paired_replay'/'SUMMARY.json'
    replay = json.loads(replay_path.read_text(encoding='utf-8')) if replay_path.exists() else None
    save(out/'SUMMARY.json', dict(status='COMPLETE_DENOMINATOR_DESCRIPTIVE_CLOSED_LOOP_SUMMARY',
        plan_sha256=sha(PLAN), analysis_source_sha256=sha(Path(__file__)), created_utc=utc(),
        planned_cases=len(plan['cases']), represented_cases=len(summaries),
        sampled_safe_cases=sum(r['sampled_safe'] for r in summaries),
        failed_cases=sum(r['status'] == 'FAILED' for r in summaries),
        cases=summaries, paired_replay_summary=replay,
        cost_interpretation='Closed-loop guard totals are different controller workloads, not paired speedup. Only separately preregistered identical-query replay supports descriptive paired costs.',
        safety_scope='CARLA ideal digital lane ranges and native CarSim dynamics, shared one-sided backup; finite sampled native body gap only',
        no_population_significance_claim=True))
    write_csv(out/'cases.csv', summaries)
    render_figure(plan, summaries, replay, out)
    manifest = {path.name: sha(path) for path in out.iterdir() if path.is_file()}
    save(out/'MANIFEST.json', dict(status='ANALYSIS_OUTPUT_FILES', sha256=manifest))
    print(json.dumps(dict(status='ANALYZED_ALL_PLANNED_CASES', planned=len(summaries),
                         sampled_safe=sum(r['sampled_safe'] for r in summaries),
                         failed=sum(r['status'] == 'FAILED' for r in summaries))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--replay', action='store_true')
    group.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(BASE))
    from validate import exclusive_lock, check_no_server
    import psutil
    with exclusive_lock(BASE):
        check_no_server(psutil, [2100, 2101, 2102])
        plan, execution = verified_inputs()
        if args.replay:
            run_replay(plan)
        else:
            run_analysis(plan, execution)


if __name__ == '__main__':
    main()
