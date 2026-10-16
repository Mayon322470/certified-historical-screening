"""Frozen 480-job serial cold-cache comparison. Only public solver input is read.

No CLI overrides of methods, grids, budget or sampling are accepted. Schema,
profile, frontends, pruning, bridge, witness and certificate construction are
inside the timed region. Imports, cache clearing and JSON disk I/O are outside.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from collections import Counter

for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
sys.set_int_max_str_digits(0)
import certified_core as cc
from budget_comparison import METHODS, certify, quick_certificate
from quick_baseline import encode
from threadpoolctl import threadpool_limits, threadpool_info
import numpy as np
import scipy
_thread_limit = threadpool_limits(limits=1)

ROOT = Path(__file__).resolve().parents[1]
GRIDS = (8, 32, 128, 512)
TIME_GRIDS = (32, 64, 128, 256)
BUDGET = 5.0
REPEATS = 3
ORDER_SEED = 712913
PUBLIC_FIELDS = {'name', 'initial', 'past', 'prior', 'packet', 'future', 'H',
                 'rho', 'limit', 'response'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_sources():
    freeze_path = ROOT/'protocol'/'METHODS_FROZEN.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    assert freeze['status'] == 'METHODS_FROZEN'
    for relative, expected in freeze['source_sha256'].items():
        assert digest(ROOT/relative) == expected, ('Frozen source drift', relative)
    assert freeze['configuration'] == dict(methods=list(METHODS), grids=list(GRIDS),
        time_grids=list(TIME_GRIDS), budget_seconds=BUDGET, repeats=REPEATS,
        order_seed=ORDER_SEED, interval_bits=cc.BITS, numerical_threads=1,
        derivative_subdivisions=1, quick_gate=True, collect_every_repeat=True)
    return freeze_path


def run(case, method):
    cc.exp_neg.cache_clear()
    cc.EXP_CERTIFICATES.clear()
    start = time.perf_counter()
    deadline = start+BUDGET
    gate = quick_certificate(case)
    stages = []
    if gate['status'] == 'QUICK_CERTIFIED_SAFE':
        result = dict(method=method, status='CERTIFIED_SAFE', quick_resolved=True,
                      upper=gate['quick_upper'], nonempty_witness=gate['nonempty_witness'],
                      historical_vertices=1, future_vertices=0, retained=0, deleted=0,
                      frontend_calls=0, completed_cells=0, grid_complete=True, budget_hit=False)
    else:
        result = None
        for n, nt in zip(GRIDS, TIME_GRIDS):
            result = certify(case, n, nt, method, collect=True, deadline=deadline)
            stages.append(result)
            if result['status'] in ('CERTIFIED_UNSAFE', 'INCONSISTENT') or (
                    result['status'] == 'CERTIFIED_SAFE' and result['nonempty_witness'] is not None):
                break
            if time.perf_counter() >= deadline:
                break
    elapsed = time.perf_counter()-start
    resolved = result['status'] in ('CERTIFIED_UNSAFE', 'INCONSISTENT') or (
        result['status'] == 'CERTIFIED_SAFE' and result.get('nonempty_witness') is not None)
    return dict(result=result, stages=stages, quick=gate, elapsed=elapsed,
                resolved=resolved, resolved_within_budget=resolved and elapsed <= BUDGET,
                budget_seconds=BUDGET, overshoot_seconds=max(0.0, elapsed-BUDGET),
                future_vertices_total=sum(s['future_vertices'] for s in stages),
                frontend_calls_total=sum(s['frontend_calls'] for s in stages),
                exponential_cache_misses=cc.exp_neg.cache_info().misses,
                unknown_reason=('CELL_BOUNDARY_DEADLINE' if elapsed >= BUDGET else
                                'FIXED_SCHEDULE_EXHAUSTED') if result['status'] == 'UNKNOWN' else None)


def main():
    freeze_path = verify_frozen_sources()
    input_path = ROOT/'protocol'/'frozen_suite'/'public_cases_32.json'
    generation_path = input_path.parent/'GENERATION_MANIFEST.json'
    generation = json.loads(generation_path.read_text(encoding='utf-8'))
    assert digest(freeze_path) == generation['method_freeze_sha256']
    assert digest(input_path) == generation['output_hashes'][input_path.name]
    # This file does not read private witnesses, groups, thresholds or seeds.
    cases = json.loads(input_path.read_text(encoding='utf-8'))
    assert len(cases) == 32 and all(set(c) == PUBLIC_FIELDS for c in cases)
    assert len(set(c['name'] for c in cases)) == 32
    output = ROOT/'results'/'formal_main'
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(status='RUNNING', input_sha256=digest(input_path),
                method_freeze_sha256=digest(freeze_path),
                generation_manifest_sha256=digest(generation_path),
                python=platform.python_version(), executable=sys.executable,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), numerical_threads=1, threadpools=threadpool_info(),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                configuration=dict(methods=list(METHODS), grids=list(GRIDS),
                    time_grids=list(TIME_GRIDS), budget_seconds=BUDGET, repeats=REPEATS,
                    order_seed=ORDER_SEED, interval_bits=cc.BITS),
                expected_records=480, hidden_metadata_read=False,
                timing_scope='Parse/quick/all stages/preparation/full certificates included; import/cache clear/disk serialization excluded',
                implementation_sha256={f.name:digest(f) for f in sorted((ROOT/'implementation').glob('*.py'))})
    (output/'run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    jobs = [(rep, i, method) for rep in range(REPEATS) for i in range(32) for method in METHODS]
    random.Random(ORDER_SEED).shuffle(jobs)
    summary = {method:dict(statuses=Counter(), elapsed_total=0.0, within_budget=0,
                           late_certificates=0, errors=0) for method in METHODS}
    seen = set()
    with (output/'records.jsonl').open('x', encoding='utf-8') as stream:
        for sequence, (rep, i, method) in enumerate(jobs):
            row = dict(sequence=sequence, case_index=i, name=cases[i]['name'], rep=rep,
                       method=method, input_sha256=digest(input_path),
                       method_freeze_sha256=digest(freeze_path))
            error_start = time.perf_counter()
            try:
                row.update(run(cases[i], method))
            except Exception:
                elapsed = time.perf_counter()-error_start
                row.update(result=dict(method=method, status='ERROR'), stages=[], quick=None,
                           elapsed=elapsed, resolved=False, resolved_within_budget=False,
                           budget_seconds=BUDGET, overshoot_seconds=max(0.0, elapsed-BUDGET),
                           future_vertices_total=0, frontend_calls_total=0,
                           error_traceback=traceback.format_exc(), unknown_reason=None)
            stream.write(json.dumps(encode(row), separators=(',', ':'))+'\n')
            stream.flush()
            seen.add((rep, i, method))
            item = summary[method]
            item['statuses'][row['result']['status']] += 1
            item['elapsed_total'] += row['elapsed']
            item['within_budget'] += row['resolved_within_budget']
            item['late_certificates'] += row['resolved'] and not row['resolved_within_budget']
            item['errors'] += row['result']['status'] == 'ERROR'
            print(f"{sequence+1}/480 rep={rep} {row['name']} {method} "
                  f"{row['result']['status']} {row['elapsed']:.4f}s within={row['resolved_within_budget']}", flush=True)
    assert seen == set(jobs) and len(seen) == 480
    (output/'summary_raw.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    meta.update(status='COMPLETED_ALL_480', records=480,
                records_sha256=digest(output/'records.jsonl'), completed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    (output/'RUN_COMPLETED.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
