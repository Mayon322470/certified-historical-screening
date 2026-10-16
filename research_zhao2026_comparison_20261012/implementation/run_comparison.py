"""Paired cold-expression-cache runs. Timing includes preparation and quick gate."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import traceback
# Fixed before numerical imports; all timed variants use one BLAS/OpenMP thread.
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable]='1'
import certified_core as cc
from comparison_core import certify, METHODS
from quick_baseline import encode, quick_certificate
from threadpoolctl import threadpool_info, threadpool_limits
_thread_limit = threadpool_limits(limits=1)

ROOT=Path(__file__).resolve().parents[1]

def run(case,method,grids,nt,hessian_pieces,budget,quick,collect):
    cc.exp_neg.cache_clear();cc.EXP_CERTIFICATES.clear()
    start=time.perf_counter()
    deadline=start+budget if budget else None
    gate=quick_certificate(case) if quick else None
    stages=[]
    if gate and gate['status']=='QUICK_CERTIFIED_SAFE':
        result=dict(method=method,status='CERTIFIED_SAFE',quick_resolved=True,
                    upper=gate['quick_upper'],nonempty_witness=dict(theta=gate['midpoint_theta']),
                    historical_vertices=1,future_vertices=0,retained=0,deleted=0,frontend_calls=0,
                    completed_cells=0,grid_complete=True,budget_hit=False)
    else:
        result=None
        for level,n in enumerate(grids):
            stage_nt=nt[level] if isinstance(nt,list) else nt
            result=certify(case,n,stage_nt,method,hessian_pieces,collect,deadline)
            stages.append(result)
            # Universal SAFE without a nonempty witness is logically valid but
            # not a resolved nonvacuous estimation-and-safety task.
            if (result['status'] in ('CERTIFIED_UNSAFE','INCONSISTENT') or
                result['status']=='CERTIFIED_SAFE' and result['nonempty_witness'] is not None):
                break
            if deadline and time.perf_counter()>=deadline:
                break
    elapsed=time.perf_counter()-start
    resolved=(result['status'] in ('CERTIFIED_UNSAFE','INCONSISTENT') or
              result['status']=='CERTIFIED_SAFE' and result.get('nonempty_witness') is not None)
    return dict(result=result,stages=stages,quick=gate,elapsed=elapsed,
                resolved=resolved,budget_seconds=budget,overshoot_seconds=max(0,elapsed-budget) if budget else 0,
                resolved_within_budget=resolved and (not budget or elapsed<=budget),
                future_vertices_total=sum(s['future_vertices'] for s in stages),
                frontend_calls_total=sum(s['frontend_calls'] for s in stages))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',default='development_43.json')
    p.add_argument('--tag',default='development')
    p.add_argument('--grids',default='32')
    p.add_argument('--nt',type=int,default=32)
    p.add_argument('--nt-schedule',default=None)
    p.add_argument('--hessian-pieces',type=int,default=1)
    p.add_argument('--budget',type=float,default=0)
    p.add_argument('--reps',type=int,default=1)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--methods',default=','.join(METHODS[:6]))
    a=p.parse_args()
    path=ROOT/'inputs'/a.inputs
    cases=json.loads(path.read_text(encoding='utf-8-sig'))
    grids=list(map(int,a.grids.split(',')))
    nt=list(map(int,a.nt_schedule.split(','))) if a.nt_schedule else a.nt
    if isinstance(nt,list) and len(nt)!=len(grids):p.error('nt-schedule and grids lengths must match')
    methods=a.methods.split(',')
    output=ROOT/'results'/a.tag
    output.mkdir(exist_ok=False)
    meta=dict(arguments=vars(a),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              python=platform.python_version(),platform=platform.platform(),
              processor=platform.processor(),logical_cpus=os.cpu_count(),
              numerical_threads=1,threadpools=threadpool_info(),
              timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'),
              implementation_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'implementation').glob('*.py'))})
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    jobs=[(rep,i,m) for rep in range(a.reps) for i in range(len(cases)) for m in methods]
    random.Random(712913).shuffle(jobs)
    rows=[]
    with (output/'records.jsonl').open('w',encoding='utf-8') as fout:
        for job,(rep,i,method) in enumerate(jobs):
            row=dict(case_index=i,name=cases[i]['name'],group=cases[i].get('group'),rep=rep,method=method)
            # Every timed repeat constructs identical certificate payloads.
            error_start=time.perf_counter()
            try:
                row.update(run(cases[i],method,grids,nt,a.hessian_pieces,a.budget,a.quick,True))
            except Exception:
                elapsed=time.perf_counter()-error_start
                row.update(result=dict(method=method,status='ERROR'),stages=[],quick=None,
                    elapsed=elapsed,resolved=False,resolved_within_budget=False,
                    budget_seconds=a.budget,overshoot_seconds=max(0,elapsed-a.budget) if a.budget else 0,
                    future_vertices_total=0,frontend_calls_total=0,error_traceback=traceback.format_exc())
            fout.write(json.dumps(encode(row),separators=(',',':'))+'\n');fout.flush()
            rows.append(dict(method=method,result=dict(status=row['result']['status']),
                             resolved=row['resolved'],elapsed=row['elapsed']))
            print(f"{job+1}/{len(jobs)} rep={rep} {row['name']} {method} {row['result']['status']} {row['elapsed']:.4f}s",flush=True)
    summary={m:dict(statuses=dict(Counter(r['result']['status'] for r in rows if r['method']==m)),
                    resolved=sum(r['resolved'] for r in rows if r['method']==m),
                    elapsed_total=sum(r['elapsed'] for r in rows if r['method']==m)) for m in methods}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
