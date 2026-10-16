"""Cold-cache end-to-end same-grid comparison, retaining slowdowns and UNKNOWN."""
from pathlib import Path
from collections import Counter
import hashlib,json,random,time
import certified_core as cc
from polynomial_contract import encode
from make_fixtures import parse
from refined_core import METHODS,certify

ROOT=Path(__file__).resolve().parent.parent

def main():
    raw=(ROOT/'inputs/fixtures.json').read_bytes()
    frozen=json.loads((ROOT/'inputs/FROZEN_INPUTS.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=frozen['sha256']:
        raise ValueError('Frozen input hash changed')
    cases=[parse(c) for c in json.loads(raw)]
    tasks=[(i,32,64) for i in range(len(cases))]
    tasks += [(i,32,256) for i in frozen['long_workload_case_indices']]
    schedule=[(rep,i,n,nt,m) for rep in range(3) for i,n,nt in tasks for m in METHODS]
    rng=random.Random(2026101202);rng.shuffle(schedule)
    results=[];exp_records={};counts=Counter()
    original_flow=cc.flow;original_endpoint=cc.endpoint;original_timeline=cc.timeline
    phase='unclassified'
    def instrumented_flow(*args,**kwargs):
        counts['flow_calls']+=1
        counts[phase+'_flow_calls']+=1
        return original_flow(*args,**kwargs)
    def instrumented_endpoint(*args,**kwargs):
        nonlocal phase
        phase='history'
        return original_endpoint(*args,**kwargs)
    def instrumented_timeline(*args,**kwargs):
        nonlocal phase
        phase='future'
        return original_timeline(*args,**kwargs)
    cc.flow=instrumented_flow
    cc.endpoint=instrumented_endpoint;cc.timeline=instrumented_timeline
    out=ROOT/'implementation/refined_results.json'
    if out.exists():
        raise FileExistsError('Results already exist; preserve them')
    for sequence,(rep,i,n,nt,method) in enumerate(schedule):
        cc.exp_neg.cache_clear();cc.EXP_CERTIFICATES.clear();counts.clear()
        start=time.perf_counter()
        result=certify(cases[i],n,nt,method,collect=rep==0)
        elapsed=time.perf_counter()-start
        info=cc.exp_neg.cache_info()
        results.append(dict(sequence=sequence,rep=rep,case=cases[i]['name'],case_index=i,
            n=n,nt=nt,method=method,elapsed_seconds=elapsed,
            work=dict(counts,exponential_evaluations=info.misses,exponential_cache_hits=info.hits),result=result))
        if rep==0:
            for x,iv in cc.EXP_CERTIFICATES.items():
                if x in exp_records and exp_records[x]!=iv:
                    raise ValueError('Inconsistent fixed-point exponential')
                exp_records[x]=iv
        if (sequence+1)%30==0:
            print(f'{sequence+1}/{len(schedule)} records completed',flush=True)
    cc.flow=original_flow;cc.endpoint=original_endpoint;cc.timeline=original_timeline
    record=dict(input_sha256=frozen['sha256'],bits=cc.BITS,methods=METHODS,
                schedule_seed=2026101202,records=results)
    out.write_text(json.dumps(encode(record),indent=2)+'\n',encoding='utf-8')
    (ROOT/'implementation/refined_exp_certificates.json').write_text(
        json.dumps(dict(bits=cc.BITS,intervals=exp_records),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(results),cases=len(cases),tasks=len(tasks),exponentials=len(exp_records))))

if __name__=='__main__':main()
