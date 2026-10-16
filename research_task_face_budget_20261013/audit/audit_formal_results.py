"""Stream all frozen 480 records and independently replay each complete proof.

No comparison solver import. Run only AFTER timers complete. Exact payload
reuses bind the whole public input and whole stage, including method/settings.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,random,sys,time

sys.dont_write_bytecode=True
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'research_zhao2026_comparison_20261012'
sys.path[:0]=[str(ROOT/'audit'),str(OLD/'qa')]
import audit_budget_result as verifier

METHODS=('ma_rank1','ma_rank1_structured','zhao1_rank1','zhao2t3_rank1','zhao2t3_rank1_structured')
GRIDS=(8,32,128,512);TIME_GRIDS=(32,64,128,256)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def solved(result):
    return result['status'] in ('CERTIFIED_UNSAFE','INCONSISTENT') or result['status']=='CERTIFIED_SAFE' and result.get('nonempty_witness') is not None


def main():
    started=time.perf_counter();directory=ROOT/'results/formal_main'
    completed_path=directory/'RUN_COMPLETED.json'
    if not completed_path.is_file():raise RuntimeError('Do not audit while formal timers are running')
    completed=json.loads(completed_path.read_text());assert completed['status']=='COMPLETED_ALL_480'
    data_path=directory/'records.jsonl';assert digest(data_path)==completed['records_sha256']
    freeze_path=ROOT/'protocol/METHODS_FROZEN.json';freeze=json.loads(freeze_path.read_text())
    assert completed['method_freeze_sha256']==digest(freeze_path)
    for name,sha in freeze['source_sha256'].items():assert digest(ROOT/name)==sha,(name,'frozen source drift')
    cases_path=ROOT/'protocol/frozen_suite/public_cases_32.json';cases=json.loads(cases_path.read_text())
    assert len(cases)==32 and completed['input_sha256']==digest(cases_path)
    input_audit=json.loads((ROOT/'audit/GENERATED_INPUT_AUDIT.json').read_text())
    assert input_audit['status']=='PASS' and input_audit['public_cases']==32
    assert input_audit['source_sha256'][str(cases_path)]==digest(cases_path)
    external=json.loads((ROOT/'audit/AUDIT_EXTERNAL_DEPENDENCIES.json').read_text())
    for path,sha in external['runtime_dependency_sha256'].items():assert digest(path)==sha
    expected=[(rep,i,m) for rep in range(3) for i in range(32) for m in METHODS]
    random.Random(712913).shuffle(expected)
    stage_cache={};stats=Counter();statuses=Counter();per_method={m:Counter() for m in METHODS}
    all_conclusions={};records=0;row_errors=[]
    for sequence,line in enumerate(data_path.open(encoding='utf-8')):
        row=json.loads(line);records+=1
        assert row['sequence']==sequence and (row['rep'],row['case_index'],row['method'])==expected[sequence]
        case=cases[row['case_index']];assert row['name']==case['name']
        assert row['input_sha256']==completed['input_sha256'] and row['method_freeze_sha256']==completed['method_freeze_sha256']
        assert row['budget_seconds']==5.0 and math.isfinite(row['elapsed']) and row['elapsed']>=0
        assert row['overshoot_seconds']==max(0.0,row['elapsed']-5.0)
        result=row['result'];method=row['method'];assert result['method']==method
        status=result['status'];statuses[status]+=1;per_method[method][status]+=1
        assert status!='INCONSISTENT','Every generated case has an independently certified original-packet witness'
        if status=='ERROR':
            assert row['resolved'] is False and row['resolved_within_budget'] is False
            assert isinstance(row['error_traceback'],str) and row['error_traceback']
            row_errors.append(dict(sequence=sequence,case=case['name'],method=method,error_traceback=row['error_traceback']))
        else:
            verifier.check_quick(case,row['quick']);stats['quick_checks']+=1
            if row['quick']['status']=='QUICK_CERTIFIED_SAFE':
                assert result['quick_resolved'] is True and status=='CERTIFIED_SAFE' and not row['stages']
                assert result['upper']==row['quick']['quick_upper'] and result['nonempty_witness']==row['quick']['nonempty_witness']
                assert result['grid_complete'] is True and result['frontend_calls']==result['future_vertices']==0
                stats['quick_resolved']+=1
            else:
                assert row['stages'] and row['stages'][-1]==result
                assert len(row['stages'])<=4
                for j,stage in enumerate(row['stages']):
                    assert stage['method']==method and (stage['n'],stage['nt'])==(GRIDS[j],TIME_GRIDS[j])
                    if j<len(row['stages'])-1:assert not solved(stage)
                    fingerprint=hashlib.sha256(json.dumps([case,stage],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    if fingerprint not in stage_cache:
                        stage_cache[fingerprint]=verifier.check_result(case,stage)
                        stats.update(stage_cache[fingerprint]);stats['unique_stage_replays']+=1
                    else:stats['identical_full_input_stage_payload_reuses']+=1
                    stats['all_stage_records']+=1
            assert row['resolved']==solved(result)
            assert row['resolved_within_budget']==(solved(result) and row['elapsed']<=5.0)
            assert row['future_vertices_total']==sum(s['future_vertices'] for s in row['stages'])
            assert row['frontend_calls_total']==sum(s['frontend_calls'] for s in row['stages'])
            reason=('CELL_BOUNDARY_DEADLINE' if row['elapsed']>=5.0 else 'FIXED_SCHEDULE_EXHAUSTED') if status=='UNKNOWN' else None
            assert row['unknown_reason']==reason
            if solved(result):all_conclusions.setdefault(case['name'],set()).add(status)
        per_method[method]['within_budget']+=row['resolved_within_budget']
        per_method[method]['late_certificates']+=row['resolved'] and not row['resolved_within_budget']
        if sequence%20==0:print(json.dumps(dict(audited=sequence+1,of=480,elapsed_seconds=time.perf_counter()-started),ensure_ascii=False),flush=True)
    assert records==480
    assert all(len(conclusions)==1 for conclusions in all_conclusions.values()),'Conflicting certified conclusions'
    for tid in range(16):
        low=all_conclusions.get(cases[2*tid]['name'],set());high=all_conclusions.get(cases[2*tid+1]['name'],set())
        assert not ('CERTIFIED_UNSAFE' in low and 'CERTIFIED_SAFE' in high),'Contradictory nested budget certificates'
    source_files=[Path(__file__)]+[ROOT/'audit'/name for name in ('audit_budget_result.py','audit_rank1_ma.py','audit_rank1_zhao.py','audit_coupling_faces.py','rank1_reference.py')]
    hashes={str(p):digest(p) for p in source_files}
    hashes.update(external['runtime_dependency_sha256'])
    # Capture every locally loaded external module, including dynamic imports.
    base=ROOT.parent.resolve()
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if path.suffix=='.py' and path.is_relative_to(base):hashes[str(path)]=digest(path)
    report=dict(status='PASS',scope='ALL_FORMAL_RESULTS_INDEPENDENT_CERTIFICATE_REPLAY',records=records,
                statuses=dict(statuses),per_method={k:dict(v) for k,v in per_method.items()},
                independent_work_counts=dict(stats),contract_cache_counts=dict(verifier.COUNTS),
                all_certified_conclusions_consistent=True,nested_budget_conclusions_consistent=True,
                error_records=row_errors,elapsed_seconds=time.perf_counter()-started,
                records_sha256=digest(data_path),method_freeze_sha256=digest(freeze_path),
                generated_input_sha256=digest(cases_path),source_sha256=hashes,
                producer_imported=False,timing_overlap=False,performance_statistics_checked=False,
                actual_vehicle_membership_proved=False,novelty_proved=False)
    with (ROOT/'audit/FORMAL_RESULTS_AUDIT.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','error_records')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
