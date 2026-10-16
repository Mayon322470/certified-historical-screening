"""Bounded development backend replay, never a formal timing experiment."""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
from collections import Counter
import hashlib,json,os,sys,time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'research_zhao2026_comparison_20261012'
sys.path[:0]=[str(ROOT/'audit'),str(ROOT/'implementation'),str(ROOT/'theory'),str(OLD/'qa')]
import budget_comparison as producer
import audit_budget_result as verifier
import audit_coupling_faces as coupling_audit


def encode(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [encode(v) for v in x]
    return x


def save(name,value):
    with (ROOT/'audit'/name).open('x',encoding='utf-8') as f:json.dump(encode(value),f,ensure_ascii=False,indent=2)


def main():
    started=time.perf_counter();stats=Counter();records=[];quick_records=[]
    cases=json.loads((ROOT/'protocol/development/public_fixtures.json').read_text())
    for case in cases:
        quick=encode(producer.quick_certificate(case));verifier.check_quick(case,quick)
        quick_records.append(dict(case=case,result=quick))
        for method in producer.METHODS:
            result=encode(producer.certify(case,8,16,method))
            record=dict(case=case,result=result);records.append(record)
            stats.update(verifier.check_result(case,result))
    # Common-beta emptiness that independent coordinate boxes cannot exclude.
    joint=json.loads((ROOT/'audit/COUPLING_INDEPENDENT_PROOF.json').read_text())['case']
    for method in producer.METHODS:
        result=encode(producer.certify(joint,2,2,method))
        assert result['status']=='INCONSISTENT'
        stats.update(verifier.check_result(joint,result));records.append(dict(case=joint,result=result))
    assert stats['common_beta_deletions']>0 and stats['coupling_deletions']>0
    mutations=[]
    safe=next(row for row in records if row['result']['status']=='CERTIFIED_SAFE')
    nonzero=next(row for row in records if row['result']['status']=='CERTIFIED_SAFE' and F(row['case']['response']['epsilon_x'])>0)
    unsafe=next(row for row in records if row['result']['status']=='CERTIFIED_UNSAFE')
    def reject(label,record,change):
        result=deepcopy(record['result']);change(result)
        try:verifier.check_result(record['case'],result)
        except (AssertionError,ValueError,KeyError,IndexError):mutations.append(label)
        else:raise AssertionError('Accepted mutation '+label)
    reject('omit_nonempty_witness',safe,lambda r:r.__setitem__('nonempty_witness',None))
    reject('omit_continuous_bridge',safe,lambda r:r.__setitem__('time_allowance','0'))
    reject('reset_global_response_clock',safe,lambda r:r.__setitem__('future_uses_global_response_clock',False))
    reject('incorrect_expanded_history',nonzero,lambda r:r['expanded_packet'][0].__setitem__(0,'0'))
    reject('incorrect_cell_beta',nonzero,lambda r:r['cells'][0].__setitem__('beta_outer',['0','0']))
    reject('understated_future_upper',safe,lambda r:r['cells'][0].__setitem__('max_node_upper','0'))
    reject('omitted_cell',safe,lambda r:r['cells'].pop())
    reject('wrong_actual_curvature',nonzero,lambda r:r['continuous_bridge'].__setitem__('response_curvature_bound','0'))
    reject('unsafe_beta_out_of_prior',unsafe,lambda r:r['witness'].__setitem__('beta','1'))
    reject('unsafe_wrong_global_time',unsafe,lambda r:r['witness'].__setitem__('global_time','0'))
    reject('nominal_only_unsafe_claim',unsafe,lambda r:r['witness'].__setitem__('position_lower',r['witness']['nominal_position_lower']))
    reject('wrong_unsafe_historical_realization',unsafe,lambda r:r['witness']['actual_historical_state'][0].__setitem__(0,'100'))
    sources=list((ROOT/'implementation').glob('*.py'))+list((ROOT/'theory').glob('*.py'))
    sources += [ROOT/'audit'/name for name in ('audit_backend_smoke.py','audit_budget_result.py','audit_coupling_faces.py','audit_rank1_ma.py','audit_rank1_zhao.py','rank1_reference.py')]
    sources += [OLD/'qa'/name for name in ('audit_comparison.py','independent_ma_reference.py','zhao_reference.py')]
    report=dict(status='PASS',scope='BOUNDED_PREFREEZE_COMPLETE_BACKEND_AUDIT',
                known_development_records=20,synthetic_joint_inconsistency_records=5,
                quick_records=4,counts=dict(stats),contract_cache_counts=dict(verifier.COUNTS),
                statuses=dict(Counter(row['result']['status'] for row in records)),
                mutations_rejected=mutations,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                elapsed_seconds=time.perf_counter()-started,
                no_formal_cases_generated_or_read=True,no_simulator=True,
                comparison_speed_measured=False,physical_membership_proved=False,novelty_proved=False)
    save('BACKEND_DEVELOPMENT_PROOFS.json',dict(records=records,quick_records=quick_records))
    save('BACKEND_DEVELOPMENT_AUDIT.json',report)
    print(json.dumps(encode({k:v for k,v in report.items() if k!='source_sha256'}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
