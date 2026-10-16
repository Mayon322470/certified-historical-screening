"""Separate audit of the post-benchmark Taylor refinement.

The immutable independent audit_efficiency.py supplies exact-rational model,
Sturm, witness and geometry-comparison checks. No producer module is imported.
New Taylor and concavity certificates are independently recomputed here.
The first 954-record dataset and its audit are never rewritten.
"""
from pathlib import Path
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial
from collections import Counter
import copy
import hashlib
import json
import random
import time

import audit_efficiency as independent

ROOT=Path(__file__).resolve().parent.parent
METHODS=('taylor_faces','taylor_curvature','hybrid_taylor')
OLD_AUDIT_SHA='8398a1e4e0ea0ea8c9cb74da446b7ebc9587d26b1c8af2207e2c616e054d9ac8'
OLD_RESULTS_SHA='87d6ff82333646e088d17b2afed2aed15a49aa244a49acbd9827ccdcd05f1570'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


@lru_cache(maxsize=200000)
def centered_reference(key,a,b,order=0):
    center=(a+b)/2; radius=(b-a)/2
    answer=independent.sparse_range(key,center,center,order)
    for j in range(1,5):
        domain=(-radius**j,radius**j) if j%2 else (Q(0),radius**j)
        term=independent.im(independent.sparse_range(key,center,center,order+j),domain)
        answer=independent.ia(answer,independent.isc(term,Q(1,factorial(j))))
    remainder=independent.im(independent.sparse_range(key,a,b,order+5),(-radius**5,radius**5))
    return independent.ia(answer,independent.isc(remainder,Q(1,factorial(5))))


def check_refined_removal(cert,meta):
    kind=cert['kind']
    if kind in ('natural','curvature','monotone_increasing','monotone_decreasing'):
        independent.check_sparse_removal(cert,meta)
        return
    l,r=map(Q,cert['theta_interval']); assert 0<l<r
    a,b=map(Q,cert['q_interval']); assert (a,b)==(1/r,1/l)
    face=meta['faces'][cert['face_index']]
    assert face['name']==cert['face_name']
    key=independent.face_key(face); lower=Q(cert['certified_lower'])
    assert lower>0
    if kind=='taylor':
        assert lower<=centered_reference(key,a,b)[0]
    elif kind=='taylor_increasing':
        assert Q(cert['point'])==a and centered_reference(key,a,b,1)[0]>=0
        assert lower<=independent.sparse_range(key,a,a)[0]
    elif kind=='taylor_decreasing':
        assert Q(cert['point'])==b and centered_reference(key,a,b,1)[1]<=0
        assert lower<=independent.sparse_range(key,b,b)[0]
    elif kind=='concave':
        assert independent.independent_second(key,a,b)[1]<=0
        assert lower<=min(independent.sparse_range(key,a,a)[0],independent.sparse_range(key,b,b)[0])
    else:
        raise AssertionError(kind)


def check_schedule(records,data,frozen,fixtures):
    assert tuple(data['methods'])==METHODS and data['schedule_seed']==2026101202
    tasks=[(i,*frozen['primary_grid']) for i in range(len(fixtures))]
    tasks += [(i,*frozen['long_horizon_time_grid']) for i in frozen['long_workload_case_indices']]
    expected=[(rep,i,n,nt,m) for rep in range(frozen['repetitions']) for i,n,nt in tasks for m in METHODS]
    random.Random(data['schedule_seed']).shuffle(expected)
    assert len(records)==len(expected)
    for sequence,(record,want) in enumerate(zip(records,expected)):
        assert record['sequence']==sequence
        assert (record['rep'],record['case_index'],record['n'],record['nt'],record['method'])==want


def main():
    started=time.perf_counter()
    assert hashlib.sha256((ROOT/'implementation/audit_efficiency.py').read_bytes()).hexdigest()==OLD_AUDIT_SHA
    oldraw=(ROOT/'implementation/benchmark_results.json').read_bytes()
    assert hashlib.sha256(oldraw).hexdigest()==OLD_RESULTS_SHA
    olddata=json.loads(oldraw)
    raw=(ROOT/'inputs/fixtures.json').read_bytes();fixtures=json.loads(raw)
    cases={c['name']:c for c in fixtures};assert len(cases)==len(fixtures)
    frozen=read('inputs/FROZEN_INPUTS.json');data=read('implementation/refined_results.json')
    digest=hashlib.sha256(raw).hexdigest()
    assert digest==frozen['sha256']==data['input_sha256']==olddata['input_sha256']
    assert len(fixtures)==frozen['cases'] and frozen['repetitions']==3
    expdata=read('implementation/refined_exp_certificates.json')
    bits=expdata['bits'];exps=expdata['intervals']
    assert bits==data['bits']==olddata['bits']
    for x,pair in exps.items():assert independent.valid_exp(x,pair,bits),x
    records=data['records'];check_schedule(records,data,frozen,fixtures)
    eager={(r['case'],r['n'],r['nt']):r['result'] for r in olddata['records']
           if r['rep']==0 and r['method']=='eager_chord'}
    checked_models=set();repeat={};decisions=Counter();kind_counts=Counter();face_counts=Counter()
    detailed_count=0;deletions=0;cells_compared=0;dominance=0;witnesses=0;sparse_models=0;sturm_chains=0
    examples={};detailed={}
    for record in records:
        name=record['case'];case=cases[name];result=record['result']
        key=(name,record['n'],record['nt'],record['method'])
        assert fixtures[record['case_index']]['name']==name
        assert (result['n'],result['nt'],result['method'])==key[1:]
        independent.check_result(result,case);independent.check_work(record,case)
        decisions[result['status']]+=1
        stable={k:v for k,v in result.items() if k not in ('cells','deletions','sparse_model','sturm_model')}
        if key in repeat:
            assert stable==repeat[key][0] and record['work']==repeat[key][1]
        else:repeat[key]=(stable,record['work'])
        if record['rep']!=0:
            assert 'cells' not in result
            continue
        detailed_count+=1;assert key not in detailed;detailed[key]=record
        assert len(result['cells'])==result['retained']
        assert len(result['deletions'])==result['filtered_cells']
        meta=result['sparse_model'];assert meta is not None
        mk=('sparse',name,json.dumps(meta,sort_keys=True))
        if mk not in checked_models:
            independent.check_sparse_model(case,meta);checked_models.add(mk);sparse_models+=1
        if result['sturm_model'] is not None:
            assert record['method']=='hybrid_taylor'
            sm=result['sturm_model'];mk=('sturm',name,json.dumps(sm,sort_keys=True))
            if mk not in checked_models:
                independent.check_sturm_model(case,sm);checked_models.add(mk);sturm_chains+=len(sm['faces'])
        elif record['method']=='hybrid_taylor':
            raise AssertionError('Missing hybrid Sturm model')
        deleted=[];prior=list(map(Q,case['prior']));n=record['n']
        for deletion in result['deletions']:
            index=deletion['index'];cert=deletion['certificate']
            assert 0<=index<n
            expected=[prior[0]+(prior[1]-prior[0])*Q(i,n) for i in (index,index+1)]
            assert list(map(Q,cert['theta_interval']))==expected
            if deletion['engine']=='sparse':
                check_refined_removal(cert,meta)
                if record['method']=='taylor_faces': assert cert['kind'] not in ('curvature','concave')
                kind_counts[cert['kind']]+=1
                examples.setdefault(cert['kind'],(copy.deepcopy(cert),meta))
            else:
                assert deletion['engine']=='sturm' and record['method']=='hybrid_taylor'
                independent.check_sturm_removal(cert,result['sturm_model'],exps,bits)
                kind_counts['sturm']+=1
            face_counts[cert['face_name']]+=1;deletions+=1;deleted.append(index)
        assert len(deleted)==len(set(deleted))
        current={c['index']:c for c in result['cells']}
        assert len(current)==result['retained'] and all(0<=i<n for i in current)
        required={j for i in current for j in (i,i+1)}
        history={j for i in range(n) if i not in deleted for j in (i,i+1)}
        assert result['future_vertices']==len(required) and result['historical_vertices']==len(history)
        if current:
            assert Q(result['upper'])==max(Q(c['max_node_upper']) for c in current.values())+Q(result['time_allowance'])
        old=eager[(name,n,record['nt'])]
        cells_compared+=independent.check_cell_relation(result,old)
        if current:dominance+=1
        if result['witness'] is not None:
            independent.check_witness(case,result);witnesses+=1
    assert set(Counter((r['case'],r['n'],r['nt'],r['method']) for r in records).values())=={3}
    mutations={}
    for kind,(cert,meta) in examples.items():
        bad=copy.deepcopy(cert);bad['certified_lower']=str(Q(bad['certified_lower'])+1)
        mutations['wrong_'+kind+'_certified_lower']=independent.reject_test(lambda b=bad,m=meta:check_refined_removal(b,m))
    if 'concave' in examples:
        cert,meta=examples['concave'];bad=copy.deepcopy(cert);bad['q_interval']=['0','1']
        mutations['wrong_concave_parameter_domain']=independent.reject_test(lambda:check_refined_removal(bad,meta))
    mutations['omitted_refined_record']=independent.reject_test(lambda:check_schedule(records[:-1],data,frozen,fixtures))
    chosen=next(r for r in detailed.values() if r['result']['retained'])
    bad=copy.deepcopy(chosen['result']);bad['cells'].pop();bad['retained']-=1
    old=eager[(chosen['case'],chosen['n'],chosen['nt'])]
    mutations['unjustified_missing_refined_cell']=independent.reject_test(lambda:independent.check_cell_relation(bad,old))
    assert all(mutations.values())
    corrupt=0
    for x,pair in list(exps.items())[:12]:
        lo,hi=independent.exp_reference(Q(x));low=hi.numerator*2**bits//hi.denominator+2
        assert not independent.valid_exp(x,[str(low),str(low+2)],bits);corrupt+=1
    out=dict(
        status='PASS',cases=len(cases),records=len(records),repetitions=3,
        detailed_first_repetition_records=detailed_count,decision_counts=dict(decisions),
        independently_rebuilt_sparse_models=sparse_models,verified_sturm_chains=sturm_chains,
        independently_checked_cell_deletions=deletions,deletions_by_kind=dict(kind_counts),
        unexercised_new_certificate_kinds=[k for k in ('taylor','taylor_increasing','taylor_decreasing','concave') if kind_counts[k]==0],
        deletions_by_face=dict(face_counts),retained_cells_identical_to_frozen_eager=cells_compared,
        nonempty_same_grid_dominance_checks=dominance,independently_recomputed_unsafe_witnesses=witnesses,
        independently_checked_exponential_enclosures=len(exps),corrupt_exponentials_rejected=corrupt,
        mutations_rejected=mutations,elapsed_seconds=time.perf_counter()-started,
        frozen_reference_hashes=dict(audit_efficiency_py=OLD_AUDIT_SHA,benchmark_results_json=OLD_RESULTS_SHA,fixtures_json=digest),
        reused_checker_scope='Imports only the previous independent checker, whose SHA-256 is bound. No producer module is imported and the old script, results and audit are not rewritten.',
        new_checker_scope='Independently recomputes fourth-order centered Taylor bounds for F and F-prime and the F-second concavity certificate using 200-bit rational exponential references.',
        limitations='Post-benchmark exploratory refinement on the same fixtures, not held-out validation. Existing chord geometry is compared, not independently reimplemented. Timing gains, general bit complexity and WCET are not certified.',
    )
    (ROOT/'implementation/REFINED_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    main()
