"""Bounded pre-freeze audit; no formal generator, simulator, or benchmark run."""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT.parent/'research_zhao2026_comparison_20261012'
sys.path[:0]=[str(HERE),str(ROOT/'implementation'),str(ROOT/'theory'),str(OLD/'qa')]
import rank1_reference as rr
import audit_rank1_ma as am
import audit_rank1_zhao as az
import rank1_ma as ma
import ma_scalar_certified as old_ma
import zhao_carleman as zhao
import shared_response as sr
import independent_ma_reference as history_ref
import audit_comparison as old_audit


def encoded(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {k:encoded(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encoded(v) for v in value]
    return value


def write(name,data):
    target=HERE/name
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(encoded(data),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def parse(raw):
    case=deepcopy(raw)
    for key in ('initial','prior'):case[key]=list(map(F,raw[key]))
    for key in ('past','future','packet'):case[key]=[list(map(F,row)) for row in raw[key]]
    for key in ('H','rho','limit'):case[key]=F(raw[key])
    return case


def main():
    started=time.perf_counter()
    groups={};records=[];mutations=[];regressions=[]
    rng=random.Random(2601013)
    # Different optimization geometry from producer: exact 2-D dual polygon.
    problems=[(1,(0,0,0),(F(1,10),)*3,10,(1,3,6)),
              (1,(1,-1,0),(F(1,10),)*3,10,(1,3,6)),
              (1,(2,0,0),(F(1,10),)*3,F(1,100),(2,0,0)),
              (0,(1,2,3),(1,1,1),0,(0,0,0))]
    for k in range(36):
        problems.append((F(rng.randrange(0,8),4),tuple(F(rng.randrange(-8,9),4) for _ in range(3)),
                         tuple(F(rng.randrange(1,9),20) for _ in range(3)),F(rng.randrange(0,8),10),
                         tuple(F(rng.randrange(0,9),4) for _ in range(3))))
    for R,C,s,b,p in problems:
        N,value,cert=ma.optimal_gain(R,C,s,p,b)
        assert am.check_gain(cert)==(N,value)
    assert ma.optimal_gain(*[problems[1][0],problems[1][1],problems[1][2],problems[1][4],problems[1][3]])[1]==F(1,10)
    groups['ma_primal_against_independent_dual']=len(problems)
    # An independent vertex representation checks FM, including collapsed sets.
    projections=[((0,2),10,[(1,1,1,F(6,5)),(-1,1,-1,F(-4,5)),(0,0,-1,1)]),
                 ((0,2),0,[(1,0,1,1),(0,0,0,0),(0,1,0,0)]),
                 ((1,1),0,[(0,0,1,2),(1,1,0,2),(0,0,0,0)]),
                 ((1,1),0,[(1,0,1,1),(0,0,0,0),(0,0,0,0)])]
    for k in range(30):
        strips=[]
        for j in range(3):
            low=F(rng.randrange(-10,11),5)
            strips.append((F(rng.randrange(-7,8),4),F(rng.randrange(0,7),4),low,low+F(rng.randrange(0,8),5)))
        projections.append(((F(1,4),F(9,4)),F(rng.randrange(0,6),5),strips))
    for prior,b,strips in projections:
        expected=rr.joint_strip_projection(prior,b,strips)
        given,_=ma.project_strips([dict(C=C,low=low,high=high) for C,p,low,high in strips],
                                 [p for C,p,low,high in strips],b,prior)
        assert given==expected
    groups['fm_against_independent_polygon']=len(projections)
    beta_problems=[([(0,1)]*3,[(F(-1,10),F(1,10))]*3,(0,0,0),1),
                   ([(0,1)]*3,[(F(1,4),F(3,4))]*3,(0,0,0),1),
                   ([(0,0),(1,1),(0,0)],[(0,0)]*3,(1,1,1),2)]
    for k in range(25):
        nominal=[];packet=[]
        for j in range(3):
            a,c=F(rng.randrange(-8,9),5),F(rng.randrange(-8,9),5)
            nominal.append((a,a+F(rng.randrange(0,6),5)))
            packet.append((c,c+F(rng.randrange(0,6),5)))
        beta_problems.append((packet,nominal,(F(1,8),F(3,8),F(3,4)),F(1)))
    for packet,nominal,p,b in beta_problems:
        for kind,method in [('outer',sr.beta_outer),('inner',sr.beta_inner)]:
            assert method(nominal,packet,p,b)==rr.beta_bounds(packet,nominal,p,b,kind)
    groups['beta_outer_inner_exact_cases']=2*len(beta_problems)
    fixtures=json.loads((ROOT/'protocol/development/public_fixtures.json').read_text(encoding='utf-8-sig'))
    parsed=[parse(raw) for raw in fixtures]
    for raw,case in zip(fixtures,parsed):
        T=sum((dt for dt,_ in case['past']),F(0));L=T+case['H'];b=F(case['response']['epsilon_x'])
        assert sr.profile(case,0)==(0,0,0) and sr.profile(case,L)==rr.cubic_psi(L,L)
        psi=rr.cubic_psi(T,L)
        expanded=tuple((lo-b*p,hi+b*p) for (lo,hi),p in zip(case['packet'],psi))
        assert sr.expanded_packet(case)==expanded
        node_step=case['H']/4
        Qnom=max(max(map(abs,expanded[2])),max(abs(u) for _,u in case['future']))
        allowance=sr.continuous_allowance(case,node_step)
        assert allowance['allowance']==(Qnom+6*b/L**2)*node_step**2/8
        assert sr.future_upper(2,(-b,b),rr.cubic_psi(T+case['H'],L)[0])==2+b
        assert sr.future_lower(2,-b,rr.cubic_psi(T+case['H'],L)[0])==2-b
        model=ma.prepare(case)
        result,cert=ma.contract(model,case['packet'],F(15,16),F(17,16),psi,b)
        assert am.check_contract(raw,cert)==result
        assert result is not None and result[0]<=1<=result[1]
        records.append(dict(case=raw,method='ma_rank1',result=result,certificate=cert))
    groups['ma_history_contracts']=len(fixtures)
    groups['global_time_and_actual_curvature']=len(fixtures)
    # A small zero-response regression and one nonzero fixture per nu.
    old_path=OLD/'implementation/zhao_carleman.py'
    spec=importlib.util.spec_from_file_location('sealed_old_zhao',old_path)
    old_zhao=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_zhao)
    for index in (0,1):
        raw,case=fixtures[index],parsed[index]
        T=sum((dt for dt,_ in case['past']),F(0));p=sr.profile(case,T)
        b=F(case['response']['epsilon_x']);response=[b*x for x in p]
        model=zhao.prepare(case)
        for nu,tight in ((1,False),(2,True)):
            result,cert=zhao.contract(model,case['packet'],F(15,16),F(17,16),nu,tight,'full',1,response)
            assert az.check_contract(raw,cert)==result
            assert result is not None and result[0]<=1<=result[1]
            records.append(dict(case=raw,method=f'zhao{nu}_rank1',result=result,certificate=cert))
            if index==0:
                old_result,old_cert=old_zhao.contract(model,case['packet'],F(15,16),F(17,16),nu,tight,'full',1)
                assert result==old_result
                for key in ('K','prediction_generators','posterior_generators','posterior_center','theorem3'):
                    assert cert[key]==old_cert[key]
                regressions.append(dict(method=f'zhao{nu}',status='PASS',comparison='exact output/gain/generators/T3 equality'))
    groups['zhao_history_contracts']=4
    case=parsed[0];model=ma.prepare(case);p=sr.profile(case,1)
    for low,high in ((F(1,2),F(2)),(F(15,16),F(17,16))):
        result,cert=ma.contract(model,case['packet'],low,high,p,0)
        previous,old_cert=old_ma.contract(model,case['packet'],low,high,strengthen=True)
        assert result==previous
    regressions.append(dict(method='ma_strong',status='PASS',comparison='exact theta result on two intervals'))
    # Each rejection exercises the checker against a proof payload mutation.
    def reject(label,check,cert,change):
        bad=deepcopy(cert);change(bad)
        try:check(bad)
        except (AssertionError,ValueError,KeyError,IndexError):mutations.append(label)
        else:raise AssertionError('Accepted mutation: '+label)
    rec=records[1];raw,cert=rec['case'],rec['certificate']
    reject('ma_wrong_response_column',lambda c:am.check_contract(raw,c),cert,lambda c:c['response_shape'].__setitem__(0,F(999)))
    reject('ma_missing_rank1_radius',lambda c:am.check_contract(raw,c),cert,lambda c:c.__setitem__('posterior_radius',F(0)))
    reject('ma_wrong_common_beta_bound',lambda c:am.check_contract(raw,c),cert,lambda c:c.__setitem__('response_bound',F(0)))
    reject('ma_invalid_optimal_gain',lambda c:am.check_contract(raw,c),cert,lambda c:c['gain_certificate']['selected'].__setitem__('N',(F(0),)*3))
    reject('ma_fm_projection_tamper',lambda c:am.check_contract(raw,c),cert,lambda c:c['all_strip_projection'].__setitem__('q_projection',(F(1),F(1))))
    reject('ma_signed_remainder_tamper',lambda c:am.check_contract(raw,c),cert,lambda c:c['strips'][0].__setitem__('remainder',(F(0),F(0))))
    rec=records[-1];raw,cert=rec['case'],rec['certificate']
    reject('zhao_missing_response_column',lambda c:az.check_contract(raw,c),cert,lambda c:c['measurement_noise_generators'][0].__setitem__(3,F(0)))
    reject('zhao_double_counted_noise',lambda c:az.check_contract(raw,c),cert,lambda c:c['measurement_noise_generators'][0].__setitem__(0,F(1)))
    reject('zhao_invalid_constant_gain',lambda c:az.check_contract(raw,c),cert,lambda c:c['K'][0].__setitem__(0,F(1)))
    reject('zhao_posterior_radius_tamper',lambda c:az.check_contract(raw,c),cert,lambda c:c.__setitem__('base_r_interval',(F(0),F(0))))
    reject('zhao_omitted_isotropic_remainder',lambda c:az.check_contract(raw,c),cert,lambda c:c['degree_remainders'][1].__setitem__('o_l',F(0)))
    reject('zhao_T3_kappa_tamper',lambda c:az.check_contract(raw,c),cert,lambda c:c['theorem3'].__setitem__('kappa',F(0)))
    sources=[Path(__file__),HERE/'rank1_reference.py',HERE/'audit_rank1_ma.py',HERE/'audit_rank1_zhao.py',
             ROOT/'implementation/rank1_ma.py',ROOT/'implementation/zhao_carleman.py',
             ROOT/'implementation/ma_scalar_certified.py',ROOT/'implementation/certified_core.py',
             ROOT/'implementation/history_derivatives.py',ROOT/'theory/shared_response.py',
             ROOT/'theory/response_budget_math.py',ROOT/'protocol/development/public_fixtures.json',
             OLD/'qa/audit_zhao_contract.py',OLD/'qa/zhao_reference.py',OLD/'qa/independent_ma_reference.py',old_path]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report=dict(status='PASS',scope='BOUNDED_PREFREEZE_UNIT_AND_CONTRACT_AUDIT',test_groups=groups,
                zero_response_regressions=regressions,proof_mutations_rejected=mutations,
                runtime_seconds=time.perf_counter()-started,python=sys.version,source_sha256=hashes,
                formal_generator_called=False,formal_suite_read=False,formal_timing_started=False,
                shared_backend_verified=False,structured_frontend_verified=False,
                actual_vehicle_membership_verified=False,performance_or_novelty_proved=False,
                sealed_directories_written=False)
    write('RANK1_CONTRACT_PROOFS.json',records)
    write('RANK1_CONTRACT_AUDIT.json',report)
    print(json.dumps(encoded(report),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
