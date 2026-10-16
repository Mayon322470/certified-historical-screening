"""Independent post-freeze input QA. Does not import or call the generator.

Nominal endpoints are independently built from command-jump expressions,
then evaluated using an unscaled rational alternating series at 384 bits.
Only the audit process reads private generating witnesses.
"""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from collections import Counter
from copy import deepcopy
import argparse,hashlib,json,sys,time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'research_zhao2026_comparison_20261012'
sys.path[:0]=[str(ROOT/'audit'),str(OLD/'qa')]
import independent_ma_reference as symbolic
import rank1_reference as rr

FIELDS={'name','initial','past','prior','packet','future','H','rho','limit','response'}
SCALE=2**384


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def down(x):return F(x.numerator*SCALE//x.denominator,SCALE)
def up(x):return -down(-x)


@lru_cache(maxsize=4096)
def exponential(x):
    x=F(x);assert x>=0
    if not x:return F(1),F(1)
    total=term=F(1);n=0
    while True:
        previous=total;n+=1;term*=(-x)/n;total+=term
        if n>2*x and abs(term)<F(1,2**408):
            return down(min(previous,total)),up(max(previous,total))


def eval_point(expression,q):
    # Combine equal exponentials exactly before interval evaluation.
    coefficients={}
    for power,decay,coefficient in expression:
        coefficients[decay]=coefficients.get(decay,F(0))+coefficient*q**power
    lo=hi=F(0)
    for decay,value in coefficients.items():
        a,b=exponential(decay*q)
        lo+=min(a*value,b*value);hi+=max(a*value,b*value)
    return lo,hi


def endpoint(case,tau,future_time=None):
    past=[tuple(map(F,row)) for row in case['past']]
    if future_time is not None:
        left=F(future_time)
        for dt,u in case['future']:
            step=min(F(dt),left)
            if step:past.append((step,F(u)))
            left-=step
            if left==0:break
        assert left==0
    model=dict(initial=case['initial'],past=past)
    return [eval_point(expression,1/F(tau)) for expression in symbolic.endpoint_terms(model)]


def contained(inner,outer):
    a,b=map(F,outer)
    return a<=inner[0]<=inner[1]<=b


def upper(case,b):
    T=sum((F(dt) for dt,_ in case['past']),F(0));H=F(case['H']);L=T+H
    p=rr.cubic_psi(T,L);packet=[tuple(map(F,row)) for row in case['packet']]
    Qnom=max(abs(packet[2][0]-b*p[2]),abs(packet[2][1]+b*p[2]),max(abs(F(u)) for _,u in case['future']))
    Qactual=Qnom+6*b/L**2
    return packet[0][1]+max(F(0),packet[1][1])*H+Qactual*H*H/2,Qnom,Qactual


def check_witness(case,private):
    T=sum((F(dt) for dt,_ in case['past']),F(0));H=F(case['H']);L=T+H
    witness=private['history_witness'];tau,beta=F(witness['tau']),F(witness['beta'])
    b=F(case['response']['epsilon_x']);p=rr.cubic_psi(T,L)
    assert F(case['prior'][0])<=tau<=F(case['prior'][1]) and abs(beta)<=b
    nominal=endpoint(case,tau)
    actual=[(lo+beta*pi,hi+beta*pi) for (lo,hi),pi in zip(nominal,p)]
    assert all(contained(row,box) for row,box in zip(nominal,witness['nominal']))
    assert all(contained(row,box) for row,box in zip(actual,witness['actual']))
    assert all(contained(row,box) for row,box in zip(actual,case['packet']))
    assert witness['original_packet_containment']==[True]*3
    assert witness['same_beta_for_all_coordinates'] is True and witness['same_beta_for_future'] is True
    s=F(private['future_witness_time']);assert 0<=s<=H
    position=endpoint(case,tau,s)[0];shift=beta*rr.cubic_psi(T+s,L)[0]
    position=(position[0]+shift,position[1]+shift)
    assert contained(position,private['future_witness_position'])
    U,Qnom,Qactual=upper(case,F(1,100))
    envelope=private['public_upper_large_budget']
    assert F(envelope['position_upper'])==U
    assert F(envelope['nominal_acceleration_abs_bound'])==Qnom
    assert F(envelope['actual_acceleration_abs_bound'])==Qactual
    assert F(envelope['epsilon'])==F(1,100)
    limit=F(case['limit']);policy=private['threshold_policy']
    assert private['public_envelope_proves_safe']==(U<=limit)
    assert private['witness_proves_unsafe']==(F(private['future_witness_position'][0])>limit)
    if private['witness_proves_unsafe']:assert position[0]>limit
    if policy==0:
        assert private['generation_label']=='CERTIFIED_SAFE_PUBLIC_ENVELOPE' and limit==U+F(1,100)
    elif policy==1:
        assert private['generation_label']=='CERTIFIED_UNSAFE_COMMON_GENERATING_WITNESS'
        assert limit==F(private['future_witness_position'][0])-F(1,1000) and position[0]>limit
    else:
        assert policy in (2,3) and private['generation_label']=='NEAR_BOUNDARY_UNCERTIFIED_DIFFICULTY_LABEL'
        assert private['numeric_reference']['certificate'] is False
    return dict(tau=str(tau),beta=str(beta),independent_history_in_original_packet=True,
                independent_reported_nominal_and_actual_enclosures=True,
                independent_future_witness_enclosure=True,threshold_policy=policy,
                certified_public_safe=U<=limit,certified_generating_unsafe=position[0]>limit,
                near_label_not_truth=policy in (2,3))


def development_self_test():
    cases=json.loads((ROOT/'protocol/development/public_fixtures.json').read_text())
    private=json.loads((ROOT/'protocol/development/private_witnesses.json').read_text())
    checks=[check_witness(case,row) for case,row in zip(cases,private)]
    return dict(status='PASS',scope='GENERATION_AUDITOR_DEVELOPMENT_ONLY',records=len(checks),
                formal_suite_opened=False,generator_imported_or_called=False,
                source_sha256=digest(__file__),checks=checks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--development-self-test',action='store_true');args=parser.parse_args()
    started=time.perf_counter()
    if args.development_self_test:
        report=development_self_test();name='GENERATION_AUDITOR_SELF_TEST.json'
    else:
        directory=ROOT/'protocol/frozen_suite';freeze_path=ROOT/'protocol/METHODS_FROZEN.json'
        freeze=json.loads(freeze_path.read_text());assert freeze['status']=='METHODS_FROZEN'
        for name,expected in freeze['source_sha256'].items():assert digest(ROOT/name)==expected,(name,'frozen source drift')
        assert digest(ROOT/'protocol/generate_taskface_suite.py')==freeze['generator_sha256']
        assert digest(ROOT/'protocol/FINITE_COMPARISON_PROTOCOL_CN.txt')==freeze['protocol_sha256']
        manifest=json.loads((directory/'GENERATION_MANIFEST.json').read_text())
        assert manifest['method_freeze_sha256']==digest(freeze_path)
        assert manifest['generator_sha256']==freeze['generator_sha256'] and manifest['protocol_sha256']==freeze['protocol_sha256']
        assert manifest['status']=='GENERATED_ALL_PREDECLARED_CASES' and manifest['formal_cases_generated']==32
        assert manifest['formal_solver_called'] is False and manifest['redraws']==0
        for name,expected in manifest['output_hashes'].items():assert digest(directory/name)==expected
        cases=json.loads((directory/'public_cases_32.json').read_text())
        private=json.loads((directory/'private_generation_witnesses.json').read_text())
        groups=json.loads((directory/'evaluation_groups.json').read_text())
        assert len(cases)==len(groups)==32 and len(private)==16
        assert [c['name'] for c in cases]==[f'case_{i:03d}' for i in range(32)]
        checks=[];family_counts=Counter();policy_counts=Counter()
        for tid,row in enumerate(private):
            assert row['template_id']==tid
            c0,c1=cases[2*tid:2*tid+2];g0,g1=groups[2*tid:2*tid+2]
            normalized=[]
            for c,g,b in ((c0,g0,F(1,1000)),(c1,g1,F(1,100))):
                assert set(c)==FIELDS and set(c['response'])=={'kind','epsilon_x','L'}
                assert c['response']['kind']=='normalized_global_cubic_v1' and F(c['response']['epsilon_x'])==b
                assert F(c['rho'])==0 and F(c['H'])>0
                assert sum(F(dt) for dt,_ in c['past'])>0
                assert F(c['response']['L'])==sum(F(dt) for dt,_ in c['past'])+F(c['H'])
                assert all(F(lo)<F(hi) for lo,hi in c['packet'])
                assert all(F(dt)>0 for dt,_ in c['past']+c['future'])
                assert sum(F(dt) for dt,_ in c['future'])==F(c['H'])
                assert g['template_id']==tid and g['name']==c['name'] and F(g['epsilon'])==b
                assert g['family']==row['design']['family'] and g['threshold_policy']==row['threshold_policy']
                value=deepcopy(c);del value['name'];del value['response']['epsilon_x'];normalized.append(value)
                checks.append(dict(name=c['name'],template_id=tid,**check_witness(c,row)))
            assert normalized[0]==normalized[1],'Nested classes must use identical other public data'
            beta=F(row['history_witness']['beta']);assert abs(beta) in (F(1,4000),F(3,4000))
            family_counts[row['design']['family']]+=1;policy_counts[row['threshold_policy']]+=1
        assert family_counts==Counter(ordinary=8,four_hold_acceleration=4,five_hold_velocity_valley=2,five_hold_position_valley=2)
        assert policy_counts==Counter({0:4,1:4,2:4,3:4})
        assert Counter(len(cases[2*tid]['past']) for tid,row in enumerate(private) if row['design']['family']=='ordinary')==Counter({1:2,2:2,4:2,8:2})
        sources=[Path(__file__),ROOT/'protocol/generate_taskface_suite.py',ROOT/'protocol/FINITE_COMPARISON_PROTOCOL_CN.txt',freeze_path,
                 directory/'GENERATION_MANIFEST.json',directory/'public_cases_32.json',directory/'private_generation_witnesses.json',directory/'evaluation_groups.json',
                 OLD/'qa/independent_ma_reference.py']
        report=dict(status='PASS',scope='INDEPENDENT_GENERATED_INPUT_AND_FREEZE_QA',public_cases=32,base_templates=16,
                    family_counts=dict(family_counts),threshold_policy_counts=dict(policy_counts),checks=checks,
                    source_sha256={str(p):digest(p) for p in sources},generator_imported_or_called=False,
                    solver_called=False,numerical_near_labels_certified=False,physics_membership_proved=False)
        name='GENERATED_INPUT_AUDIT.json'
    report['elapsed_seconds']=time.perf_counter()-started
    with (ROOT/'audit'/name).open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({key:value for key,value in report.items() if key not in ('checks','source_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
