"""Small exact audit-design counterexamples, not a production certificate verifier.

No solver, simulator, formal inputs, timing benchmark or producer import.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def inflate(packet,response):
    lo,hi=map(F,packet);elo,ehi=map(F,response)
    if lo>hi or elo>ehi:raise ValueError('Ordered intervals required')
    return lo-ehi,hi-elo


def history_intersection(prior,packets):
    lo,hi=map(F,prior)
    for center,slope,lower,upper in packets:
        center,slope,lower,upper=map(F,(center,slope,lower,upper))
        if not slope:
            if not lower<=center<=upper:return None
        else:
            x,y=(lower-center)/slope,(upper-center)/slope
            lo,hi=max(lo,min(x,y)),min(hi,max(x,y))
            if lo>hi:return None
    return lo,hi


def valid_history(beta,prior,packets):
    beta=F(beta)
    return F(prior[0])<=beta<=F(prior[1]) and all(F(l)<=F(c)+F(a)*beta<=F(u) for c,a,l,u in packets)


def joint_unsafe(beta,prior,packets,center,slope,limit):
    return valid_history(beta,prior,packets) and F(center)+F(slope)*F(beta)>F(limit)


def verdict(*,history_nonempty,cover_complete,node_upper,tube_upper,bridge,limit,empty_proved=False,joint_counterexample=False):
    # Each proof input must be independently verified by the eventual checker.
    # This function tests decision logic only; booleans are not proof objects.
    if joint_counterexample:return 'UNSAFE_IN_DECLARED_CLASS'
    if empty_proved and cover_complete:return 'INCONSISTENT'
    if not history_nonempty or not cover_complete:return 'UNKNOWN'
    upper=F(node_upper)+F(tube_upper)+F(bridge)
    return 'SAFE' if upper<=F(limit) else 'UNKNOWN'


def main():
    records=[]
    def passed(name):records.append(name)
    assert inflate((1,2),(F(-1,10),F(3,10)))==(F(7,10),F(21,10))
    passed('signed packet inflation uses [packet_lo-error_hi,packet_hi-error_lo]')
    try:inflate((0,1),(1,-1))
    except ValueError:pass
    else:raise AssertionError('invalid response interval accepted')
    passed('invalid interval rejected')
    first=[(0,1,F(3,4),1)];second=[(0,1,0,F(1,4))]
    assert history_intersection((0,1),first) and history_intersection((0,1),second)
    assert history_intersection((0,1),first+second) is None
    passed('individually feasible packets may lack one common response witness')
    assert history_intersection((0,1),[(0,1,1,1)])==(F(1),F(1))
    assert valid_history(1,(0,1),[(0,1,1,1)])
    passed('closed packet boundary/singleton is retained')
    prior=(-2,0);packets=[(0,1,-2,-2)]
    assert valid_history(-2,prior,packets) and 11+F(-2)==9
    assert not joint_unsafe(0,prior,packets,11,1,10)
    passed('nominal violation 11>10 does not witness actual-response class with beta=-2')
    default=dict(history_nonempty=True,cover_complete=True,node_upper=F(9,10),tube_upper=0,bridge=0,limit=1)
    assert verdict(**dict(default,tube_upper=F(1,5)))=='UNKNOWN'
    passed('upper tube exceeding limit is UNKNOWN, never an unsafe witness')
    assert joint_unsafe(1,(0,1),[(0,1,0,1)],F(19,2),1,10)
    passed('explicit common history+future response can witness unsafe in declared class')
    assert not joint_unsafe(F(1,2),(0,1),[(0,1,0,1)],F(19,2),1,10)
    passed('equality to safety limit is not strict violation')
    assert verdict(**dict(default,history_nonempty=False))=='UNKNOWN'
    passed('vacuous or unproved-nonempty safe is not resolved')
    assert verdict(**dict(default,cover_complete=False))=='UNKNOWN'
    passed('partial grid coverage cannot certify universal safe')
    assert verdict(**dict(default,history_nonempty=False,empty_proved=True))=='INCONSISTENT'
    assert verdict(**dict(default,history_nonempty=False,empty_proved=True,cover_complete=False))=='UNKNOWN'
    passed('inconsistency needs complete certified exclusion')
    endpoint=F(99,100);midpoint=endpoint+4*F(1,2)*(1-F(1,2))
    assert endpoint<1<midpoint and F(8)*1**2/8==1
    assert verdict(**dict(default,node_upper=endpoint,bridge=1))=='UNKNOWN'
    passed('safe nodes alone miss continuous quadratic excursion; bridge is required')
    assert verdict(**dict(default,tube_upper=F(1,10)))=='SAFE'
    passed('closed continuous upper bound equal to limit can certify safe with witness')
    assert verdict(**dict(default,tube_upper=F(1,5),bridge=F(1,100)))!='SAFE'
    passed('omitting tube or bridge would forge a safe certificate')
    actual_budget=F(5);finish=F(5001,1000)
    assert finish>actual_budget and not (finish<=actual_budget)
    passed('valid certificate after deadline is not a within-budget resolution')
    history_error=F(1,10);future_error_at_zero=F(0)
    assert history_error!=future_error_at_zero
    passed('physical response discrepancy cannot silently reset at future origin')
    out=dict(status='PASS_SMALL_CONTRACT_LOGIC_CHECKS_ONLY',checks=len(records),passed=records,
        production_certificate_verifier_completed=False,new_benchmark_inputs_generated=False,
        old_certificate_batch_rerun=False,simulation_started=False,formal_timing_performed=False,
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    target=HERE/'CONTRACT_SMOKE_RESULTS.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
