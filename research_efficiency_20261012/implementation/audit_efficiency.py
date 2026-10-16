"""Independent exact certificate audit for the efficiency comparison.

Does not import certified_core, sparse_faces, efficient_core, or polynomial_contract.
Sparse endpoint models are rebuilt from command jumps, rather than the producer's
state recursion. Exponentials use unscaled Fraction Taylor sums at 200-bit error.
Sturm identities reuse the mathematical checks from the previous independent audit.
Existing chord geometry is checked for exact equality to eager cells, not reimplemented.
This script audits certificates and work counts, not the validity of wall-clock timing.
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

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_METHODS=('eager_chord','lazy_chord','interval_faces','curvature_faces','sturm_faces','hybrid_faces')


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def tidy(p):
    p = list(map(Q,p))
    while len(p)>1 and p[-1]==0:
        p.pop()
    return p or [Q(0)]


def padd(p,q):
    out=[Q(0)]*max(len(p),len(q))
    for i,a in enumerate(p): out[i]+=a
    for i,a in enumerate(q): out[i]+=a
    return tidy(out)


def pmul(p,q):
    out=[Q(0)]*(len(p)+len(q)-1)
    for i,a in enumerate(p):
        for j,b in enumerate(q): out[i+j]+=a*b
    return tidy(out)


def pscale(p,a): return tidy([a*x for x in p])


def pvalue(p,x):
    out=Q(0)
    for a in reversed(p): out=out*x+a
    return out


@lru_cache(maxsize=300000)
def exp_reference(x):
    x=Q(x)
    assert x>=0
    if x==0: return Q(1),Q(1)
    term=total=Q(1); n=0
    while True:
        previous=total; n+=1; term*=(-x)/n; total+=term
        if n>2*x and abs(term)<Q(1,2**200):
            return min(total,previous),max(total,previous)


def ia(a,b): return a[0]+b[0],a[1]+b[1]
def im(a,b):
    ps=[x*y for x in a for y in b]
    return min(ps),max(ps)
def isc(a,c): return im(a,(c,c))
def isin(small,large): return large[0]<=small[0]<=small[1]<=large[1]


def valid_exp(x,pair,bits):
    lo,hi=exp_reference(Q(x))
    return Q(int(pair[0]),2**bits)<=lo<=hi<=Q(int(pair[1]),2**bits)


def independent_sparse(case):
    x0,v0,a0=map(Q,case['initial'])
    segments=[(Q(dt),Q(u)) for dt,u in case['past']]
    assert all(dt>=0 for dt,u in segments)
    active=[(dt,u) for dt,u in segments if dt]
    T=sum((dt for dt,u in active),Q(0))
    area=sum((dt*u for dt,u in active),Q(0))
    clock=Q(0); weighted=Q(0)
    for dt,u in active:
        weighted+=u*dt*(T-clock-dt/2)
        clock+=dt
    terms={}
    if active:
        constant=active[-1][1]
        terms[T]=a0-active[0][1]
        clock=Q(0)
        for j,(dt,u) in enumerate(active[:-1]):
            clock+=dt
            tau=T-clock
            terms[tau]=terms.get(tau,Q(0))+u-active[j+1][1]
    else:
        constant=a0
    terms=sorted((tau,c) for tau,c in terms.items() if c)
    f0=constant-a0
    vs=v0+area; xs=x0+v0*T+weighted; moment=area-a0*T
    packet=[list(map(Q,b)) for b in case['packet']]
    specs=[('a_lower',[-(constant-packet[2][0])],-1),
           ('a_upper',[constant-packet[2][1]],1),
           ('v_lower',[f0,-(vs-packet[1][0])],1),
           ('v_upper',[-f0,vs-packet[1][1]],-1),
           ('x_lower',[-f0,moment,-(xs-packet[0][0])],-1),
           ('x_upper',[f0,-moment,xs-packet[0][1]],1)]
    faces=[dict(name=name,poly=poly,terms=[(tau,sgn*c) for tau,c in terms])
           for name,poly,sgn in specs]
    return dict(faces=faces,sparse_terms=len(terms),past_holds=len(segments),
                moments=dict(T=T,Q=moment,v_star=vs,x_star=xs)),constant,terms


def check_sparse_model(case,meta):
    expected,constant,terms=independent_sparse(case)
    assert meta['sparse_terms']==expected['sparse_terms']
    assert meta['past_holds']==expected['past_holds']
    assert {k:Q(v) for k,v in meta['moments'].items()}==expected['moments']
    assert len(meta['faces'])==6
    for face,want in zip(meta['faces'],expected['faces']):
        assert face['name']==want['name']
        assert list(map(Q,face['poly']))==want['poly']
        assert [tuple(map(Q,t)) for t in face['terms']]==want['terms']


def face_key(face):
    return tuple(map(Q,face['poly'])),tuple(tuple(map(Q,t)) for t in face['terms'])


@lru_cache(maxsize=300000)
def sparse_range(key,a,b,order=0):
    poly,terms=key
    poly=list(poly)
    for _ in range(order):
        poly=[i*poly[i] for i in range(1,len(poly))] or [Q(0)]
    p=poly+[Q(0)]*(3-len(poly))
    vals=[pvalue(p,a),pvalue(p,b)]
    if p[2]:
        vertex=-p[1]/(2*p[2])
        if a<vertex<b: vals.append(pvalue(p,vertex))
    out=(min(vals),max(vals))
    for tau,c in terms:
        e=(exp_reference(tau*b)[0],exp_reference(tau*a)[1])
        out=ia(out,isc(e,c*(-tau)**order))
    return out


def independent_second(key,a,b):
    direct=sparse_range(key,a,b,2)
    c=(a+b)/2; r=(b-a)/2
    centered=sparse_range(key,c,c,2)
    for j in range(1,5):
        power=(-r**j,r**j) if j%2 else (Q(0),r**j)
        centered=ia(centered,isc(im(sparse_range(key,c,c,2+j),power),Q(1,factorial(j))))
    remainder=isc(im(sparse_range(key,a,b,7),(-r**5,r**5)),Q(1,factorial(5)))
    centered=ia(centered,remainder)
    result=max(direct[0],centered[0]),min(direct[1],centered[1])
    assert result[0]<=result[1]
    return result


def check_sparse_removal(cert,meta):
    l,r=map(Q,cert['theta_interval'])
    assert 0<l<r
    a,b=map(Q,cert['q_interval'])
    assert (a,b)==(1/r,1/l)
    face=meta['faces'][cert['face_index']]
    assert face['name']==cert['face_name']
    key=face_key(face); kind=cert['kind']; claimed=Q(cert['certified_lower'])
    assert claimed>0
    if kind=='natural':
        assert claimed<=sparse_range(key,a,b)[0]
    elif kind=='monotone_increasing':
        assert Q(cert['point'])==a
        assert sparse_range(key,a,b,1)[0]>=0
        assert claimed<=sparse_range(key,a,a)[0]
    elif kind=='monotone_decreasing':
        assert Q(cert['point'])==b
        assert sparse_range(key,a,b,1)[1]<=0
        assert claimed<=sparse_range(key,b,b)[0]
    elif kind=='curvature':
        mu=Q(cert['mu']); c=Q(cert['center'])
        assert a<=c<=b and 0<mu<=independent_second(key,a,b)[0]
        val=Q(cert['value_lower'])
        assert val<=sparse_range(key,c,c)[0]
        slope=tuple(map(Q,cert['slope_interval']))
        assert isin(sparse_range(key,c,c,1),slope)
        magnitude=max(map(abs,slope))
        assert claimed==val-magnitude*magnitude/(2*mu)
        assert isinstance(cert['stationary_steps'],int) and 0<=cert['stationary_steps']<=32
    else:
        raise AssertionError(kind)


def changes(chain,x):
    signs=[(pvalue(p,x)>0)-(pvalue(p,x)<0) for p in chain]
    nonzero=[v for v in signs if v]
    return sum(a!=b for a,b in zip(nonzero,nonzero[1:])),signs


def check_chain(face):
    p=tidy(face['violation_polynomial'])
    chain=[tidy(a) for a in face['sturm']['chain']]
    steps=face['sturm']['steps']
    assert chain[0]==p
    if len(p)==1:
        assert len(chain)==1 and not steps
        return
    assert chain[1]==tidy([i*p[i] for i in range(1,len(p))])
    assert len(steps)==len(chain)-1
    for i,step in enumerate(steps):
        quotient=tidy(step['quotient'])
        if step['next'] is None:
            assert i==len(steps)-1 and step['scale'] is None
            assert chain[i]==pmul(quotient,chain[i+1])
        else:
            factor=Q(step['scale']); nxt=tidy(step['next'])
            assert factor>0 and nxt==chain[i+2] and len(nxt)<len(chain[i+1])
            assert chain[i]==padd(pmul(quotient,chain[i+1]),pscale(nxt,-factor))


def check_sturm_model(case,meta):
    canonical={
        'initial':[str(Q(x)) for x in case['initial']],
        'past':[[str(Q(x)) for x in pair] for pair in case['past']],
        'prior':[str(Q(x)) for x in case['prior']],
        'packet':[[str(Q(x)) for x in pair] for pair in case['packet']],
        'future':[[str(Q(x)) for x in pair] for pair in case['future']],
        **{k:str(Q(case[k])) for k in ('H','rho','limit')},
    }
    assert meta['input_signature']==hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
    initial=list(map(Q,case['initial']))
    segments=[tuple(map(Q,s)) for s in case['past']]
    h=Q(meta['h']); powers=meta['powers']
    assert h>0 and len(powers)==len(segments)
    assert all(dt==h*m and m>=0 for (dt,u),m in zip(segments,powers))
    assert sum(powers)==meta['degree']
    sparse,constant,terms=independent_sparse(case)
    want=sparse['moments'].copy()
    want['area']=want['Q']+initial[2]*want['T']
    assert {k:Q(v) for k,v in meta['moments'].items()}==want
    if meta['status']!='READY':
        assert not meta['faces'] and meta['status'].startswith('FALLBACK_')
        if meta['status']=='FALLBACK_DEGREE_BUDGET':
            assert meta['degree']>meta['degree_budget']
        return
    # Sparse jump expansion directly supplies the polynomial coefficients.
    p=[Q(0)]*(meta['degree']+1); p[0]=constant
    for tau,c in terms:
        k=tau/h
        assert k.denominator==1
        p[int(k)]+=c
    p=tidy(p)
    assert p==tidy(meta['acceleration_polynomial'])
    packet=[list(map(Q,z)) for z in case['packet']]
    faces={'a_lower':(padd(pscale(p,-1),[packet[2][0]]),0),
           'a_upper':(padd(p,[-packet[2][1]]),0)}
    assert meta['mode']=='aligned_sturm'
    f=padd(p,[-initial[2]])
    if packet[1][0]==want['v_star']: faces['v_lower']=(f,1)
    if packet[1][1]==want['v_star']: faces['v_upper']=(pscale(f,-1),1)
    if want['Q']==0:
        if packet[0][0]==want['x_star']: faces['x_lower']=(pscale(f,-1),2)
        if packet[0][1]==want['x_star']: faces['x_upper']=(f,2)
    assert set(faces)=={f['name'] for f in meta['faces']}
    for face in meta['faces']:
        assert (tidy(face['violation_polynomial']),face['positive_theta_power'])==faces[face['name']]
        check_chain(face)


def check_sturm_removal(cert,meta,exps,bits):
    assert meta['status']=='READY'
    l,r=map(Q,cert['theta_interval']); assert 0<l<r
    h=Q(meta['h']); zl,zu=map(Q,cert['z_interval']); assert zl<zu
    assert zl==Q(int(exps[str(h/l)][0]),2**bits)
    assert zu==Q(int(exps[str(h/r)][1]),2**bits)
    face=meta['faces'][cert['face_index']]
    assert face['name']==cert['face_name']
    p=tidy(face['violation_polynomial'])
    assert pvalue(p,zl)>0 and pvalue(p,zu)>0
    chain=[tidy(p) for p in face['sturm']['chain']]
    vl,sl=changes(chain,zl); vu,su=changes(chain,zu)
    assert vl==vu and [vl,vu]==cert['variation'] and [sl,su]==cert['signs']


def expected_allowance(case,nt):
    packet=max(abs(Q(v)) for v in case['packet'][2])
    command=max(abs(Q(u)) for dt,u in case['future'])
    return (max(packet,command)+Q(case['rho'])*Q(case['prior'][1]))*(Q(case['H'])/nt)**2/8


def check_result(result,case):
    assert result['n']>0 and result['nt']>0
    assert Q(result['limit'])==Q(case['limit'])
    if result['retained']:
        assert Q(result['time_allowance'])==expected_allowance(case,result['nt'])
    else:
        assert Q(result['time_allowance'])==0
    status=result['status']
    if status=='CERTIFIED_SAFE':
        assert result['retained']>0 and Q(result['upper'])<=Q(case['limit'])
        assert result['witness'] is None
    elif status=='CERTIFIED_UNSAFE':
        assert result['retained']>0 and result['witness'] is not None
        assert Q(result['upper'])>Q(case['limit'])
        assert Q(result['witness']['position_lower'])>Q(case['limit'])
    elif status=='INCONSISTENT':
        assert result['retained']==0 and result['upper'] is None
    else:
        assert status=='UNKNOWN' and result['retained']>0 and result['witness'] is None
        assert Q(result['upper'])>Q(case['limit'])


def future_intervals(case,nt):
    horizon=Q(case['H'])
    events={horizon*Q(i,nt) for i in range(nt+1)}
    clock=Q(0)
    for dt,u in case['future']:
        clock+=Q(dt)
        if 0<clock<horizon: events.add(clock)
    assert clock>=horizon
    return len(events)-1


def check_work(record,case):
    result=record['result']; work=record['work']
    history=result['historical_vertices']*len(case['past'])
    future=result['future_vertices']*future_intervals(case,result['nt'])
    assert work.get('history_flow_calls',0)==history
    assert work.get('future_flow_calls',0)==future
    assert work.get('flow_calls',0)==history+future
    assert result['requested_time_intervals']==result['nt']
    assert work['exponential_evaluations']>=0 and work['exponential_cache_hits']>=0


def endpoint_reference(case,theta):
    meta,constant,terms=independent_sparse(case)
    accel=(constant,constant)
    for tau,c in terms: accel=ia(accel,isc(exp_reference(tau/theta),c))
    f=ia(accel,(-Q(case['initial'][2]),-Q(case['initial'][2])))
    moments=meta['moments']
    velocity=ia((moments['v_star'],moments['v_star']),isc(f,-theta))
    base=moments['x_star']-theta*moments['Q']
    position=ia((base,base),isc(f,theta*theta))
    return position,velocity,accel


def flow_reference(theta,state,dt,u,rho):
    e=exp_reference(dt/theta); one=(1-e[1],1-e[0])
    eq=u+theta*rho
    delta=ia(state[2],(-eq,-eq))
    b=ia((theta*dt,theta*dt),isc(one,-theta*theta))
    x=ia(ia(state[0],isc(state[1],dt)),(eq*dt*dt/2,eq*dt*dt/2))
    x=ia(x,im(delta,b))
    v=ia(ia(state[1],(eq*dt,eq*dt)),isc(im(delta,one),theta))
    a=ia((eq,eq),im(delta,e))
    return x,v,a


def check_witness(case,result):
    witness=result['witness']
    if witness is None:return
    theta=Q(witness['theta']); timepoint=Q(witness['time'])
    lo,hi=map(Q,case['prior'])
    assert lo<=theta<=hi and 0<=timepoint<=Q(case['H'])
    assert ((theta-lo)*result['n']/(hi-lo)).denominator==1
    assert (timepoint*result['nt']/Q(case['H'])).denominator==1
    state=endpoint_reference(case,theta)
    assert all(isin(v,tuple(map(Q,b))) for v,b in zip(state,case['packet']))
    elapsed=Q(0)
    for duration,command in case['future']:
        dt=min(Q(duration),timepoint-elapsed)
        if dt>0:
            state=flow_reference(theta,state,dt,Q(command),Q(case['rho']))
            elapsed+=dt
        if elapsed==timepoint:break
    assert elapsed==timepoint
    assert Q(witness['position_lower'])<=state[0][0]
    assert state[0][0]>Q(case['limit'])


def reject_test(fn):
    try: fn()
    except (AssertionError,ValueError,KeyError,IndexError,ZeroDivisionError): return True
    return False


def check_input_binding(raw,data,frozen,fixtures):
    digest=hashlib.sha256(raw).hexdigest()
    assert digest==frozen['sha256']==data['input_sha256']
    assert frozen['cases']==len(fixtures)
    assert tuple(data['methods'])==EXPECTED_METHODS


def check_schedule(records,data,frozen,fixtures):
    assert data['schedule_seed']==2026101201 and frozen['repetitions']==3
    primary=tuple(frozen['primary_grid']); extra=tuple(frozen['long_horizon_time_grid'])
    tasks=[(i,*primary) for i in range(len(fixtures))]
    tasks += [(i,*extra) for i in frozen['long_workload_case_indices']]
    expected=[(rep,i,n,nt,m) for rep in range(3) for i,n,nt in tasks for m in EXPECTED_METHODS]
    random.Random(data['schedule_seed']).shuffle(expected)
    assert len(records)==len(expected)
    for sequence,(record,want) in enumerate(zip(records,expected)):
        got=record['rep'],record['case_index'],record['n'],record['nt'],record['method']
        assert record['sequence']==sequence and got==want


def check_cell_relation(result,old):
    oldcells={c['index']:c for c in old['cells']}
    newcells={c['index']:c for c in result['cells']}
    deleted={c['index'] for c in result['deletions']}
    assert set(newcells)==set(oldcells)-deleted
    for i,c in newcells.items(): assert c==oldcells[i]
    assert result['witness']==old['witness']
    if result['retained']:
        assert old['retained'] and Q(result['upper'])<=Q(old['upper'])
        assert result['time_allowance']==old['time_allowance']
    return len(newcells)


def main():
    started=time.perf_counter()
    raw=(ROOT/'inputs/fixtures.json').read_bytes()
    fixtures=json.loads(raw); cases={c['name']:c for c in fixtures}
    assert len(cases)==len(fixtures)
    data=read('implementation/benchmark_results.json')
    frozen=read('inputs/FROZEN_INPUTS.json')
    check_input_binding(raw,data,frozen,fixtures)
    expdata=read('implementation/benchmark_exp_certificates.json')
    exps=expdata['intervals']; bits=expdata['bits']
    assert bits==data['bits']
    for x,pair in exps.items(): assert valid_exp(x,pair,bits),x
    records=data['records']; detailed={}; repeats={}; seen=set()
    check_schedule(records,data,frozen,fixtures)
    decisions=Counter(); deletion_kinds=Counter(); deletion_faces=Counter()
    models_checked=set(); sturm_chains=0; sparse_models=0; compared_cells=0
    dominance_checks=0; witness_checks=0; deletion_total=0
    sparse_example=None; curvature_example=None; chain_example=None
    for record in records:
        name=record['case']; case=cases[name]; result=record['result']
        key=(name,record['n'],record['nt'],record['method'])
        fullkey=key+(record['rep'],)
        assert fullkey not in seen; seen.add(fullkey)
        assert result['method']==record['method'] and result['n']==record['n'] and result['nt']==record['nt']
        assert fixtures[record['case_index']]['name']==name
        assert record['elapsed_seconds']>=0
        check_result(result,case); check_work(record,case)
        decisions[result['status']]+=1
        stable={k:v for k,v in result.items() if k not in ('cells','deletions','sparse_model','sturm_model')}
        if key in repeats:
            assert stable==repeats[key][0] and record['work']==repeats[key][1]
        else: repeats[key]=(stable,record['work'])
        if record['rep']!=0:
            assert 'cells' not in result
            continue
        assert key not in detailed; detailed[key]=record
        assert len(result['cells'])==result['retained']
        assert len(result['deletions'])==result['filtered_cells']
        if result['witness'] is not None:
            check_witness(case,result); witness_checks+=1
        if result['sparse_model'] is not None:
            meta=result['sparse_model']; mk=('sparse',name,json.dumps(meta,sort_keys=True))
            if mk not in models_checked:
                check_sparse_model(case,meta); models_checked.add(mk); sparse_models+=1
            if sparse_example is None: sparse_example=(case,copy.deepcopy(meta))
        if result['sturm_model'] is not None:
            meta=result['sturm_model']; mk=('sturm',name,json.dumps(meta,sort_keys=True))
            if mk not in models_checked:
                check_sturm_model(case,meta); models_checked.add(mk); sturm_chains+=len(meta['faces'])
            if chain_example is None:
                chain_example=next((copy.deepcopy(f) for f in meta['faces'] if len(f['sturm']['chain'])>2),None)
        prior=list(map(Q,case['prior'])); n=result['n']; deleted=[]
        for deletion in result['deletions']:
            index=deletion['index']; cert=deletion['certificate']
            assert 0<=index<n
            interval=[prior[0]+(prior[1]-prior[0])*Q(j,n) for j in (index,index+1)]
            assert list(map(Q,cert['theta_interval']))==interval
            if deletion['engine']=='sparse':
                assert record['method'] in ('interval_faces','curvature_faces','hybrid_faces')
                if record['method']=='interval_faces': assert cert['kind']!='curvature'
                check_sparse_removal(cert,result['sparse_model'])
                deletion_kinds[cert['kind']]+=1
                if cert['kind']=='curvature' and curvature_example is None:
                    curvature_example=(copy.deepcopy(cert),result['sparse_model'])
            else:
                assert deletion['engine']=='sturm' and record['method'] in ('sturm_faces','hybrid_faces')
                check_sturm_removal(cert,result['sturm_model'],exps,bits)
                deletion_kinds['sturm']+=1
            deletion_faces[cert['face_name']]+=1; deletion_total+=1; deleted.append(index)
        assert len(deleted)==len(set(deleted))
        cells={c['index']:c for c in result['cells']}
        assert len(cells)==result['retained']
        assert all(0<=i<n for i in cells)
        needed={k for i in cells for k in (i,i+1)}
        historical={k for i in range(n) if i not in deleted for k in (i,i+1)}
        if record['method']=='eager_chord':
            assert result['future_vertices']==n+1 and result['historical_vertices']==n+1
        else:
            assert result['future_vertices']==len(needed)
            assert result['historical_vertices']==len(historical)
        if cells:
            assert Q(result['upper'])==max(Q(c['max_node_upper']) for c in cells.values())+Q(result['time_allowance'])
    for key,record in detailed.items():
        name,n,nt,method=key; result=record['result']
        old=detailed[(name,n,nt,'eager_chord')]['result']
        compared_cells+=check_cell_relation(result,old)
        if result['retained']:
            dominance_checks+=1
        if method=='lazy_chord':
            assert result['upper']==old['upper'] and result['status']==old['status']
    reps=Counter((r['case'],r['n'],r['nt'],r['method']) for r in records)
    assert set(reps.values())=={3}
    corrupt_exp=0
    for x,pair in list(exps.items())[:12]:
        lo,hi=exp_reference(Q(x)); low=hi.numerator*2**bits//hi.denominator+2
        assert not valid_exp(x,[str(low),str(low+2)],bits); corrupt_exp+=1
    mutations={}
    if sparse_example:
        case,meta=sparse_example; bad=copy.deepcopy(meta)
        bad['faces'][0]['poly'][0]=str(Q(bad['faces'][0]['poly'][0])+1)
        mutations['wrong_sparse_face_coefficient']=reject_test(lambda:check_sparse_model(case,bad))
    if curvature_example:
        cert,meta=curvature_example; bad=copy.deepcopy(cert); bad['mu']='-1'
        mutations['negative_curvature_bound']=reject_test(lambda:check_sparse_removal(bad,meta))
        bad=copy.deepcopy(cert); bad['certified_lower']=str(Q(bad['certified_lower'])+1)
        mutations['wrong_curvature_quadratic_lower']=reject_test(lambda:check_sparse_removal(bad,meta))
        bad=copy.deepcopy(cert); bad['q_interval']=['0','1']
        mutations['wrong_reciprocal_parameter_image']=reject_test(lambda:check_sparse_removal(bad,meta))
    if chain_example:
        bad=copy.deepcopy(chain_example)
        bad['sturm']['steps'][0]['quotient'][0]=str(Q(bad['sturm']['steps'][0]['quotient'][0])+1)
        mutations['wrong_sturm_quotient']=reject_test(lambda:check_chain(bad))
        bad=copy.deepcopy(chain_example); bad['sturm']['steps'][0]['scale']=str(-Q(bad['sturm']['steps'][0]['scale']))
        mutations['negative_sturm_normalization']=reject_test(lambda:check_chain(bad))
    chosen=next(r for r in records if r['result']['retained'])
    bad=copy.deepcopy(chosen['result']); bad['time_allowance']='-1'
    mutations['understated_time_remainder']=reject_test(lambda:check_result(bad,cases[chosen['case']]))
    bad=copy.deepcopy(chosen); bad['work']['future_flow_calls']=bad['work'].get('future_flow_calls',0)+1
    mutations['wrong_future_work_count']=reject_test(lambda:check_work(bad,cases[chosen['case']]))
    bad=copy.deepcopy(data);bad['input_sha256']='0'*64
    mutations['wrong_frozen_input_binding']=reject_test(lambda:check_input_binding(raw,bad,frozen,fixtures))
    mutations['omitted_scheduled_record']=reject_test(lambda:check_schedule(records[:-1],data,frozen,fixtures))
    pair=next((k,r) for k,r in detailed.items() if k[-1]!='eager_chord' and r['result']['retained'])
    k,r=pair;old=detailed[(k[0],k[1],k[2],'eager_chord')]['result']
    bad=copy.deepcopy(r['result']);bad['cells'].pop();bad['retained']-=1
    mutations['unjustified_missing_retained_cell']=reject_test(lambda:check_cell_relation(bad,old))
    for k,r in detailed.items():
        old=detailed[(k[0],k[1],k[2],'eager_chord')]['result']
        old_indices={c['index'] for c in old['cells']}
        deletion_index=next((j for j,d in enumerate(r['result']['deletions']) if d['index'] in old_indices),None)
        if deletion_index is not None:
            bad=copy.deepcopy(r['result']);bad['deletions'].pop(deletion_index)
            mutations['missing_exclusion_certificate']=reject_test(lambda:check_cell_relation(bad,old))
            break
    assert all(mutations.values())
    output=dict(
        status='PASS',cases=len(cases),records=len(records),repetitions=3,
        decision_counts=dict(decisions),detailed_first_repetition_records=len(detailed),
        independently_rebuilt_sparse_models=sparse_models,verified_sturm_chains=sturm_chains,
        independently_verified_deletions=deletion_total,deletions_by_kind=dict(deletion_kinds),
        deletions_by_face=dict(deletion_faces),retained_cells_identical_to_eager=compared_cells,
        nonempty_same_grid_dominance_checks=dominance_checks,
        independently_recomputed_witnesses=witness_checks,
        checked_exponential_enclosures=len(exps),corrupt_exponentials_rejected=corrupt_exp,
        mutations_rejected=mutations,elapsed_seconds=time.perf_counter()-started,
        input_binding_scope='Frozen fixture SHA-256, every Sturm input signature, the complete prescribed task/method/repetition schedule, and every filtered-versus-eager cell set are checked; omissions are mutation-tested.',
        sparse_scope='All six faces rebuilt by command-jump expansion. Natural, monotonicity and strong-convexity removals independently recomputed; F-second derivative uses an independently evaluated fourth-order Taylor enclosure with seventh-derivative remainder.',
        geometry_scope='Every retained cell equals the same-grid eager cell; common packet time allowance and node-max combination are recomputed. The existing chord interpolation geometry is not independently reimplemented.',
        work_scope='History/future flow calls and unique required vertices checked from inputs and cell indices. Counts and all mathematical results agree across three repetitions. General polynomial-operation or bit-cost complexity is not certified.',
        timing_scope='No wall-clock speedup, cold-process isolation, or WCET is certified by this arithmetic audit.',
    )
    (ROOT/'implementation/EFFICIENCY_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    main()
