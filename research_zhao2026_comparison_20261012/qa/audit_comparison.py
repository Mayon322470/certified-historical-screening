"""Independent checks for detailed Ma2026/Zhao2026 comparison certificates.

No production module is imported. The old structural auditors are read-only,
SHA256 bound dependencies. New Ma algebra is independently reconstructed.
The benchmark-record loader is intentionally separate from these predicates.
"""
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
import independent_ma_reference as ref

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent/'research_efficiency_20261012'/'implementation'
AUDIT_HASHES = {
    'audit_efficiency': '8398a1e4e0ea0ea8c9cb74da446b7ebc9587d26b1c8af2207e2c616e054d9ac8',
    'audit_refined': 'fa71c95f6c51f0fc7fe12e48351d3813ca905ca1f2be0815452dbcfe00c8c69c',
}


def structural_auditors():
    loaded = []
    for name, expected in AUDIT_HASHES.items():
        path = OLD/(name+'.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name,path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        loaded.append(sys.modules[name])
    return loaded


@lru_cache(maxsize=1000)
def models(serialized):
    return ref.endpoint_terms(json.loads(serialized))


def get_models(case):
    return models(json.dumps({'initial':case['initial'],'past':case['past']},sort_keys=True))


def interval_equal(a,b):
    assert tuple(map(Q,a)) == tuple(map(Q,b)), (a,b)


def check_contract(case, cert):
    assert cert['kind'] == 'ma2026_scalar'
    tl,th = map(Q,cert['theta_input'])
    assert 0 < tl < th
    ql,qh = Q(1)/th,Q(1)/tl
    interval_equal(cert['q_input'],(ql,qh))
    c,r = (ql+qh)/2,(qh-ql)/2
    assert Q(cert['center'])==c and Q(cert['radius'])==r
    pieces = cert['hessian_pieces']
    assert isinstance(pieces,int) and pieces >= 1
    expressions = get_models(case)
    assert len(cert['strips']) == 3
    strips = []
    options = [(r,'prior',(ql,qh))]
    incompatible = None
    for j,(expression,strip) in enumerate(zip(expressions,cert['strips'])):
        assert strip['coordinate']==j
        value = tuple(map(Q,strip['value']))
        jacobian = tuple(map(Q,strip['derivative']))
        hessian = tuple(map(Q,strip['hessian']))
        assert ref.contained(ref.natural_range(expression,c,c),value), ('center',j)
        assert ref.contained(ref.natural_range(ref.derivative(expression),c,c),jacobian), ('jacobian',j)
        subbounds=[]
        for k in range(pieces):
            low = ql+(qh-ql)*Q(k,pieces)
            high = ql+(qh-ql)*Q(k+1,pieces)
            subbounds.append(ref.strong_range(ref.derivative(expression,2),low,high))
        reference_hessian = min(v[0] for v in subbounds),max(v[1] for v in subbounds)
        if not ref.contained(reference_hessian,hessian):
            assert ref.prove_range_contained(ref.derivative(expression,2),ql,qh,hessian)
        C = sum(jacobian)/2
        assert Q(strip['C'])==C
        rounding = (jacobian[1]-jacobian[0])*r/2
        assert Q(strip['slope_rounding'])==rounding
        rem = min(Q(0),hessian[0])*r*r/2,max(Q(0),hessian[1])*r*r/2
        interval_equal(strip['remainder'],rem)
        bl,bh = map(Q,case['packet'][j])
        low = bl-value[1]-rem[1]-rounding+C*c
        high = bh-value[0]-rem[0]+rounding+C*c
        assert Q(strip['low'])==low and Q(strip['high'])==high and low<=high
        simple = dict(C=C,b=(low+high)/2,s=(high-low)/2)
        strips.append(simple)
        if C:
            ends = tuple(sorted((low/C,high/C)))
            interval_equal(strip['q_interval'],ends)
            options.append(((high-low)/(2*abs(C)),j,ends))
        elif not low<=0<=high:
            incompatible=j
    best = min(options,key=lambda v:v[0])
    assert cert['lp_choice']==best[1]
    assert Q(cert['lp_radius'])==best[0]
    interval_equal(cert['lp_q_interval'],best[2])
    # Verify the independent exact optimum, including a valid gain equality.
    optimum=ref.scalar_lp((ql,qh),strips)
    assert optimum['radius']==best[0]
    ref.validate_gain((ql,qh),strips,optimum['T'],optimum['N'],optimum['center'],optimum['radius'])
    low,high=max(ql,best[2][0]),min(qh,best[2][1])
    if cert['strengthened']:
        for strip in cert['strips']:
            if Q(strip['C']):
                low=max(low,Q(strip['q_interval'][0]))
                high=min(high,Q(strip['q_interval'][1]))
    interval_equal(cert['q_intersection'],(low,high))
    assert cert['zero_row_inconsistent']==incompatible
    rejected=incompatible is not None or low>high
    assert cert['rejected']==rejected
    if rejected:
        assert 'theta_output' not in cert
        return None
    assert 0<low<=high
    answer=Q(1)/high,Q(1)/low
    interval_equal(cert['theta_output'],answer)
    assert tl<=answer[0]<=answer[1]<=th
    return answer


def endpoint(case,theta):
    q=Q(1)/Q(theta)
    return tuple(ref.natural_range(expr,q,q) for expr in get_models(case))


def flow(theta,state,dt,u,rho):
    theta,dt,u,rho=map(Q,(theta,dt,u,rho))
    if dt==0:
        return state
    e=ref.exp_minus(dt/theta)
    one=1-e[1],1-e[0]
    eq=u+theta*rho
    delta=ref.add(state[2],(-eq,-eq))
    b=ref.add((theta*dt,theta*dt),ref.scale(one,-theta*theta))
    x=ref.add(ref.add(state[0],ref.scale(state[1],dt)),(eq*dt*dt/2,eq*dt*dt/2))
    x=ref.add(x,ref.mul(delta,b))
    v=ref.add(ref.add(state[1],(eq*dt,eq*dt)),ref.scale(ref.mul(delta,one),theta))
    a=ref.add((eq,eq),ref.mul(delta,e))
    # Independent 208-bit outward rounding keeps long timelines manageable.
    # Production has 96 bits, so this remains a materially sharper reference.
    scale_factor=2**208
    result=[]
    for lo,hi in (x,v,a):
        result.append((Q(lo.numerator*scale_factor//lo.denominator,scale_factor),
                       Q(-((-hi.numerator*scale_factor)//hi.denominator),scale_factor)))
    return tuple(result)


def future(case,theta,timepoint):
    state=endpoint(case,theta)
    elapsed=Q(0)
    for duration,command in case['future']:
        dt=min(Q(duration),Q(timepoint)-elapsed)
        if dt>0:
            state=flow(theta,state,dt,Q(command),Q(case['rho']))
            elapsed+=dt
        if elapsed==timepoint:
            break
    assert elapsed==timepoint
    return state


@lru_cache(maxsize=10000)
def future_series(serialized,theta,nt):
    case=json.loads(serialized)
    theta=Q(theta)
    state=endpoint(case,theta)
    times=[Q(case['H'])*Q(j,nt) for j in range(nt+1)]
    elapsed=Q(0);j=0
    end=Q(case['future'][0][0])
    series=[]
    for t in times:
        while elapsed<t:
            while elapsed==end and j+1<len(case['future']):
                j+=1
                end+=Q(case['future'][j][0])
            stop=min(t,end)
            assert stop>elapsed
            state=flow(theta,state,stop-elapsed,Q(case['future'][j][1]),Q(case['rho']))
            elapsed=stop
        series.append(state)
    return tuple(series)


def history_curvature(case):
    tm=Q(case['prior'][0])
    duration=sum((Q(dt) for dt,u in case['past']),Q(0))
    acceleration=[Q(case['initial'][2])]+[Q(u) for dt,u in case['past']]
    A0=max(map(abs,acceleration))
    difference=max(acceleration)-min(acceleration)
    e=min(Q(1),duration/tm)
    La=difference*e/tm
    Ma=2*difference*e*(1+e)/(tm*tm)
    return dict(A0=A0,La=La,M=(duration*duration*Ma/2,duration*Ma,Ma))


def linear(theta,l,r,xl,xr):
    assert l<r
    return ((r-theta)*xl+(theta-l)*xr)/(r-l)


def clip(interval,l,r,yl,yr,lower,upper):
    a,b=interval
    slope=(yr-yl)/(r-l)
    if slope==0:
        return (a,b) if lower<=yl<=upper else None
    points=l+(lower-yl)/slope,l+(upper-yl)/slope
    a,b=max(a,min(points)),min(b,max(points))
    return None if a>b else (a,b)


def check_geometry(case,item,is_deletion=False,nt=None):
    """Rebuild projected chord geometry and future bounds independently."""
    l,r=map(Q,item['theta_interval'] if is_deletion else item['interval'])
    endpoint_bounds=item['endpoint_state_bounds']
    assert len(endpoint_bounds)==2
    endpoint_bounds=[[tuple(map(Q,v)) for v in side] for side in endpoint_bounds]
    assert all(len(v)==3 for v in endpoint_bounds)
    for theta,reported in zip((l,r),endpoint_bounds):
        for actual,outer in zip(endpoint(case,theta),reported):
            assert ref.contained(actual,outer),('endpoint_bound',theta)
    if l==r:
        assert not is_deletion
        interval_equal(item['projection'],(l,r))
        assert all(Q(v)==0 for v in item['radii'])
        series=future_series(json.dumps(case,sort_keys=True),l,nt)
        assert Q(item['max_node_upper'])>=max(state[0][1] for state in series)
        return
    bounds=history_curvature(case)
    width=r-l
    mids=[[sum(v)/2 for v in side] for side in endpoint_bounds]
    radii=[bounds['M'][j]*width*width/8+
           max((endpoint_bounds[k][j][1]-endpoint_bounds[k][j][0])/2 for k in (0,1))
           for j in range(3)]
    projection=l,r
    for j,(packet_low,packet_high) in enumerate(case['packet']):
        projection=clip(projection,l,r,mids[0][j],mids[1][j],Q(packet_low)-radii[j],Q(packet_high)+radii[j])
        if projection is None:
            break
    if is_deletion:
        assert projection is None
        return
    assert projection is not None
    interval_equal(item['projection'],projection)
    assert list(map(Q,item['radii']))==radii
    key=json.dumps(case,sort_keys=True)
    left=future_series(key,l,nt)
    right=future_series(key,r,nt)
    U=max(abs(Q(u)) for _,u in case['future'])
    rho=Q(case['rho'])
    mx,mv,ma=bounds['M']
    rx,rv,ra=radii
    node_bounds=[]
    for j in range(nt+1):
        s=Q(case['H'])*Q(j,nt)
        bmax=s*s/2
        Mpsi=mx+s*mv+bmax*ma+2*s*bounds['La']+2*(bounds['A0']+U+rho*s)
        high=max(linear(theta,l,r,left[j][0][1],right[j][0][1]) for theta in projection)
        node_bounds.append(high+Mpsi*width*width/8+2*rx+2*s*rv+2*bmax*ra)
    assert Q(item['max_node_upper'])>=max(node_bounds), 'future_chord_upper'


def check_nonempty(case,witness):
    if witness is None:
        return
    theta=Q(witness['theta'])
    assert Q(case['prior'][0])<=theta<=Q(case['prior'][1])
    reference=endpoint(case,theta)
    reported=[tuple(map(Q,v)) for v in witness['historical_state']]
    assert len(reported)==3
    for actual,outer,packet in zip(reference,reported,case['packet']):
        assert ref.contained(actual,outer)
        assert ref.contained(outer,tuple(map(Q,packet)))


def check_unsafe(case,witness,nt):
    if witness is None:
        return
    theta,t=Q(witness['theta']),Q(witness['time'])
    assert Q(case['prior'][0])<=theta<=Q(case['prior'][1])
    H=Q(case['H'])
    assert 0<=t<=H
    assert H==0 or (t*nt/H).denominator==1
    for actual,packet in zip(endpoint(case,theta),case['packet']):
        assert ref.contained(actual,tuple(map(Q,packet)))
    actual=future(case,theta,t)
    assert Q(case['limit'])<Q(witness['position_lower'])<=actual[0][0]


def check_result(case,result,exp_intervals=None,bits=96):
    """Validate named frontends, structural removals, coverage and semantics.

    Common-backend geometry verification is added separately when endpoint
    enclosure certificates are available. A return value is a count, not a
    claim to validate missing backend fields or unrecorded timing.
    """
    method=result['method']
    assert method in ('ma_strip','ma_strip_structured','zhao1','zhao2','zhao2t3',
                      'zhao2t3_structured','zhao2t3_sturm','zhao2t3_sparse')
    mode=method.split('_')[-1]
    use_sturm=mode in ('structured','sturm')
    use_sparse=mode in ('structured','sparse')
    structured=use_sturm or use_sparse
    is_zhao=method.startswith('zhao')
    strengthened=method.startswith('ma_strip')
    n,nt=result['n'],result['nt']
    assert isinstance(n,int) and n>0 and isinstance(nt,int) and nt>0
    a,b=map(Q,case['prior'])
    assert 0<a<b
    grid=[a+(b-a)*Q(i,n) for i in range(n+1)]
    contracts={}
    stats=Counter()
    for record in result['frontend_certificates']:
        i=record['index'];cert=record['certificate']
        assert i not in contracts and 0<=i<n
        interval_equal(cert['theta_input'],(grid[i],grid[i+1]))
        assert cert['hessian_pieces']==result['hessian_pieces']
        if is_zhao:
            from audit_zhao_contract import check_contract as check_zhao_contract
            assert cert['nu']==(1 if method.startswith('zhao1') else 2)
            assert cert['tighten']==method.startswith('zhao2t3')
            contracts[i]=check_zhao_contract(case,cert)
            stats['zhao_contracts']+=1
        else:
            assert cert['strengthened']==strengthened
            contracts[i]=check_contract(case,cert)
            stats['ma_contracts']+=1
    assert len(contracts)==result['frontend_calls']
    previous,refined=structural_auditors()
    if use_sparse:
        previous.check_sparse_model(case,result['sparse_model'])
    else:
        assert result['sparse_model'] is None
    if use_sturm:
        previous.check_sturm_model(case,result['sturm_model'])
    else:
        assert result['sturm_model'] is None
    visited=set()
    for deletion in result['deletions']:
        i=deletion['index'];engine=deletion['engine']
        assert i not in visited and 0<=i<n
        visited.add(i)
        if engine in ('ma','zhao'):
            assert engine==('zhao' if is_zhao else 'ma')
            assert contracts[i] is None
            assert result['frontend_certificates'][deletion['certificate_index']]['index']==i
        elif engine in ('sparse','sturm'):
            assert structured and i not in contracts
            cert=deletion['certificate']
            interval_equal(cert['theta_interval'],(grid[i],grid[i+1]))
            if engine=='sparse':
                assert use_sparse
                if mode=='sparse':
                    assert cert['kind']!='curvature'
                refined.check_refined_removal(cert,result['sparse_model'])
            else:
                assert use_sturm
                meta=result['sturm_model']
                assert meta['status']=='READY'
                h=Q(meta['h']);tl,tr=map(Q,cert['theta_interval'])
                zl,zh=map(Q,cert['z_interval'])
                assert zl<zh
                assert zl<=ref.exp_minus(h/tl)[0]
                assert zh>=ref.exp_minus(h/tr)[1]
                face=meta['faces'][cert['face_index']]
                assert cert['face_name']==face['name']
                polynomial=previous.tidy(face['violation_polynomial'])
                assert previous.pvalue(polynomial,zl)>0 and previous.pvalue(polynomial,zh)>0
                chain=[previous.tidy(p) for p in face['sturm']['chain']]
                vl,sl=previous.changes(chain,zl);vh,sh=previous.changes(chain,zh)
                assert vl==vh and cert['variation']==[vl,vh] and cert['signs']==[sl,sh]
        else:
            assert engine=='common_chord'
            assert contracts[i] is not None
            interval_equal(deletion['theta_interval'],contracts[i])
            check_geometry(case,deletion,is_deletion=True)
        stats[engine+'_deletions']+=1
    for cell in result['cells']:
        i=cell['index']
        assert i not in visited and 0<=i<n
        visited.add(i)
        assert contracts[i] is not None
        interval_equal(cell['interval'],contracts[i])
        l,r=contracts[i]
        gl,gh=map(Q,cell['projection'])
        assert l<=gl<=gh<=r
        assert len(cell['radii'])==3 and all(Q(v)>=0 for v in cell['radii'])
        check_geometry(case,cell,nt=nt)
        stats['retained_cells']+=1
    assert len(result['cells'])==result['retained']
    assert len(result['deletions'])==result['deleted']
    assert visited==set(range(result['completed_cells']))
    assert set(contracts)<=visited
    assert result['grid_complete']==(result['completed_cells']==n)
    U=max(abs(Q(u)) for _,u in case['future'])
    Ap=max(abs(Q(v)) for v in case['packet'][2])
    allowance=(max(Ap,U)+Q(case['rho'])*b)*(Q(case['H'])/nt)**2/8
    assert Q(result['time_allowance'])==allowance
    check_nonempty(case,result['nonempty_witness'])
    check_unsafe(case,result['witness'],nt)
    stats['nonempty_witnesses']+=result['nonempty_witness'] is not None
    stats['unsafe_witnesses']+=result['witness'] is not None
    status=result['status']
    if result['grid_complete']:
        expected_upper=max((Q(v['max_node_upper']) for v in result['cells']),default=None)
        if expected_upper is not None:
            expected_upper+=allowance
        assert result['upper'] is None if expected_upper is None else Q(result['upper'])==expected_upper
    else:
        assert result['upper'] is None
    if status=='CERTIFIED_SAFE':
        assert result['grid_complete'] and result['retained']>0
        assert Q(result['upper'])<=Q(case['limit']) and result['witness'] is None
    elif status=='CERTIFIED_UNSAFE':
        assert result['witness'] is not None and result['nonempty_witness'] is not None
    elif status=='INCONSISTENT':
        assert result['grid_complete'] and result['retained']==0 and result['witness'] is None
        assert result['nonempty_witness'] is None
    else:
        assert status=='UNKNOWN' and result['witness'] is None
        assert not result['grid_complete'] or (result['retained']>0 and Q(result['upper'])>Q(case['limit']))
    return stats
