"""Independent complete result predicates; no producer imports.

Inherited history dynamics use independently reconstructed 220-bit exponential
expressions and a separate 208-bit future flow. Rank-one geometry, common beta,
original-packet witnesses and continuous actual-position bridge are explicit.
"""
from fractions import Fraction as F
from collections import Counter
import hashlib
import json
import audit_comparison as old
import independent_ma_reference as ref
import audit_rank1_ma as am
import audit_rank1_zhao as az
import audit_coupling_faces as ac
import rank1_reference as rr

CACHE={}
COUNTS=Counter()


def pair(value):return tuple(map(F,value))
def equal(a,b):assert pair(a)==pair(b),(a,b)


def contract_context(case):
    T=sum((F(dt) for dt,_ in case['past']),F(0));H=F(case['H']);L=T+H
    b=F(case['response']['epsilon_x']);assert b>=0
    assert F(case['response']['L'])==L and case['response']['kind']=='normalized_global_cubic_v1'
    assert F(case['rho'])==0
    p=rr.cubic_psi(T,L)
    packet=list(map(pair,case['packet']))
    expanded=[(lo-b*pi,hi+b*pi) for (lo,hi),pi in zip(packet,p)]
    return T,H,L,b,p,packet,expanded


def bridge(case,step,reported=None):
    T,H,L,b,p,packet,expanded=contract_context(case)
    U=max(abs(F(u)) for _,u in case['future'])
    Qnom=max(*map(abs,expanded[2]),U);Qresponse=6*b/L**2
    result=(Qnom+Qresponse)*step**2/8
    if reported is not None:
        assert F(reported['nominal_acceleration_bound'])==Qnom
        assert F(reported['response_curvature_bound'])==Qresponse
        assert F(reported['actual_acceleration_bound'])==Qnom+Qresponse
        assert F(reported['allowance'])==result and F(reported['rho'])==0
        assert reported['continuous_contract_required'] is True
    return result,Qnom+Qresponse


def check_nonempty(case,witness):
    if witness is None:return
    T,H,L,b,p,packet,expanded=contract_context(case)
    theta,beta=F(witness['theta']),F(witness['beta'])
    assert F(case['prior'][0])<=theta<=F(case['prior'][1]) and -b<=beta<=b
    nominal=list(map(pair,witness['nominal_historical_state']))
    for actual,outer in zip(old.endpoint(case,theta),nominal):assert ref.contained(actual,outer)
    actual=[(lo+beta*pi,hi+beta*pi) for (lo,hi),pi in zip(nominal,p)]
    assert list(map(pair,witness['actual_historical_state']))==actual
    assert all(ref.contained(actual_row,packet_row) for actual_row,packet_row in zip(actual,packet))
    expected=rr.beta_bounds(packet,nominal,p,b,'inner')
    assert expected is not None and expected[0]<=beta<=expected[1]
    equal(witness['beta_inner'],expected)
    assert list(map(pair,witness['original_packet']))==packet and F(witness['global_history_time'])==T


def check_unsafe(case,witness,nt):
    if witness is None:return
    T,H,L,b,p,packet,expanded=contract_context(case)
    theta,beta,s=map(F,(witness['theta'],witness['beta'],witness['time']))
    assert F(case['prior'][0])<=theta<=F(case['prior'][1]) and -b<=beta<=b and 0<=s<=H
    assert not H or (s*nt/H).denominator==1
    assert F(witness['global_time'])==T+s
    recorded=list(map(pair,witness['actual_historical_state']))
    for nominal,actual,pi,box in zip(old.endpoint(case,theta),recorded,p,packet):
        shifted=nominal[0]+beta*pi,nominal[1]+beta*pi
        assert ref.contained(shifted,actual) and ref.contained(actual,box)
    # The saved beta_inner was based on the producer nominal enclosures.
    # Derive those same enclosures by subtracting the single declared beta.
    nominal=[(lo-beta*pi,hi-beta*pi) for (lo,hi),pi in zip(recorded,p)]
    expected=rr.beta_bounds(packet,nominal,p,b,'inner')
    assert expected is not None and expected[0]<=beta<=expected[1]
    equal(witness['beta_inner'],expected)
    reference=old.future(case,theta,s)[0][0]
    lower=F(witness['nominal_position_lower'])
    assert lower<=reference
    actual_lower=lower+beta*rr.cubic_psi(T+s,L)[0]
    assert F(witness['position_lower'])==actual_lower and actual_lower>F(case['limit'])


def geometry(case,item,kind,nt):
    T,H,L,b,p,packet,expanded=contract_context(case)
    l,r=pair(item['interval'] if kind=='retained' else item['theta_interval'])
    ends=[[pair(v) for v in side] for side in item['endpoint_state_bounds']]
    assert len(ends)==2 and all(len(side)==3 for side in ends)
    for theta,side in zip((l,r),ends):
        for actual,outer in zip(old.endpoint(case,theta),side):assert ref.contained(actual,outer)
    bounds=old.history_curvature(case)
    if l==r:
        assert kind!='common_chord'
        radii=[F(0)]*3;projection=(l,r);nominal=ends[0]
    else:
        w=r-l
        mids=[[sum(v)/2 for v in side] for side in ends]
        radii=[bounds['M'][j]*w*w/8+max((ends[k][j][1]-ends[k][j][0])/2 for k in (0,1)) for j in range(3)]
        projection=(l,r)
        for j,(lo,hi) in enumerate(expanded):
            projection=old.clip(projection,l,r,mids[0][j],mids[1][j],lo-radii[j],hi+radii[j])
            if projection is None:break
        if kind=='common_chord':
            assert projection is None
            return
        assert projection is not None
        nominal=[]
        for j,(lo,hi) in enumerate(expanded):
            values=[old.linear(theta,l,r,mids[0][j],mids[1][j]) for theta in projection]
            nominal.append((max(lo,min(values)-radii[j]),min(hi,max(values)+radii[j])))
    equal(item['projection'],projection);assert list(map(F,item['radii']))==radii
    assert list(map(pair,item['nominal_historical_ranges']))==nominal
    beta=rr.beta_bounds(packet,nominal,p,b,'outer')
    if kind=='common_beta':
        assert beta is None
        return
    assert beta is not None;equal(item['beta_outer'],beta)
    serialized=json.dumps(case,sort_keys=True)
    left=old.future_series(serialized,l,nt);right=old.future_series(serialized,r,nt)
    U=max(abs(F(u)) for _,u in case['future']);node_bounds=[]
    for j in range(nt+1):
        s=H*F(j,nt)
        if l==r:upper=left[j][0][1]
        else:
            mx,mv,ma=bounds['M'];rx,rv,ra=radii
            M=mx+s*mv+s*s*ma/2+2*s*bounds['La']+2*(bounds['A0']+U)
            upper=max(old.linear(theta,l,r,left[j][0][1],right[j][0][1]) for theta in projection)
            upper+=M*(r-l)**2/8+2*rx+2*s*rv+s*s*ra
        node_bounds.append(upper+beta[1]*rr.cubic_psi(T+s,L)[0])
    assert F(item['max_node_upper'])>=max(node_bounds),'future common beta/global-time node upper'


def check_quick(case,result):
    T,H,L,b,p,packet,expanded=contract_context(case)
    _,A=bridge(case,H,result['continuous_bound'])
    expected=packet[0][1]+max(F(0),packet[1][1])*H+A*H*H/2
    assert F(result['quick_upper'])==expected
    assert F(result['midpoint_theta'])==sum(map(F,case['prior']))/2
    check_nonempty(case,result['nonempty_witness'])
    safe=expected<=F(case['limit']) and result['nonempty_witness'] is not None
    assert result['status']==('QUICK_CERTIFIED_SAFE' if safe else 'QUICK_NO_CERTIFICATE')
    assert result['shared_beta_contract'] is True and result['future_vertices']==0


def check_result(case,result):
    method=result['method'];structured=method.endswith('_structured');is_zhao=method.startswith('zhao')
    assert method in ('ma_rank1','ma_rank1_structured','zhao1_rank1','zhao2t3_rank1','zhao2t3_rank1_structured')
    n,nt=result['n'],result['nt'];assert isinstance(n,int) and n>0 and isinstance(nt,int) and nt>0
    T,H,L,b,p,packet,expanded=contract_context(case)
    assert result['response_contract']==case['response'] and result['future_uses_global_response_clock'] is True
    assert list(map(pair,result['expanded_packet']))==expanded
    low,high=pair(case['prior']);grid=[low+(high-low)*F(i,n) for i in range(n+1)]
    contracts={};stats=Counter()
    for row in result['frontend_certificates']:
        i,cert=row['index'],row['certificate'];assert i not in contracts and 0<=i<n
        equal(cert['theta_input'],(grid[i],grid[i+1]));assert cert['hessian_pieces']==1
        key=hashlib.sha256(json.dumps([case,cert],sort_keys=True).encode()).hexdigest()
        if key not in CACHE:
            if is_zhao:
                assert cert['nu']==(1 if method.startswith('zhao1') else 2)
                assert cert['tighten']==method.startswith('zhao2t3')
                CACHE[key]=az.check_contract(case,cert)
            else:CACHE[key]=am.check_contract(case,cert)
            COUNTS['new_contract_replays']+=1
        else:COUNTS['full_input_payload_cache_reuses']+=1
        contracts[i]=CACHE[key];stats['frontend_contracts']+=1
    assert len(contracts)==result['frontend_calls']
    previous,refined=old.structural_auditors()
    outer_case=dict(case,packet=[[str(v) for v in row] for row in expanded])
    if structured:
        previous.check_sturm_model(outer_case,result['sturm_model'])
        previous.check_sparse_model(outer_case,result['sparse_model'])
    visited=set()
    for deletion in result['deletions']:
        i,engine=deletion['index'],deletion['engine'];assert i not in visited and 0<=i<n
        visited.add(i)
        if engine in ('ma','zhao'):
            assert engine==('zhao' if is_zhao else 'ma') and contracts[i] is None
            assert result['frontend_certificates'][deletion['certificate_index']]['index']==i
        elif engine in ('sturm','sparse','coupling'):
            assert structured and i not in contracts
            cert=deletion['certificate'];equal(cert['theta_interval'],(grid[i],grid[i+1]))
            if engine=='coupling':ac.check_certificate(case,cert)
            elif engine=='sparse':refined.check_refined_removal(cert,result['sparse_model'])
            else:
                meta=result['sturm_model'];assert meta['status']=='READY'
                zl,zh=pair(cert['z_interval']);h=F(meta['h'])
                assert zl<zh and zl<=ref.exp_minus(h/grid[i])[0] and zh>=ref.exp_minus(h/grid[i+1])[1]
                face=meta['faces'][cert['face_index']];assert face['name']==cert['face_name']
                polynomial=previous.tidy(face['violation_polynomial'])
                assert previous.pvalue(polynomial,zl)>0 and previous.pvalue(polynomial,zh)>0
                chain=[previous.tidy(poly) for poly in face['sturm']['chain']]
                vl,sl=previous.changes(chain,zl);vh,sh=previous.changes(chain,zh)
                assert vl==vh and cert['variation']==[vl,vh] and cert['signs']==[sl,sh]
        else:
            assert engine in ('common_chord','common_beta') and contracts[i] is not None
            equal(deletion['theta_interval'],contracts[i]);geometry(case,deletion,engine,nt)
        stats[engine+'_deletions']+=1
    for cell in result['cells']:
        i=cell['index'];assert i not in visited and 0<=i<n and contracts[i] is not None
        visited.add(i);equal(cell['interval'],contracts[i]);geometry(case,cell,'retained',nt)
        stats['retained_cells']+=1
    assert len(result['cells'])==result['retained'] and len(result['deletions'])==result['deleted']
    assert visited==set(range(result['completed_cells'])) and set(contracts)<=visited
    assert result['grid_complete']==(result['completed_cells']==n)
    allowance,_=bridge(case,H/nt,result['continuous_bridge']);assert F(result['time_allowance'])==allowance
    check_nonempty(case,result['nonempty_witness']);check_unsafe(case,result['witness'],nt)
    if result['grid_complete']:
        expected=max((F(cell['max_node_upper']) for cell in result['cells']),default=None)
        if expected is None:assert result['upper'] is None
        else:assert F(result['upper'])==expected+allowance
    else:assert result['upper'] is None
    status=result['status']
    if status=='CERTIFIED_SAFE':
        assert result['grid_complete'] and result['retained']>0 and result['nonempty_witness'] is not None
        assert F(result['upper'])<=F(case['limit']) and result['witness'] is None
    elif status=='CERTIFIED_UNSAFE':assert result['witness'] is not None and result['nonempty_witness'] is not None
    elif status=='INCONSISTENT':
        assert result['grid_complete'] and result['retained']==0 and result['witness'] is None and result['nonempty_witness'] is None
    else:
        assert status=='UNKNOWN' and result['witness'] is None
        assert not result['grid_complete'] or result['retained']>0 and (F(result['upper'])>F(case['limit']) or result['nonempty_witness'] is None)
    stats['nonempty_witnesses']+=result['nonempty_witness'] is not None
    stats['unsafe_witnesses']+=result['witness'] is not None
    return stats
