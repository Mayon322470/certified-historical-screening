"""Independent rational verification of compact Zhao2026 adapter certificates."""
from fractions import Fraction as Q
from math import factorial
import hashlib
import json

import independent_ma_reference as ref
import zhao_reference as zr

CERTIFICATE_CACHE={}
CACHE_COUNTS={'independent_contract_computations':0,'identical_contract_payloads_checked_by_hash':0}


def equal_interval(recorded,expected):
    assert tuple(map(Q,recorded))==tuple(expected),(recorded,expected)


def equal_vector(recorded,expected):
    assert list(map(Q,recorded))==list(expected)


def equal_matrix(recorded,expected):
    assert zr.matrix(recorded)==expected


def drop_zero_columns(A):
    columns=[j for j in range(len(A[0])) if any(row[j] for row in A)]
    return [[row[j] for j in columns] for row in A]


def radius(A):
    return [sum(map(abs,row),Q(0)) for row in A]


def binary_rational_matrix(value,rows,columns):
    A=zr.matrix(value)
    assert len(A)==rows and all(len(row)==columns for row in A)
    assert all((v.denominator & (v.denominator-1))==0 for row in A for v in row)
    return A


def check_contract(case,cert):
    # Budget-boundary variation can produce different stage prefixes containing
    # exactly the same contracts. Bind both the complete input and certificate;
    # no producer data is treated as trusted without an earlier full check.
    fingerprint=hashlib.sha256(json.dumps([case,cert],sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
    if fingerprint in CERTIFICATE_CACHE:
        CACHE_COUNTS['identical_contract_payloads_checked_by_hash']+=1
        return CERTIFICATE_CACHE[fingerprint]
    answer=_check_contract(case,cert)
    CERTIFICATE_CACHE[fingerprint]=answer
    CACHE_COUNTS['independent_contract_computations']+=1
    return answer


def _check_contract(case,cert):
    assert cert['kind']=='zhao_carleman_single_history'
    nu=cert['nu'];tight=cert['tighten']
    assert nu in (1,2) and isinstance(tight,bool) and (nu==2 or not tight)
    assert cert['isotropic_per_degree_remainder'] is True
    assert cert['ideal_single_sensor'] is True
    assert cert['certificate_detail'] in ('compact','full')
    tl,th=map(Q,cert['theta_input'])
    assert 0<tl<=th
    ql,qh=1/th,1/tl
    center,R=(ql+qh)/2,(qh-ql)/2
    equal_interval(cert['q_input'],(ql,qh))
    assert Q(cert['q_center'])==center and Q(cert['q_radius'])==Q(cert['radius'])==R
    assert isinstance(cert['hessian_pieces'],int) and cert['hessian_pieces']>=1
    indices,models=zr.get_model(case,center,nu)
    nbar=len(models)
    assert cert['full_lift_dimension']==nbar
    active=[0,1]+([5] if nu==2 else [])
    assert cert['initial_active_indices']==active
    initial_center,initial_G=zr.initial_lift(R,nu)
    c0=[initial_center[i] for i in active]
    G0=[initial_G[i] for i in active]
    equal_vector(cert['initial_lift_center_active'],c0)
    equal_matrix(cert['initial_lift_generators_active'],G0)
    coefficients=cert['coefficient_intervals']
    assert len(coefficients)==nbar
    A=zr.zeros(nbar,nu+1);rounding=[Q(0)]*nbar
    for row,(expression,bounds) in enumerate(zip(models,coefficients)):
        assert len(bounds)==nu+1
        for order,interval in enumerate(bounds):
            enclosure=tuple(map(Q,interval))
            assert enclosure[0]<=enclosure[1]
            actual=ref.scale(zr.point_range(ref.derivative(expression,order),center),Q(1,factorial(order)))
            assert ref.contained(actual,enclosure),('Taylor coefficient',row,order,actual,enclosure)
            A[row][order]=sum(enclosure)/2
            rounding[row]+=(enclosure[1]-enclosure[0])*R**order/2
    equal_matrix(cert['A_active_midpoint'],A)
    equal_vector(cert['coefficient_rounding_radius'],rounding)
    degree_data=cert['degree_remainders']
    assert len(degree_data)==nu+1
    remainder=[];offset=0
    for degree,data in enumerate(degree_data):
        assert data['degree']==degree
        intervals=data['derivative_component_intervals']
        assert len(intervals)==4**degree
        biggest=Q(0)
        for within_degree,pair in enumerate(intervals):
            outer=tuple(map(Q,pair))
            assert outer[0]<=outer[1]
            expression=ref.derivative(models[offset+within_degree],nu+1)
            assert zr.prove_range_contained(expression,ql,qh,outer)
            biggest=max(biggest,abs(outer[0]),abs(outer[1]))
        assert Q(data['F_l_upper'])==biggest
        o=R**(nu+1)*biggest/factorial(nu+1)
        assert Q(data['o_l'])==o
        remainder.extend([o]*(4**degree));offset+=4**degree
    predicted_center=zr.mv(A,c0)
    predicted_G=drop_zero_columns(zr.hcat(zr.mm(A,G0),zr.diagonal(remainder),zr.diagonal(rounding)))
    equal_vector(cert['prediction_center'],predicted_center)
    if cert['certificate_detail']=='full':
        equal_matrix(cert['prediction_generators'],predicted_G)
    H=zr.zeros(3,nbar)
    for i in range(3):
        H[i][i+2]=Q(1)
    packet=[tuple(map(Q,pair)) for pair in case['packet']]
    y=[(low+high)/2 for low,high in packet]
    widths=[(high-low)/2 for low,high in packet]
    assert all(w>0 for w in widths)
    equal_vector(cert['measurement_center'],y)
    equal_vector(cert['measurement_halfwidth'],widths)
    V=zr.diagonal(widths)
    K=binary_rational_matrix(cert['K'],nbar,3)
    assert all(v==0 for v in K[0]),'Constant lift must have zero measurement gain'
    residual=[value-prediction for value,prediction in zip(y,zr.mv(H,predicted_center))]
    correction=zr.mv(K,residual)
    posterior_center=[a+b for a,b in zip(predicted_center,correction)]
    posterior_G=drop_zero_columns(zr.hcat(zr.mm(zr.add(zr.identity(nbar),zr.mm(K,H),-1),predicted_G),
                                        zr.scale(zr.mm(K,V),Q(-1))))
    equal_vector(cert['posterior_center'],posterior_center)
    if cert['certificate_detail']=='full':
        equal_matrix(cert['posterior_generators'],posterior_G)
    physical=posterior_center[1:5];E1=posterior_G[1:5]
    radii=radius(E1)
    base=(physical[0]-radii[0],physical[0]+radii[0])
    equal_interval(cert['base_r_interval'],base)
    corrected=None
    if tight:
        data=cert['theorem3']
        Xi=zr.xi1(physical)
        equal_matrix(data['Xi1'],Xi)
        kappa=zr.infinity_norm(E1)**2
        assert Q(data['kappa'])==kappa
        L=binary_rational_matrix(data['L'],4,16)
        if data['gain_proposal'] in ('zero_physical_error_boundary_gain','zero_gain_fallback_singular_float_solve'):
            assert not any(v for row in L for v in row)
        else:
            assert data['gain_proposal']=='float_Eq50_gain_then_exact_binary_rational'
        if kappa==0:
            assert data['gain_proposal']=='zero_physical_error_boundary_gain'
        high_center=posterior_center[5:21];Ehigh=posterior_G[5:21]
        residual=[a-b for a,b in zip(high_center,zr.lifted_square(physical))]
        correction=zr.mv(L,residual)
        new_center=[a+b for a,b in zip(physical,correction)]
        new_G=drop_zero_columns(zr.hcat(zr.mm(zr.add(zr.identity(4),zr.mm(L,Xi),-1),E1),
                                       zr.scale(L,-kappa),zr.mm(L,Ehigh)))
        equal_vector(data['center'],new_center)
        if cert['certificate_detail']=='full':
            equal_matrix(data['generators'],new_G)
        assert Q(data['physical_f_radius_squared_before'])==zr.squared_frobenius(E1)
        assert Q(data['physical_f_radius_squared_after'])==zr.squared_frobenius(new_G)
        radii=radius(new_G)
        corrected=(new_center[0]-radii[0],new_center[0]+radii[0])
        equal_interval(cert['tightened_r_interval'],corrected)
    else:
        assert cert['theorem3'] is None and cert['tightened_r_interval'] is None
    base_q=(center+base[0],center+base[1])
    equal_interval(cert['base_q_interval'],base_q)
    retained=max(ql,base_q[0]),min(qh,base_q[1])
    if corrected is not None:
        corrected_q=(center+corrected[0],center+corrected[1])
        equal_interval(cert['tightened_q_interval'],corrected_q)
        retained=max(retained[0],corrected_q[0]),min(retained[1],corrected_q[1])
    else:
        assert cert['tightened_q_interval'] is None
    assert cert['domain_intersections']==['prior','Eq43_projection']+(['Eq49_projection'] if tight else [])
    equal_interval(cert['q_intersection'],retained)
    rejected=retained[0]>retained[1]
    assert cert['rejected']==rejected
    if rejected:
        assert 'theta_output' not in cert
        return None
    assert 0<retained[0]<=retained[1]
    theta=1/retained[1],1/retained[0]
    equal_interval(cert['theta_output'],theta)
    assert tl<=theta[0]<=theta[1]<=th
    return theta
