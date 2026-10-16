"""Independent rank-one Ma adapter replay using a two-variable exact dual.

History ranges use the sealed independent Laurent-exponential reference.
No production module is imported here. Candidate optimization uses the dual;
strip projection uses 2-D vertex geometry, not producer Fourier--Motzkin.
"""
from fractions import Fraction as F
import independent_ma_reference as ref
import rank1_reference as rr


def eq(a,b):
    assert tuple(map(F,a))==tuple(map(F,b)),(a,b)


def check_gain(cert):
    R,C,s,p,b=(cert['radius'],cert['C'],cert['diagonal_noise_halfwidth'],
               cert['response_shape'],cert['response_bound'])
    N=tuple(map(F,cert['selected']['N']))
    optimum=rr.verify_ma_optimum(N,R,C,s,b,p)['value']
    assert F(cert['selected']['objective'])==optimum
    assert F(cert['T'])==1-sum(n*F(c) for n,c in zip(N,C))
    assert cert['kind']=='exact_rank1_weighted_l1_gain'
    assert 1<=len(cert['candidates'])<=11
    for candidate in cert['candidates']:
        assert F(candidate['objective'])==rr.ma_objective(candidate['N'],R,C,s,b,p)
    return N,optimum


def check_contract(case,cert):
    assert cert['kind']=='ma2026_shared_rank1_scalar' and cert['strengthened'] is True
    tl,th=map(F,cert['theta_input']);assert 0<tl<=th
    ql,qh=1/th,1/tl;c,R=(ql+qh)/2,(qh-ql)/2
    eq(cert['q_input'],(ql,qh));assert cert['center']==c and cert['radius']==R
    response=case['response'];T=sum((F(dt) for dt,_ in case['past']),F(0))
    b,L=F(response['epsilon_x']),F(response['L'])
    assert response['kind']=='normalized_global_cubic_v1' and L==T+F(case['H'])
    assert F(case['rho'])==0 and b>=0
    p=rr.cubic_psi(T,L)
    eq(cert['response_shape'],p);assert F(cert['response_bound'])==b
    expressions=ref.endpoint_terms(case)
    strips=[];widths=[];centers=[];measurement=[];slopes=[]
    assert len(cert['strips'])==3
    for j,(expression,row) in enumerate(zip(expressions,cert['strips'])):
        assert row['coordinate']==j
        value,derivative,hessian=[tuple(map(F,row[k])) for k in ('value','derivative','hessian')]
        assert ref.contained(ref.natural_range(expression,c,c),value)
        assert ref.contained(ref.natural_range(ref.derivative(expression),c,c),derivative)
        derivative2=ref.derivative(expression,2)
        if not ref.contained(ref.strong_range(derivative2,ql,qh),hessian):
            assert ref.prove_range_contained(derivative2,ql,qh,hessian)
        C=sum(derivative)/2;rounding=(derivative[1]-derivative[0])*R/2
        assert F(row['C'])==C and F(row['slope_rounding'])==rounding
        remainder=min(F(0),hessian[0])*R*R/2,max(F(0),hessian[1])*R*R/2
        eq(row['remainder'],remainder)
        pl,pu=map(F,case['packet'][j])
        low=pl-value[1]-remainder[1]-rounding+C*c
        high=pu-value[0]-remainder[0]+rounding+C*c
        assert F(row['low'])==low and F(row['high'])==high
        strips.append((C,p[j],low,high));slopes.append(C)
        centers.append(sum(value)/2+sum(remainder)/2)
        widths.append((pu-pl)/2+(value[1]-value[0])/2+(remainder[1]-remainder[0])/2+rounding)
        measurement.append((pl+pu)/2)
    eq(cert['diagonal_noise_halfwidth'],widths)
    eq(cert['prediction_center'],centers);eq(cert['measurement_center'],measurement)
    gain=cert['gain_certificate']
    assert F(gain['radius'])==R and F(gain['response_bound'])==b
    eq(gain['C'],slopes);eq(gain['diagonal_noise_halfwidth'],widths);eq(gain['response_shape'],p)
    N,post_radius=check_gain(gain)
    post_center=c+sum(n*(y-g) for n,y,g in zip(N,measurement,centers))
    assert F(cert['posterior_center'])==post_center and F(cert['posterior_radius'])==post_radius
    interval=max(ql,post_center-post_radius),min(qh,post_center+post_radius)
    eq(cert['gain_q_interval'],interval)
    projection=rr.joint_strip_projection((ql,qh),b,strips)
    assert cert['all_strip_projection']['rejected']==(projection is None)
    if projection is not None: eq(cert['all_strip_projection']['q_projection'],projection)
    rejected=projection is None or interval[0]>interval[1]
    if not rejected:interval=max(interval[0],projection[0]),min(interval[1],projection[1])
    rejected=rejected or interval[0]>interval[1]
    eq(cert['q_intersection'],interval);assert cert['rejected']==rejected
    if rejected:
        assert 'theta_output' not in cert
        return None
    theta=1/interval[1],1/interval[0]
    eq(cert['theta_output'],theta)
    return theta
