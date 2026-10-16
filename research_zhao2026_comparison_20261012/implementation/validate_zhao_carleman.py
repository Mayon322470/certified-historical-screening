"""Formula-identity regression, not a reproduction of paper Monte Carlo plots.

Rebuilds a fully specified polynomial F(r,z) and the printed matrices with
independent elementary coefficient operations. Also checks actual actuator
history cells retain a known compatible parameter. No production changes.
"""
from fractions import Fraction as Q
from math import factorial
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import certified_core as cc
import zhao_carleman as zhao

HERE=Path(__file__).resolve().parent


def obj(v):return np.asarray(v,dtype=object)
def zeros(s):return np.full(s,Q(0),dtype=object)
def eye(n):
    out=zeros((n,n))
    for i in range(n):out[i,i]=Q(1)
    return out
def diag(v):
    out=zeros((len(v),len(v)))
    for i,a in enumerate(v):out[i,i]=a
    return out
def nonzero(a):return a[:,[j for j in range(a.shape[1]) if any(a[:,j])]]
def rowradius(a):return obj([sum((abs(x) for x in r),Q(0)) for r in a])
def norm2(a):return sum((x*x for x in a.flat),Q(0))
def equal(a,b):assert np.array_equal(obj(a),obj(b))
def poly_mul(a,b):
    result=[Q(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):result[i+j]+=x*y
    return result
def diff(a):return [i*a[i] for i in range(1,len(a))] or [Q(0)]
def interval_poly(a,lo,hi):
    x=cc.IV(cc.IV.of(lo).lo,cc.IV.of(hi).hi);value=cc.IV.of(0)
    for coefficient in reversed(a):value=value*x+coefficient
    return value
def exact_poly(a,r):return sum((c*r**i for i,c in enumerate(a)),Q(0))
def jet(polynomials,lo,hi,order):
    result=[];derivatives=[list(a) for a in polynomials]
    for h in range(order+1):
        result.append(tuple(interval_poly(a,lo,hi) for a in derivatives))
        derivatives=[diff(a) for a in derivatives]
    return result


def generic_membership(c,G,x):
    G=np.asarray(G,float);rhs=np.asarray(obj(x)-obj(c),float)
    if not G.shape[1]:return bool(np.max(abs(rhs))<1e-9)
    result=linprog(np.zeros(G.shape[1]),A_eq=G,b_eq=rhs,bounds=[(-1,1)]*G.shape[1],method='highs')
    return bool(result.success)


def formula_case(nu,tighten):
    # Zero initial auxiliary coordinates; full 4-dimensional output, degree6
    # in its second lift, so this cannot collapse to ordinary linear Taylor.
    polynomials=[list(map(Q,a)) for a in [[0,1],[1,1,1],[1,2,-1],[2,0,0,1]]]
    R=Q(1,4);witness=Q(1,10)
    measured=[exact_poly(a,witness) for a in polynomials[1:]]
    vr=Q(3,100);packet=[(y-vr,y+vr) for y in measured]
    center=jet(polynomials,Q(0),Q(0),nu+1);domain=jet(polynomials,-R,R,nu+1)
    base,tight,cert=zhao.from_jets(center,[domain],R,packet,nu,tighten,collect='full')
    rows=[[Q(1)]]+polynomials
    if nu==2:rows += [poly_mul(a,b) for a in polynomials for b in polynomials]
    n=len(rows);A=zeros((n,nu+1))
    for i,a in enumerate(rows):
        for h in range(min(nu+1,len(a))):A[i,h]=a[h]
    equal(cert['A_active_midpoint'],A)
    assert all(x==0 for x in cert['coefficient_rounding_radius'])
    c0=obj([Q(1),Q(0)]+([R*R/2] if nu==2 else []))
    G0=zeros((nu+1,nu));G0[1,0]=R
    if nu==2:G0[2,1]=R*R/2
    equal(cert['initial_lift_center_active'],c0);equal(cert['initial_lift_generators_active'],G0)
    assert cert['initial_active_indices']==([0,1] if nu==1 else [0,1,5])
    # Recompute each printed isotropic layer radius from certificate derivatives.
    remainders=[]
    for degree in cert['degree_remainders']:
        bound=max(max(abs(lo),abs(hi)) for lo,hi in degree['derivative_component_intervals'])
        assert degree['F_l_upper']==bound
        assert degree['o_l']==R**(nu+1)*bound/Q(factorial(nu+1))
        remainders += [degree['o_l']]*(4**degree['degree'])
    predc=A@c0;predG=nonzero(np.column_stack((A@G0,diag(remainders))))
    equal(cert['prediction_center'],predc);equal(cert['prediction_generators'],predG)
    # The r row gets the layer remainder although its own true remainder=0.
    assert cert['degree_remainders'][1]['o_l']>0
    assert sum(abs(x) for x in predG[1])>R
    if nu==2:
        assert predc[2]==1+R*R/2 and predc[3]==1-R*R/2
    H=zeros((3,n));H[:,2:5]=eye(3);V=diag([vr]*3);y=obj(measured)
    K=obj(cert['K']);equal(K[0],[Q(0)]*3)
    postc=predc+K@(y-H@predc)
    postG=nonzero(np.column_stack(((eye(n)-K@H)@predG,-K@V)))
    equal(cert['posterior_center'],postc);equal(cert['posterior_generators'],postG)
    P=np.array(predG,float)@np.array(predG,float).T
    hf=np.array(H,float);kf=np.array(K,float);vf=np.array(V,float)
    innovation=hf@P@hf.T+vf@vf.T
    normal_equation_residual=float(np.max(abs(kf@innovation-P@hf.T)))
    assert normal_equation_residual<1e-10
    # Eq42 minimizes F-radius, which is a different objective from Ma's LP.
    post_norm=float(norm2(postG));zero_gain_norm=float(norm2(predG))
    assert post_norm<=zero_gain_norm+1e-12
    rng=np.random.default_rng(20261012+nu)
    perturbation_objectives=[]
    for j in range(8):
        trial=kf+rng.normal(0,.01,size=kf.shape);trial[0]=0
        tg=np.column_stack(((np.eye(n)-trial@hf)@np.array(predG,float),-trial@vf))
        objective=float(np.sum(tg*tg));assert objective>=post_norm-1e-10
        perturbation_objectives.append(objective)
    corrected=None;thm3_info=None
    if tighten and nu==2:
        t=cert['theorem3'];L=obj(t['L']);X=zeros((16,4));physical=postc[1:5]
        E1=postG[1:5];Eh=postG[5:21];kappa=max(rowradius(E1))**2
        for i in range(4):
            for j in range(4):X[4*i+j,i]+=physical[j];X[4*i+j,j]+=physical[i]
        equal(t['Xi1'],X);assert t['kappa']==kappa
        corrected=physical+L@(postc[5:21]-np.kron(physical,physical))
        Gtight=nonzero(np.column_stack(((eye(4)-L@X)@E1,-kappa*L,L@Eh)))
        equal(t['center'],corrected);equal(t['generators'],Gtight)
        assert norm2(Gtight)<=norm2(E1)
        assert any(L[0])  # High-order constraints actually feed back to r.
        thm3_info={'q_gain_nonzero':True,'kappa':kappa,
            'F_radius_squared_before':norm2(E1),'F_radius_squared_after':norm2(Gtight),
            'r_halfwidth_before':rowradius(E1)[0],'r_halfwidth_after':rowradius(Gtight)[0],
            'r_center_before':physical[0],'r_center_after':corrected[0]}
    # Diagnostic membership of full ordered lifts, not just their interval hulls.
    sampled=0;compatible=0
    for i in range(81):
        r=-R+2*R*Q(i,80);state=obj([exact_poly(a,r) for a in polynomials])
        lifted=np.r_[obj([Q(1)]),state,np.kron(state,state)] if nu==2 else np.r_[obj([Q(1)]),state]
        assert generic_membership(predc,predG,lifted)
        sampled+=1
        if all(lo<=state[j+1]<=hi for j,(lo,hi) in enumerate(packet)):
            assert generic_membership(postc,postG,lifted)
            if corrected is not None:assert generic_membership(corrected,Gtight,state)
            compatible+=1
    assert compatible>0 and base[0]<=witness<=base[1]
    if tight is not None:assert tight[0]<=witness<=tight[1]
    return {'nu':nu,'tighten':tighten,'status':'PASS','polynomials':polynomials,
            'radius':R,'packet':packet,'compatible_witness_r':witness,
            'full_lift_dimension':n,'normal_equation_residual_float':normal_equation_residual,
            'prediction_F_radius_squared':zero_gain_norm,'posterior_F_radius_squared':post_norm,
            'perturbed_gain_objectives':perturbation_objectives,'theorem3_checks':thm3_info,
            'mapped_samples_diagnostic':sampled,'compatible_samples_diagnostic':compatible,
            'base_r_interval':base,'tightened_r_interval':tight,'full_certificate':cert}


def history_checks():
    import history_derivatives as hd
    case={'initial':['0','0','0'],'past':[['1/2','1'],['1/3','-1/2']],
          'packet':[['-1','1']]*3}
    model=zhao.prepare(case);qtruth=Q(3,2)
    exact=hd.derivatives(model,qtruth,qtruth,0)[0]
    packet=[(x.lower()-Q(1,1000),x.upper()+Q(1,1000)) for x in exact]
    cases=[]
    for theta in [(Q(1,2),Q(1)),(Q(2,3),Q(2,3))]:
        for nu,tighten in [(1,False),(2,False),(2,True)]:
            answer,cert=zhao.contract(model,packet,*theta,nu=nu,tighten=tighten,collect=True)
            assert answer is not None and answer[0]<=1/qtruth<=answer[1]
            cases.append({'theta_input':theta,'nu':nu,'tighten':tighten,'output':answer,
                          'q_truth':qtruth,'status':'compatible_parameter_retained',
                          'certificate_detail':cert['certificate_detail']})
    # Validate unsupported zero-width measurement is rejected explicitly,
    # rather than silently inverting a singular measurement covariance.
    bad=[(Q(0),Q(0)),(Q(-1),Q(1)),(Q(-1),Q(1))]
    try:zhao.contract(model,bad,Q(1,2),Q(1),nu=2)
    except ValueError as exc:unsupported=str(exc)
    else:raise AssertionError('zero-width observation silently accepted')
    return {'cases':cases,'zero_width_measurement_policy':unsupported}


def serial(x):
    if isinstance(x,Q):return str(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


def main():
    tests=[formula_case(1,False),formula_case(2,False),formula_case(2,True)]
    histories=history_checks()
    report={'status':'PASS','scope':'formula_identity_regression_not_paper_Example2_MC_reproduction',
            'production_sha256':hashlib.sha256((HERE/'zhao_carleman.py').read_bytes()).hexdigest(),
            'formula_tests':tests,'history_checks':histories,
            'numerical_membership_checks_are_diagnostics':True,
            'scope_limit':'zero process noise and single-sensor ideal communication boundary adapter; no positive-noise simulation claim'}
    (HERE/'zhao_formula_validation.json').write_text(json.dumps(serial(report),indent=2)+'\n',encoding='utf-8')
    lines=['Zhao2026 Carleman-SME: formula identity regression PASS.',
           'Implemented nu=1, nu=2, nu=2 plus Theorem3; full output lifts 5 and 21.',
           'Checked Lemma4 r^2 center, initial independent generators, exact A coefficients,',
           'isotropic per-degree Eq38-40 remainder, Eq25/26/33/34 prediction/update,',
           'Eq42 F-radius gain normal equations, and Eq47/49/50 Theorem3 correction.',
           'Eight gain perturbations per polynomial run confirm the numerical F-radius minimum.',
           'Theorem3 r-gain is nonzero and physical F-radius decreases in the regression.',
           'Six real-history parameter-retention tests cover ordinary and point priors.',
           'Full-lift LP membership samples are diagnostics, not the mathematical enclosure proof.',
           'This is a fully reconstructible polynomial formula regression, NOT a paper Example2 Monte Carlo reproduction.',
           'Process noise=0, zero initial auxiliary radii and ideal one-sensor communication are explicit boundary adaptations.',
           'No simulator, installation, old-file mutation, or performance benchmark was performed.']
    (HERE/'zhao_formula_validation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','formula_variants':3,'real_history_checks':6,
                      'theorem3':serial(tests[-1]['theorem3_checks'])},indent=2))


if __name__=='__main__':main()
