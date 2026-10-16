"""Zhao et al. TAC 2026: certified single-step 4-state history adapter.

Independent implementation of Algorithm 1, Theorems 1--3, nu=1/2.
F(r,z)=(r,f_T(qc+r)), W=0, ideal one sensor C=[0 I3]. Initial
auxiliary coordinates and process noise are degenerate boundary adaptations.
Keeps full 5/21 dimensional OUTPUT lifts and the paper's isotropic remainder
per degree. Only identically-zero INITIAL lift columns are eliminated.
Float F-radius-optimal gains are proposals; their exact binary rational
values are used in all subsequent center/generator calculations.
"""
from fractions import Fraction as Q
from math import comb, factorial
import numpy as np
import certified_core as cc
import history_derivatives as hd


def prepare(case):
    return hd.prepare(case)


def _zeros(shape):
    return np.full(shape,Q(0),dtype=object)


def _eye(n):
    a=_zeros((n,n))
    for i in range(n):a[i,i]=Q(1)
    return a


def _diag(v):
    out=_zeros((len(v),len(v)))
    for i,x in enumerate(v):out[i,i]=Q(x)
    return out


def _float(a):
    return np.asarray(a,dtype=float)


def _rational(a):
    a=np.asarray(a,dtype=float)
    if not np.isfinite(a).all():raise ArithmeticError('nonfinite gain proposal')
    return np.array([Q.from_float(float(x)) for x in a.flat],dtype=object).reshape(a.shape)


def _drop_zero_columns(a):
    return a[:,[j for j in range(a.shape[1]) if any(a[:,j])]]


def _radius(a):
    return np.array([sum(map(abs,row),Q(0)) for row in a],dtype=object)


def _frobenius_squared(a):
    return sum((x*x for x in a.flat),Q(0))


def _iv_interval(lo,hi):
    return cc.IV(cc.IV.of(lo).lo,cc.IV.of(hi).hi)


def _fjet(history_jet,rlo,rhi):
    """Derivative h of F, ordered as [r,x,v,a]."""
    return [(_iv_interval(rlo,rhi) if h==0 else cc.IV.of(1 if h==1 else 0),*values)
            for h,values in enumerate(history_jet)]


def lifted_derivatives(fjet,l,order):
    """All ordered output Kronecker rows, differentiated via Leibniz."""
    if l==0:return [[cc.IV.of(1 if h==0 else 0)] for h in range(order+1)]
    if l==1:return [list(v) for v in fjet[:order+1]]
    if l!=2:raise ValueError('this implementation freezes nu to 1 or 2')
    return [[sum((comb(h,j)*fjet[j][a]*fjet[h-j][b] for j in range(h+1)),cc.IV.of(0))
             for a in range(4) for b in range(4)] for h in range(order+1)]


def _innovation_gain(generators,measurement,noise):
    """Eq42 float proposal; physical constant row fixed at zero by caller."""
    G,H,V=map(_float,(generators,measurement,noise))
    P=G@G.T; cross=P@H.T; innovation=H@cross+V@V.T
    if np.any(np.linalg.eigvalsh(innovation)<=0):
        raise ArithmeticError('positive measurement radii required for nonsingular Eq42')
    return _rational(np.linalg.solve(innovation,cross.T).T)


def _tighten(center,generators):
    """Theorem3 Eqs47--51; independent concatenation exactly as Eq49."""
    physical=center[1:5]; E1=generators[1:5]; chigh=center[5:21]; Ehigh=generators[5:21]
    Xi=_zeros((16,4))
    for i in range(4):
        for j in range(4):
            Xi[4*i+j,i]+=physical[j]
            Xi[4*i+j,j]+=physical[i]
    kappa=max(_radius(E1))**2
    P=E1@E1.T; B=P@Xi.T
    innovation=Xi@B+(kappa*kappa)*_eye(16)+Ehigh@Ehigh.T
    if kappa==0:
        # E1=0: zero gain is the exact zero-error boundary optimum.
        L=_zeros((4,16)); proposal='zero_physical_error_boundary_gain'
    else:
        try:
            L=_rational(np.linalg.solve(_float(innovation),_float(B).T).T)
            proposal='float_Eq50_gain_then_exact_binary_rational'
        except np.linalg.LinAlgError:
            # Any finite gain preserves Eq49 containment; fallback is explicit,
            # never mislabeled F-radius optimal or used as infeasibility.
            L=_zeros((4,16)); proposal='zero_gain_fallback_singular_float_solve'
    c=physical+L@(chigh-np.kron(physical,physical))
    G=np.column_stack(((_eye(4)-L@Xi)@E1,-kappa*L,L@Ehigh))
    G=_drop_zero_columns(G)
    return c,G,{'Xi1':Xi.tolist(),'kappa':kappa,'L':L.tolist(),
        'gain_proposal':proposal,'generator_blocks':'[(I-L Xi1)E1, -kappa L, L Ehigh]',
        'physical_f_radius_squared_before':_frobenius_squared(E1),
        'physical_f_radius_squared_after':_frobenius_squared(G)}


def from_jets(center_fjet,domain_fjets,radius,packet,nu=2,tighten=True,collect=True,response_generator=None):
    """Shared formula kernel, independent of the actuator history model.

    center_fjet: orders 0..nu+1, each length 4 of cc.IV at r=0.
    domain_fjets: list of jets, each covering one subinterval of [-R,R].
    Assumes F depends only on r, initial state=(r,0,0,0), no process noise.
    Returns physical interval projections and complete formula certificate.
    """
    if nu not in (1,2):raise ValueError('nu must be 1 or 2')
    radius=Q(radius)
    if radius<0:raise ValueError('negative initial radius')
    ny=3; nbar=5 if nu==1 else 21
    # Lemma4, Eq23: r and r^2 use independent initial generators.
    initial_center=np.array([Q(1),Q(0)]+([radius*radius/2] if nu==2 else []),dtype=object)
    initial_G=_zeros((nu+1,nu));initial_G[1,0]=radius
    if nu==2:initial_G[2,1]=radius*radius/2
    active=[0,1]+([5] if nu==2 else [])
    A=_zeros((nbar,nu+1)); coefficient_error=_zeros(nbar)
    remainder=_zeros(nbar);degree_data=[];offset=0;coefficients=[]
    for l in range(nu+1):
        center_lift=lifted_derivatives(center_fjet,l,nu+1)
        domain_lift=[lifted_derivatives(jet,l,nu+1) for jet in domain_fjets]
        rows=4**l
        derivative_intervals=[]
        for i in range(rows):
            bound=cc.IV(min(jet[nu+1][i].lo for jet in domain_lift),
                        max(jet[nu+1][i].hi for jet in domain_lift))
            derivative_intervals.append((bound.lower(),bound.upper()))
            c_row=[]
            for h in range(nu+1):
                coefficient=center_lift[h][i]/Q(factorial(h))
                A[offset+i,h]=coefficient.midpoint()
                coefficient_error[offset+i]+=coefficient.radius()*radius**h
                c_row.append((coefficient.lower(),coefficient.upper()))
            coefficients.append(c_row)
        largest=max(max(abs(lo),abs(hi)) for lo,hi in derivative_intervals)
        o=radius**(nu+1)*largest/Q(factorial(nu+1))
        # EXACTLY Eqs38--40: same o on every row at degree l, including r/rr.
        remainder[offset:offset+rows]=o
        degree_data.append({'degree':l,'F_l_upper':largest,'o_l':o,
                            'derivative_component_intervals':derivative_intervals})
        offset+=rows
    predicted_center=A@initial_center
    predicted_G=_drop_zero_columns(np.column_stack((A@initial_G,_diag(remainder),_diag(coefficient_error))))
    H=_zeros((3,nbar));H[:,2:5]=_eye(3)
    y=np.array([(Q(lo)+Q(hi))/2 for lo,hi in packet],dtype=object)
    vr=np.array([(Q(hi)-Q(lo))/2 for lo,hi in packet],dtype=object)
    if any(x<=0 for x in vr):raise ValueError('this adapter requires strictly positive measurement-box halfwidths')
    V=_diag(vr)
    if response_generator is not None:
        response_generator=np.asarray(response_generator,dtype=object)
        if response_generator.shape!=(3,):raise ValueError('One shared response column required')
        V=np.column_stack((V,response_generator))
    K=_innovation_gain(predicted_G,H,V);K[0,:]=Q(0)
    posterior_center=predicted_center+K@(y-H@predicted_center)
    posterior_G=_drop_zero_columns(np.column_stack(((_eye(nbar)-K@H)@predicted_G,-K@V)))
    physical_center=posterior_center[1:5];physical_G=posterior_G[1:5]
    physical_radius=_radius(physical_G)
    base_interval=(physical_center[0]-physical_radius[0],physical_center[0]+physical_radius[0])
    corrected_interval=None;tight_data=None
    if nu==2 and tighten:
        corrected_center,corrected_G,tight_data=_tighten(posterior_center,posterior_G)
        corrected_radius=_radius(corrected_G)
        corrected_interval=(corrected_center[0]-corrected_radius[0],corrected_center[0]+corrected_radius[0])
        if collect:
            tight_data.update({'center':corrected_center.tolist()})
            if collect=='full':tight_data['generators']=corrected_G.tolist()
    cert={'kind':'zhao_carleman_single_history','nu':nu,'tighten':bool(nu==2 and tighten),
          'full_lift_dimension':nbar,'initial_active_indices':active,'radius':radius,
          'base_r_interval':base_interval,'tightened_r_interval':corrected_interval,
          'gain_guarantee':'floating_F_radius_optimal_proposal_exact_rational_containment_recomputation',
          'isotropic_per_degree_remainder':True,
          'certificate_detail':'full' if collect=='full' else 'compact' if collect else 'minimal',
          'initial_boundary_adapter':'zero_process_noise_and_zero_auxiliary_initial_radii',
          'ideal_single_sensor':True}
    if collect:
        cert.update({'initial_lift_center_active':initial_center.tolist(),
            'initial_lift_generators_active':initial_G.tolist(),
            'A_active_midpoint':A.tolist(),'coefficient_intervals':coefficients,
            'coefficient_rounding_radius':coefficient_error.tolist(),'degree_remainders':degree_data,
            'prediction_center':predicted_center.tolist(),
            'measurement_center':y.tolist(),'measurement_halfwidth':vr.tolist(),
            'K':K.tolist(),'posterior_center':posterior_center.tolist(),'theorem3':tight_data,
            'measurement_noise_generators':V.tolist(),
            'generator_reconstruction':{
                'prediction':'[A_active G0_active, diag(repeat_each_degree(o_l)), diag(coefficient_rounding_radius)]; drop zero columns',
                'posterior':'[(I-K H) prediction_G, -K V]; H selects rows2:5; drop zero columns',
                'theorem3':'[(I-L Xi1) E1, -kappa L, L Ehigh]; drop zero columns'}})
        if collect=='full':
            cert.update({'prediction_generators':predicted_G.tolist(),
                         'posterior_generators':posterior_G.tolist()})
    return base_interval,corrected_interval,cert


def contract(model,packet,theta_lo,theta_hi,nu=2,tighten=True,collect=True,hessian_pieces=1,response_generator=None):
    """Return retained theta interval/None and a rational certificate.

    Prior intersection and optional intersection of both Eq43 and Eq49
    projections are explicit common-domain operations, not new paper gains.
    """
    theta_lo,theta_hi=Q(theta_lo),Q(theta_hi)
    if not 0<theta_lo<=theta_hi:raise ValueError('theta interval must be strictly positive')
    pieces=int(hessian_pieces)
    if pieces<1:raise ValueError('hessian_pieces must be positive')
    qlo,qhi=1/theta_hi,1/theta_lo;center=(qlo+qhi)/2;radius=(qhi-qlo)/2
    center_jet=_fjet(hd.derivatives(model,center,center,nu+1),Q(0),Q(0))
    domain=[]
    for j in range(pieces):
        lo=qlo+(qhi-qlo)*Q(j,pieces);hi=qlo+(qhi-qlo)*Q(j+1,pieces)
        domain.append(_fjet(hd.derivatives(model,lo,hi,nu+1),lo-center,hi-center))
    base,tight,cert=from_jets(center_jet,domain,radius,packet,nu,tighten,collect,response_generator)
    base_q=(center+base[0],center+base[1])
    tight_q=(center+tight[0],center+tight[1]) if tight is not None else None
    retained=(max(qlo,base_q[0]),min(qhi,base_q[1]))
    if tight_q is not None:retained=(max(retained[0],tight_q[0]),min(retained[1],tight_q[1]))
    rejected=retained[0]>retained[1]
    cert.update({'theta_input':[theta_lo,theta_hi],'q_input':[qlo,qhi],
        'q_center':center,'q_radius':radius,'hessian_pieces':pieces,
        'base_q_interval':base_q,'tightened_q_interval':tight_q,
        'q_intersection':retained,'rejected':rejected,
        'domain_intersections':['prior','Eq43_projection']+(['Eq49_projection'] if tight_q is not None else [])})
    if rejected:return None,cert
    result=(1/retained[1],1/retained[0]);cert['theta_output']=result
    return result,cert
