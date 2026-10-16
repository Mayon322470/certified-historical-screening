"""Independent exact 2-D references for the new rank-one response contract.

No producer imports. Dual verification is independent of a 3-D primal
arrangement search. Small dimensions only; this is audit work, not timing.
"""
from fractions import Fraction as F
from itertools import combinations


def vertices_2d(rows):
    rows=[(tuple(map(F,a)),F(b)) for a,b in rows]
    candidates=set()
    for (a,b),(c,d) in combinations(rows,2):
        det=a[0]*c[1]-a[1]*c[0]
        if not det:continue
        point=((b*c[1]-a[1]*d)/det,(a[0]*d-b*c[0])/det)
        if all(sum(x*y for x,y in zip(normal,point))<=rhs for normal,rhs in rows):candidates.add(point)
    return tuple(sorted(candidates))


def ma_dual(radius,slopes,noise,amplitude,psi):
    R,b=F(radius),F(amplitude);C,s,p=map(lambda z:tuple(map(F,z)),(slopes,noise,psi))
    if len(C)!=3 or len(s)!=3 or len(p)!=3 or R<0 or b<0 or any(x<=0 for x in s):
        raise ValueError('3 positive diagonal noise widths and nonnegative radii required')
    rows=[((1,0),R),((-1,0),R),((0,1),b),((0,-1),b)]
    for ci,si,pi in zip(C,s,p):rows.extend([((ci,-pi),si),((-ci,pi),si)])
    vertices=vertices_2d(rows)
    if not vertices:raise AssertionError('Zero is feasible and dual box is compact')
    witness=max(vertices,key=lambda x:x[0])
    return dict(optimum=witness[0],witness=witness,vertices=vertices)


def ma_objective(N,radius,slopes,noise,amplitude,psi):
    N,C,s,p=map(lambda z:tuple(map(F,z)),(N,slopes,noise,psi))
    if any(len(x)!=3 for x in (N,C,s,p)):raise ValueError('3-D gain required')
    return F(radius)*abs(1-sum(n*c for n,c in zip(N,C)))+sum(v*abs(n) for v,n in zip(s,N))+F(amplitude)*abs(sum(n*pi for n,pi in zip(N,p)))


def verify_ma_optimum(N,radius,slopes,noise,amplitude,psi):
    dual=ma_dual(radius,slopes,noise,amplitude,psi)
    primal=ma_objective(N,radius,slopes,noise,amplitude,psi)
    if primal!=dual['optimum']:raise AssertionError('Primal/dual objective gap')
    return dict(value=primal,dual_witness=dual['witness'])


def estimator_interval(N,q_center,radius,slopes,noise,amplitude,psi,measurement_center):
    N,C,y=map(lambda z:tuple(map(F,z)),(N,slopes,measurement_center))
    center=(1-sum(n*c for n,c in zip(N,C)))*F(q_center)+sum(n*yi for n,yi in zip(N,y))
    r=ma_objective(N,radius,C,noise,amplitude,psi)
    return center-r,center+r


def joint_strip_projection(q_prior,amplitude,strips):
    """Independent polygon projection for low<=C*q+psi*beta<=high.

    This directly enumerates a compact 2-D polygon; it does not use Fourier-
    Motzkin elimination and is intended to audit that production elimination.
    """
    lo,hi=map(F,q_prior);b=F(amplitude)
    if lo>hi or b<0:raise ValueError('Ordered prior/nonnegative amplitude required')
    rows=[((1,0),hi),((-1,0),-lo),((0,1),b),((0,-1),b)]
    for slope,psi,lower,upper in strips:
        slope,psi,lower,upper=map(F,(slope,psi,lower,upper))
        if lower>upper:raise ValueError('Ordered strip required')
        rows.extend([((slope,psi),upper),((-slope,-psi),-lower)])
    vertices=vertices_2d(rows)
    return None if not vertices else (min(x[0] for x in vertices),max(x[0] for x in vertices))


def beta_bounds(packet,nominal_intervals,psi,amplitude,kind):
    """Necessary outer vs sufficient inner beta, for nonnegative cubic psi."""
    if kind not in ('outer','inner'):raise ValueError('Specify outer or inner')
    lo,hi=-F(amplitude),F(amplitude)
    for (pl,pu),(gl,gu),p in zip(packet,nominal_intervals,psi):
        pl,pu,gl,gu,p=map(F,(pl,pu,gl,gu,p))
        if pl>pu or gl>gu or p<0:raise ValueError('Ordered intervals and nonnegative psi required')
        if not p:
            compatible=not (gu<pl or gl>pu) if kind=='outer' else pl<=gl<=gu<=pu
            if not compatible:return None
            continue
        left=(pl-gu)/p if kind=='outer' else (pl-gl)/p
        right=(pu-gl)/p if kind=='outer' else (pu-gu)/p
        lo,hi=max(lo,left),min(hi,right)
        if lo>hi:return None
    return lo,hi


def cubic_psi(t,length):
    t,L=F(t),F(length)
    if not 0<=t<=L or L<=0:raise ValueError('0<=t<=L, L>0 required')
    return t**3/L**3,3*t*t/L**3,6*t/L**3
