"""Sound tests of all six coordinate packet faces in q=1/theta.

Natural interval, derivative monotonicity, and positive-curvature certificates.
No common time lattice, floating sign decision, or feasibility inference.
"""
from fractions import Fraction as F
from math import factorial
import certified_core as cc

MAX_STATIONARY_STEPS = 32

def prepare(case):
    segments=[(F(dt),F(u)) for dt,u in case['past']]
    p = {F(0): F(case['initial'][2])}
    T = sum((t for t, u in segments),F(0))
    area = F(0); weighted = F(0); clock = F(0)
    for dt, u in segments:
        if dt < 0:
            raise ValueError('Negative hold duration')
        if not dt:
            continue
        p[0] = p.get(F(0), F(0)) - u
        p = {tau + dt: c for tau, c in p.items() if c}
        p[0] = p.get(F(0), F(0)) + u
        area += dt * u
        weighted += u * ((T-clock)*dt-dt*dt/2)
        clock += dt
    x0, v0, a0 = map(F,case['initial'])
    f0 = p.get(F(0), F(0))-a0
    terms = sorted((tau, c) for tau, c in p.items() if tau and c)
    vs = v0+area; xs = x0+v0*T+weighted; Q = area-a0*T
    faces = []
    # Feasibility is F(q)<=0 for each face. Positive q factors preserve signs.
    for coord, name in ((2,'a'), (1,'v'), (0,'x')):
        for side, sigma, bound in (('lower', -1, case['packet'][coord][0]),
                                   ('upper', 1, case['packet'][coord][1])):
            bound=F(bound)
            if coord == 2:
                poly = [f0+a0-bound]; sign_exp = 1
            elif coord == 1:
                poly = [-f0, vs-bound]; sign_exp = -1
            else:
                poly = [f0, -Q, xs-bound]; sign_exp = 1
            faces.append(dict(name=name+'_'+side,
                poly=[sigma*c for c in poly],
                terms=[(tau, sigma*sign_exp*c) for tau,c in terms]))
    return dict(faces=faces, sparse_terms=len(terms), past_holds=len(case['past']),
                moments=dict(T=T,Q=Q,v_star=vs,x_star=xs))

def polynomial_range(coefficients, lo, hi):
    a = list(coefficients)+[F(0)]*(3-len(coefficients))
    def val(x):
        return a[0]+a[1]*x+a[2]*x*x
    values = [val(lo), val(hi)]
    if a[2]:
        vertex = -a[1]/(2*a[2])
        if lo < vertex < hi:
            values.append(val(vertex))
    return cc.IV(cc.IV.of(min(values)).lo, cc.IV.of(max(values)).hi)

def evaluate(face, lo, hi, derivative=0):
    poly = list(face['poly'])
    for _ in range(derivative):
        poly = [i*poly[i] for i in range(1,len(poly))] or [F(0)]
    result = polynomial_range(poly,lo,hi)
    for tau, c in face['terms']:
        coefficient = c*(-tau)**derivative
        e = cc.IV(cc.exp_neg(tau*hi).lo, cc.exp_neg(tau*lo).hi)
        result += coefficient*e
    return result

def positive_curvature(face, a, b, mu, steps):
    """Every trial uses a global lower quadratic on the ORIGINAL [a,b]."""
    left, right = a, b
    for j in range(steps+1):
        center = (left+right)/2
        value = evaluate(face,center,center)
        slope = evaluate(face,center,center,1)
        G = max(abs(slope.lower()),abs(slope.upper()))
        lower = value.lower()-G*G/(2*mu)
        if lower > 0:
            return dict(kind='curvature',center=center,mu=mu,
                        value_lower=value.lower(),slope_interval=[slope.lower(),slope.upper()],
                        certified_lower=lower,stationary_steps=j)
        if slope.upper() < 0:
            left = center
        elif slope.lower() > 0:
            right = center
        else:
            break
    return None

def second_enclosure(face,a,b):
    """Intersection with a fourth-order centered Taylor enclosure of F''."""
    direct=evaluate(face,a,b,2)
    if direct.lower()>0:
        return direct
    center=(a+b)/2;radius=(b-a)/2
    result=evaluate(face,center,center,2)
    for j in range(1,5):
        power=cc.IV.of(radius**j)
        variation=cc.IV(-power.hi if j%2 else 0,power.hi)
        result += evaluate(face,center,center,2+j)*variation/F(factorial(j))
    power=cc.IV.of(radius**5)
    result += evaluate(face,a,b,7)*cc.IV(-power.hi,power.hi)/F(factorial(5))
    return cc.IV(max(direct.lo,result.lo),min(direct.hi,result.hi))

def reject_cell(meta,l,r,curvature=False):
    a,b=F(1)/r,F(1)/l
    for idx,face in enumerate(meta['faces']):
        certificate = None
        direct = evaluate(face,a,b)
        if direct.lower() > 0:
            certificate = dict(kind='natural',certified_lower=direct.lower())
        elif direct.upper() <= 0:
            continue
        else:
            slope = evaluate(face,a,b,1)
            if slope.lower() >= 0:
                point = evaluate(face,a,a)
                if point.lower() > 0:
                    certificate = dict(kind='monotone_increasing',point=a,
                        certified_lower=point.lower())
            elif slope.upper() <= 0:
                point = evaluate(face,b,b)
                if point.lower() > 0:
                    certificate = dict(kind='monotone_decreasing',point=b,
                        certified_lower=point.lower())
            elif curvature:
                second = second_enclosure(face,a,b)
                if second.lower() > 0:
                    # No sign or root is assumed from the bisection branch.
                    # Each successful bound certifies the entire original cell.
                    certificate = positive_curvature(face,a,b,second.lower(),MAX_STATIONARY_STEPS)
        if certificate:
            return dict(face_index=idx,face_name=face['name'],theta_interval=[l,r],
                        q_interval=[a,b],**certificate)
    return None
