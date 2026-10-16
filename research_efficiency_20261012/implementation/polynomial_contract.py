"""Exact rational packet-face exclusion before the unchanged chord query bound.

No samples imply feasibility. A cell is deleted only after a strict polynomial
sign proof on a certified exponential image of the ENTIRE parameter cell.
"""
from fractions import Fraction as F
from math import gcd,lcm
from functools import reduce
import hashlib,json
from certified_core import exp_neg

MAX_DEGREE=32
MAX_COEFFICIENT_BITS=4096

class BudgetExceeded(Exception): pass

def trim(a):
    a=list(map(F,a))
    while len(a)>1 and not a[-1]: a.pop()
    return a or [F(0)]

def zero(a): return len(a)==1 and a[0]==0
def neg(a): return [-x for x in a]
def shift_constant(a,c):
    a=list(a);a[0]+=c;return trim(a)
def value(a,x):
    r=F(0)
    for c in reversed(a):r=r*x+c
    return r
def sign(x):return (x>0)-(x<0)
def derivative(a):return trim([i*a[i] for i in range(1,len(a))])
def check_size(a):
    if len(a)-1>MAX_DEGREE:raise BudgetExceeded('DEGREE_BUDGET')
    if any(max(abs(x.numerator).bit_length(),x.denominator.bit_length())>MAX_COEFFICIENT_BITS for x in a):
        raise BudgetExceeded('COEFFICIENT_BIT_BUDGET')

def divide(a,b):
    a=trim(a);b=trim(b);assert not zero(b)
    q=[F(0)]*max(1,len(a)-len(b)+1)
    while not zero(a) and len(a)>=len(b):
        k=len(a)-len(b);c=a[-1]/b[-1];q[k]+=c
        for j,v in enumerate(b):a[k+j]-=c*v
        a=trim(a);check_size(a);check_size(q)
    return trim(q),a

def primitive(a):
    """Return integer polynomial b and POSITIVE scale c with a=c*b."""
    assert not zero(a)
    den=reduce(lcm,(x.denominator for x in a),1)
    ints=[int(x*den) for x in a]
    g=reduce(gcd,(abs(x) for x in ints),0)
    return [F(x//g) for x in ints],F(g,den)

def sturm(a):
    a=trim(a);check_size(a)
    if len(a)==1:return dict(chain=[a],steps=[])
    chain=[a,derivative(a)];steps=[]
    while True:
        q,r=divide(chain[-2],chain[-1])
        if zero(r):
            steps.append(dict(quotient=q,scale=None,next=None));break
        nxt,scale=primitive(neg(r));check_size(nxt)
        steps.append(dict(quotient=q,scale=scale,next=nxt))
        chain.append(nxt)
    return dict(chain=chain,steps=steps)

def variation(chain,x):
    signs=[sign(value(p,x)) for p in chain]
    nz=[s for s in signs if s]
    return sum(a!=b for a,b in zip(nz,nz[1:])),signs

def time_lattice(segments):
    durations=[F(t) for t,u in segments if F(t)>0]
    if not durations:return F(1),0,[0]*len(segments)
    den=reduce(lcm,(t.denominator for t in durations),1)
    g=reduce(gcd,(int(t*den) for t in durations),0)
    h=F(g,den);powers=[int(F(t)/h) for t,u in segments]
    return h,sum(powers),powers

def moments(case):
    segments=[tuple(map(F,z)) for z in case['past']]
    T=sum(t for t,u in segments);t0=F(0);area=F(0);weighted=F(0)
    for dt,u in segments:
        area+=dt*u
        weighted+=u*((T-t0)*dt-dt*dt/2)
        t0+=dt
    x0,v0,a0=map(F,case['initial'])
    return dict(T=T,area=area,Q=area-a0*T,v_star=v0+area,
                x_star=x0+v0*T+weighted)

def signature(case):
    data={k:case[k] for k in ('initial','past','prior','packet','future','H','rho','limit')}
    return hashlib.sha256(json.dumps(encode(data),sort_keys=True).encode()).hexdigest()

def prepare(case,mode):
    assert mode in ('accel_sturm','aligned_sturm')
    h,degree,powers=time_lattice(case['past'])
    meta=dict(mode=mode,input_signature=signature(case),h=h,degree=degree,powers=powers,moments=moments(case),
              degree_budget=MAX_DEGREE,coefficient_bit_budget=MAX_COEFFICIENT_BITS)
    try:
        if degree>MAX_DEGREE:raise BudgetExceeded('DEGREE_BUDGET')
        p=[case['initial'][2]]
        for (dt,u),m in zip(case['past'],powers):
            p=shift_constant([F(0)]*m+shift_constant(p,-u),u)
        p=trim(p);check_size(p)
        faces=[]
        def face(name,poly,power):
            faces.append(dict(name=name,violation_polynomial=trim(poly),positive_theta_power=power,sturm=sturm(poly)))
        face('a_lower',shift_constant(neg(p),case['packet'][2][0]),0)
        face('a_upper',shift_constant(p,-case['packet'][2][1]),0)
        if mode=='aligned_sturm':
            f=shift_constant(p,-case['initial'][2]);mom=meta['moments']
            if case['packet'][1][0]==mom['v_star']:face('v_lower',f,1)
            if case['packet'][1][1]==mom['v_star']:face('v_upper',neg(f),1)
            if mom['Q']==0:
                if case['packet'][0][0]==mom['x_star']:face('x_lower',neg(f),2)
                if case['packet'][0][1]==mom['x_star']:face('x_upper',f,2)
        saved=[c for face_data in faces for poly in face_data['sturm']['chain'] for c in poly]
        saved += [c for face_data in faces for step in face_data['sturm']['steps'] for c in step['quotient']]
        meta.update(status='READY',acceleration_polynomial=p,faces=faces,
                    stored_sturm_max_coefficient_bits=max((max(abs(c.numerator).bit_length(),c.denominator.bit_length()) for c in saved),default=0),
                    acceleration_nonzero_terms=sum(c!=0 for c in p))
    except BudgetExceeded as e:
        meta.update(status='FALLBACK_'+str(e),faces=[])
    return meta

def reject_cell(meta,l,r):
    if meta['status']!='READY':return None
    h=meta['h']
    zl=exp_neg(h/l).lower();zu=exp_neg(h/r).upper()
    for idx,f in enumerate(meta['faces']):
        p=f['violation_polynomial']
        # Strict positivity at both endpoints also rules out roots there.
        if value(p,zl)<=0 or value(p,zu)<=0:continue
        vl,sl=variation(f['sturm']['chain'],zl)
        vu,su=variation(f['sturm']['chain'],zu)
        if vl==vu:
            return dict(face_index=idx,face_name=f['name'],theta_interval=[l,r],
                        z_interval=[zl,zu],variation=[vl,vu],signs=[sl,su])
    return None

def encode(o):
    if isinstance(o,F):return str(o)
    if isinstance(o,dict):return {k:encode(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [encode(v) for v in o]
    return o
