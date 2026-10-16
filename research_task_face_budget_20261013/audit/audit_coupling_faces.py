"""Independent exact reconstruction/replay of common-beta necessary faces."""
from fractions import Fraction as F
from math import factorial
import independent_ma_reference as ref


def canonical(expression):
    terms={}
    for power,value in enumerate(expression['poly']):terms[power,F(0)]=F(value)
    for decay,poly in expression['terms']:
        for power,value in enumerate(poly):
            key=power,F(decay);terms[key]=terms.get(key,F(0))+F(value)
    return ref.tidy(terms)


def combine(parts,constant=F(0)):
    terms={(2,F(0)):F(constant)}
    for scale,expression in parts:
        for power,decay,value in expression:
            key=power+2,decay;terms[key]=terms.get(key,F(0))+F(scale)*value
    return ref.tidy(terms)


def reference_faces(case):
    g=ref.endpoint_terms(case)
    T=sum((F(dt) for dt,_ in case['past']),F(0));L=T+F(case['H'])
    response=case['response'];b=F(response['epsilon_x'])
    assert response['kind']=='normalized_global_cubic_v1' and F(response['L'])==L and L>0
    assert b>=0 and F(case['rho'])==0
    p=(T**3/L**3,3*T*T/L**3,6*T/L**3)
    packet=[tuple(map(F,pair)) for pair in case['packet']]
    faces=[]
    for i,axis in enumerate(('x','v','a')):
        for side in ('lower','upper'):
            expr=combine([(-1,g[i])],packet[i][0]-b*p[i]) if side=='lower' else combine([(1,g[i])],-packet[i][1]-b*p[i])
            faces.append((f'{axis}_{side}_beta_prior','beta_prior',expr))
    if all(pi>0 for pi in p):
        for i,axis in enumerate(('x','v','a')):
            for j,other in enumerate(('x','v','a')):
                if i!=j:
                    faces.append((f'{axis}_lower_vs_{other}_upper','same_beta',combine([(-p[j],g[i]),(p[i],g[j])],p[j]*packet[i][0]-p[i]*packet[j][1])))
    return faces


def qrange(expression,lo,hi,order=0):
    """Different symbolic derivative, exact quadratic extrema, 220-bit exp."""
    groups={}
    for power,decay,value in ref.derivative(expression,order):
        assert power in (0,1,2)
        groups.setdefault(decay,[F(0)]*3)[power]+=value
    result=F(0),F(0)
    for decay,(a,b,c) in groups.items():
        candidates=[a+b*q+c*q*q for q in (lo,hi)]
        if c and lo < -b/(2*c) < hi:
            q=-b/(2*c);candidates.append(a+b*q+c*q*q)
        e=(ref.exp_minus(decay*hi)[0],ref.exp_minus(decay*lo)[1])
        result=ref.add(result,ref.mul((min(candidates),max(candidates)),e))
    return result


def centered(expression,lo,hi,order=0):
    c,r=(lo+hi)/2,(hi-lo)/2
    value=qrange(expression,c,c,order)
    for j in range(1,5):
        power=(-r**j,r**j) if j%2 else (F(0),r**j)
        value=ref.add(value,ref.scale(ref.mul(qrange(expression,c,c,order+j),power),F(1,factorial(j))))
    remainder=ref.mul(qrange(expression,lo,hi,order+5),(-r**5,r**5))
    return ref.add(value,ref.scale(remainder,F(1,factorial(5))))


def intersect_range(expression,lo,hi,order):
    a,b=qrange(expression,lo,hi,order),centered(expression,lo,hi,order)
    return max(a[0],b[0]),min(a[1],b[1])


def check_certificate(case,cert):
    tl,th=map(F,cert['theta_interval']);assert 0<tl<=th
    lo,hi=map(F,cert['q_interval']);assert (lo,hi)==(1/th,1/tl)
    faces=reference_faces(case)
    name,family,expression=faces[cert['face_index']]
    assert name==cert['face_name'] and family==cert['face_family']
    assert cert['face_scale']=='q_squared_positive'
    assert canonical(cert['expression'])==expression
    source=cert['source']
    for key in ('initial',):assert tuple(map(F,source[key]))==tuple(map(F,case[key]))
    for key,target in [('past','past'),('original_packet','packet')]:
        assert [tuple(map(F,row)) for row in source[key]]==[tuple(map(F,row)) for row in case[target]]
    for key in ('H','rho'):assert F(source[key])==F(case[key])
    assert source['response']['kind']==case['response']['kind']
    for key in ('L','epsilon_x'):assert F(source['response'][key])==F(case['response'][key])
    lower=F(cert['certified_lower']);assert lower>0
    kind=cert['kind']
    if kind=='natural':assert lower<=qrange(expression,lo,hi)[0]
    elif kind=='taylor':assert lower<=centered(expression,lo,hi)[0]
    elif kind=='monotone_increasing':
        assert F(cert['point'])==lo and intersect_range(expression,lo,hi,1)[0]>=0
        assert lower<=qrange(expression,lo,lo)[0]
    elif kind=='monotone_decreasing':
        assert F(cert['point'])==hi and intersect_range(expression,lo,hi,1)[1]<=0
        assert lower<=qrange(expression,hi,hi)[0]
    elif kind=='concave':
        assert intersect_range(expression,lo,hi,2)[1]<=0
        assert lower<=min(qrange(expression,lo,lo)[0],qrange(expression,hi,hi)[0])
    elif kind=='curvature':
        mu,c=F(cert['mu']),F(cert['center'])
        assert lo<=c<=hi and 0<mu<=intersect_range(expression,lo,hi,2)[0]
        value,slope=tuple(map(F,cert['value_range'])),tuple(map(F,cert['slope_range']))
        assert ref.contained(qrange(expression,c,c),value)
        assert ref.contained(qrange(expression,c,c,1),slope)
        magnitude=max(map(abs,slope));assert F(cert['slope_abs_upper'])==magnitude
        assert lower==value[0]-magnitude*magnitude/(2*mu)
        assert isinstance(cert['stationary_attempts'],int) and 1<=cert['stationary_attempts']<=32
    else:raise AssertionError(kind)
    return True
