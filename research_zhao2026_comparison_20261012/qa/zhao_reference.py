"""Independent ordered Carleman algebra; never imports producer modules."""
from fractions import Fraction as Q
from functools import lru_cache
import itertools
import json

import independent_ma_reference as ref


def expression_sum(*expressions):
    result={}
    for expression in expressions:
        for power,tau,coefficient in expression:
            key=power,tau
            result[key]=result.get(key,Q(0))+coefficient
    return ref.tidy(result)


def expression_scale(expression,coefficient):
    return ref.tidy({(power,tau):value*Q(coefficient) for power,tau,value in expression})


@lru_cache(maxsize=50000)
def expression_product(left,right):
    result={}
    for p,t,c in left:
        for pp,tt,cc in right:
            key=p+pp,t+tt
            result[key]=result.get(key,Q(0))+c*cc
    return ref.tidy(result)


@lru_cache(maxsize=2000)
def model(serialized,q_center,nu):
    assert nu in (1,2)
    base=(((1,Q(0),Q(1)),(0,Q(0),-q_center)),)+ref.endpoint_terms(json.loads(serialized))
    ordered=[((0,Q(0),Q(1)),)]
    indices=[()]
    for degree in range(1,nu+1):
        for word in itertools.product(range(4),repeat=degree):
            expression=((0,Q(0),Q(1)),)
            for letter in word:
                expression=expression_product(expression,base[letter])
            indices.append(word)
            ordered.append(expression)
    return tuple(indices),tuple(ordered)


def get_model(case,q_center,nu):
    serialized=json.dumps({'initial':case['initial'],'past':case['past']},sort_keys=True)
    return model(serialized,Q(q_center),nu)


@lru_cache(maxsize=100000)
def point_range(expression,point):
    """Combine coefficients of identical exponentials before enclosing them.

    This proves exact cancellations such as (q-c)*g(q)=0 at q=c without
    asking an interval representation to cancel uncertain exponentials.
    """
    coefficients={}
    for power,tau,value in expression:
        coefficients[tau]=coefficients.get(tau,Q(0))+value*point**power
    result=(Q(0),Q(0))
    for tau,value in coefficients.items():
        if value:
            result=ref.add(result,ref.scale(ref.exp_minus(tau*point),value))
    return result


@lru_cache(maxsize=100000)
def prove_range_contained(expression,low,high,outer,depth=0):
    enclosed=ref.strong_range(expression,low,high)
    if ref.contained(enclosed,outer):
        return True
    # A natural producer expression can attain its tight extreme at a domain
    # endpoint. Certify monotonicity, then use endpoint enclosures, rather than
    # waiting for an arbitrary Taylor dependency width to vanish by subdivision.
    slope=ref.strong_range(ref.derivative(expression),low,high)
    if slope[0]>=0:
        sharp=(point_range(expression,low)[0],point_range(expression,high)[1])
        if ref.contained(sharp,outer):
            return True
    elif slope[1]<=0:
        sharp=(point_range(expression,high)[0],point_range(expression,low)[1])
        if ref.contained(sharp,outer):
            return True
    midpoint=(low+high)/2
    point=point_range(expression,midpoint)
    assert ref.contained(point,outer),('point outside derivative certificate',midpoint,point,outer)
    assert depth<110,('independent derivative coverage exhausted',low,high,enclosed,outer)
    return (prove_range_contained(expression,low,midpoint,outer,depth+1) and
            prove_range_contained(expression,midpoint,high,outer,depth+1))


def matrix(value):
    result=[list(map(Q,row)) for row in value]
    assert result and all(len(row)==len(result[0]) for row in result)
    return result


def zeros(rows,columns):
    return [[Q(0) for _ in range(columns)] for _ in range(rows)]


def identity(n):
    return [[Q(i==j) for j in range(n)] for i in range(n)]


def transpose(A):
    return list(map(list,zip(*A)))


def mm(A,B):
    assert A and B and len(A[0])==len(B)
    return [[sum((a*b for a,b in zip(row,column)),Q(0))
             for column in zip(*B)] for row in A]


def mv(A,b):
    assert A and len(A[0])==len(b)
    return [sum((a*v for a,v in zip(row,b)),Q(0)) for row in A]


def add(A,B,sign=1):
    assert len(A)==len(B) and all(len(a)==len(b) for a,b in zip(A,B))
    return [[x+sign*y for x,y in zip(a,b)] for a,b in zip(A,B)]


def scale(A,c):
    return [[v*c for v in row] for row in A]


def hcat(*matrices):
    rows=len(matrices[0])
    assert all(len(A)==rows for A in matrices)
    return [list(itertools.chain.from_iterable(A[i] for A in matrices)) for i in range(rows)]


def diagonal(values):
    return [[value if i==j else Q(0) for j in range(len(values))] for i,value in enumerate(values)]


def infinity_norm(A):
    return max((sum(map(abs,row),Q(0)) for row in A),default=Q(0))


def squared_frobenius(A):
    return sum((entry*entry for row in A for entry in row),Q(0))


def initial_lift(radius,nu):
    size=5 if nu==1 else 21
    center=[Q(0)]*size;center[0]=Q(1)
    generator=zeros(size,1 if nu==1 else 2)
    generator[1][0]=radius
    if nu==2:
        center[5]=radius*radius/2
        generator[5][1]=radius*radius/2
    return center,generator


def xi1(center):
    assert len(center)==4
    return [[(center[j] if k==i else 0)+(center[i] if k==j else 0)
             for k in range(4)] for i,j in itertools.product(range(4),repeat=2)]


def lifted_square(center):
    return [a*b for a,b in itertools.product(center,repeat=2)]


def self_check():
    c=Q(3,2)
    case=dict(initial=['1','2','3'],past=[['1/2','3']])
    indices,expressions=get_model(case,c,2)
    assert len(indices)==len(expressions)==21
    assert indices[5]==(0,0) and indices[6]==(0,1) and indices[9]==(1,0)
    for index,expression in enumerate(expressions):
        # All physical endpoint coordinates are constant in q for this case.
        assert ref.derivative(expression,3)==()
        if index not in (1,5,6,7,8,9,13,17):
            assert ref.derivative(expression)==()
    assert ref.natural_range(expressions[5],c,c)==(Q(0),Q(0))
    assert ref.natural_range(ref.derivative(expressions[5],2),c,c)==(Q(2),Q(2))
    center,generator=initial_lift(Q(2),2)
    assert center[5]==generator[5][1]==2
    assert infinity_norm([[Q(1),Q(-2)],[Q(2),Q(0)]])==3
    point=[Q(1),Q(2),Q(3),Q(4)]
    assert mv(xi1(point),point)==[2*v for v in lifted_square(point)]
    assert hcat(identity(2),diagonal([Q(3),Q(4)]))==[[1,0,3,0],[0,1,0,4]]
    return dict(status='PASS',scope='ordered polynomial lift and exact matrix helpers; no production records')


if __name__=='__main__':
    print(json.dumps(self_check(),indent=2))
