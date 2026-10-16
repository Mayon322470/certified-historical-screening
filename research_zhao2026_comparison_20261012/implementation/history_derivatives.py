"""Certified derivatives of the shared exact history map with respect to q.

Uses the exponential sum A(q)=a(T;q)-a(0), physical moments Q,v*,x*.
All natural interval bounds are outward 96-bit and cover the entire input cell.
"""
from fractions import Fraction as F
from math import comb,factorial
import certified_core as cc
import sparse_faces as sf
from ma_scalar_certified import prepare,iv_range


def derivatives(model,lo,hi,order):
    assert 0<lo<=hi and order>=0
    A=[sf.evaluate(model['A'],lo,hi,k) for k in range(order+1)]
    inv=cc.IV.of(1)/iv_range(lo,hi)
    powers=[cc.IV.of(1)]
    for k in range(order+2):powers.append(powers[-1]*inv)
    def negative_power(m,k):
        coefficient=(-1)**k*F(factorial(m+k-1),factorial(m-1))
        return coefficient*powers[m+k]
    result=[]
    for k in range(order+1):
        x=cc.IV.of(model['x_star'] if k==0 else 0)-model['Q']*negative_power(1,k)
        v=cc.IV.of(model['v_star'] if k==0 else 0)
        for j in range(k+1):
            x += comb(k,j)*A[j]*negative_power(2,k-j)
            v -= comb(k,j)*A[j]*negative_power(1,k-j)
        a=A[k]+(model['a0'] if k==0 else 0)
        result.append((x,v,a))
    return result


def derivative_hulls(model,lo,hi,order,pieces=1):
    assert pieces>=1
    part=[derivatives(model,lo+(hi-lo)*F(j,pieces),lo+(hi-lo)*F(j+1,pieces),order)
          for j in range(pieces)]
    return [tuple(cc.IV(min(p[k][i].lo for p in part),max(p[k][i].hi for p in part))
                  for i in range(3)) for k in range(order+1)]
