"""Refined standard Taylor baseline and optional positive-curvature exclusion.

Added after the first frozen benchmark exposed natural-interval cancellation.
All old methods/results remain unchanged. Inputs remain the same frozen cases.
"""
from fractions import Fraction as F
from math import factorial
import certified_core as cc
import sparse_faces as sf

prepare=sf.prepare

def centered_enclosure(face,a,b,derivative=0):
    center=(a+b)/2;radius=(b-a)/2
    result=sf.evaluate(face,center,center,derivative)
    for j in range(1,5):
        power=cc.IV.of(radius**j)
        delta=cc.IV(-power.hi if j%2 else 0,power.hi)
        result += sf.evaluate(face,center,center,derivative+j)*delta/F(factorial(j))
    power=cc.IV.of(radius**5)
    result += sf.evaluate(face,a,b,derivative+5)*cc.IV(-power.hi,power.hi)/F(factorial(5))
    return result

def reject_cell(meta,l,r,curvature=False):
    a,b=F(1)/r,F(1)/l
    for idx,face in enumerate(meta['faces']):
        cert=None
        natural=sf.evaluate(face,a,b)
        if natural.lower()>0:
            cert=dict(kind='natural',certified_lower=natural.lower())
        elif natural.upper()<=0:
            continue
        else:
            # A point proven feasible for this face precludes strict whole-cell
            # exclusion by this face. This is only a cheap producer shortcut.
            if any(sf.evaluate(face,p,p).upper()<=0 for p in (a,b)):
                continue
            direct=centered_enclosure(face,a,b)
            if direct.lower()>0:
                cert=dict(kind='taylor',certified_lower=direct.lower())
            else:
                slope=centered_enclosure(face,a,b,1)
                if slope.lower()>=0:
                    lower=sf.evaluate(face,a,a).lower()
                    if lower>0:
                        cert=dict(kind='taylor_increasing',point=a,certified_lower=lower)
                elif slope.upper()<=0:
                    lower=sf.evaluate(face,b,b).lower()
                    if lower>0:
                        cert=dict(kind='taylor_decreasing',point=b,certified_lower=lower)
                elif curvature:
                    second=sf.second_enclosure(face,a,b)
                    if second.lower()>0:
                        cert=sf.positive_curvature(face,a,b,second.lower(),sf.MAX_STATIONARY_STEPS)
                    elif second.upper()<=0:
                        lower=min(sf.evaluate(face,p,p).lower() for p in (a,b))
                        if lower>0:
                            cert=dict(kind='concave',certified_lower=lower)
        if cert:
            return dict(face_index=idx,face_name=face['name'],theta_interval=[l,r],
                        q_interval=[a,b],**cert)
    return None
