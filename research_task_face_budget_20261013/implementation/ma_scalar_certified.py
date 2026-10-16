"""Certified one-parameter batch adaptation of Ma et al., Automatica 2026.

Implements Theorems 1, 4, 5, including the signed square remainder.
For a scalar state and diagonal measurement/remainder generators, Eq. (18)
has the exact weighted-l1 solution min(r, s_i / abs(C_i)).
Optional intersection of all scalar consistency strips strengthens the paper
LP result and is explicitly an added adapter operation, not Eq. (18).
"""
from fractions import Fraction as F
import certified_core as cc
import sparse_faces as sf


def iv_range(lo, hi):
    return cc.IV(cc.IV.of(lo).lo, cc.IV.of(hi).hi)


def prepare(case):
    meta = sf.prepare(case)
    face = meta['faces'][1]  # a_upper = a(q) - packet upper
    a0 = F(case['initial'][2])
    A = dict(poly=[face['poly'][0] + F(case['packet'][2][1]) - a0],
             terms=face['terms'])
    return dict(A=A, a0=a0, **meta['moments'])


def history_jet(model, lo, hi):
    """Outward value, first derivative, second derivative wrt q."""
    A = sf.evaluate(model['A'], lo, hi, 0)
    A1 = sf.evaluate(model['A'], lo, hi, 1)
    A2 = sf.evaluate(model['A'], lo, hi, 2)
    qi = cc.IV.of(1) / iv_range(lo, hi)
    qi2, qi3, qi4 = qi*qi, qi*qi*qi, qi*qi*qi*qi
    Q, vs, xs = model['Q'], model['v_star'], model['x_star']
    value = (xs-Q*qi+A*qi2, vs-A*qi, model['a0']+A)
    first = (Q*qi2+A1*qi2-2*A*qi3, -A1*qi+A*qi2, A1)
    second = (-2*Q*qi3+A2*qi2-4*A1*qi3+6*A*qi4,
              -A2*qi+2*A1*qi2-2*A*qi3, A2)
    return value, first, second


def hessian_bounds(model, lo, hi, pieces=1):
    """Natural interval evaluation, optionally over a fixed partition."""
    pieces = int(pieces)
    assert pieces >= 1
    bounds = []
    for j in range(pieces):
        a = lo + (hi-lo)*F(j, pieces)
        b = lo + (hi-lo)*F(j+1, pieces)
        bounds.append(history_jet(model, a, b)[2])
    return tuple(cc.IV(min(row[i].lo for row in bounds),
                       max(row[i].hi for row in bounds)) for i in range(3))


def contract(model, packet, theta_lo, theta_hi, strengthen=True, hessian_pieces=1):
    """Return retained theta interval or None, plus exact proof data.

Rounding error in g(c) and g'(c) is included in the remainder/strip radius.
The scalar LP uses rational point slopes, so equality T+N C=1 is exact.
"""
    qlo, qhi = F(1)/theta_hi, F(1)/theta_lo
    c, radius = (qlo+qhi)/2, (qhi-qlo)/2
    value, deriv, _ = history_jet(model, c, c)
    hess = hessian_bounds(model, qlo, qhi, hessian_pieces)
    strips = []
    candidates = [(radius, 'prior', (qlo, qhi))]
    incompatible = None
    for j in range(3):
        C = deriv[j].midpoint()
        rounding_radius = deriv[j].radius()*radius
        rem_lo = min(F(0), hess[j].lower())*radius*radius/2
        rem_hi = max(F(0), hess[j].upper())*radius*radius/2
        blo, bhi = map(F, packet[j])
        # C*q is in this interval whenever g(q) is in the packet.
        low = blo-value[j].upper()-rem_hi-rounding_radius+C*c
        high = bhi-value[j].lower()-rem_lo+rounding_radius+C*c
        strip = dict(coordinate=j, C=C, low=low, high=high,
                     value=[value[j].lower(), value[j].upper()],
                     derivative=[deriv[j].lower(), deriv[j].upper()],
                     hessian=[hess[j].lower(), hess[j].upper()],
                     remainder=[rem_lo, rem_hi], slope_rounding=rounding_radius)
        if C:
            ends = sorted((low/C, high/C))
            strip['q_interval'] = tuple(ends)
            candidates.append(((high-low)/(2*abs(C)), j, tuple(ends)))
        elif not low <= 0 <= high:
            incompatible = j
        strips.append(strip)
    best = min(candidates, key=lambda x: x[0])  # deterministic prior/coordinate ties
    lp_q = best[2]
    lo, hi = max(qlo, lp_q[0]), min(qhi, lp_q[1])
    if strengthen:
        for strip in strips:
            if strip['C']:
                lo = max(lo, strip['q_interval'][0])
                hi = min(hi, strip['q_interval'][1])
    # Constant inconsistent rows of C*q must be checked even in native LP mode.
    # This is a direct consistency check, not an optimizer termination inference.
    rejected = incompatible is not None or lo > hi
    cert = dict(kind='ma2026_scalar', theta_input=[theta_lo, theta_hi],
                q_input=[qlo, qhi], center=c, radius=radius,
                hessian_pieces=hessian_pieces, strips=strips,
                lp_choice=best[1], lp_radius=best[0], lp_q_interval=lp_q,
                strengthened=bool(strengthen), zero_row_inconsistent=incompatible,
                q_intersection=[lo, hi], rejected=rejected)
    if rejected:
        return None, cert
    result = (F(1)/hi, F(1)/lo)
    cert['theta_output'] = result
    return result, cert
