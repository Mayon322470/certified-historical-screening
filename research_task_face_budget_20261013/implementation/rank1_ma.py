"""Ma2026 scalar ZFO with the full shared rank-one response generator.

Eq18 weighted-l1 gain is solved exactly, including the response column.
The all-strip Fourier-Motzkin projection is an explicit strengthening shared by
both Ma configurations; it is not mislabeled as the paper's original update.
"""
from fractions import Fraction as F
from itertools import combinations
import ma_scalar_certified as base

prepare = base.prepare


def det3(m):
    a, b, c = m
    return (a[0] * (b[1]*c[2] - b[2]*c[1])
            - a[1] * (b[0]*c[2] - b[2]*c[0])
            + a[2] * (b[0]*c[1] - b[1]*c[0]))


def solve3(matrix, rhs):
    determinant = det3(matrix)
    if determinant == 0:
        return None
    return tuple(det3([[rhs[i] if k == j else matrix[i][k] for k in range(3)]
                       for i in range(3)]) / determinant for j in range(3))


def optimal_gain(radius, C, noise_halfwidth, response_shape, response_bound):
    """Coercive 3D piecewise-affine objective; at most ten arrangement vertices."""
    R, b = F(radius), F(response_bound)
    C, s, psi = [tuple(map(F, v)) for v in (C, noise_halfwidth, response_shape)]
    if R < 0 or b < 0 or any(v <= 0 for v in s) or any(len(v) != 3 for v in (C, s, psi)):
        raise ValueError('Positive diagonal noise widths and three measurement rows required')
    hyperplanes = [((F(1), F(0), F(0)), F(0)),
                   ((F(0), F(1), F(0)), F(0)),
                   ((F(0), F(0), F(1)), F(0)), (psi, F(0)), (C, F(1))]

    def objective(N):
        T = 1 - sum((n*c for n, c in zip(N, C)), F(0))
        return abs(T)*R + sum((abs(n)*v for n, v in zip(N, s)), F(0)) + b*abs(sum((n*p for n, p in zip(N, psi)), F(0)))

    candidates = [dict(N=(F(0),)*3, objective=R, active_planes=None)]
    for ids in combinations(range(5), 3):
        N = solve3([hyperplanes[i][0] for i in ids], [hyperplanes[i][1] for i in ids])
        if N is not None:
            candidates.append(dict(N=N, objective=objective(N), active_planes=ids))
    best = min(candidates, key=lambda row: row['objective'])
    N = best['N']
    return N, best['objective'], dict(kind='exact_rank1_weighted_l1_gain', radius=R,
                                     C=C, diagonal_noise_halfwidth=s, response_shape=psi,
                                     response_bound=b, candidates=candidates, selected=best,
                                     T=1-sum((n*c for n, c in zip(N, C)), F(0)))


def project_strips(strips, psi, bound, q_interval):
    """Exact projection of low_j<=C_j*q+psi_j*beta<=high_j, |beta|<=b."""
    if len(strips) != 3 or len(psi) != 3 or len(q_interval) != 2 or F(bound) < 0:
        raise ValueError('Three strips, three shapes, and a nonnegative response bound required')
    lo, hi = map(F, q_interval)
    if lo > hi:
        raise ValueError('Ordered q interval required')
    beta_lower = [(F(0), -F(bound), 'prior_lower')]
    beta_upper = [(F(0), F(bound), 'prior_upper')]
    inequalities = []
    for j, (row, p) in enumerate(zip(strips, psi)):
        p = F(p)
        C, low, high = row['C'], row['low'], row['high']
        if p < 0:
            raise ValueError('This normalized-profile adapter requires nonnegative shape')
        if p == 0:
            inequalities.extend([(C, -high, ('zero_shape_upper', j)),
                                 (-C, low, ('zero_shape_lower', j))])
        else:
            beta_lower.append((-C/p, low/p, ('packet_lower', j)))
            beta_upper.append((-C/p, high/p, ('packet_upper', j)))
    for a0, b0, id0 in beta_lower:
        for a1, b1, id1 in beta_upper:
            inequalities.append((a0-a1, b0-b1, (id0, id1)))
    rejected_by = None
    for slope, intercept, ids in inequalities:
        if slope > 0:
            hi = min(hi, -intercept/slope)
        elif slope < 0:
            lo = max(lo, -intercept/slope)
        elif intercept > 0:
            rejected_by = ids
        if lo > hi or rejected_by is not None:
            return None, dict(inequalities=inequalities, rejected_by=ids,
                              q_projection=(lo, hi), rejected=True)
    return (lo, hi), dict(inequalities=inequalities, rejected_by=None,
                         q_projection=(lo, hi), rejected=False)


def contract(model, packet, theta_lo, theta_hi, response_shape, response_bound,
             hessian_pieces=1, collect=True):
    qlo, qhi = F(1)/F(theta_hi), F(1)/F(theta_lo)
    center, radius = (qlo+qhi)/2, (qhi-qlo)/2
    value, derivative, _ = base.history_jet(model, center, center)
    hessian = base.hessian_bounds(model, qlo, qhi, hessian_pieces)
    strips, slopes, centers, widths = [], [], [], []
    measurement_center = []
    for j in range(3):
        C = derivative[j].midpoint()
        rounding = derivative[j].radius()*radius
        rlo = min(F(0), hessian[j].lower())*radius*radius/2
        rhi = max(F(0), hessian[j].upper())*radius*radius/2
        blo, bhi = map(F, packet[j])
        low = blo-value[j].upper()-rhi-rounding+C*center
        high = bhi-value[j].lower()-rlo+rounding+C*center
        strips.append(dict(coordinate=j, C=C, low=low, high=high,
                           value=(value[j].lower(), value[j].upper()),
                           derivative=(derivative[j].lower(), derivative[j].upper()),
                           hessian=(hessian[j].lower(), hessian[j].upper()),
                           remainder=(rlo, rhi), slope_rounding=rounding))
        slopes.append(C)
        centers.append(value[j].midpoint()+(rlo+rhi)/2)
        widths.append((bhi-blo)/2+value[j].radius()+(rhi-rlo)/2+rounding)
        measurement_center.append((blo+bhi)/2)
    N, post_radius, gain = optimal_gain(radius, slopes, widths, response_shape, response_bound)
    post_center = center + sum((n*(y-h) for n, y, h in zip(N, measurement_center, centers)), F(0))
    gain_interval = (max(qlo, post_center-post_radius), min(qhi, post_center+post_radius))
    projected, projection = project_strips(strips, response_shape, response_bound, (qlo, qhi))
    rejected = projected is None or gain_interval[0] > gain_interval[1]
    if not rejected:
        retained = (max(gain_interval[0], projected[0]), min(gain_interval[1], projected[1]))
        rejected = retained[0] > retained[1]
    else:
        retained = gain_interval
    cert = dict(kind='ma2026_shared_rank1_scalar', theta_input=(theta_lo, theta_hi),
                q_input=(qlo, qhi), center=center, radius=radius, hessian_pieces=hessian_pieces,
                gain_certificate=gain, posterior_center=post_center, posterior_radius=post_radius,
                gain_q_interval=gain_interval, all_strip_projection=projection,
                q_intersection=retained, rejected=rejected, strengthened=True,
                noise_contract='original packet diagonal + same response beta rank1 column')
    if collect:
        cert.update(strips=strips, prediction_center=centers, measurement_center=measurement_center,
                    diagonal_noise_halfwidth=widths, response_shape=list(response_shape),
                    response_bound=F(response_bound))
    if rejected:
        return None, cert
    result = (1/retained[1], 1/retained[0])
    cert['theta_output'] = result
    return result, cert
