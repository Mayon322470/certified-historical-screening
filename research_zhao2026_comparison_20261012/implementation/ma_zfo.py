"""Independent implementation of Ma et al., Automatica 192 (2026) 113189.

Implements Theorem 1 / Eqs. (8)--(10), prediction (Theorem 3),
consistent measurements (Theorem 4), and interval-hull LP (Theorem 5).
This is ordinary floating-point algorithm code, not a rounding certificate.
The caller supplies a VALID interval Hessian oracle; sampling is not one.
No sampled or failed LP can establish an empty compatible set here.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
from scipy.optimize import linprog


@dataclass
class Zonotope:
    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.generators = np.asarray(self.generators, dtype=float)
        if self.generators.ndim != 2 or self.generators.shape[0] != len(self.center):
            raise ValueError('generators must have shape (state_dimension, count)')
        if not (np.isfinite(self.center).all() and np.isfinite(self.generators).all()):
            raise ValueError('zonotope entries must be finite')

    @property
    def radius(self):
        return np.abs(self.generators).sum(axis=1)

    @property
    def bounds(self):
        return self.center - self.radius, self.center + self.radius

    def as_dict(self):
        return {'center': self.center.tolist(), 'generators': self.generators.tolist(),
                'interval_lower': self.bounds[0].tolist(),
                'interval_upper': self.bounds[1].tolist()}


def _groups(n, order=None, groups=None):
    if order is not None and groups is not None:
        raise ValueError('specify either scalar order or ordered groups')
    answer = [[int(i)] for i in (range(n) if order is None else order)] if groups is None else [list(map(int, g)) for g in groups]
    if any(not g for g in answer) or sorted(i for g in answer for i in g) != list(range(n)):
        raise ValueError('ordered groups must partition all coordinates exactly once')
    return answer


def ordered_hessian(z, hessian_bounds: Callable, output_dimension, order=None, groups=None):
    """Eq. (10), recursively split in the ordered group sequence.

    For entry (i,j), fix all groups BEFORE the later of their two groups
    at the center. Keep that later group and every subsequent group as
    intervals. In particular the diagonal uses its own group as later.
    All blocks in a common group use the same interval box.
    """
    n, m = len(z.center), output_dimension
    ordered = _groups(n, order, groups)
    positions = {i: k for k, group in enumerate(ordered) for i in group}
    lower = np.empty((m, n, n)); upper = np.empty((m, n, n))
    box_lo, box_hi = (x.copy() for x in z.bounds)
    boxes = []
    for stage, group in enumerate(ordered):
        hlo, hhi = (np.asarray(x, dtype=float) for x in hessian_bounds(box_lo.copy(), box_hi.copy()))
        if hlo.shape != (m, n, n) or hhi.shape != hlo.shape:
            raise ValueError('Hessian oracle must return two arrays with shape (m,n,n)')
        if np.any(hlo > hhi) or not (np.isfinite(hlo).all() and np.isfinite(hhi).all()):
            raise ValueError('invalid Hessian enclosure')
        for i in range(n):
            for j in range(i, n):
                if max(positions[i], positions[j]) == stage:
                    # Symmetric hull tolerates nonidentical interval evaluation
                    # of the two analytic mixed partials without narrowing.
                    lower[:, i, j] = lower[:, j, i] = np.minimum(hlo[:, i, j], hlo[:, j, i])
                    upper[:, i, j] = upper[:, j, i] = np.maximum(hhi[:, i, j], hhi[:, j, i])
        boxes.append({'stage': stage, 'group': group, 'lower': box_lo.tolist(), 'upper': box_hi.tolist()})
        box_lo[group] = z.center[group]
        box_hi[group] = z.center[group]
    return lower, upper, ordered, boxes


def improved_remainder(z, hessian_bounds, output_dimension, order=None, groups=None):
    """Eq. (9), retaining delta_i**2 in [0,r_i**2]."""
    low, high, ordered, boxes = ordered_hessian(z, hessian_bounds, output_dimension, order, groups)
    r = z.radius
    absolute = np.maximum(np.abs(low), np.abs(high))
    idx = np.arange(len(r))
    absolute[:, idx, idx] = 0.0
    offdiag = 0.5 * np.einsum('i,mij,j->m', r, absolute, r)
    diaglow = np.diagonal(low, axis1=1, axis2=2)
    diaghigh = np.diagonal(high, axis1=1, axis2=2)
    remainder_low = 0.5 * np.minimum(0., diaglow * r**2).sum(axis=1) - offdiag
    remainder_high = 0.5 * np.maximum(0., diaghigh * r**2).sum(axis=1) + offdiag
    return remainder_low, remainder_high, {
        'method': 'Ma2026_Theorem1_Eq8_10', 'groups': ordered,
        'hessian_lower': low.tolist(), 'hessian_upper': high.tolist(),
        'hessian_evaluation_boxes': boxes,
        'remainder_lower': remainder_low.tolist(), 'remainder_upper': remainder_high.tolist(),
        'numeric_guarantee': 'ordinary_float_requires_external_rounding_verification'}


def improved_zfo(z, fun, jacobian, hessian_bounds, order=None, groups=None):
    fp = np.asarray(fun(z.center), dtype=float).reshape(-1)
    jac = np.asarray(jacobian(z.center), dtype=float)
    if jac.shape != (len(fp), len(z.center)):
        raise ValueError('Jacobian shape mismatch')
    low, high, info = improved_remainder(z, hessian_bounds, len(fp), order, groups)
    out = Zonotope(fp + (low + high) / 2,
                   np.column_stack((jac @ z.generators, np.diag((high - low) / 2))))
    info.update({'function_at_center': fp.tolist(), 'jacobian': jac.tolist()})
    return out, info


def existing_zfo(z, fun, jacobian, hessian_bounds):
    """Paper Proposition 1, retained ONLY as its Example 1 reference."""
    fp = np.asarray(fun(z.center), dtype=float).reshape(-1)
    jac = np.asarray(jacobian(z.center), dtype=float)
    low, high = (np.asarray(x, dtype=float) for x in hessian_bounds(*z.bounds))
    radius = 0.5 * np.einsum('i,mij,j->m', z.radius, np.maximum(np.abs(low), np.abs(high)), z.radius)
    return Zonotope(fp, np.column_stack((jac @ z.generators, np.diag(radius)))), {
        'method': 'Althoff2008_as_Ma2026_Proposition1',
        'remainder_lower': (-radius).tolist(), 'remainder_upper': radius.tolist()}


def prediction(z, fun, jacobian, hessian_bounds, process_generators=None, order=None, groups=None):
    out, info = improved_zfo(z, fun, jacobian, hessian_bounds, order, groups)
    if process_generators is not None:
        noise = np.asarray(process_generators, dtype=float)
        out = Zonotope(out.center, np.column_stack((out.generators, noise)))
    info['estimation_step'] = 'Theorem3_prediction'
    return out, info


def consistent_measurement(z, y, fun, jacobian, hessian_bounds, noise_generators, order=None, groups=None):
    """Theorem 4 / Eq. (16): returns (C,p_tilde,H_tilde,info)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    gp = np.asarray(fun(z.center), dtype=float).reshape(-1)
    C = np.asarray(jacobian(z.center), dtype=float)
    Hv = np.asarray(noise_generators, dtype=float)
    low, high, info = improved_remainder(z, hessian_bounds, len(y), order, groups)
    p_tilde = y - gp + C @ z.center - (low + high) / 2
    H_tilde = np.column_stack((-np.diag((high - low) / 2), -Hv))
    info.update({'estimation_step': 'Theorem4_consistent_set', 'C': C.tolist(),
                 'p_tilde': p_tilde.tolist(), 'H_tilde': H_tilde.tolist()})
    return C, p_tilde, H_tilde, info


def combine_lp(z, C, p_tilde, H_tilde, method='highs', scalar_analytic=False):
    """Theorem 5 / Eqs. (17)--(18), one L1 LP per state coordinate.

    scalar_analytic=True solves the EXACT same LP analytically if n=1 and
    each measurement-generator column has at most one nonzero. This is
    a solver shortcut, not a different gain/ellipsoid/F-radius objective.
    If infeasible/numerically unsuccessful, raises RuntimeError: NEVER a
    conclusion that the true information set is empty.
    """
    C = np.asarray(C, dtype=float); p_tilde = np.asarray(p_tilde, dtype=float).reshape(-1)
    H_tilde = np.asarray(H_tilde, dtype=float)
    n, ny = len(z.center), len(p_tilde)
    if C.shape != (ny, n) or H_tilde.shape[0] != ny:
        raise ValueError('measurement shape mismatch')
    k, j = z.generators.shape[1], H_tilde.shape[1]
    T = np.zeros((n, n)); N = np.zeros((n, ny)); rows = []
    independent_noise = np.all(np.count_nonzero(H_tilde, axis=0) <= 1)
    if scalar_analytic and n == 1 and independent_noise:
        radius = float(z.radius[0]); mr = np.abs(H_tilde).sum(axis=1)
        choices = np.full(ny, np.inf)
        nonzero = C[:, 0] != 0
        choices[nonzero] = mr[nonzero] / np.abs(C[nonzero, 0])
        which = int(np.argmin(choices)) if ny else -1
        if which >= 0 and choices[which] < radius:
            N[0, which] = 1 / C[which, 0]
            objective = float(choices[which])
        else:
            T[0, 0] = 1.; objective = radius
        rows.append({'solver': 'exact_scalar_L1_formula_evaluated_in_float', 'objective': objective,
                     'measurement_radius_candidates': [float(c) if np.isfinite(c) else None for c in choices]})
    else:
        # variable = [T_i (n), N_i (ny), a_T (k), a_N (j)]
        variables = n + ny + k + j
        objective = np.r_[np.zeros(n + ny), np.ones(k + j)]
        equality = np.zeros((n, variables))
        equality[:, :n] = np.eye(n); equality[:, n:n+ny] = C.T
        base = np.zeros((k + j, variables))
        base[:k, :n] = z.generators.T
        base[k:, n:n+ny] = H_tilde.T
        positive = base.copy(); negative = -base
        positive[:, n+ny:] = -np.eye(k + j)
        negative[:, n+ny:] = -np.eye(k + j)
        for i in range(n):
            rhs = np.eye(n)[i]
            result = linprog(objective, A_ub=np.vstack((positive, negative)),
                             b_ub=np.zeros(2*(k+j)), A_eq=equality, b_eq=rhs,
                             bounds=[(None,None)]*(n+ny)+[(0,None)]*(k+j), method=method)
            if not result.success:
                raise RuntimeError(f'Ma Eq18 LP failed (not an emptiness certificate): {result.message}')
            T[i] = result.x[:n]; N[i] = result.x[n:n+ny]
            rows.append({'solver': method, 'objective': float(result.fun), 'status': int(result.status),
                         'iterations': int(result.nit)})
    out = Zonotope(T @ z.center + N @ p_tilde,
                   np.column_stack((T @ z.generators, N @ H_tilde)))
    residual = T + N @ C - np.eye(n)
    return out, {'estimation_step': 'Theorem5_Eq17_18', 'T': T.tolist(), 'N': N.tolist(),
                 'identity_residual': residual.tolist(), 'row_solver_results': rows,
                 'output_halfwidth': out.radius.tolist(),
                 'numeric_guarantee': 'ordinary_float_requires_external_rounding_verification'}


def measurement_update(z, y, fun, jacobian, hessian_bounds, noise_generators,
                       order=None, groups=None, scalar_analytic=False):
    C, pc, Hc, consistent = consistent_measurement(z, y, fun, jacobian, hessian_bounds,
                                                 noise_generators, order, groups)
    out, combination = combine_lp(z, C, pc, Hc, scalar_analytic=scalar_analytic)
    return out, {'consistent_set': consistent, 'combination': combination}
