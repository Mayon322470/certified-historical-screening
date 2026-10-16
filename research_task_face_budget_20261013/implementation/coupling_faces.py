"""Strict whole-cell exclusions for one shared normalized-cubic beta.

All faces are multiplied by the positive q**2, q=1/theta. Expressions have
degree-at-most-two polynomial coefficients, including the coefficients of
each exponential. Derivatives therefore use (D-decay)**k on EVERY exponential
coefficient. This module never infers a feasible trajectory from an outer set.
"""
from fractions import Fraction as F
from math import factorial
from numbers import Integral

import certified_core as cc

KIND = "normalized_global_cubic_v1"
MAX_STATIONARY_STEPS = 32
TAYLOR_ORDER = 4


def _poly(values):
    values = [F(value) for value in values]
    if len(values) > 3:
        raise ValueError("Degree greater than two is not supported")
    return values + [F(0)]*(3-len(values))


def _add_poly(left, right):
    return [a+b for a, b in zip(_poly(left), _poly(right))]


def _scaled_poly(poly, scale):
    return [F(scale)*value for value in _poly(poly)]


def derivative_polynomial(poly, decay=F(0), order=1):
    """Polynomial factor of D**order(poly(q)*exp(-decay*q))."""
    if not isinstance(order, Integral) or order < 0:
        raise ValueError("A nonnegative integral derivative order is required")
    decay = F(decay)
    if decay < 0:
        raise ValueError("Nonnegative decay required")
    result = _poly(poly)
    for _ in range(order):
        a, b, c = result
        result = [b-decay*a, 2*c-decay*b, -decay*c]
    return result


def polynomial_range(coefficients, lo, hi):
    """Exact quadratic range before outward conversion to the shared lattice."""
    a, b, c = _poly(coefficients)
    lo, hi = F(lo), F(hi)
    if lo > hi:
        raise ValueError("Reversed polynomial interval")

    def value(q):
        return a+b*q+c*q*q

    values = [value(lo), value(hi)]
    if c:
        stationary = -b/(2*c)
        if lo < stationary < hi:
            values.append(value(stationary))
    return cc.IV(cc.IV.of(min(values)).lo, cc.IV.of(max(values)).hi)


def _expression(poly, terms):
    combined = {}
    for decay, coefficient in terms:
        decay = F(decay)
        if decay < 0:
            raise ValueError("Negative exponential decay")
        combined[decay] = _add_poly(combined.get(decay, []), coefficient)
    base = _poly(poly)
    if F(0) in combined:
        base = _add_poly(base, combined.pop(F(0)))
    return dict(poly=base, terms=[(decay, coefficient)
                                for decay, coefficient in sorted(combined.items())
                                if any(coefficient)])


def _linear_combination(parts, constant_q_squared=F(0)):
    poly = [F(0), F(0), F(constant_q_squared)]
    terms = []
    for multiplier, expression in parts:
        poly = _add_poly(poly, _scaled_poly(expression["poly"], multiplier))
        terms.extend((decay, _scaled_poly(coefficient, multiplier))
                     for decay, coefficient in expression["terms"])
    return _expression(poly, terms)


def prepare(case):
    """Build six beta-prior faces and six same-beta FM coupling faces.

    Feasibility requires every resulting face <= 0. Original packets are used;
    they must not already have been expanded by the discrepancy radius.
    """
    initial = tuple(map(F, case["initial"]))
    packet = [tuple(map(F, row)) for row in case["packet"]]
    segments = [(F(dt), F(u)) for dt, u in case["past"]]
    if len(initial) != 3 or len(packet) != 3 or any(len(row) != 2 for row in packet):
        raise ValueError("Three state/packet coordinates required")
    if any(lo > hi for lo, hi in packet) or any(dt < 0 for dt, _ in segments):
        raise ValueError("Invalid packet or past duration")
    response = case["response"]
    if response["kind"] != KIND:
        raise ValueError("Unsupported response class")
    epsilon, length, horizon = F(response["epsilon_x"]), F(response["L"]), F(case["H"])
    total = sum((dt for dt, _ in segments), F(0))
    if epsilon < 0 or horizon < 0 or length <= 0 or length != total+horizon:
        raise ValueError("Invalid normalized global response contract")
    if F(case.get("rho", 0)) != 0:
        raise ValueError("This response contract has rho=0")
    psi = (total**3/length**3, 3*total**2/length**3, 6*total/length**3)

    x0, v0, a0 = initial
    acceleration_terms = {F(0): a0}
    area = weighted = clock = F(0)
    for dt, u in segments:
        if not dt:
            continue
        acceleration_terms[0] = acceleration_terms.get(F(0), F(0))-u
        acceleration_terms = {decay+dt: value for decay, value in acceleration_terms.items() if value}
        acceleration_terms[0] = acceleration_terms.get(F(0), F(0))+u
        area += dt*u
        weighted += u*((total-clock)*dt-dt*dt/2)
        clock += dt
    f0 = acceleration_terms.get(F(0), F(0))-a0
    terms = sorted((decay, value) for decay, value in acceleration_terms.items() if decay and value)
    qmoment = area-a0*total
    vs, xs = v0+area, x0+v0*total+weighted
    # Nominal rows P_i=q^2*p_i, all represented in the SAME positive scale.
    nominal = [
        _expression([f0, -qmoment, xs], [(t, [c]) for t, c in terms]),
        _expression([0, -f0, vs], [(t, [0, -c]) for t, c in terms]),
        _expression([0, 0, a0+f0], [(t, [0, 0, c]) for t, c in terms]),
    ]
    faces = []
    axes = ("x", "v", "a")
    for i, axis in enumerate(axes):
        lo, hi = packet[i]
        lower = _linear_combination([(-1, nominal[i])], lo-epsilon*psi[i])
        upper = _linear_combination([(1, nominal[i])], -(hi+epsilon*psi[i]))
        for side, expression in (("lower", lower), ("upper", upper)):
            faces.append(dict(name=f"{axis}_{side}_beta_prior", family="beta_prior",
                              coordinate=i, side=side, **expression))
    positive = all(value > 0 for value in psi)
    if positive:
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                expression = _linear_combination(
                    [(-psi[j], nominal[i]), (psi[i], nominal[j])],
                    psi[j]*packet[i][0]-psi[i]*packet[j][1])
                faces.append(dict(name=f"{axes[i]}_lower_vs_{axes[j]}_upper", family="same_beta",
                                  lower_coordinate=i, upper_coordinate=j, **expression))
    return dict(faces=faces, nominal_q_squared=nominal, psi=psi, epsilon_x=epsilon,
                coupling_enabled=positive, zero_profile_fallback=not positive,
                face_scale="q_squared_positive", feasible_face_sign="<=0",
                moments=dict(T=total, Q=qmoment, v_star=vs, x_star=xs, f0=f0),
                source=dict(initial=initial, past=segments, original_packet=packet,
                            H=horizon, rho=F(0), response=dict(kind=KIND, epsilon_x=epsilon, L=length)))


def evaluate(face, lo, hi, derivative=0):
    lo, hi = F(lo), F(hi)
    if lo < 0 or lo > hi:
        raise ValueError("Ordered nonnegative q interval required")
    result = polynomial_range(derivative_polynomial(face["poly"], order=derivative), lo, hi)
    for decay, coefficients in face["terms"]:
        decay = F(decay)
        coefficient = derivative_polynomial(coefficients, decay, derivative)
        exponential = cc.IV(cc.exp_neg(decay*hi).lo, cc.exp_neg(decay*lo).hi)
        result += polynomial_range(coefficient, lo, hi)*exponential
    return result


def _pair(interval):
    return [interval.lower(), interval.upper()]


def centered_enclosure(face, lo, hi, derivative=0, with_details=False):
    lo, hi = F(lo), F(hi)
    center, radius = (lo+hi)/2, (hi-lo)/2
    if radius < 0:
        raise ValueError("Reversed Taylor domain")
    value = evaluate(face, center, center, derivative)
    result = value
    coefficients = [_pair(value)]
    for k in range(1, TAYLOR_ORDER+1):
        derivative_value = evaluate(face, center, center, derivative+k)
        power = cc.IV.of(radius**k)
        variation = cc.IV(-power.hi if k % 2 else 0, power.hi)
        result += derivative_value*variation/F(factorial(k))
        coefficients.append(_pair(derivative_value))
    k = TAYLOR_ORDER+1
    remainder_derivative = evaluate(face, lo, hi, derivative+k)
    power = cc.IV.of(radius**k)
    result += remainder_derivative*cc.IV(-power.hi, power.hi)/F(factorial(k))
    if not with_details:
        return result
    return result, dict(center=center, radius=radius, derivative_order=derivative,
                        polynomial_order=TAYLOR_ORDER, point_derivative_ranges=coefficients,
                        remainder_derivative_order=derivative+k,
                        whole_domain_remainder_derivative_range=_pair(remainder_derivative),
                        result_range=_pair(result))


def _derivative_enclosure(face, lo, hi, derivative):
    natural = evaluate(face, lo, hi, derivative)
    taylor, details = centered_enclosure(face, lo, hi, derivative, True)
    lower, upper = max(natural.lo, taylor.lo), min(natural.hi, taylor.hi)
    if lower > upper:
        raise ArithmeticError("Disjoint certified derivative enclosures")
    combined = cc.IV(lower, upper)
    return combined, dict(natural_range=_pair(natural), taylor=details,
                          intersection_range=_pair(combined))


def positive_curvature(face, lo, hi, mu, second_proof=None):
    """At most 32 attempts, all using curvature on the ORIGINAL whole domain."""
    lo, hi, mu = F(lo), F(hi), F(mu)
    if mu <= 0 or lo > hi:
        raise ValueError("Positive whole-domain curvature required")
    left, right = lo, hi
    for attempt in range(MAX_STATIONARY_STEPS):
        center = (left+right)/2
        value = evaluate(face, center, center)
        slope = evaluate(face, center, center, 1)
        slope_abs = max(abs(slope.lower()), abs(slope.upper()))
        lower = value.lower()-slope_abs*slope_abs/(2*mu)
        if lower > 0:
            return dict(kind="curvature", certified_lower=lower,
                        center=center, mu=mu, value_range=_pair(value),
                        slope_range=_pair(slope), slope_abs_upper=slope_abs,
                        stationary_attempts=attempt+1, second_derivative_proof=second_proof)
        if slope.upper() < 0:
            left = center
        elif slope.lower() > 0:
            right = center
        else:
            break
    return None


def reject_cell(meta, theta_lo, theta_hi):
    """Return a strict-positive necessary-face certificate, or None.

    Point evaluations can only avoid unnecessary work or be combined with a
    certified monotonicity/curvature bound; samples alone never delete a cell.
    """
    theta_lo, theta_hi = F(theta_lo), F(theta_hi)
    if not 0 < theta_lo <= theta_hi:
        raise ValueError("Ordered strictly positive theta interval required")
    lo, hi = 1/theta_hi, 1/theta_lo
    for index, face in enumerate(meta["faces"]):
        certificate = None
        natural = evaluate(face, lo, hi)
        if natural.lower() > 0:
            certificate = dict(kind="natural", certified_lower=natural.lower(),
                               function_range=_pair(natural))
        elif natural.upper() <= 0:
            continue
        else:
            endpoints = [evaluate(face, q, q) for q in (lo, hi)]
            if any(value.upper() <= 0 for value in endpoints):
                continue
            taylor, details = centered_enclosure(face, lo, hi, with_details=True)
            if taylor.lower() > 0:
                certificate = dict(kind="taylor", certified_lower=taylor.lower(), taylor=details)
            else:
                slope, slope_proof = _derivative_enclosure(face, lo, hi, 1)
                if slope.lower() >= 0 and endpoints[0].lower() > 0:
                    certificate = dict(kind="monotone_increasing", certified_lower=endpoints[0].lower(),
                                       point=lo, endpoint_range=_pair(endpoints[0]), derivative_proof=slope_proof)
                elif slope.upper() <= 0 and endpoints[1].lower() > 0:
                    certificate = dict(kind="monotone_decreasing", certified_lower=endpoints[1].lower(),
                                       point=hi, endpoint_range=_pair(endpoints[1]), derivative_proof=slope_proof)
                elif slope.lower() < 0 < slope.upper():
                    second, second_proof = _derivative_enclosure(face, lo, hi, 2)
                    if second.lower() > 0:
                        certificate = positive_curvature(face, lo, hi, second.lower(), second_proof)
                    elif second.upper() <= 0:
                        lower = min(value.lower() for value in endpoints)
                        if lower > 0:
                            certificate = dict(kind="concave", certified_lower=lower,
                                               endpoint_ranges=list(map(_pair, endpoints)),
                                               second_derivative_proof=second_proof)
        if certificate:
            if certificate["certified_lower"] <= 0:
                raise ArithmeticError("Non-strict deletion is forbidden")
            return dict(face_index=index, face_name=face["name"], face_family=face["family"],
                        theta_interval=[theta_lo, theta_hi], q_interval=[lo, hi],
                        face_scale="q_squared_positive", expression=face,
                        source=meta["source"], **certificate)
    return None
