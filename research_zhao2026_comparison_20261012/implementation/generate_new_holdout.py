"""Predeclared new 48-case holdout; independent numerical feasible-set reference.

Default invocation only prints the protocol and source hash. Generation requires
--generate and a pre-existing frozen-method manifest. Hidden generating parameters
are written only to qa/holdout_generation_private.json, never to solver inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import mpmath as mp
from scipy.optimize import brentq, minimize_scalar
import scipy


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "analysis" / "NEW_HOLDOUT_PROTOCOL_CN.txt"
SEED = 202610120947
VERSION = "zhao2026-new-holdout-v1"
SCALE = 10 ** 110
ALLOWED_FIELDS = {"name", "group", "initial", "past", "prior", "packet",
                  "future", "H", "rho", "limit"}


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def json_bytes(value):
    return (json.dumps(encode(value), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def mpq(value):
    value = F(value)
    return mp.mpf(value.numerator) / value.denominator


def rational_nearest(value, digits=60):
    scale = 10 ** digits
    return F(int(mp.nint(value * scale)), scale)


def down(value):
    value = F(value)
    return F((value.numerator * SCALE) // value.denominator, SCALE)


def up(value):
    value = F(value)
    return F(-((-value.numerator * SCALE) // value.denominator), SCALE)


@dataclass(frozen=True)
class RBox:
    """Independent decimal-lattice rational intervals, used only in generation QA."""
    lo: F
    hi: F

    @staticmethod
    def point(value):
        return value if isinstance(value, RBox) else RBox(F(value), F(value))

    def __add__(self, other):
        other = self.point(other)
        return RBox(down(self.lo + other.lo), up(self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return RBox(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.point(other)

    def __rsub__(self, other):
        return self.point(other) + -self

    def __mul__(self, other):
        other = self.point(other)
        values = [a * b for a in (self.lo, self.hi)
                  for b in (other.lo, other.hi)]
        return RBox(down(min(values)), up(max(values)))

    __rmul__ = __mul__

    def pair(self):
        return [self.lo, self.hi]


def exp_minus_rational(value):
    """Alternating-series enclosure of exp(-value), then interval squaring."""
    value = F(value)
    assert value >= 0
    if not value:
        return RBox.point(1)
    squarings = 0
    while value > F(1, 2):
        value /= 2
        squarings += 1
    term = total = F(1)
    index = 0
    while True:
        previous = total
        index += 1
        term *= -value / index
        total += term
        if abs(term) < F(1, 10 ** 120):
            result = RBox(down(min(previous, total)), up(max(previous, total)))
            break
    for _ in range(squarings):
        result = result * result
    return result


def rational_flow(theta, state, duration, command, rho=F(0)):
    """Independent enclosure for a_dot=(u-a)/theta+rho."""
    theta, duration, command, rho = map(F, (theta, duration, command, rho))
    assert theta > 0 and duration >= 0
    x, v, a = map(RBox.point, state)
    decay = exp_minus_rational(duration / theta)
    equilibrium = command + theta * rho
    delta = a - equilibrium
    return (x + v * duration + equilibrium * duration ** 2 / 2
            + delta * (theta * duration - theta ** 2 * (1 - decay)),
            v + equilibrium * duration + delta * theta * (1 - decay),
            equilibrium + delta * decay)


def rational_endpoint(theta, initial, segments, rho=F(0), until=None):
    state = tuple(map(RBox.point, initial))
    remaining = sum(dt for dt, _ in segments) if until is None else F(until)
    assert 0 <= remaining <= sum(dt for dt, _ in segments)
    for duration, command in segments:
        step = min(duration, remaining)
        if step:
            state = rational_flow(theta, state, step, command, rho)
        remaining -= step
        if not remaining:
            break
    assert remaining == 0
    return state


def mp_flow(theta, state, duration, command, rho=F(0)):
    theta, duration, command, rho = map(mpq, (theta, duration, command, rho))
    x, v, a = state
    equilibrium = command + theta * rho
    delta = a - equilibrium
    one_minus_decay = -mp.expm1(-duration / theta)
    return (x + v * duration + equilibrium * duration ** 2 / 2
            + delta * (theta * duration - theta ** 2 * one_minus_decay),
            v + equilibrium * duration + delta * theta * one_minus_decay,
            equilibrium + delta * (1 - one_minus_decay))


def mp_endpoint(theta, initial, segments):
    state = tuple(map(mpq, initial))
    for duration, command in segments:
        state = mp_flow(theta, state, duration, command)
    return state


def point_peak(case, theta, dps):
    """Enumerate all stationary times for one hidden theta, independently.

    Acceleration is monotone in each hold. Splitting at its possible zero makes
    velocity monotone, so each piece has at most one velocity zero. This oracle
    sets difficulty; its floating output is not a solver safety certificate.
    """
    with mp.workdps(dps):
        state = mp_endpoint(theta, case["initial"], case["past"])
        initial_terminal = tuple(state)
        candidates = [(mp.mpf(0), state[0])]
        clock = mp.mpf(0)
        theta_mp = mpq(theta)
        for duration, command in case["future"]:
            dt = mpq(duration)
            equilibrium = mpq(command) + theta_mp * mpq(case["rho"])
            delta = state[2] - equilibrium

            def at(time):
                one = -mp.expm1(-time / theta_mp)
                return (state[0] + state[1] * time + equilibrium * time ** 2 / 2
                        + delta * (theta_mp * time - theta_mp ** 2 * one),
                        state[1] + equilibrium * time + delta * theta_mp * one)

            split = [mp.mpf(0), dt]
            if delta and -equilibrium / delta > 0:
                acceleration_zero = -theta_mp * mp.log(-equilibrium / delta)
                if 0 < acceleration_zero < dt:
                    split.append(acceleration_zero)
            split.sort()
            times = list(split)
            for left, right in zip(split, split[1:]):
                vl, vr = at(left)[1], at(right)[1]
                if vl == 0:
                    times.append(left)
                if vr == 0:
                    times.append(right)
                if vl * vr < 0:
                    lo, hi = left, right
                    for _ in range(4 * dps):
                        mid = (lo + hi) / 2
                        vm = at(mid)[1]
                        if vm == 0:
                            lo = hi = mid
                            break
                        if vl * vm > 0:
                            lo, vl = mid, vm
                        else:
                            hi = mid
                    times.append((lo + hi) / 2)
            candidates.extend((clock + time, at(time)[0]) for time in times)
            state = mp_flow(theta, state, duration, command, case["rho"])
            clock += dt
        time, maximum = max(candidates, key=lambda item: item[1])
        return dict(time=rational_nearest(time, 70), peak=rational_nearest(maximum, 70),
                    terminal=[rational_nearest(item, 70) for item in initial_terminal],
                    candidate_count=len(candidates))


def public_upper(case):
    """A rational upper bound valid for every compatible theta and future time."""
    history_accel = max(abs(case["initial"][2]),
                        *(abs(command) for _, command in case["past"]))
    future_accel = max(abs(command) for _, command in case["future"])
    accel = max(history_accel, future_accel + case["prior"][1] * case["rho"])
    horizon = case["H"]
    upper = case["packet"][0][1] + max(F(0), case["packet"][1][1]) * horizon
    upper += accel * horizon ** 2 / 2
    return upper, accel


class NumericalHistory:
    """Independent analytic first q-sensitivities, evaluated with 90 decimal digits."""
    def __init__(self, case):
        self.case = case
        self.cache = {}
        self.peak_cache = {}

    def jet(self, q):
        q = float(q)
        if q not in self.cache:
            with mp.workdps(90):
                qm = mpq(F(q))
                x, v, a = map(mpq, self.case["initial"])
                dx = dv = da = mp.mpf(0)
                for duration, command in self.case["past"]:
                    dt, u = mpq(duration), mpq(command)
                    decay = mp.exp(-dt*qm)
                    one = -mp.expm1(-dt*qm)
                    b = one/qm
                    db = dt*decay/qm-one/qm**2
                    c = dt/qm-one/qm**2
                    dc = -dt*(1+decay)/qm**2+2*one/qm**3
                    delta = a-u
                    x, v, a, dx, dv, da = (
                        x+v*dt+u*dt**2/2+delta*c,
                        v+u*dt+delta*b, u+delta*decay,
                        dx+dv*dt+da*c+delta*dc,
                        dv+da*b+delta*db, da*decay-delta*dt*decay)
                self.cache[q] = ((x, v, a), (dx, dv, da))
        return self.cache[q]

    def face(self, q, coordinate, side):
        with mp.workdps(90):
            value = self.jet(q)[0][coordinate]
            low, high = self.case["packet"][coordinate]
            residual = mpq(low)-value if side == 0 else value-mpq(high)
            return float(residual)

    def violation(self, q):
        with mp.workdps(90):
            values = self.jet(q)[0]
            return float(max(max(mpq(low)-value, value-mpq(high))/(1+abs(mpq(low))+abs(mpq(high)))
                             for value, (low, high) in zip(values, self.case["packet"])))

    def future_peak(self, q):
        """Double precision time optimization for a fixed q, with all velocity roots."""
        q = float(q)
        if q in self.peak_cache:
            return self.peak_cache[q]
        theta = 1/q
        state = tuple(map(float, self.jet(q)[0]))
        candidates = [(state[0], 0.0)]
        clock = 0.0
        for duration, command in self.case["future"]:
            dt = float(duration)
            equilibrium = float(command)+theta*float(self.case["rho"])
            delta = state[2]-equilibrium

            def at(time):
                one = -math.expm1(-time/theta)
                return (state[0]+state[1]*time+equilibrium*time*time/2
                        +delta*(theta*time-theta*theta*one),
                        state[1]+equilibrium*time+delta*theta*one,
                        equilibrium+delta*(1-one))

            split = [0.0, dt]
            if delta and -equilibrium/delta > 0:
                az = -theta*math.log(-equilibrium/delta)
                if 0 < az < dt:
                    split.append(az)
            split.sort()
            times = list(split)
            for left, right in zip(split, split[1:]):
                if at(left)[1]*at(right)[1] < 0:
                    times.append(brentq(lambda time: at(time)[1], left, right,
                                        xtol=1e-13, rtol=1e-14))
            candidates.extend((at(time)[0], clock+time) for time in times)
            state = at(dt)
            clock += dt
        result = max(candidates)
        self.peak_cache[q] = result
        return result


def unique_sorted(values):
    """Retain distinct float roots; do not merge a narrow feasible component by tolerance."""
    return sorted(set(map(float, values)))


def feasible_peak_sweep(case, oracle, hidden_theta, grid_size):
    """Finite numerical search only; completeness is deliberately not claimed."""
    qlow, qhigh = float(1/case["prior"][1]), float(1/case["prior"][0])
    grid = [qlow+(qhigh-qlow)*j/grid_size for j in range(grid_size+1)]
    critical = []
    issues = []
    for coordinate in range(3):
        deriv = [float(oracle.jet(q)[1][coordinate]) for q in grid]
        for j, (left, right) in enumerate(zip(grid, grid[1:])):
            if deriv[j]*deriv[j+1] < 0:
                try:
                    critical.append(brentq(lambda q: float(oracle.jet(q)[1][coordinate]),
                                           left, right, xtol=1e-13, rtol=1e-14, maxiter=150))
                except Exception as exc:
                    issues.append(dict(kind="critical_root_failure", coordinate=coordinate,
                                       interval=[left, right], error=str(exc)))
    # Critical points recover near-tangent packet boundaries that a grid alone misses.
    root_grid = unique_sorted(grid+critical)
    boundaries = []
    for coordinate in range(3):
        for side in range(2):
            values = [oracle.face(q, coordinate, side) for q in root_grid]
            for j, (left, right) in enumerate(zip(root_grid, root_grid[1:])):
                if values[j] == 0:
                    boundaries.append(left)
                if values[j]*values[j+1] < 0:
                    try:
                        boundaries.append(brentq(lambda q: oracle.face(q, coordinate, side),
                                                 left, right, xtol=1e-13, rtol=1e-14, maxiter=150))
                    except Exception as exc:
                        issues.append(dict(kind="packet_root_failure", coordinate=coordinate,
                                           side=side, interval=[left, right], error=str(exc)))
            if values[-1] == 0:
                boundaries.append(root_grid[-1])
    cuts = unique_sorted([qlow, qhigh]+boundaries+critical)
    tolerance = 1e-24
    branches = [[left, right] for left, right in zip(cuts, cuts[1:])
                if oracle.violation((left+right)/2) <= tolerance]
    near_contact = [q for q in cuts if abs(oracle.violation(q)) <= tolerance]
    candidates = [float(1/hidden_theta)]
    candidates.extend(q for q in grid+cuts if oracle.violation(q) <= tolerance)
    for left, right in branches:
        sample = [left+(right-left)*j/16 for j in range(17)]
        candidates.extend(q for q in sample if oracle.violation(q) <= tolerance)
        peaks = [oracle.future_peak(q)[0] for q in sample]
        search_intervals = [(left, right)]
        search_intervals.extend((sample[j-1], sample[j+1]) for j in range(1, 16)
                                if peaks[j] >= peaks[j-1] and peaks[j] >= peaks[j+1])
        for lo, hi in search_intervals:
            if hi-lo <= 1e-13:
                continue
            optimum = minimize_scalar(lambda q: -oracle.future_peak(q)[0], bounds=(lo, hi),
                                      method="bounded", options=dict(xatol=1e-12, maxiter=300))
            if not optimum.success:
                issues.append(dict(kind="peak_optimizer_failure", interval=[lo, hi],
                                   message=str(optimum.message)))
            # Never turn numerical near-feasibility into a safety certificate.
            if oracle.violation(optimum.x) <= tolerance:
                candidates.append(float(optimum.x))
    candidates = unique_sorted(candidates)
    best_q = max(candidates, key=lambda q: oracle.future_peak(q)[0])
    best_peak, best_time = oracle.future_peak(best_q)
    return dict(grid_size=grid_size, derivative_critical_points=unique_sorted(critical),
                packet_boundary_roots=unique_sorted(boundaries), numerical_q_branches=branches,
                near_contact_q=near_contact, near_contact_count=len(near_contact),
                candidate_count=len(candidates), best_q=best_q, best_peak=best_peak,
                best_time=best_time, numerical_violation=oracle.violation(best_q), issues=issues)


def numerical_compatible_reference(case, hidden_theta):
    oracle = NumericalHistory(case)
    coarse = feasible_peak_sweep(case, oracle, hidden_theta, 512)
    fine = feasible_peak_sweep(case, oracle, hidden_theta, 1024)
    best = max((coarse, fine), key=lambda record: record["best_peak"])
    q = min(1/case["prior"][0], max(1/case["prior"][1], F(best["best_q"])))
    theta = 1/q
    reference120 = point_peak(case, theta, 120)
    reference160 = point_peak(case, theta, 160)
    # Include the independently generated compatible point even if numerical search loses it.
    hidden = point_peak(case, hidden_theta, 160)
    if hidden["peak"] > reference160["peak"]:
        theta = hidden_theta
        reference120 = point_peak(case, theta, 120)
        reference160 = hidden
    terminal = rational_endpoint(theta, case["initial"], case["past"])
    verified_history = all(low <= item.lo <= item.hi <= high
                           for item, (low, high) in zip(terminal, case["packet"]))
    witness_time = min(case["H"], max(F(0), reference160["time"]))
    witness = rational_endpoint(theta, terminal, case["future"], case["rho"], witness_time)[0]
    difference = abs(coarse["best_peak"]-fine["best_peak"])
    disagreement = difference > 1e-7*(1+abs(best["best_peak"]))
    precision_failure = abs(reference120["peak"]-reference160["peak"]) > (1+abs(reference160["peak"]))/10**65
    return reference160["peak"], dict(
        status="NUMERICAL_REFERENCE_UNRESOLVED" if disagreement or precision_failure or coarse["issues"] or fine["issues"]
        else "NUMERICAL_REFERENCE_UNCERTIFIED", coarse=coarse, fine=fine,
        grid_peak_difference=difference, grid_disagreement=disagreement,
        high_precision_disagreement=precision_failure, reference_theta=theta,
        reference120=reference120, reference160=reference160,
        reference_history_certified=verified_history,
        reference_terminal_certificate=[item.pair() for item in terminal],
        reference_witness_time=witness_time, reference_witness_position=witness.pair(),
        missing_components_or_isolated_points_not_ruled_out=True,
        guarantee="Finite stationary-point and boundary search plus local optimization; not a certified global maximum.")


def seal_case(case, hidden_theta, index, design):
    """Verify generation independently; never select, replace, or discard cases."""
    assert set(case) == ALLOWED_FIELDS - {"limit"}
    assert case["prior"][0] < hidden_theta < case["prior"][1]
    assert all(dt > 0 for dt, _ in case["past"] + case["future"])
    assert sum(dt for dt, _ in case["future"]) == case["H"]
    endpoint = rational_endpoint(hidden_theta, case["initial"], case["past"])
    inside = [low <= item.lo <= item.hi <= high
              for item, (low, high) in zip(endpoint, case["packet"])]
    assert all(inside), (case["name"], "hidden truth failed certified packet containment")
    peak120 = point_peak(case, hidden_theta, 120)
    peak160 = point_peak(case, hidden_theta, 160)
    hidden_scale = 1 + abs(peak160["peak"])
    assert abs(peak120["peak"] - peak160["peak"]) <= hidden_scale / 10 ** 65
    policy = (index + design.get("threshold_rotation", 0)) % 4
    broad, acceleration = public_upper(case)
    try:
        reference_peak, reference = numerical_compatible_reference(case, hidden_theta)
    except Exception as exc:
        # Retain this case with a fixed public-envelope fallback, never redraw or skip.
        reference_peak = broad
        reference = dict(status="NUMERICAL_SEARCH_FAILED_PUBLIC_BOUND_FALLBACK",
                         error=type(exc).__name__+": "+str(exc), reference_peak=broad,
                         guarantee="Numerical search failed; public rational envelope used without case selection.")
    scale = 1 + abs(reference_peak)
    if policy == 0:
        margin = (1 + abs(broad)) / 20
        case["limit"] = broad + margin
        threshold_name = "public_envelope_plus_5percent_scale"
    else:
        margin = scale * (F(1, 100), F(1, 10000), F(-1, 1000))[policy - 1]
        case["limit"] = reference_peak + margin
        threshold_name = ("compatible_reference_plus_1e-2_scale", "compatible_reference_plus_1e-4_scale",
                          "compatible_reference_minus_1e-3_scale")[policy - 1]
    witness_time = min(case["H"], max(F(0), peak160["time"]))
    witness = rational_endpoint(hidden_theta, endpoint, case["future"],
                                rho=case["rho"], until=witness_time)[0]
    metadata = dict(name=case["name"], index=index, design=design,
                    hidden_theta=hidden_theta, certified_terminal=[item.pair() for item in endpoint],
                    history_packet_containment=inside, peak_at_120_dps=peak120,
                    peak_at_160_dps=peak160, threshold_policy=threshold_name,
                    margin=margin, numerical_compatible_reference=reference,
                    threshold_reference_peak=reference_peak,
                    public_upper=broad, public_acceleration_bound=acceleration,
                    witness_time=witness_time, certified_witness_position=witness.pair(),
                    hidden_witness_strictly_unsafe=witness.lo > case["limit"],
                    global_safe_by_public_envelope=case["limit"] > broad,
                    certification_scope="Rational enclosures certify only the recorded hidden point; "
                    "public-envelope inequality separately proves all compatible points safe when marked. "
                    "Numerical compatible-set reference and its margin are not global safety labels.")
    if reference.get("reference_history_certified"):
        metadata["reference_point_strictly_unsafe_certified"] = reference["reference_witness_position"][0] > case["limit"]
    else:
        metadata["reference_point_strictly_unsafe_certified"] = False
    assert set(case) == ALLOWED_FIELDS
    return case, metadata


def random_cases():
    rng = random.Random(SEED)
    result = []
    for n_index, holds in enumerate((1, 2, 4, 8)):
        for alignment in range(2):
            for replicate in range(4):
                index = len(result)
                initial = (F(rng.randint(-4, 4), 8), F(rng.randint(-6, 10), 8),
                           F(rng.randint(-8, 8), 8))
                if alignment == 0:
                    past = [(F(rng.randint(1, 4), 4), F(rng.randint(-12, 12), 8))
                            for _ in range(holds)]
                else:
                    primes = (7, 11, 13, 17, 19, 23, 29, 31)
                    past = [(F(rng.randint(2, 6), primes[j]), F(rng.randint(-12, 12), 8))
                            for j in range(holds)]
                prior = ((F(2, 5), F(9, 5)), (F(4, 5), F(14, 5)),
                         (F(1, 4), F(15, 4)), (F(11, 10), F(11, 5)))[(n_index + replicate) % 4]
                hidden = prior[0] + (prior[1] - prior[0]) * F(rng.randint(2, 8), 10)
                with mp.workdps(120):
                    state = mp_endpoint(hidden, initial, past)
                    centers = [rational_nearest(item) for item in state]
                width = (F(1, 5000), F(1, 200), F(1, 20), F(1, 2))[(replicate + n_index + alignment) % 4]
                radii = [width * factor for factor in (F(2), F(1), F(1, 2))]
                packet = [(center - radius, center + radius)
                          for center, radius in zip(centers, radii)]
                horizon = (F(3, 4), F(5, 4), F(2), F(7, 3))[(replicate + alignment + n_index) % 4]
                future_holds = (1, 2, 2, 4)[(replicate + n_index) % 4]
                weights = [rng.randint(1, 5) for _ in range(future_holds)]
                future = [(horizon * F(weight, sum(weights)), F(rng.randint(-10, 8), 8))
                          for weight in weights]
                rho = (F(0), F(1, 200), F(1, 40), F(1, 10))[(n_index + replicate + alignment) % 4]
                case = dict(name=f"holdout_random_{index:02d}_N{holds}", group="holdout_random",
                            initial=initial, past=past, prior=prior, packet=packet,
                            future=future, H=horizon, rho=rho)
                design = dict(family="general_random", past_holds=holds,
                              time_design="quarter_lattice" if alignment == 0 else "distinct_prime_denominators",
                              replicate=replicate, measurement_base_radius=width)
                result.append(seal_case(case, hidden, index, design))
    return result


def cubic_coefficients(r0, r1, delta):
    return (F(1), -(2 * r1 + r0), r1 ** 2 + delta + 2 * r0 * r1,
            -r0 * (r1 ** 2 + delta))


def structural_cases():
    result = []
    # New rational roots/scales/amplitudes, not the old frozen fixture inputs.
    profiles = [(F(7, 10), F(7, 8), F(1, 700), F(47, 100), F(61, 100)),
                (F(13, 10), F(9, 8), F(1, 450), F(52, 100), F(66, 100))]
    deltas = (F(1, 10 ** 5), F(1, 10 ** 14), F(0), F(-1, 10 ** 10))
    for profile, (duration, base, amplitude, r0, r1) in enumerate(profiles):
        for variant, delta in enumerate(deltas):
            coefficients = cubic_coefficients(r0, r1, delta)
            perturb = [amplitude * value for value in coefficients]
            initial = (F(profile - 1, 5), F(3 + profile, 4), base)
            total = 4 * duration
            max_perturb = max(map(abs, perturb))
            xbase = initial[0] + initial[1] * total + base * total ** 2 / 2
            vbase = initial[1] + base * total
            packet = [(xbase - 1 - max_perturb * total ** 2 / 2,
                       xbase + 1 + max_perturb * total ** 2 / 2),
                      (vbase - F(1, 2) - max_perturb * total,
                       vbase + F(1, 2) + max_perturb * total),
                      (base - 2 * max_perturb, base)]
            horizon = duration * F(3 + variant, 4)
            case = dict(name=f"holdout_four_hold_{profile}_{variant}", group="holdout_four_hold",
                        initial=initial, past=[(duration, base + value) for value in perturb],
                        prior=(duration * F(7, 10), duration * F(33 + profile, 10)),
                        packet=packet, future=[(horizon / 3, F(-1, 4)),
                                               (2 * horizon / 3, F(1, 8))],
                        H=horizon, rho=F(variant % 2, 100))
            design = dict(family="four_hold_acceleration_face", r0=r0, r1=r1, delta=delta,
                          threshold_rotation=2*profile,
                          duration=duration, base=base, amplitude=amplitude,
                          endpoint_identity="aT-base = amplitude*(1-z)*(z-r0)*((z-r1)^2+delta), z=exp(-duration/theta)")
            result.append(seal_case(case, duration, 32 + len(result), design))
    epsilons = (F(1, 10 ** 7), F(1, 10 ** 16), F(0), F(-1, 10 ** 10))
    for face in ("velocity", "position"):
        for variant, epsilon in enumerate(epsilons):
            duration = (F(5, 6), F(7, 6), F(11, 10), F(2, 3))[variant]
            base = (F(9, 10), F(11, 10), F(4, 5), F(6, 5))[variant]
            amplitude = F(variant + 3, 1400)
            r0, r1 = F(47 + variant, 100), F(61 + variant, 100)
            a3, a2, a1, a0 = cubic_coefficients(r0, r1, F(0))
            perturb = [amplitude * value for value in (-a3, a3-a2, a2-a1, a1-a0, a0)]
            assert sum(perturb) == 0
            initial = (F(variant - 2, 7), F(5 + variant, 6), base)
            total = 5 * duration
            vstar = initial[1] + base * total
            xstar = initial[0] + initial[1] * total + base * total ** 2 / 2
            xstar += sum(value * duration * (total - (F(j) + F(1, 2)) * duration)
                         for j, value in enumerate(perturb))
            max_perturb = max(map(abs, perturb))
            xwidth = 1 + max_perturb * total ** 2
            vwidth = 1 + max_perturb * total
            packet = [(xstar-xwidth, xstar+xwidth), (vstar-vwidth, vstar+vwidth),
                      (base-2*max_perturb, base+2*max_perturb)]
            if face == "velocity":
                packet[1] = (vstar + epsilon, vstar + vwidth)
            else:
                packet[0] = (xstar - xwidth, xstar - epsilon)
            horizon = duration * F(variant + 3, 3)
            case = dict(name=f"holdout_valley_{face}_{variant}", group="holdout_valley_" + face,
                        initial=initial, past=[(duration, base+value) for value in perturb],
                        prior=(duration * F(3, 4), duration * F(16 + variant, 5)),
                        packet=packet, future=[(horizon/2, F(0)), (horizon/2, F(-1, 5))],
                        H=horizon, rho=F(variant % 3, 200))
            design = dict(family="five_hold_"+face+"_valley", r0=r0, r1=r1, epsilon=epsilon,
                          threshold_rotation=1 if face == "velocity" else 3,
                          duration=duration, base=base, amplitude=amplitude,
                          endpoint_identity="A=amplitude*(1-z)^2*(z-r0)*(z-r1)^2; vT=vstar-theta*A; xT=xstar+theta^2*A",
                          vstar=vstar, xstar=xstar)
            result.append(seal_case(case, duration, 32 + len(result), design))
    return result


def self_test():
    """Analytic kernel checks only: does not instantiate holdout cases or write files."""
    assert exp_minus_rational(0).pair() == [F(1), F(1)]
    state = rational_flow(F(3, 2), (F(2), F(-1), F(2)), F(3, 4), F(2))
    expected = (F(29, 16), F(1, 2), F(2))
    assert all(value.lo <= truth <= value.hi for value, truth in zip(state, expected))
    with mp.workdps(160):
        for argument in (F(1, 7), F(2), F(31, 3)):
            enclosure = exp_minus_rational(argument)
            exact = mp.exp(-mpq(argument))
            assert mpq(enclosure.lo) <= exact <= mpq(enclosure.hi)
        synthetic = dict(initial=(F(0), F(1), F(2)),
                         past=[(F(3, 4), F(-1)), (F(2, 3), F(1, 2))])
        q = 0.7
        oracle = NumericalHistory(synthetic)
        step = F(1, 10**30)
        right = mp_endpoint(1/(F(q)+step), synthetic["initial"], synthetic["past"])
        left = mp_endpoint(1/(F(q)-step), synthetic["initial"], synthetic["past"])
        numerical_derivatives = [(a-b)/(2*mpq(step)) for a, b in zip(right, left)]
        assert all(abs(a-b) < mp.mpf("1e-45")
                   for a, b in zip(numerical_derivatives, oracle.jet(q)[1]))
    print("Analytic generation kernels passed; no holdout cases generated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--frozen-methods", type=Path,
                        help="Existing JSON manifest frozen before this generation")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.generate:
            parser.error("--self-test and --generate are separate operations")
        self_test()
        return
    source_hash = sha256(Path(__file__).read_bytes())
    protocol_hash = sha256(PROTOCOL.read_bytes())
    if not args.generate:
        print(json.dumps(dict(version=VERSION, seed=SEED, count=48,
                              generator_sha256=source_hash, protocol_sha256=protocol_hash,
                              status="PREGENERATION_ONLY_NO_CASES_CREATED"), indent=2))
        return
    if args.frozen_methods is None or not args.frozen_methods.is_file():
        parser.error("Generation requires a pre-existing --frozen-methods manifest")
    frozen_bytes = args.frozen_methods.read_bytes()
    json.loads(frozen_bytes)  # Require a readable JSON record; preserve its exact bytes/hash.
    destinations = [ROOT/"inputs"/"holdout_48.json", ROOT/"qa"/"holdout_generation_private.json",
                    ROOT/"inputs"/"HOLDOUT_FROZEN.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Holdout outputs already exist; no overwrite or regeneration")
    pairs = random_cases() + structural_cases()
    assert len(pairs) == 48
    cases, private = map(list, zip(*pairs))
    assert len({case["name"] for case in cases}) == 48
    assert all(set(case) == ALLOWED_FIELDS for case in cases)
    inputs_bytes = json_bytes(cases)
    private_bytes = json_bytes(dict(version=VERSION, seed=SEED,
                                    disclosure="Generation-only hidden parameters; do not provide to solvers.",
                                    cases=private))
    manifest = dict(version=VERSION, seed=SEED, count=48, random_count=32,
                    structural_count=16, generator_sha256=source_hash,
                    protocol_sha256=protocol_hash, frozen_methods_path=str(args.frozen_methods.resolve()),
                    frozen_methods_sha256=sha256(frozen_bytes), inputs_sha256=sha256(inputs_bytes),
                    private_generation_sha256=sha256(private_bytes), python=sys.version.split()[0],
                    mpmath=mp.__version__, scipy=scipy.__version__, selection="All 48 predefined cases retained; no solver calls, "
                    "redraws, skips, or outcome-dependent thresholds.")
    for path, data in zip(destinations, (inputs_bytes, private_bytes, json_bytes(manifest))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
