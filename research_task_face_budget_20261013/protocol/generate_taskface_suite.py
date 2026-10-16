"""Independent generation and witness QA for a declared response class.

Default: describe the design without instantiating formal cases. --self-test and
--write-development never draw from the formal RNG. Formal generation requires
a matching pre-existing method freeze and refuses all output replacement.
Only standard-library arithmetic is used; no comparison solver is imported.
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

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "FINITE_COMPARISON_PROTOCOL_CN.txt"
VERSION = "taskface-cubic-paired-v1"
SEED = 20261013137
EPSILONS = (F(1, 1000), F(1, 100))
SCALE = 10**80
PUBLIC_FIELDS = {"name", "initial", "past", "prior", "packet", "future", "H",
                 "rho", "limit", "response"}


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def as_bytes(value):
    return (json.dumps(encode(value), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def down(value):
    value = F(value)
    return F(value.numerator * SCALE // value.denominator, SCALE)


def up(value):
    return -down(-F(value))


@dataclass(frozen=True)
class RI:
    lo: F
    hi: F

    @staticmethod
    def point(value):
        return value if isinstance(value, RI) else RI(F(value), F(value))

    def __add__(self, other):
        other = self.point(other)
        return RI(down(self.lo + other.lo), up(self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return RI(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.point(other)

    def __rsub__(self, other):
        return self.point(other) + -self

    def __mul__(self, other):
        other = self.point(other)
        products = [x*y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return RI(down(min(products)), up(max(products)))

    __rmul__ = __mul__

    def pair(self):
        return [self.lo, self.hi]


def exp_minus(value):
    """Rational alternating-series enclosure, scaled by interval squaring."""
    value = F(value)
    if value < 0:
        raise ValueError("Nonnegative exponent required")
    if not value:
        return RI.point(1)
    squarings = 0
    while value > F(1, 2):
        value /= 2
        squarings += 1
    total = term = F(1)
    n = 0
    while True:
        previous = total
        n += 1
        term *= -value / n
        total += term
        if abs(term) < F(1, 10**90):
            result = RI(down(min(previous, total)), up(max(previous, total)))
            break
    for _ in range(squarings):
        result = result * result
    return result


def flow(tau, state, duration, command):
    tau, duration, command = map(F, (tau, duration, command))
    if tau <= 0 or duration < 0:
        raise ValueError("Invalid flow parameter")
    x, v, a = map(RI.point, state)
    decay = exp_minus(duration/tau)
    delta = a - command
    return (x + v*duration + command*duration**2/2
            + delta*(tau*duration - tau**2*(1-decay)),
            v + command*duration + delta*tau*(1-decay),
            command + delta*decay)


def endpoint(tau, initial, segments, until=None):
    total = sum((F(dt) for dt, _ in segments), F(0))
    left = total if until is None else F(until)
    if not 0 <= left <= total:
        raise ValueError("Query not covered by held commands")
    state = tuple(map(RI.point, initial))
    for duration, command in segments:
        step = min(F(duration), left)
        if step:
            state = flow(tau, state, step, command)
        left -= step
        if not left:
            break
    return state


def profile(time, length):
    t, length = F(time), F(length)
    if length <= 0 or not 0 <= t <= length:
        raise ValueError("Invalid global response clock")
    return (t**3/length**3, 3*t*t/length**3, 6*t/length**3)


def clocks(case):
    past = sum((dt for dt, _ in case["past"]), F(0))
    return past, past + case["H"]


def history_certificate(case, tau, beta):
    t, length = clocks(case)
    nominal = endpoint(tau, case["initial"], case["past"])
    actual = [p + beta*psi for p, psi in zip(nominal, profile(t, length))]
    inside = [lo <= item.lo <= item.hi <= hi
              for item, (lo, hi) in zip(actual, case["packet"])]
    return dict(tau=tau, beta=beta, nominal=[p.pair() for p in nominal],
                actual=[p.pair() for p in actual], original_packet_containment=inside,
                same_beta_for_all_coordinates=True, same_beta_for_future=True)


def future_point(case, tau, beta, time):
    t, length = clocks(case)
    past = endpoint(tau, case["initial"], case["past"])
    state = endpoint(tau, past, case["future"], time)
    return state[0] + beta*profile(t + time, length)[0]


def public_upper(case, epsilon):
    """All actual trajectories, original current packet, rho fixed to zero."""
    t, length = clocks(case)
    psi_a = profile(t, length)[2]
    alo, ahi = case["packet"][2]
    nominal_initial_a = max(abs(alo-epsilon*psi_a), abs(ahi+epsilon*psi_a))
    umax = max(abs(u) for _, u in case["future"])
    nominal_acceleration = max(nominal_initial_a, umax)
    actual_acceleration = nominal_acceleration + 6*epsilon/length**2
    h = case["H"]
    upper = case["packet"][0][1] + max(F(0), case["packet"][1][1])*h
    upper += actual_acceleration*h*h/2
    return dict(position_upper=upper, actual_acceleration_abs_bound=actual_acceleration,
                nominal_acceleration_abs_bound=nominal_acceleration, epsilon=epsilon)


def numeric_flow(tau, state, duration, command):
    dt, u = float(duration), float(command)
    x, v, a = state
    one = -math.expm1(-dt/tau)
    d = a-u
    return (x+v*dt+u*dt*dt/2+d*(tau*dt-tau*tau*one),
            v+u*dt+d*tau*one, u+d*(1-one))


def numeric_endpoint(tau, initial, segments, until=None):
    state = tuple(map(float, initial))
    left = sum(float(dt) for dt, _ in segments) if until is None else float(until)
    for duration, command in segments:
        step = min(float(duration), max(0., left))
        state = numeric_flow(tau, state, step, command)
        left -= step
        if left <= 1e-15:
            break
    return state


def numerical_reference(case, hidden_tau):
    """Finite numerical joint search: not a global bound or a truth label."""
    t, length = clocks(case)
    psi = tuple(map(float, profile(t, length)))
    h, length = float(case["H"]), float(length)
    epsilon = float(EPSILONS[0])
    qlo, qhi = float(1/case["prior"][1]), float(1/case["prior"][0])
    records = []
    for n in (128, 256):
        qs = [qlo+(qhi-qlo)*j/n for j in range(n+1)] + [float(1/hidden_tau)]
        times = {h*j/n for j in range(n+1)}
        clock = 0.
        for dt, _ in case["future"]:
            clock += float(dt)
            times.add(min(h, clock))
        best = None
        feasible_points = 0
        for q in qs:
            tau = 1/q
            nominal = numeric_endpoint(tau, case["initial"], case["past"])
            lower, upper = -epsilon, epsilon
            for value, coefficient, (lo, hi) in zip(nominal, psi, case["packet"]):
                lower = max(lower, (float(lo)-value)/coefficient)
                upper = min(upper, (float(hi)-value)/coefficient)
            if lower > upper:
                continue
            feasible_points += 1
            for time in sorted(times):
                value = numeric_endpoint(tau, nominal, case["future"], time)[0]
                value += upper*((float(t)+time)/length)**3
                if best is None or value > best["position"]:
                    best = dict(position=value, tau=tau, beta=upper, future_time=time)
        records.append(dict(grid=n, feasible_q_points=feasible_points, best=best))
    valid = [row["best"] for row in records if row["best"] is not None]
    if not valid:
        raise ArithmeticError("No numerically feasible q points")
    best = max(valid, key=lambda row: row["position"])
    disagreement = (abs(records[0]["best"]["position"]-records[1]["best"]["position"])
                    if all(row["best"] is not None for row in records) else None)
    return F.from_float(best["position"]), dict(
        status="UNCERTIFIED_NUMERICAL_REFERENCE", grids=records,
        coarse_fine_difference=disagreement, finite_search_may_miss_components=True,
        finite_search_may_miss_continuous_time_maximum=True,
        uses_shared_beta=True, epsilon=EPSILONS[0], certificate=False)


def center_packets(case, tau, beta, radius, offset_sign=1):
    t, length = clocks(case)
    terminal = endpoint(tau, case["initial"], case["past"])
    boxes = []
    for k, (value, psi, r) in enumerate(zip(terminal, profile(t, length), radius)):
        center = (value.lo+value.hi)/2
        rounding_scale = 10**40
        center = F(center.numerator*rounding_scale//center.denominator, rounding_scale)
        shift = beta*psi + offset_sign*((-1)**k)*r/4
        boxes.append((center+shift-r, center+shift+r))
    case["packet"] = boxes


def ordinary_template(rng, policy, local):
    index = 2*policy + local
    holds = ((1, 2), (4, 8), (1, 4), (2, 8))[policy][local]
    initial = (F(rng.randint(-4, 4), 8), F(rng.randint(-6, 10), 8),
               F(rng.randint(-8, 8), 8))
    aligned = (policy//2+local) % 2 == 0
    primes = (7, 11, 13, 17, 19, 23, 29, 31)
    past = [(F(rng.randint(1, 4), 4) if aligned else F(rng.randint(2, 6), primes[j]),
             F(rng.randint(-12, 12), 8)) for j in range(holds)]
    prior = ((F(2, 5), F(9, 5)), (F(4, 5), F(14, 5)),
             (F(1, 4), F(15, 4)), (F(11, 10), F(11, 5)))[index % 4]
    tau = prior[0] + (prior[1]-prior[0])*F(rng.randint(2, 8), 10)
    h = (F(3, 4), F(5, 4), F(2), F(7, 3))[index % 4]
    weights = [rng.randint(1, 5) for _ in range((1, 2, 2, 4)[index % 4])]
    future = [(h*F(w, sum(weights)), F(rng.randint(-10, 8), 8)) for w in weights]
    case = dict(initial=initial, past=past, prior=prior, future=future, H=h, rho=F(0))
    beta = EPSILONS[0]*(F(-3, 4), F(-1, 4), F(1, 4), F(3, 4))[index % 4]
    width = (F(1, 5000), F(1, 200), F(1, 20), F(1, 2))[(index+policy) % 4]
    center_packets(case, tau, beta, [2*width, width, width/2], (-1)**index)
    return case, tau, beta, dict(family="ordinary", past_holds=holds,
                                aligned=aligned, measurement_radius=width)


def structural_template(rng, policy, local):
    duration = (F(3, 5), F(4, 5), F(6, 5), F(7, 5))[rng.randrange(4)]
    base = (F(7, 8), F(9, 8), F(11, 8))[rng.randrange(3)]
    amplitude = (F(1, 600), F(1, 450), F(1, 300), F(1, 200))[rng.randrange(4)]
    r0, r1 = F(rng.randint(44, 49), 100), F(rng.randint(60, 66), 100)
    delta = (F(1, 10**5), F(1, 10**14), F(0), F(-1, 10**10))[policy]
    a3, a2, a1, a0 = (F(1), -(2*r1+r0), r1*r1+delta+2*r0*r1,
                       -r0*(r1*r1+delta))
    initial = (F(rng.randint(-2, 2), 7), F(rng.randint(3, 8), 6), base)
    if local == 0:
        family = "four_hold_acceleration"
        perturb = [amplitude*x for x in (a3, a2, a1, a0)]
    else:
        family = "five_hold_velocity_valley" if policy % 2 == 0 else "five_hold_position_valley"
        # Valley has an exact repeated factor; the face offset varies separately.
        delta = (F(1, 10**7), F(1, 10**16), F(0), F(-1, 10**10))[policy]
        a3, a2, a1, a0 = F(1), -(2*r1+r0), r1*r1+2*r0*r1, -r0*r1*r1
        perturb = [amplitude*x for x in (-a3, a3-a2, a2-a1, a1-a0, a0)]
    total = len(perturb)*duration
    size = max(map(abs, perturb))
    vstar = initial[1]+base*total
    xstar = initial[0]+initial[1]*total+base*total*total/2
    xstar += sum(value*duration*(total-(F(j)+F(1, 2))*duration)
                 for j, value in enumerate(perturb)) if local else F(0)
    xwidth, vwidth = 1+size*total*total, 1+size*total
    packet = [(xstar-xwidth, xstar+xwidth), (vstar-vwidth, vstar+vwidth),
              (base-2*size, base+2*size)]
    if local == 0:
        packet[2] = (base-2*size, base)
    elif "velocity" in family:
        packet[1] = (vstar+delta, vstar+vwidth)
    else:
        packet[0] = (xstar-xwidth, xstar-delta)
    h = duration*F(policy+3, 3)
    case = dict(initial=initial, past=[(duration, base+p) for p in perturb],
                prior=(duration*F(3, 4), duration*F(17, 5)),
                future=[(h/2, F(0)), (h/2, F(-1, 5))], H=h, rho=F(0))
    beta = EPSILONS[0]*(F(-3, 4), F(1, 4), F(3, 4), F(-1, 4))[(policy+local) % 4]
    psi = profile(total, total+h)
    case["packet"] = [(lo+beta*p, hi+beta*p) for (lo, hi), p in zip(packet, psi)]
    return case, duration, beta, dict(family=family, duration=duration, base=base,
                                     amplitude=amplitude, r0=r0, r1=r1, face_offset=delta)


def finish_template(case, tau, beta, policy):
    history = history_certificate(case, tau, beta)
    if not all(history["original_packet_containment"]):
        raise ValueError("Generated declared trajectory does not satisfy original packet")
    times = {case["H"]*F(j, 4) for j in range(5)}
    clock = F(0)
    for dt, _ in case["future"]:
        clock += dt
        times.add(clock)
    values = [(time, future_point(case, tau, beta, time)) for time in sorted(times)]
    witness_time, witness_position = max(values, key=lambda row: row[1].lo)
    envelope = public_upper(case, EPSILONS[1])
    reference = None
    if policy == 0:
        case["limit"] = envelope["position_upper"]+F(1, 100)
        label = "CERTIFIED_SAFE_PUBLIC_ENVELOPE"
    elif policy == 1:
        case["limit"] = witness_position.lo-F(1, 1000)
        label = "CERTIFIED_UNSAFE_COMMON_GENERATING_WITNESS"
    else:
        try:
            peak, reference = numerical_reference(case, tau)
        except Exception as exc:
            peak = envelope["position_upper"]
            reference = dict(status="REFERENCE_FAILED_PUBLIC_BOUND_FALLBACK",
                             error=type(exc).__name__+": "+str(exc), certificate=False)
        case["limit"] = peak + (F(1, 10000) if policy == 2 else F(-1, 10000))
        label = "NEAR_BOUNDARY_UNCERTIFIED_DIFFICULTY_LABEL"
    return dict(history_witness=history, future_witness_time=witness_time,
                future_witness_position=witness_position.pair(), threshold_policy=policy,
                generation_label=label, numeric_reference=reference,
                public_upper_large_budget=envelope,
                witness_proves_unsafe=witness_position.lo > case["limit"],
                public_envelope_proves_safe=envelope["position_upper"] <= case["limit"])


def build_formal_suite():
    """Called only after formal freeze checks in main; no redraw or solver use."""
    rng = random.Random(SEED)
    cases, private, groups = [], [], []
    for policy in range(4):
        for structured in (False, True):
            for local in range(2):
                template_id = len(private)
                factory = structural_template if structured else ordinary_template
                template, tau, beta, design = factory(rng, policy, local)
                qa = finish_template(template, tau, beta, policy)
                private.append(dict(template_id=template_id, design=design, **qa))
                for epsilon in EPSILONS:
                    _, length = clocks(template)
                    case = dict(template, name=f"case_{len(cases):03d}",
                                response=dict(kind="normalized_global_cubic_v1",
                                              epsilon_x=epsilon, L=length))
                    if set(case) != PUBLIC_FIELDS or abs(beta) > epsilon:
                        raise AssertionError("Invalid public contract")
                    cases.append(case)
                    groups.append(dict(name=case["name"], template_id=template_id,
                                       family=design["family"], structured=structured,
                                       epsilon=epsilon, threshold_policy=policy))
    assert len(cases) == 32 and len(private) == 16
    return cases, private, groups


def development_fixtures():
    """Four deterministic analytic development examples; no formal RNG."""
    result, witnesses = [], []
    for index, (epsilon, beta) in enumerate(((F(0), F(0)), (EPSILONS[0], F(1, 2000)),
                                           (EPSILONS[1], F(-1, 2000)), (EPSILONS[1], F(1, 2000)))):
        case = dict(initial=(F(0), F(1), F(1, 5)),
                    past=[(F(1, 2), F(-1, 3)), (F(1, 2), F(1, 4))],
                    prior=(F(1, 2), F(2)), future=[(F(1, 2), F(-1, 2)), (F(1, 2), F(0))],
                    H=F(1), rho=F(0))
        tau = F(1)
        center_packets(case, tau, beta, (F(1, 10000), F(1, 10000), F(1, 10000)))
        qa = finish_template(case, tau, beta, 0 if index < 3 else 1)
        case.update(name=f"development_{index}",
                    response=dict(kind="normalized_global_cubic_v1", epsilon_x=epsilon, L=F(2)))
        result.append(case)
        witnesses.append(qa)
    return result, witnesses


def self_test():
    exact = flow(F(3, 2), (F(2), F(-1), F(2)), F(3, 4), F(2))
    assert all(iv.lo <= point <= iv.hi for iv, point in zip(exact, (F(29, 16), F(1, 2), F(2))))
    assert exp_minus(0).pair() == [F(1), F(1)]
    assert profile(F(2), F(2)) == (F(1), F(3, 2), F(3, 2))
    assert profile(F(0), F(2)) == (F(0), F(0), F(0))
    # Exact semigroup inclusion from two independent range-reduction paths.
    for a, b in ((F(1, 7), F(2, 9)), (F(1), F(9)), (F(31, 3), F(2, 5))):
        direct, product = exp_minus(a+b), exp_minus(a)*exp_minus(b)
        assert max(direct.lo, product.lo) <= min(direct.hi, product.hi)
    cases, qa = development_fixtures()
    assert len(cases) == 4
    assert all(all(row["history_witness"]["original_packet_containment"]) for row in qa)
    assert all(row["public_envelope_proves_safe"] for row in qa[:3])
    assert qa[3]["witness_proves_unsafe"]
    assert all(set(case) == PUBLIC_FIELDS for case in cases)
    return dict(status="PASS", scope="ANALYTIC_KERNEL_AND_4_DEVELOPMENT_FIXTURES",
                formal_rng_called=False, formal_cases_generated=0,
                test_count=8, physical_vehicle_membership_proved=False)


def write_exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--write-development", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--frozen-methods", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.write_development:
        check = self_test()
        cases, qa = development_fixtures()
        paths = [ROOT/"development"/name for name in
                 ("public_fixtures.json", "private_witnesses.json", "GENERATOR_SELF_TEST.json")]
        if any(path.exists() for path in paths):
            raise FileExistsError("Development outputs already exist")
        for path, value in zip(paths, (cases, qa, check)):
            write_exclusive(path, as_bytes(value))
        print(json.dumps(dict(status="DEVELOPMENT_ONLY", files=list(map(str, paths)))))
        return
    metadata = dict(version=VERSION, seed=SEED, base_templates=16, public_cases=32,
                    epsilon_values=list(map(str, EPSILONS)), formal_cases_generated=0,
                    generator_sha256=digest(__file__), protocol_sha256=digest(PROTOCOL))
    if not args.generate:
        print(json.dumps(dict(status="PREGENERATION_ONLY", **metadata), indent=2))
        return
    if args.frozen_methods is None or not args.frozen_methods.is_file():
        parser.error("Generation requires a pre-existing --frozen-methods JSON")
    freeze = json.loads(args.frozen_methods.read_text(encoding="utf-8-sig"))
    if freeze.get("status") != "METHODS_FROZEN":
        raise ValueError("Method manifest must have status METHODS_FROZEN")
    for key in ("generator_sha256", "protocol_sha256"):
        if freeze.get(key) != metadata[key]:
            raise ValueError("Freeze/source mismatch: "+key)
    paths = [ROOT/"frozen_suite"/name for name in
             ("public_cases_32.json", "private_generation_witnesses.json",
              "evaluation_groups.json", "GENERATION_MANIFEST.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Formal outputs already exist; regeneration is forbidden")
    if (ROOT/"frozen_suite"/"GENERATION_FAILURE.json").exists():
        raise FileExistsError("A failed generation is preserved; start a new version, not a redraw")
    self_test()
    try:
        cases, private, groups = build_formal_suite()
    except Exception as exc:
        write_exclusive(ROOT/"frozen_suite"/"GENERATION_FAILURE.json", as_bytes(dict(
            status="GENERATION_FAILED_NO_REDRAW", error=type(exc).__name__+": "+str(exc), **metadata)))
        raise
    payloads = [as_bytes(value) for value in (cases, private, groups)]
    manifest = dict(metadata, formal_cases_generated=32,
                    status="GENERATED_ALL_PREDECLARED_CASES", python=sys.version,
                    method_freeze_path=str(args.frozen_methods.resolve()),
                    method_freeze_sha256=digest(args.frozen_methods),
                    output_hashes={path.name: hashlib.sha256(data).hexdigest()
                                   for path, data in zip(paths, payloads)},
                    formal_solver_called=False, redraws=0, physical_membership_claimed=False)
    for path, data in zip(paths, payloads+[as_bytes(manifest)]):
        write_exclusive(path, data)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
