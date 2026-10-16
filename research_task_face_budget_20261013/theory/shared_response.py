"""Root-consumable shared normalized-cubic response contract, exact rationals.

Case schema: response={kind:'normalized_global_cubic_v1',epsilon_x,L};
L must equal sum(past durations)+H, rho must be zero. Public interval APIs use
(Fraction lower,Fraction upper) pairs and return pairs or None.
"""
from fractions import Fraction as F
import response_budget_math as rb

KIND = "normalized_global_cubic_v1"


def _contract(case):
    response = case.get("response")
    if not isinstance(response, dict) or response.get("kind") != KIND:
        raise ValueError("Use the frozen normalized_global_cubic_v1 response kind")
    epsilon = rb.rational(response["epsilon_x"])
    length = rb.rational(response["L"])
    horizon = rb.rational(case["H"])
    durations = tuple(rb.rational(segment[0]) for segment in case["past"])
    if epsilon < 0 or horizon < 0 or any(dt < 0 for dt in durations):
        raise ValueError("Nonnegative amplitude/horizon/past durations required")
    if length <= 0 or length != sum(durations, F(0))+horizon:
        raise ValueError("L must equal the full task duration sum(past)+H > 0")
    if rb.rational(case.get("rho", 0)) != 0:
        raise ValueError("This frozen response interface requires future rho=0")
    return rb.CubicResponseClass(length, rb.Interval(-epsilon, epsilon))


def profile(case, t_abs):
    contract = _contract(case)
    return tuple(contract.profile(axis, t_abs) for axis in ("x", "v", "a"))


def expanded_packet(case):
    contract = _contract(case)
    time = sum((rb.rational(segment[0]) for segment in case["past"]), F(0))
    psi = profile(case, time)
    packet = tuple(case["packet"])
    if len(packet) != 3:
        raise ValueError("Single end-history packet must contain x/v/a intervals")
    result = []
    for row, coefficient in zip(packet, psi):
        expanded = rb.expanded_packet(rb.Interval(*row), coefficient, contract.beta_prior)
        result.append((expanded.lower, expanded.upper))
    return tuple(result)


def _beta(nominal_ranges, original_packet, psi, bound, inner):
    nominal_ranges, original_packet, psi = tuple(nominal_ranges), tuple(original_packet), tuple(psi)
    bound = rb.rational(bound)
    if bound < 0 or not psi or len({len(nominal_ranges), len(original_packet), len(psi)}) != 1:
        raise ValueError("Matching nonempty rows and nonnegative beta bound required")
    allowed = rb.Interval(-bound, bound)
    for nominal_pair, packet_pair, coefficient in zip(nominal_ranges, original_packet, psi):
        nominal, packet, coefficient = rb.Interval(*nominal_pair), rb.Interval(*packet_pair), rb.rational(coefficient)
        if not coefficient:
            compatible = (packet.lower <= nominal.lower and nominal.upper <= packet.upper) if inner \
                         else nominal.intersect(packet) is not None
            if not compatible:
                return None
            continue
        lower = packet.lower-(nominal.lower if inner else nominal.upper)
        upper = packet.upper-(nominal.upper if inner else nominal.lower)
        if lower > upper:
            return None
        allowed = allowed.intersect(rb.Interval(lower, upper).scaled(1/coefficient))
        if allowed is None:
            return None
    return allowed.lower, allowed.upper


def beta_outer(nominal_ranges, original_packet, psi, bound):
    """Necessary common beta outer, using whole-tau-cell nominal ranges."""
    return _beta(nominal_ranges, original_packet, psi, bound, inner=False)


def beta_inner(nominal_ranges, original_packet, psi, bound):
    """Sufficient common beta inner for one certified nominal tau point."""
    return _beta(nominal_ranges, original_packet, psi, bound, inner=True)


def future_upper(nominal_upper, beta_interval, psi_x):
    if beta_interval is None:
        raise ValueError("A nonempty beta outer is required")
    shift = rb.Interval(*beta_interval).scaled(psi_x)
    return rb.rational(nominal_upper)+shift.upper


def future_lower(nominal_lower, beta, psi_x):
    """Caller must independently verify beta is a common history witness."""
    return rb.rational(nominal_lower)+rb.rational(beta)*rb.rational(psi_x)


def response_curvature_bound(case):
    return _contract(case).curvature_bound()


def continuous_allowance(case, node_step, expanded_nominal_a=None):
    """Actual-position curvature, including response even for negative beta."""
    contract = _contract(case)
    if expanded_nominal_a is None:
        expanded_nominal_a = expanded_packet(case)[2]
    commands = tuple(abs(rb.rational(segment[1])) for segment in case["future"])
    if not commands:
        raise ValueError("A complete declared future held-input record is required")
    return rb.continuous_position_allowance(contract, rb.Interval(*expanded_nominal_a), max(commands), node_step)


def contract_metadata():
    return dict(rb.contract_metadata(), response_kind=KIND, case_L_validated=True,
                single_end_history_packet_implementation=True, shared_beta_sensor_rank_one=True)
