"""Exact affine nuisance elimination around a shared one-tau nominal model.

All intervals supplied by the nominal oracle must cover the claimed point/cell.
This module does not fit or qualify a physical model, evaluate exponentials,
or claim a CarSim discrepancy budget is sound. No old files are imported.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from numbers import Integral


def rational(value):
    if isinstance(value, float):
        raise TypeError("Use exact fractions/decimal strings or an outward nominal interval")
    return value if isinstance(value, F) else F(value)


@dataclass(frozen=True)
class Interval:
    lower: F
    upper: F

    def __post_init__(self):
        lower, upper = rational(self.lower), rational(self.upper)
        if lower > upper:
            raise ValueError("Ordered finite interval required")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def midpoint(self):
        return (self.lower+self.upper)/2

    def intersect(self, other):
        lower, upper = max(self.lower, other.lower), min(self.upper, other.upper)
        return None if lower > upper else Interval(lower, upper)

    def scaled(self, scalar):
        scalar = rational(scalar)
        a, b = scalar*self.lower, scalar*self.upper
        return Interval(min(a, b), max(a, b))

    def contains(self, value):
        return self.lower <= rational(value) <= self.upper


def from_outward_nominal(value):
    """Adapt old certified_core.IV without importing/mutating that package."""
    return Interval(value.lower(), value.upper())


@dataclass(frozen=True)
class CubicResponseClass:
    total_time: F
    beta_prior: Interval

    def __post_init__(self):
        total_time = rational(self.total_time)
        if total_time <= 0 or not isinstance(self.beta_prior, Interval):
            raise ValueError("Positive total time and exact beta interval required")
        object.__setattr__(self, "total_time", total_time)

    def profile(self, axis, absolute_time):
        t, length = rational(absolute_time), self.total_time
        if not 0 <= t <= length:
            raise ValueError("Profile uses the full task absolute clock in [0,L]")
        if axis == "x":
            return (t/length)**3
        if axis == "v":
            return 3*t*t/(length**3)
        if axis == "a":
            return 6*t/(length**3)
        raise ValueError("Physical axis must be x, v, or a")

    def amplitude(self):
        return max(abs(self.beta_prior.lower), abs(self.beta_prior.upper))

    def curvature_bound(self):
        return 6*self.amplitude()/(self.total_time**2)


@dataclass(frozen=True)
class HistoryRow:
    name: str
    axis: str
    absolute_time: F
    nominal: Interval
    packet: Interval

    def __post_init__(self):
        if not self.name or self.axis not in ("x", "v", "a") or not isinstance(self.nominal, Interval) or not isinstance(self.packet, Interval):
            raise ValueError("Named physical history row with validated intervals required")
        object.__setattr__(self, "absolute_time", rational(self.absolute_time))


def expanded_packet(packet, profile_coefficient, beta_prior):
    """Necessary nominal interval; does not itself produce a shared witness."""
    if not isinstance(packet, Interval) or not isinstance(beta_prior, Interval):
        raise ValueError("Exact intervals required")
    shift = beta_prior.scaled(profile_coefficient)
    return Interval(packet.lower-shift.upper, packet.upper-shift.lower)


def shared_beta_interval(response_class, rows, kind="outer"):
    """One beta shared by ALL supplied history rows and times.

    outer is necessary for a whole tau cell, given sound nominal ranges.
    inner is sufficient for a tau POINT, given sound nominal point enclosures:
    every beta it returns puts each exact nominal value in its original packet.
    A failed inner check is UNKNOWN feasibility, not proof of an empty class.
    """
    if not isinstance(response_class, CubicResponseClass) or kind not in ("outer", "inner"):
        raise ValueError("Declared cubic class and inner/outer kind required")
    allowed = response_class.beta_prior
    for row in rows:
        if not isinstance(row, HistoryRow):
            raise ValueError("Validated history rows required")
        coefficient = response_class.profile(row.axis, row.absolute_time)
        if not coefficient:
            if kind == "outer":
                compatible = row.nominal.intersect(row.packet) is not None
            else:
                compatible = row.packet.lower <= row.nominal.lower and row.nominal.upper <= row.packet.upper
            if not compatible:
                return None
            continue
        if kind == "outer":
            low, high = row.packet.lower-row.nominal.upper, row.packet.upper-row.nominal.lower
        else:
            low, high = row.packet.lower-row.nominal.lower, row.packet.upper-row.nominal.upper
        if low > high:
            return None
        row_beta = Interval(low, high).scaled(1/coefficient)
        allowed = allowed.intersect(row_beta)
        if allowed is None:
            return None
    return allowed


def point_history_witness(response_class, rows):
    """Choose independent beta from inner interval; no hidden generator used."""
    rows = tuple(rows)
    inner = shared_beta_interval(response_class, rows, kind="inner")
    if inner is None:
        return None
    beta = inner.midpoint()
    records = []
    for row in rows:
        shift = beta*response_class.profile(row.axis, row.absolute_time)
        actual = Interval(row.nominal.lower+shift, row.nominal.upper+shift)
        if not (row.packet.lower <= actual.lower and actual.upper <= row.packet.upper):
            raise ArithmeticError("Internal witness replay failed")
        records.append(dict(name=row.name, actual=actual, original_packet=row.packet))
    return dict(beta=beta, beta_inner=inner, original_history_records=tuple(records),
                scope="Declared nominal-plus-cubic response class only", physical_vehicle_verified=False)


def future_position_cell_upper(response_class, beta_outer, nominal_upper, absolute_time):
    """Sound node upper when nominal_upper covers the WHOLE retained tau cell.

    Using one independently maximizing beta is an outer relaxation; it is not
    an attaining trajectory and cannot establish UNSAFE.
    """
    if not isinstance(beta_outer, Interval):
        raise ValueError("A nonempty shared beta outer interval is required")
    return rational(nominal_upper)+beta_outer.upper*response_class.profile("x", absolute_time)


def future_position_point_lower(response_class, beta, nominal_lower, absolute_time):
    """One shared history-witness beta, for a certified nominal tau POINT."""
    beta = rational(beta)
    if not response_class.beta_prior.contains(beta):
        raise ValueError("Witness beta outside the declared prior")
    return rational(nominal_lower)+beta*response_class.profile("x", absolute_time)


def continuous_position_allowance(response_class, expanded_nominal_acceleration_packet,
                                  maximum_held_input, node_step):
    """Common rho=0 allowance for ACTUAL x, not just nominal x.

    First-order nominal a lies between its query a and held inputs; cubic
    actual acceleration adds beta*psi_a. Requires the declared continuous
    response class, positive tau, complete input record and this dynamics.
    """
    if not isinstance(expanded_nominal_acceleration_packet, Interval):
        raise ValueError("Expanded nominal query acceleration packet required")
    u, h = rational(maximum_held_input), rational(node_step)
    if u < 0 or h < 0:
        raise ValueError("Nonnegative input magnitude and node step required")
    nominal_bound = max(abs(expanded_nominal_acceleration_packet.lower),
                        abs(expanded_nominal_acceleration_packet.upper), u)
    actual_bound = nominal_bound+response_class.curvature_bound()
    return dict(nominal_acceleration_bound=nominal_bound, response_curvature_bound=response_class.curvature_bound(),
                actual_acceleration_bound=actual_bound, allowance=actual_bound*h*h/8,
                continuous_contract_required=True, rho=F(0))


def fourier_motzkin_pair(response_class, lower_row, upper_row):
    """Outward value of psi_j(L_i-p_i)-psi_i(U_j-p_j).

    Exact nominal row values make <=0 the shared-beta pairwise necessary
    constraint. A strict positive LOWER bound over the whole tau cell proves
    rejection. Nonpositive upper bound does not prove full feasibility.
    """
    pi = response_class.profile(lower_row.axis, lower_row.absolute_time)
    pj = response_class.profile(upper_row.axis, upper_row.absolute_time)
    if pi <= 0 or pj <= 0:
        raise ValueError("Zero-time profile rows need separate direct consistency checks")
    # V=constant-pj*p_i+pi*p_j; respect both signs in outward interval algebra.
    constant = pj*lower_row.packet.lower-pi*upper_row.packet.upper
    left = lower_row.nominal.scaled(-pj)
    right = upper_row.nominal.scaled(pi)
    return Interval(constant+left.lower+right.lower, constant+left.upper+right.upper)


def contract_metadata():
    return {
        "status": "CONDITIONAL_SHARED_TAU_CUBIC_RESPONSE_INTERFACE",
        "tau_grid_added": False,
        "beta_grid_added": False,
        "future_rho": "0",
        "response_origin": "full-task absolute time, never reset at query",
        "packet_expansion_is_history_witness": False,
        "upper_tube_exceedance_is_unsafe_witness": False,
        "empty_inner_beta_proves_inconsistency": False,
        "physical_vehicle_discrepancy_qualified": False,
        "empirical_budget_used": False,
        "ma_zhao_adapter_implemented_here": False,
        "novelty_claim": False,
        "simulator_called": False,
    }
