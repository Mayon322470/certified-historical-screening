"""Independent generation-certificate recomputation. Run only after timed jobs end.

No production/generator imports. Uses a positive exponential series plus a
geometric tail and reciprocal, and a whole-history convolution identity rather
than the generator's alternating exponential and sequential state recurrence.
"""
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = 1 << 440


def lower(value):
    value = F(value)
    return F((value.numerator*UNIT)//value.denominator, UNIT)


def upper(value):
    return -lower(-F(value))


@dataclass(frozen=True)
class Interval:
    lo: F
    hi: F

    @staticmethod
    def scalar(value):
        return value if isinstance(value, Interval) else Interval(F(value), F(value))

    def __add__(self, other):
        other = self.scalar(other)
        return Interval(lower(self.lo+other.lo), upper(self.hi+other.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self+-self.scalar(other)

    def __rsub__(self, other):
        return self.scalar(other)+-self

    def __mul__(self, other):
        other = self.scalar(other)
        values = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(lower(min(values)), upper(max(values)))

    __rmul__ = __mul__

    def pair(self):
        return [str(self.lo), str(self.hi)]


@lru_cache(maxsize=30000)
def negative_exponential(value):
    """Bound e^-value via the reciprocal of a positive Taylor series with tail."""
    value = F(value)
    assert value >= 0
    if value == 0:
        return Interval.scalar(1)
    squarings = 0
    while value > F(1, 4):
        value /= 2
        squarings += 1
    total = term = Interval.scalar(1)
    degree = 0
    while True:
        # exp(x)-sum_{k=0}^n x^k/k! <= t_{n+1}/(1-x/(n+2)).
        next_term = term*(value/F(degree+1))
        tail = next_term.hi/(1-value/F(degree+2))
        if tail < F(1, 1 << 400):
            pos_lo, pos_hi = total.lo, total.hi+tail
            result = Interval(lower(1/pos_hi), upper(1/pos_lo))
            break
        total += next_term
        term = next_term
        degree += 1
        assert degree < 1000
    for _ in range(squarings):
        result = result*result
    return result


def closed_history(theta, initial, segments, rho=F(0), until=None):
    """Whole-record moment identity; all exponential evaluations are certified."""
    theta = F(theta)
    rho = F(rho)
    assert theta > 0
    segments = [tuple(map(F, row)) for row in segments]
    total = sum(dt for dt, _ in segments)
    horizon = total if until is None else F(until)
    assert 0 <= horizon <= total
    x0, v0, a0 = map(Interval.scalar, initial)
    a = a0*negative_exponential(horizon/theta)
    first_moment = second_moment = F(0)
    start = F(0)
    for duration, command in segments:
        end = min(horizon, start+duration)
        if end > start:
            equilibrium = command+theta*rho
            a += equilibrium*(negative_exponential((horizon-end)/theta)
                              - negative_exponential((horizon-start)/theta))
            first_moment += equilibrium*(end-start)
            second_moment += equilibrium*(end-start)*(horizon-(start+end)/2)
        start = end
        if start == horizon:
            break
    v = v0+first_moment+theta*(a0-a)
    x = x0+v0*horizon+second_moment+theta*horizon*a0-theta*first_moment-theta**2*(a0-a)
    return x, v, a


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


def encloses(stored, recomputed):
    return all(F(lo) <= interval.lo <= interval.hi <= F(hi)
               for (lo, hi), interval in zip(stored, recomputed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-after-timing", action="store_true")
    args = parser.parse_args()
    if not args.run_after_timing:
        print("Prepared only. Add --run-after-timing after ALL timed main/ablation jobs finish.")
        return
    inputs = read("inputs/holdout_48.json")
    private = read("qa/holdout_generation_private.json")["cases"]
    frozen = read("inputs/HOLDOUT_FROZEN.json")
    assert hashlib.sha256((ROOT/"inputs/holdout_48.json").read_bytes()).hexdigest() == frozen["inputs_sha256"]
    assert hashlib.sha256((ROOT/"qa/holdout_generation_private.json").read_bytes()).hexdigest() == frozen["private_generation_sha256"]
    rows = []
    mutation_rejected = False
    for case, meta in zip(inputs, private):
        assert case["name"] == meta["name"]
        failures = []

        def require(label, condition):
            if not condition:
                failures.append(label)

        theta = F(meta["hidden_theta"])
        initial = tuple(map(F, case["initial"]))
        state = closed_history(theta, initial, case["past"])
        require("stored_hidden_terminal_bounds", encloses(meta["certified_terminal"], state))
        require("hidden_packet_inclusion", encloses(case["packet"], state))
        time = F(meta["witness_time"])
        witness = closed_history(theta, state, case["future"], F(case["rho"]), time)[0]
        require("stored_hidden_witness_bounds", encloses([meta["certified_witness_position"]], [witness]))
        if meta["hidden_witness_strictly_unsafe"]:
            require("claimed_hidden_unsafe_witness", witness.lo > F(case["limit"]))
        m = F(meta["peak_at_160_dps"]["peak"])
        tolerance = (1+abs(m))/10**60
        require("hidden_peak_point_consistency_only", witness.lo-tolerance <= m <= witness.hi+tolerance)
        numerical = meta["numerical_compatible_reference"]
        ref_row = None
        if "reference_theta" in numerical:
            ref_theta = F(numerical["reference_theta"])
            ref_state = closed_history(ref_theta, initial, case["past"])
            require("stored_reference_terminal_bounds", encloses(numerical["reference_terminal_certificate"], ref_state))
            ref_in_packet = encloses(case["packet"], ref_state)
            if numerical["reference_history_certified"]:
                require("claimed_reference_history_inclusion", ref_in_packet)
            ref_time = F(numerical["reference_witness_time"])
            ref_witness = closed_history(ref_theta, ref_state, case["future"], F(case["rho"]), ref_time)[0]
            require("stored_reference_witness_bounds", encloses([numerical["reference_witness_position"]], [ref_witness]))
            if meta["reference_point_strictly_unsafe_certified"]:
                require("claimed_reference_unsafe_witness", ref_in_packet and ref_witness.lo > F(case["limit"]))
            reference_m = F(numerical["reference160"]["peak"])
            tolerance = (1+abs(reference_m))/10**60
            require("reference_peak_point_consistency_only", ref_witness.lo-tolerance <= reference_m <= ref_witness.hi+tolerance)
            ref_row = dict(theta=str(ref_theta), terminal=[item.pair() for item in ref_state],
                           witness=ref_witness.pair(), packet_inclusion_certified=ref_in_packet)
        if not rows:
            corrupted = [list(pair) for pair in meta["certified_terminal"]]
            corrupted[0] = [str(state[0].hi+1), str(state[0].hi+2)]
            mutation_rejected = not encloses(corrupted, state)
        rows.append(dict(name=case["name"], passed=not failures, failures=failures,
                         hidden_terminal=[item.pair() for item in state],
                         hidden_witness=witness.pair(), numerical_reference_point=ref_row))
        print(case["name"], "PASS" if not failures else failures, flush=True)
    passed = len(rows) == 48 and all(row["passed"] for row in rows) and mutation_rejected
    report = dict(status="PASS" if passed else "FAIL", cases=len(rows), mutation_rejected=mutation_rejected,
                  input_sha256=frozen["inputs_sha256"], private_sha256=frozen["private_generation_sha256"],
                  arithmetic="440-bit outward rational intervals; positive-exp geometric tail then reciprocal",
                  model_reconstruction="Independent whole-history convolution/moment identities; no production or generator imports",
                  scope="Stored hidden/reference point trajectories and reported point witnesses; not numerical feasible-set/global-maximum completeness",
                  numerical_global_reference_remains_uncertified=True, rows=rows)
    output = ROOT/"qa"/"GENERATION_MATH_AUDIT.json"
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
