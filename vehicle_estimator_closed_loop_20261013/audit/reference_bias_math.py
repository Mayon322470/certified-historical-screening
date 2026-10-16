"""Independent rational reference for the declared digital-observation gate.

No simulator imports, source estimator imports, or historical-package writes.
This is a component check; it does not establish CarSim response membership.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def square_packet(reading, range_error=F(3, 50)):
    reading, range_error = F(reading), F(range_error)
    if reading < 0 or range_error < 0:
        raise ValueError("nonnegative reading and declared error required")
    return max(F(0), reading-range_error)**2, (reading+range_error)**2


def quadratic_bounds(odo, s0, landmark_s, lateral, t, bias_interval):
    odo, s0, landmark_s, lateral, t = map(F, (odo, s0, landmark_s, lateral, t))
    left, right = map(F, bias_interval)
    if t < 0 or left > right:
        raise ValueError("nonnegative time and ordered bias interval required")
    zleft = odo+s0-landmark_s-left*t
    zright = odo+s0-landmark_s-right*t
    zlo, zhi = min(zleft, zright), max(zleft, zright)
    minimum = F(0) if zlo <= 0 <= zhi else min(zlo*zlo, zhi*zhi)
    return minimum+lateral*lateral, max(zlo*zlo, zhi*zhi)+lateral*lateral


def reserve(v):
    # Independently copied mathematical constants, not imported source code.
    v = max(F(0), F(v))
    A, waiting, latency, tau, B, padding = F(5,2), F(1,20), F(1,20), F(3,10), F(2), F(1,4)
    travel = v*waiting+A*waiting*waiting/2
    next_v = v+A*waiting
    delay = tau+latency
    return travel+next_v*delay+A*delay*delay/2+(next_v+A*delay)**2/(2*B)+padding


def position_task_face(odo, s0, t, speed_reading, bias, odom_pad=F(1,50), speed_pad=F(1,100)):
    odo, s0, t, speed_reading, bias = map(F, (odo, s0, t, speed_reading, bias))
    if t < 0:
        raise ValueError("nonnegative time required")
    return s0+odo-bias*t+F(odom_pad)+reserve(speed_reading-bias+F(speed_pad))


def main():
    checks = 0
    # Exact Pythagorean triples avoid numerical roots in the containment test.
    for z, c, r in ((3,4,5), (-3,4,5), (5,12,13), (-5,12,13), (0,7,7), (8,15,17)):
        for eodo in (F(-1,50), F(-1,100), F(0), F(1,100), F(1,50)):
            for erange in (F(-1,25), F(-1,50), F(0), F(1,50), F(1,25)):
                packet = square_packet(F(r)+erange)
                h = (F(z)+eodo)**2+F(c)**2
                assert packet[0] <= h <= packet[1]
                checks += 1
    # Every interval bound encloses exact evaluations; include vertex crossings.
    for odo in (F(-2), F(0), F(3), F(7)):
        for t in (F(0), F(1,20), F(1), F(4)):
            for interval in ((F(-1),F(1)), (F(0),F(1,4)), (F(1,3),F(7,3))):
                bounds = quadratic_bounds(odo, F(2), F(1), F(3), t, interval)
                for j in range(17):
                    b = interval[0]+(interval[1]-interval[0])*F(j,16)
                    h = (odo+F(2)-F(1)-b*t)**2+F(3)**2
                    assert bounds[0] <= h <= bounds[1]
                    checks += 1
    # Lowest retained bias is the worst upper face, including clipped low speed.
    for t in (F(0), F(1,20), F(1), F(7)):
        for speed in (F(0), F(1,10), F(8), F(18)):
            previous = None
            for j in range(17):
                b = F(-1,5)+F(j,40)
                face = position_task_face(F(3),F(2),t,speed,b)
                assert previous is None or face <= previous
                previous = face
                checks += 1
    status = {
        "status": "PASS_INDEPENDENT_RATIONAL_OBSERVATION_COMPONENT_CHECKS",
        "checks": checks,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Digital range+odom packet inclusion, exact quadratic bounds, monotone task face. No simulator, method kernel, or physical-response qualification.",
        "range_composite_error_m": "3/50",
        "odom_error_m": "1/50",
        "speed_error_mps": "1/100",
        "future_cubic_vehicle_model_used": False,
    }
    output = Path(__file__).with_name("OBSERVATION_MATH_CHECKS.json")
    if output.exists():
        raise FileExistsError("Preserve existing audit record")
    output.write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
