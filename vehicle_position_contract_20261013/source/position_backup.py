"""Conditional/empirical one-sided stopping-position envelope.

The virtual deceleration B is a declared envelope shape, not an established
pointwise CarSim brake acceleration bound. Membership is tested separately.
"""
import math

A_PLUS = 2.5
BRAKE_DELAY = .30
VIRTUAL_B = 2.0
POSITION_PADDING = .25
BRAKE_PRESSURE = 2.5
TERMINAL_SPEED = .5
STOP_SPEED = .05
STOP_PERSISTENCE = .5
STOP_TAIL = 3.0


def prefix_upper(v, t, extra_delay=0.0):
    if not math.isfinite(v) or not math.isfinite(t) or v < 0 or t < 0 or extra_delay < 0:
        raise ValueError('Nonnegative finite speed/time required')
    delay = BRAKE_DELAY+extra_delay
    first = min(t, delay)
    distance = v*first+A_PLUS*first*first/2
    if t > delay:
        peak = v+A_PLUS*delay
        s = min(t-delay, peak/VIRTUAL_B)
        distance += peak*s-VIRTUAL_B*s*s/2
    return distance+POSITION_PADDING


def stopping_reserve(v, extra_delay=0.0):
    if v < 0 or not math.isfinite(v) or extra_delay < 0:
        raise ValueError('Nonnegative finite speed/delay required')
    delay = BRAKE_DELAY+extra_delay
    peak = v+A_PLUS*delay
    return v*delay+A_PLUS*delay*delay/2+peak*peak/(2*VIRTUAL_B)+POSITION_PADDING


def next_period_required(v_upper, decision_period=.05, measurement_pad=.005, deadline_delay=.05):
    """Conditional safe normal step followed by this same backup response class."""
    v=max(0.0,v_upper)
    travel=v*decision_period+A_PLUS*decision_period*decision_period/2
    return travel+stopping_reserve(v+A_PLUS*decision_period, deadline_delay)+measurement_pad
