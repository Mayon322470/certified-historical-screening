"""Independent rational reference for the scalar Ma2026 batch adaptation.

No production module is imported. Endpoint expressions are reconstructed from
command jumps. Differentiation uses Laurent-exponential terms in q=1/theta;
exp(-x) is bounded by unscaled rational alternating Taylor sums (220 bits).
This module validates mathematical enclosures, not wall-clock performance.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial


def interval(value):
    if isinstance(value, (list, tuple)):
        a, b = map(Q, value)
    else:
        a = b = Q(value)
    assert a <= b
    return a, b


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul(a, b):
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def scale(a, c):
    return mul(a, (c, c))


def contained(inner, outer):
    return outer[0] <= inner[0] <= inner[1] <= outer[1]


@lru_cache(maxsize=200000)
def exp_minus(x):
    x = Q(x)
    assert x >= 0
    if x == 0:
        return Q(1), Q(1)
    term = total = Q(1)
    n = 0
    while True:
        previous = total
        n += 1
        term *= -x / n
        total += term
        if n > 2 * x and abs(term) < Q(1, 2**220):
            # Discard huge Taylor denominators using independent outward
            # rounding; the reference still has over twice production bits.
            lo,hi=min(previous,total),max(previous,total)
            scale_factor=2**228
            return (Q(lo.numerator*scale_factor//lo.denominator,scale_factor),
                    Q(-((-hi.numerator*scale_factor)//hi.denominator),scale_factor))


def tidy(terms):
    return tuple(sorted((int(power), Q(tau), Q(c))
                        for (power, tau), c in terms.items() if c))


def endpoint_terms(case):
    """Return x,v,a expressions as tuples (power,tau,coefficient)."""
    x0, v0, a0 = map(Q, case['initial'])
    history = [(Q(dt), Q(u)) for dt, u in case['past']]
    assert all(dt >= 0 for dt, _ in history)
    holds = [(dt, u) for dt, u in history if dt > 0]
    duration = sum((dt for dt, _ in holds), Q(0))
    area = sum((dt * u for dt, u in holds), Q(0))
    clock = Q(0)
    position_area = Q(0)
    for dt, u in holds:
        position_area += u * dt * (duration - clock - dt / 2)
        clock += dt
    constant = holds[-1][1] if holds else a0
    jumps = {}
    if holds:
        jumps[duration] = a0 - holds[0][1]
        clock = Q(0)
        for i, (dt, u) in enumerate(holds[:-1]):
            clock += dt
            tau = duration - clock
            jumps[tau] = jumps.get(tau, Q(0)) + u - holds[i + 1][1]
    av = {(0, Q(0)): constant}
    vv = {(0, Q(0)): v0 + area, (-1, Q(0)): -(constant - a0)}
    xv = {(0, Q(0)): x0 + v0 * duration + position_area,
          (-1, Q(0)): -(area - a0 * duration),
          (-2, Q(0)): constant - a0}
    for tau, c in jumps.items():
        av[0, tau] = av.get((0, tau), Q(0)) + c
        vv[-1, tau] = vv.get((-1, tau), Q(0)) - c
        xv[-2, tau] = xv.get((-2, tau), Q(0)) + c
    return tuple(tidy(v) for v in (xv, vv, av))


@lru_cache(maxsize=100000)
def derivative(terms, order=1):
    assert order >= 0
    for _ in range(order):
        out = {}
        for power, tau, c in terms:
            if power:
                key = power - 1, tau
                out[key] = out.get(key, Q(0)) + power * c
            if tau:
                key = power, tau
                out[key] = out.get(key, Q(0)) - tau * c
        terms = tidy(out)
    return terms


@lru_cache(maxsize=100000)
def natural_range(terms, low, high):
    low, high = Q(low), Q(high)
    assert 0 < low <= high
    result = Q(0), Q(0)
    for power, tau, c in terms:
        a, b = low**power, high**power
        powers = min(a, b), max(a, b)
        exponentials = exp_minus(tau * high)[0], exp_minus(tau * low)[1]
        result = add(result, scale(mul(powers, exponentials), c))
    return result


def centered_range(terms, low, high, degree=4):
    """Independent Taylor enclosure; used only as a verified intersection."""
    low, high = Q(low), Q(high)
    assert 0 < low <= high and degree >= 0
    midpoint, radius = (low + high) / 2, (high - low) / 2
    result = natural_range(terms, midpoint, midpoint)
    for j in range(1, degree + 1):
        powers = (-radius**j, radius**j) if j % 2 else (Q(0), radius**j)
        coefficient = natural_range(derivative(terms, j), midpoint, midpoint)
        result = add(result, scale(mul(coefficient, powers), Q(1, factorial(j))))
    j = degree + 1
    powers = (-radius**j, radius**j) if j % 2 else (Q(0), radius**j)
    remainder = natural_range(derivative(terms, j), low, high)
    return add(result, scale(mul(remainder, powers), Q(1, factorial(j))))


@lru_cache(maxsize=100000)
def strong_range(terms, low, high):
    plain = natural_range(terms, low, high)
    centered = centered_range(terms, low, high)
    result = max(plain[0], centered[0]), min(plain[1], centered[1])
    assert result[0] <= result[1]
    return result


@lru_cache(maxsize=100000)
def prove_range_contained(terms, low, high, outer, depth=0):
    """Cover the whole interval by certified bounds; never use sampling.

    Distinct algebraic interval expressions can have different dependency
    widths. Adaptive subdivision lets the independent Laurent expression
    certify a tighter producer interval without adopting its formulas.
    """
    enclosed = strong_range(terms, low, high)
    if contained(enclosed, outer):
        return True
    midpoint = (low + high) / 2
    point = natural_range(terms, midpoint, midpoint)
    assert contained(point, outer), ('point outside claimed range',midpoint,point,outer)
    assert depth < 20, ('independent interval proof exhausted',low,high,enclosed,outer)
    return (prove_range_contained(terms,low,midpoint,outer,depth+1) and
            prove_range_contained(terms,midpoint,high,outer,depth+1))


def make_scalar_strips(q_interval, packet, g_values, g_derivatives, hessians):
    """Exact Theorem 4 strips including center/Jacobian enclosure errors.

    Inputs g_values/g_derivatives must enclose exact values at the center;
    hessians must enclose the whole closed q interval. C is chosen as the
    rational midpoint of the Jacobian enclosure. No floating LP is used.
    """
    low, high = map(Q, q_interval)
    assert 0 < low <= high
    p, r = (low + high) / 2, (high - low) / 2
    strips = []
    for measured, value, jacobian, hessian in zip(packet, g_values, g_derivatives, hessians):
        yl, yh = interval(measured)
        gl, gh = interval(value)
        dl, dh = interval(jacobian)
        hl, hh = interval(hessian)
        gm, C = (gl + gh) / 2, (dl + dh) / 2
        ell = min(Q(0), hl) * r*r / 2, max(Q(0), hh) * r*r / 2
        error = (gh - gl) / 2 + (dh - dl) * r / 2
        remainder = ell[0] - error, ell[1] + error
        b = (yl + yh) / 2 - gm + C*p - sum(remainder)/2
        s = (yh - yl) / 2 + (remainder[1] - remainder[0])/2
        strips.append(dict(C=C, b=b, s=s, remainder=remainder, signed_square=ell))
    assert len(strips) == len(packet) == len(g_values) == len(g_derivatives) == len(hessians)
    return strips


def scalar_lp(q_interval, strips):
    """Analytic exact optimum of Ma Eq.18 for 1D and independent boxes."""
    low, high = map(Q, q_interval)
    p, r = (low + high)/2, (high - low)/2
    options = [(r, -1, p)]
    for i, strip in enumerate(strips):
        C, b, s = (Q(strip[k]) for k in ('C', 'b', 's'))
        assert s >= 0
        if C:
            options.append((s/abs(C), i, b/C))
    radius, chosen, center = min(options)
    T = Q(chosen == -1)
    N = [Q(0)]*len(strips)
    if chosen != -1:
        N[chosen] = 1/Q(strips[chosen]['C'])
    return dict(center=center, radius=radius, interval=(center-radius, center+radius),
                chosen=chosen, T=T, N=N)


def validate_gain(q_interval, strips, T, N, claimed_center, claimed_radius):
    low, high = map(Q, q_interval)
    p, r = (low+high)/2, (high-low)/2
    T = Q(T)
    N = list(map(Q, N))
    assert len(N) == len(strips)
    assert T + sum((n*Q(s['C']) for n,s in zip(N,strips)), Q(0)) == 1
    center = T*p + sum((n*Q(s['b']) for n,s in zip(N,strips)), Q(0))
    radius = abs(T)*r + sum((abs(n)*Q(s['s']) for n,s in zip(N,strips)), Q(0))
    assert center == Q(claimed_center)
    assert Q(claimed_radius) >= radius
    return radius


def intersect_strips(q_interval, strips):
    low, high = map(Q, q_interval)
    for strip in strips:
        C, b, s = (Q(strip[k]) for k in ('C','b','s'))
        assert s >= 0
        if C == 0:
            if not b-s <= 0 <= b+s:
                return None
            continue
        endpoints = (b-s)/C, (b+s)/C
        low, high = max(low,min(endpoints)), min(high,max(endpoints))
        if low > high:  # Equality is a retained point, never an empty set.
            return None
    return low, high


def self_check():
    # Constant acceleration: all q derivatives vanish exactly.
    c = dict(initial=['1','2','3'], past=[['1/2','3'],['1/3','3']])
    models = endpoint_terms(c)
    t = Q(5,6)
    expected = Q(1)+2*t+3*t*t/2, Q(2)+3*t, Q(3)
    for expression, value in zip(models,expected):
        assert natural_range(expression,Q(1,3),Q(2)) == (value,value)
        assert derivative(expression) == ()
    strips = [dict(C=Q(2),b=Q(3),s=Q(1,4)),dict(C=Q(-1),b=Q(-2),s=Q(2))]
    result = scalar_lp((Q(1),Q(3)),strips)
    assert result['chosen']==0 and result['radius']==Q(1,8)
    validate_gain((Q(1),Q(3)),strips,result['T'],result['N'],result['center'],result['radius'])
    assert intersect_strips((Q(1),Q(2)),[dict(C=Q(1),b=Q(3),s=Q(1))]) == (Q(2),Q(2))
    assert intersect_strips((Q(1),Q(2)),[dict(C=Q(1),b=Q(3),s=Q(1,2))]) is None
    # Deliberately invalid gain must not pass its equality constraint.
    try:
        validate_gain((Q(1),Q(3)),strips,Q(0),[Q(0),Q(0)],Q(0),Q(0))
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt gain accepted')
    return {'status':'PASS','scope':'independent reference algebra only; no benchmark records audited'}


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(),indent=2))
