"""Matched Ma/Zhao query with one beta shared by history and future response.

All methods use original packet information, a full rank-one measurement
noise column, common beta cell bounds, witnesses and actual-x bridge.
"""
from pathlib import Path
from fractions import Fraction as F
import sys
import time
sys.path.append(str(Path(__file__).resolve().parents[1] / 'theory'))
import shared_response as response
import certified_core as cc
import polynomial_contract as pc
import taylor_faces as tf
import rank1_ma as ma
import zhao_carleman as zhao

METHODS = ('ma_rank1', 'ma_rank1_structured', 'zhao1_rank1',
           'zhao2t3_rank1', 'zhao2t3_rank1_structured')


def parse_case(raw):
    case = dict(raw)
    for key in ('initial', 'prior'):
        case[key] = tuple(map(F, case[key]))
    for key in ('past', 'future', 'packet'):
        case[key] = [tuple(map(F, row)) for row in case[key]]
    for key in ('H', 'rho', 'limit'):
        case[key] = F(case[key])
    if len(case['initial']) != 3 or len(case['packet']) != 3 or not 0 < case['prior'][0] < case['prior'][1]:
        raise ValueError('Three output coordinates and positive ordered tau prior required')
    if case['H'] < 0 or any(dt < 0 for dt, _ in case['past'] + case['future']):
        raise ValueError('Nonnegative times required')
    if not case['future'] or sum((dt for dt, _ in case['future']), F(0)) < case['H']:
        raise ValueError('Future plan must cover the horizon')
    if any(lo >= hi for lo, hi in case['packet']):
        raise ValueError('Strictly positive original measurement widths required')
    response.expanded_packet(case)  # Validates the complete declared class.
    return case


def point_witness(case, theta, state):
    T = sum((dt for dt, _ in case['past']), F(0))
    psi = response.profile(case, T)
    b = F(case['response']['epsilon_x'])
    ranges = [(v.lower(), v.upper()) for v in state]
    beta = response.beta_inner(ranges, case['packet'], psi, b)
    if beta is None:
        return None
    chosen = (beta[0]+beta[1])/2
    actual = [(lo+chosen*p, hi+chosen*p) for (lo, hi), p in zip(ranges, psi)]
    if not all(lo <= a <= c <= hi for (a, c), (lo, hi) in zip(actual, case['packet'])):
        raise ArithmeticError('Joint point witness failed original packet replay')
    return dict(theta=theta, beta=chosen, beta_inner=beta,
                nominal_historical_state=ranges, actual_historical_state=actual,
                original_packet=case['packet'], global_history_time=T)


def quick_certificate(raw):
    case = parse_case(raw)
    H = case['H']
    physical_packet = case['packet']
    curvature = response.continuous_allowance(case, H)
    A = curvature['actual_acceleration_bound']
    upper = physical_packet[0][1]+max(F(0), physical_packet[1][1])*H+A*H*H/2
    theta = sum(case['prior'])/2
    state = cc.endpoint(theta, case['initial'], case['past'])
    witness = point_witness(case, theta, state)
    return dict(status='QUICK_CERTIFIED_SAFE' if upper <= case['limit'] and witness else 'QUICK_NO_CERTIFICATE',
                quick_upper=upper, continuous_bound=curvature,
                nonempty_witness=witness, midpoint_theta=theta,
                shared_beta_contract=True, future_vertices=0)


def certify(raw, n, nt, method, collect=True, deadline=None):
    if method not in METHODS or n <= 0 or nt <= 0:
        raise ValueError('Frozen method and positive grids required')
    case = parse_case(raw)
    structured = method.endswith('_structured')
    is_zhao = method.startswith('zhao')
    expanded = response.expanded_packet(case)
    outer_case = dict(case, packet=expanded)
    model = ma.prepare(outer_case)
    sturm = pc.prepare(outer_case, 'aligned_sturm') if structured else None
    sparse = tf.prepare(outer_case) if structured else None
    if structured:
        import coupling_faces
        coupling = coupling_faces.prepare(case)
    else:
        coupling = None
    bounds = cc.history_bounds(outer_case)
    T, H = bounds['T'], case['H']
    b = F(case['response']['epsilon_x'])
    psi = response.profile(case, T)
    response_generator = [b*p for p in psi]
    bridge = response.continuous_allowance(case, H/F(nt))
    allowance = bridge['allowance']
    U = max(abs(u) for _, u in case['future'])
    times = [H*F(j, nt) for j in range(nt+1)]
    lo, hi = case['prior']
    grid = [lo+(hi-lo)*F(j, n) for j in range(n+1)]
    states, queries = {}, {}
    deleted, retained, front_records = [], [], []
    nonempty, unsafe = None, None
    completed, hit = 0, False

    def state(theta):
        if theta not in states:
            states[theta] = cc.endpoint(theta, case['initial'], case['past'])
        return states[theta]

    def query(theta):
        if theta not in queries:
            queries[theta] = cc.timeline(theta, state(theta), case['future'], times, F(0))
        return queries[theta]

    def check_witness(theta):
        nonlocal nonempty, unsafe
        candidate = point_witness(case, theta, state(theta))
        if candidate is None:
            return
        if nonempty is None:
            nonempty = candidate
        # One realizable beta used for all history coordinates and this future.
        beta = candidate['beta_inner'][1]
        for s, y in zip(times, query(theta)):
            lower = response.future_lower(y[0].lower(), beta, response.profile(case, T+s)[0])
            if lower > case['limit']:
                shifted = [(v.lower()+beta*p, v.upper()+beta*p) for v, p in zip(state(theta), psi)]
                if not all(a <= l <= r <= z for (l, r), (a, z) in zip(shifted, case['packet'])):
                    raise ArithmeticError('UNSAFE beta does not replay all original history')
                unsafe = dict(theta=theta, beta=beta, time=s, global_time=T+s,
                              nominal_position_lower=y[0].lower(), position_lower=lower,
                              actual_historical_state=shifted, beta_inner=candidate['beta_inner'])
                return

    for i in range(n):
        if deadline is not None and time.perf_counter() >= deadline:
            hit = True
            break
        l, r = grid[i], grid[i+1]
        rejection = None
        if structured:
            cert = pc.reject_cell(sturm, l, r)
            if cert is not None:
                rejection = dict(index=i, engine='sturm', certificate=cert)
            else:
                cert = tf.reject_cell(sparse, l, r, True)
                if cert is not None:
                    rejection = dict(index=i, engine='sparse', certificate=cert)
                else:
                    cert = coupling_faces.reject_cell(coupling, l, r)
                    if cert is not None:
                        rejection = dict(index=i, engine='coupling', certificate=cert)
        if rejection is not None:
            deleted.append(rejection)
            completed += 1
            continue
        if is_zhao:
            domain, cert = zhao.contract(model, case['packet'], l, r,
                                         nu=1 if method.startswith('zhao1') else 2,
                                         tighten=method.startswith('zhao2t3'),
                                         response_generator=response_generator)
        else:
            domain, cert = ma.contract(model, case['packet'], l, r, psi, b)
        front_records.append(dict(index=i, certificate=cert))
        if domain is None:
            deleted.append(dict(index=i, engine='zhao' if is_zhao else 'ma',
                                certificate_index=len(front_records)-1))
            completed += 1
            continue
        l, r = domain
        if l == r:
            cell = dict(l=l, r=r, gamma=(l, r), radii=[F(0)]*3)
            nominal_ranges = [(v.lower(), v.upper()) for v in state(l)]
        else:
            cell = cc.make_cell(l, r, state(l), state(r), expanded, bounds['M'])
            if cell is None:
                deleted.append(dict(index=i, engine='common_chord', theta_interval=(l, r)))
                if collect:
                    deleted[-1]['endpoint_state_bounds'] = [[(v.lower(), v.upper()) for v in state(t)] for t in (l, r)]
                completed += 1
                continue
            nominal_ranges = cc.box_bounds(cell, expanded)
        beta = response.beta_outer(nominal_ranges, case['packet'], psi, b)
        if beta is None:
            deleted.append(dict(index=i, engine='common_beta', theta_interval=(l, r),
                                nominal_historical_ranges=nominal_ranges,
                                projection=cell['gamma'], radii=cell['radii']))
            if collect:
                deleted[-1]['endpoint_state_bounds'] = [[(v.lower(), v.upper()) for v in state(t)] for t in (l, r)]
            completed += 1
            continue
        ql, qr = query(l), query(r)
        node_bounds = []
        for j, s in enumerate(times):
            nominal_upper = ql[j][0].upper() if l == r else cc.chord_upper(cell, ql[j][0], qr[j][0], s, bounds, U, F(0))
            node_bounds.append(response.future_upper(nominal_upper, beta, response.profile(case, T+s)[0]))
        node = max(node_bounds)
        record = dict(index=i, interval=(l, r), projection=cell['gamma'], radii=cell['radii'],
                      beta_outer=beta, nominal_historical_ranges=nominal_ranges, max_node_upper=node)
        if collect:
            record.update(endpoint_state_bounds=[[(v.lower(), v.upper()) for v in state(t)] for t in (l, r)])
        retained.append(record)
        for t in (l, r):
            check_witness(t)
        if nonempty is None or unsafe is None and node+allowance > case['limit']:
            check_witness(sum(cell['gamma'])/2)
        completed += 1
        if unsafe is not None:
            break
    complete = completed == n
    upper = max((row['max_node_upper'] for row in retained), default=None)
    if upper is not None:
        upper += allowance
    if unsafe is not None:
        status = 'CERTIFIED_UNSAFE'
        if not complete:
            upper = None
    elif not complete:
        status, upper = 'UNKNOWN', None
    elif not retained:
        status = 'INCONSISTENT'
    elif upper <= case['limit'] and nonempty is not None:
        status = 'CERTIFIED_SAFE'
    else:
        status = 'UNKNOWN'
    result = dict(method=method, status=status, n=n, nt=nt, upper=upper,
                  time_allowance=allowance, continuous_bridge=bridge, witness=unsafe,
                  nonempty_witness=nonempty, completed_cells=completed, budget_hit=hit,
                  grid_complete=complete, historical_vertices=len(states), future_vertices=len(queries),
                  retained=len(retained), deleted=len(deleted), frontend_calls=len(front_records),
                  sturm_status=sturm['status'] if sturm else None,
                  coupling_faces=len(coupling['faces']) if coupling else 0,
                  response_contract=case['response'], expanded_packet=expanded,
                  future_uses_global_response_clock=True)
    if collect:
        result.update(cells=retained, deletions=deleted, frontend_certificates=front_records,
                      sturm_model=sturm, sparse_model=sparse, coupling_model=coupling)
    return result
