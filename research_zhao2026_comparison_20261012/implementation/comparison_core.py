"""Shared certified future backend for independent Ma2026 and Zhao2026 adapters."""
from fractions import Fraction as F
import time
import certified_core as cc
import polynomial_contract as pc
import taylor_faces as tf
import ma_scalar_certified as ma
import zhao_carleman as zhao

METHODS = ('ma_strip', 'ma_strip_structured', 'zhao1', 'zhao2',
           'zhao2t3', 'zhao2t3_structured', 'zhao2t3_sturm', 'zhao2t3_sparse')


def parse_case(raw):
    case = dict(raw)
    for key in ('initial', 'prior'):
        case[key] = tuple(map(F, case[key]))
    for key in ('past', 'future', 'packet'):
        case[key] = [tuple(map(F, x)) for x in case[key]]
    for key in ('H', 'rho', 'limit'):
        case[key] = F(case[key])
    return case


def certify(raw, n, nt, method, hessian_pieces=1, collect=True, deadline=None):
    if method not in METHODS:
        raise ValueError(method)
    case = parse_case(raw)
    mode=method.split('_')[-1]
    structured = mode in ('structured','sturm','sparse')
    use_sturm = mode in ('structured','sturm')
    use_sparse = mode in ('structured','sparse')
    is_zhao = method.startswith('zhao')
    strengthened = method.startswith('ma_strip')
    mm = ma.prepare(case)
    poly = pc.prepare(case, 'aligned_sturm') if use_sturm else None
    sparse = tf.prepare(case) if use_sparse else None
    bounds = cc.history_bounds(case)
    a, b = case['prior']
    H, rho = case['H'], case['rho']
    U = max(abs(u) for _, u in case['future'])
    allowance = (max(max(abs(c) for c in case['packet'][2]), U)+rho*b)*(H/nt)**2/8
    times = [H*F(j, nt) for j in range(nt+1)]
    grid = [a+(b-a)*F(j, n) for j in range(n+1)]
    states, queries, deleted, retained, ma_records = {}, {}, [], [], []
    nonempty = None
    witness = None

    def over_budget():
        return deadline is not None and time.perf_counter() >= deadline

    def state(theta):
        if theta not in states:
            states[theta] = cc.endpoint(theta, case['initial'], case['past'])
        return states[theta]

    def query(theta):
        if theta not in queries:
            queries[theta] = cc.timeline(theta, state(theta), case['future'], times, rho)
        return queries[theta]

    def feasible(theta):
        st = state(theta)
        return all(case['packet'][k][0] <= st[k].lower() and
                   st[k].upper() <= case['packet'][k][1] for k in range(3))

    def check_witness(theta):
        nonlocal nonempty, witness
        if feasible(theta):
            if nonempty is None:
                nonempty = dict(theta=theta, historical_state=[(v.lower(), v.upper()) for v in state(theta)])
            if witness is None:
                for s, y in zip(times, query(theta)):
                    if y[0].lower() > case['limit']:
                        witness = dict(theta=theta, time=s, position_lower=y[0].lower())
                        break

    completed_cells = 0
    budget_hit = False
    for i in range(n):
        if over_budget():
            budget_hit = True
            break
        l, r = grid[i], grid[i+1]
        rejection = None
        if structured:
            cert = pc.reject_cell(poly, l, r) if use_sturm else None
            if cert is not None:
                rejection = dict(index=i, engine='sturm', certificate=cert)
            elif use_sparse:
                cert = tf.reject_cell(sparse, l, r, mode=='structured')
                if cert is not None:
                    rejection = dict(index=i, engine='sparse', certificate=cert)
        if rejection:
            deleted.append(rejection)
            completed_cells += 1
            continue
        if is_zhao:
            domain,cert = zhao.contract(mm,case['packet'],l,r,
                                       nu=1 if method.startswith('zhao1') else 2,
                                       tighten=method.startswith('zhao2t3'),
                                       hessian_pieces=hessian_pieces)
        else:
            domain, cert = ma.contract(mm, case['packet'], l, r, strengthened, hessian_pieces)
        ma_records.append(dict(index=i, certificate=cert))
        if domain is None:
            deleted.append(dict(index=i, engine='zhao' if is_zhao else 'ma', certificate_index=len(ma_records)-1))
            completed_cells += 1
            continue
        l, r = domain
        if l == r:
            cell = dict(l=l, r=r, gamma=(l, r), radii=[F(0)]*3)
            node = max(v[0].upper() for v in query(l))
        else:
            cell = cc.make_cell(l, r, state(l), state(r), case['packet'], bounds['M'])
            if cell is None:
                rec = dict(index=i, engine='common_chord', theta_interval=[l, r])
                if collect:
                    rec['endpoint_state_bounds'] = [[(v.lower(), v.upper()) for v in state(t)] for t in (l, r)]
                deleted.append(rec)
                completed_cells += 1
                continue
            ql, qr = query(l), query(r)
            node = max(cc.chord_upper(cell, ql[j][0], qr[j][0], s, bounds, U, rho)
                       for j, s in enumerate(times))
        rec = dict(index=i, interval=[l, r], projection=cell['gamma'],
                   radii=cell['radii'], max_node_upper=node)
        if collect:
            rec['endpoint_state_bounds'] = [[(v.lower(), v.upper()) for v in state(t)] for t in (l, r)]
        retained.append(rec)
        # Same witness policy for all methods; no hidden generating parameter.
        for t in (l, r):
            check_witness(t)
        if nonempty is None or (witness is None and node+allowance > case['limit']):
            check_witness((cell['gamma'][0]+cell['gamma'][1])/2)
        completed_cells += 1
        if witness is not None:
            # A real counterexample is sufficient; no remaining cells can undo it.
            break

    complete = completed_cells == n
    upper = max((v['max_node_upper'] for v in retained), default=None)
    if upper is not None:
        upper += allowance
    if witness is not None:
        status = 'CERTIFIED_UNSAFE'
        upper = upper if complete else None
    elif not complete:
        status, upper = 'UNKNOWN', None
    elif not retained:
        status = 'INCONSISTENT'
    else:
        status = 'CERTIFIED_SAFE' if upper <= case['limit'] else 'UNKNOWN'
    result = dict(method=method, status=status, n=n, nt=nt, upper=upper,
                  time_allowance=allowance, witness=witness, nonempty_witness=nonempty,
                  hessian_pieces=hessian_pieces, completed_cells=completed_cells,
                  budget_hit=budget_hit, grid_complete=complete,
                  historical_vertices=len(states), future_vertices=len(queries),
                  retained=len(retained), deleted=len(deleted), frontend_calls=len(ma_records),
                  sturm_status=poly['status'] if poly else None)
    if collect:
        result.update(cells=retained, deletions=deleted, frontend_certificates=ma_records,
                      sparse_model=sparse, sturm_model=poly)
    return result
