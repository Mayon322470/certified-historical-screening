"""Independent analytic feasible-point tests of all five observation adapters.

This imports the new estimator only, with bytecode disabled by the caller.
It never starts a simulator or changes historical files. Finite tests supplement
the source algebra audit; they are not an exhaustive containment proof.
"""
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source'/'sensor_estimators.py'


def evaluate(row, b):
    return (F(row['a'])-F(row['stamp_s'])*b)**2+F(row['c2'])


def independent_bounds(row, lo, hi):
    a, t, c2 = map(F, (row['a'], row['stamp_s'], row['c2']))
    zlo, zhi = sorted((a-t*lo, a-t*hi))
    floor = F(0) if zlo <= 0 <= zhi else min(zlo*zlo, zhi*zhi)
    return floor+c2, max(zlo*zlo, zhi*zhi)+c2


def load_source():
    spec = importlib.util.spec_from_file_location('independent_estimator_subject', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    source = load_source()
    checks, runs, records = 0, 0, []
    prior = (F(-4,5), F(4,5))
    truths = [F(-4,5), F(-2,5), F(-1,10), F(0), F(1,10), F(2,5), F(4,5)]
    # Rows include sign changes and stationary observation vertices, with
    # different historical clocks, avoiding exclusively monotone examples.
    patterns = [
        [(F(1,5),F(1,5),F(1)), (F(1),F(1),F(4)), (F(-1,5),F(2),F(9))],
        [(F(10),F(1,2),F(4)), (F(7),F(3,2),F(9)), (F(3),F(4),F(1))],
    ]
    for pattern_id, pattern in enumerate(patterns):
        for true_b in truths:
            rows = []
            for j, (a, t, c2) in enumerate(pattern):
                center = (a-t*true_b)**2+c2
                width = F(1,50) if pattern_id == 0 else F(1,5)
                rows.append({'a':a, 'stamp_s':t, 'c2':c2,
                             'packet':[center-width, center+width]})
            feasible = [true_b]
            for j in range(161):
                b = prior[0]+(prior[1]-prior[0])*F(j,160)
                if all(F(row['packet'][0]) <= evaluate(row,b) <= F(row['packet'][1]) for row in rows):
                    feasible.append(b)
            for method in source.METHODS:
                result = source.estimate(method,rows,prior=prior)
                assert result['status'] == 'OK', (method,pattern_id,str(true_b),result['status'])
                lower, upper = map(F,result['b_interval_rational'])
                for b in feasible:
                    assert lower <= b <= upper, (method,pattern_id,str(b),str(lower),str(upper))
                    checks += 1
                for cert in result['certificates']:
                    if cert.get('stage') == 'quadratic_history_face':
                        cell_lo, cell_hi = map(F,cert['b_input'])
                        row = rows[cert['row']]
                        bound = independent_bounds(row,cell_lo,cell_hi)
                        assert tuple(map(F,cert['quadratic_range'])) == bound
                        p,q = map(F,row['packet'])
                        assert bound[1] < p or bound[0] > q
                        assert not cell_lo <= true_b <= cell_hi
                        checks += 3
                runs += 1
                records.append({'pattern':pattern_id, 'truth_b':str(true_b), 'method':method,
                                'status':result['status'], 'b_interval':result['b_interval_rational'],
                                'feasible_reference_points':len(feasible),
                                'frontend_calls':result['frontend_calls'], 'deleted':result['deleted']})
    # At zero time, all observations are independent of b and must keep prior.
    rows = [{'a':F(j), 'stamp_s':F(0), 'c2':F(1),
             'packet':[F(j*j+1)-F(1,50),F(j*j+1)+F(1,50)]} for j in range(3)]
    for method in source.METHODS:
        result = source.estimate(method,rows,prior=prior)
        assert result['status'] == 'OK'
        assert tuple(map(F,result['b_interval_rational'])) == prior
        runs += 1
        checks += 2
    after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert before == after, 'Source changed during independent audit'
    report = {'status':'PASS_INDEPENDENT_OBSERVATION_ADAPTER_COMPONENT_AUDIT',
              'checks':checks, 'runs':runs, 'estimator_source_sha256':before,
              'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'Exact rational truth and feasible grid points, independent quadratic deletion range, branch-changing histories and t=0 prior. Finite tests and algebra review; no vehicle or deadline qualification.',
              'records':records}
    output = Path(__file__).with_name('ESTIMATOR_COMPONENT_AUDIT.json')
    if output.exists():
        raise FileExistsError('Preserve previous audit')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','checks','runs','estimator_source_sha256')}))


if __name__ == '__main__':
    main()
