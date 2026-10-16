"""Finite integration checks on four predeclared development fixtures."""
from pathlib import Path
import json
import time
from fractions import Fraction as F
import budget_comparison as bc
from quick_baseline import encode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'protocol/development/ADAPTER_SMOKE.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve integration development check')
    cases = json.loads((ROOT / 'protocol/development/public_fixtures.json').read_text(encoding='utf-8'))
    records = []
    for i, case in enumerate(cases):
        for method in bc.METHODS:
            start = time.perf_counter()
            result = bc.certify(case, 8, 16, method, True)
            record = dict(case_index=i, method=method, result=result, wall_s=time.perf_counter()-start)
            records.append(record)
            parsed = bc.parse_case(case)
            for key in ('witness', 'nonempty_witness'):
                w = result.get(key)
                if w:
                    nominal = bc.cc.endpoint(w['theta'], parsed['initial'], parsed['past'])
                    T = sum((dt for dt, _ in parsed['past']), F(0))
                    psi = bc.response.profile(parsed, T)
                    assert abs(w['beta']) <= F(parsed['response']['epsilon_x'])
                    assert all(lo <= value.lower()+w['beta']*p <= value.upper()+w['beta']*p <= hi
                               for value, p, (lo, hi) in zip(nominal, psi, parsed['packet']))
            if result['status'] == 'CERTIFIED_SAFE':
                assert result['nonempty_witness'] is not None and result['grid_complete']
                assert result['upper'] <= parsed['limit'] and result['witness'] is None
            if result['status'] == 'CERTIFIED_UNSAFE':
                assert result['witness']['position_lower'] > parsed['limit']
            print(json.dumps(dict(case=i, method=method, status=result['status'],
                                  front=result['frontend_calls'], deleted=result['deleted'],
                                  wall_s=record['wall_s'])), flush=True)
    for i in range(len(cases)):
        statuses = {row['result']['status'] for row in records if row['case_index'] == i}
        assert not {'CERTIFIED_SAFE', 'CERTIFIED_UNSAFE'} <= statuses
    OUT.write_text(json.dumps(encode(dict(status='PASS', scope='20 integration runs on known development only; no timing comparison',
                                         records=records)), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
