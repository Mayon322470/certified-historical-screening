"""Read-only check of the strong exact-range comparator on existing replays."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'analysis'/'paired_replay'/'runs.jsonl'
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    lookup={(r['query_id'],r['repetition'],r['method']):r for r in records}
    outcomes=[]
    for method in ('ma','ma_structured','zhao2t3_structured'):
        compared=subset=tighter=0;ratios=[]
        for key,record in lookup.items():
            if key[2]!=method or record['estimate']['status']!='OK':continue
            exact=lookup[(key[0],key[1],'exact_range')]
            if exact['estimate']['status']!='OK':raise AssertionError('Exact range unavailable')
            lo,hi=map(F,record['estimate']['b_interval_rational'])
            elo,ehi=map(F,exact['estimate']['b_interval_rational'])
            compared+=1
            subset+=lo<=elo<=ehi<=hi
            tighter+=(ehi-elo)<(hi-lo)
            ratios.append(float((hi-lo)/(ehi-elo)))
        if not compared==subset==tighter:raise AssertionError('Stronger interval claim not verified for '+method)
        outcomes.append(dict(method=method,qualified_comparisons=compared,
                             exact_interval_subset_count=subset,exact_strictly_narrower_count=tighter,
                             comparator_to_exact_width_ratio_range=[min(ratios),max(ratios)]))
    report=dict(status='PASS_EXACT_RANGE_STRONG_BASELINE_INTERVAL_AUDIT',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=outcomes,
        scope='Exact rational interval endpoints on existing successful same-input replay runs. Does not claim a globally optimal method beyond this quadratic observation class.')
    output=ROOT/'audit'/'EXACT_RANGE_STRENGTH_AUDIT.json'
    if output.exists():raise FileExistsError('Preserve previous audit')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
