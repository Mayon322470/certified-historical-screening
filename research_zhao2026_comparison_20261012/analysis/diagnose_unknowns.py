"""Supplementary diagnostic: budget exhaustion versus exhausted grid schedule.

Does not change the preregistered method, solved-rate, or speed-ratio analysis.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args()
    folder=ROOT/'results'/a.tag
    meta=json.loads((folder/'run_metadata.json').read_text())
    grids=list(map(int,meta['arguments']['grids'].split(',')))
    counts=defaultdict(Counter);details=[]
    with (folder/'records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);result=row['result']
            if row['resolved']:continue
            status=result['status']
            if status=='ERROR':reason='error'
            elif status=='CERTIFIED_SAFE':reason='safe_without_nonempty_witness'
            elif row['budget_seconds'] and row['elapsed']>=row['budget_seconds']:
                reason='unknown_budget_reached'
            elif result.get('grid_complete') and result.get('n')==grids[-1]:
                reason='unknown_schedule_exhausted_before_budget'
            else:reason='unknown_other'
            counts[row['method']][reason]+=1
            details.append(dict(case_index=row['case_index'],name=row['name'],method=row['method'],
                rep=row['rep'],reason=reason,elapsed=row['elapsed'],n=result.get('n'),nt=result.get('nt'),
                grid_complete=result.get('grid_complete'),completed_cells=result.get('completed_cells'),
                nonempty_witness_found=result.get('nonempty_witness') is not None))
    data=dict(tag=a.tag,scope='Supplementary unresolved-trial diagnostics; no cases excluded from primary analysis',
        counts_by_method={k:dict(v) for k,v in counts.items()},details=details)
    (folder/'unknown_diagnostics.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data['counts_by_method'],indent=2))

if __name__=='__main__':main()
