"""Freeze the prespecified experiment before fresh inputs exist."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]

def main():
    dest=ROOT/'inputs'/'method_freeze.json'
    if dest.exists() or (ROOT/'inputs'/'holdout_48.json').exists():
        raise FileExistsError('Refuse to refreeze or freeze after fresh data generation')
    files=list((ROOT/'implementation').glob('*.py'))+[
        ROOT/'inputs'/'generation_protocol.json',
        ROOT/'analysis'/'NEW_HOLDOUT_PROTOCOL_CN.txt',
        ROOT/'analysis'/'FORMAL_COMPARISON_PROTOCOL_CN.txt']
    data=dict(version='zhao2026-formal-v1',timestamp_utc=datetime.now(timezone.utc).isoformat(),
      main_methods=['ma_strip','ma_strip_structured','zhao1','zhao2','zhao2t3','zhao2t3_structured'],
      ablation_methods=['zhao2t3_sturm','zhao2t3_sparse'],
      main_repetitions=3,ablation_repetitions=1,grids=[8,32,128,512],
      time_grids=[32,64,128,256],budget_seconds=5,quick_gate=True,
      hessian_pieces=1,numerical_threads=1,job_order_seed=712913,bootstrap_seed=20261012947,
      development_data=['development_43.json','development_ma48.json'],
      analysis_protocol='analysis/FORMAL_COMPARISON_PROTOCOL_CN.txt',
      file_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(files)})
    dest.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(dest)
    print(hashlib.sha256(dest.read_bytes()).hexdigest())

if __name__=='__main__':main()
