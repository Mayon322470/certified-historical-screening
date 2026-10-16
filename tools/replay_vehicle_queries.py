"""Portable entry point to the unchanged archived identical-query replay.

It redirects only the output directory in memory. No historical source, input,
freeze, timing, or simulator configuration is overwritten. No simulator starts.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from repository_data import ROOT,PROVENANCE,read_json
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
C='vehicle_estimator_closed_loop_20261013'
needed=[C+'/source/sensor_estimators.py',C+'/analysis/analyze_and_replay.py','vehicle_position_contract_20261013/source/position_backup.py']
needed += [e['path'] for e in PROVENANCE['entries'] if e['path'].startswith('research_task_face_budget_20261013/implementation/') and e['path'].endswith('.py')]
index={e['path']:e for e in PROVENANCE['entries']}
for rel in needed:assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==index[rel]['source_sha256'],rel
source=ROOT/C/'analysis/analyze_and_replay.py';spec=importlib.util.spec_from_file_location('public_original_vehicle_replay',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=read_json(C+'/inputs/COSIM_ESTIMATOR_PLAN_FROZEN.json');queries,_=module.select_replay_queries(plan);assert len(queries)==6
original=read_json(C+'/analysis/paired_replay/QUERIES.json')['queries'];assert queries==original,'Selection differs from archived six queries'
# Import the original estimation module without making any estimator call.
module.load('public_replay_import_smoke',ROOT/C/'source/sensor_estimators.py')
if a.check_only:
    print(json.dumps(dict(status='PASS',unchanged_source_files=len(needed),queries=6,simulator_started=False,estimator_called=False)));raise SystemExit
if a.output is None:p.error('--output NEW_DIRECTORY is required unless --check-only is used')
a.output.mkdir(parents=True,exist_ok=False);module.ANALYSIS=a.output.resolve();module.run_replay(plan)
