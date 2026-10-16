"""Copy immutable model sources/inputs to a fresh result-free workspace."""
import argparse,shutil
from pathlib import Path
from repository_data import ROOT
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
names=('research_zhao2026_comparison_20261012','research_task_face_budget_20261013','research_efficiency_20261012')
for name in names:
    src=ROOT/name;dst=a.output/name
    for item in src.iterdir():
        if item.name=='results' or item.name=='__pycache__':continue
        target=dst/item.name
        if item.is_dir():shutil.copytree(item,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item,target)
print('Prepared frozen model source and inputs in',a.output.resolve())
