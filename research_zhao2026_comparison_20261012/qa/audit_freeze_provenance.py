"""Read-only checks of recorded freeze, input, runtime-setting provenance."""
from collections import Counter
from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path
import argparse
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PUBLIC_FIELDS={'name','group','initial','past','prior','packet','future','H','rho','limit'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def local_import_closure(start):
    found=set();pending=[start]
    while pending:
        name=pending.pop()
        if name in found:
            continue
        found.add(name)
        source=(ROOT/'implementation'/(name+'.py')).read_text(encoding='utf-8-sig')
        assert 'holdout_generation_private' not in source and 'hidden_theta' not in source
        for node in ast.walk(ast.parse(source)):
            names=([entry.name.split('.')[0] for entry in node.names] if isinstance(node,ast.Import) else
                   [(node.module or '').split('.')[0]] if isinstance(node,ast.ImportFrom) else [])
            pending.extend(item for item in names if (ROOT/'implementation'/(item+'.py')).exists())
    assert 'generate_new_holdout' not in found
    return sorted(found)


def normalize(value):
    if isinstance(value,list):
        return tuple(map(normalize,value))
    return Q(value)


def physical_signature(case):
    return tuple((key,normalize(case[key])) for key in sorted(PUBLIC_FIELDS-{'name','group','limit'}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tags',required=True)
    args=parser.parse_args()
    frozen=load(ROOT/'inputs/method_freeze.json')
    analysis_frozen=load(ROOT/'inputs/analysis_freeze.json')
    generated=load(ROOT/'inputs/HOLDOUT_FROZEN.json')
    protocol=load(ROOT/'inputs/generation_protocol.json')
    cases=load(ROOT/'inputs/holdout_48.json')
    assert generated['frozen_methods_sha256']==sha(ROOT/'inputs/method_freeze.json')
    assert analysis_frozen['methods_freeze_sha256']==generated['frozen_methods_sha256']
    assert analysis_frozen['before_first_holdout_solver'] is True
    assert datetime.fromisoformat(analysis_frozen['timestamp_utc'])>=datetime.fromisoformat(frozen['timestamp_utc'])
    assert generated['inputs_sha256']==sha(ROOT/'inputs/holdout_48.json')
    assert generated['private_generation_sha256']==sha(ROOT/'qa/holdout_generation_private.json')
    assert generated['generator_sha256']==protocol['generator_sha256']==sha(ROOT/'implementation/generate_new_holdout.py')
    assert generated['protocol_sha256']==protocol['full_protocol_sha256']==sha(ROOT/'analysis/NEW_HOLDOUT_PROTOCOL_CN.txt')
    assert generated['seed']==protocol['seed']==202610120947
    assert generated['count']==protocol['count']==len(cases)==48
    assert len({case['name'] for case in cases})==48
    assert dict(Counter(case['group'] for case in cases))==protocol['groups']
    for case in cases:
        assert set(case)==PUBLIC_FIELDS
        assert len(case['initial'])==len(case['packet'])==3
        assert all(Q(dt)>0 for dt,_ in case['past']+case['future'])
        assert sum(Q(dt) for dt,_ in case['future'])==Q(case['H'])
        assert 0<Q(case['prior'][0])<Q(case['prior'][1]) and Q(case['rho'])>=0
        assert all(Q(low)<Q(high) for low,high in case['packet'])
    old_signatures=set()
    for filename in frozen['development_data']:
        old_signatures.update(map(physical_signature,load(ROOT/'inputs'/filename)))
    assert not old_signatures.intersection(map(physical_signature,cases))
    for filename,want in frozen['file_sha256'].items():
        assert sha(ROOT/filename)==want,('frozen file changed',filename)
    for filename,want in analysis_frozen['files'].items():
        assert sha(ROOT/filename)==want,('frozen analysis file changed',filename)
    closure=local_import_closure('run_comparison')
    runs={}
    for tag in args.tags.split(','):
        metadata=load(ROOT/'results'/tag/'run_metadata.json');settings=metadata['arguments']
        methods=settings['methods'].split(',')
        if methods==frozen['main_methods']:
            repetitions=frozen['main_repetitions']
        else:
            assert methods==frozen['ablation_methods']
            repetitions=frozen['ablation_repetitions']
        assert settings['inputs']=='holdout_48.json' and settings['tag']==tag
        assert metadata['input_sha256']==generated['inputs_sha256']
        assert datetime.strptime(metadata['timestamp'],'%Y-%m-%d %H:%M:%S %z')>datetime.fromisoformat(frozen['timestamp_utc'])
        assert datetime.strptime(metadata['timestamp'],'%Y-%m-%d %H:%M:%S %z')>datetime.fromisoformat(analysis_frozen['timestamp_utc'])
        assert list(map(int,settings['grids'].split(',')))==frozen['grids']
        assert list(map(int,settings['nt_schedule'].split(',')))==frozen['time_grids']
        assert settings['hessian_pieces']==frozen['hessian_pieces']
        assert settings['budget']==frozen['budget_seconds'] and settings['quick']==frozen['quick_gate']
        assert settings['reps']==repetitions
        assert metadata['numerical_threads']==frozen['numerical_threads']==1
        assert metadata['threadpools'] and all(pool['num_threads']==1 for pool in metadata['threadpools'])
        for filename,want in metadata['implementation_sha256'].items():
            assert frozen['file_sha256']['implementation/'+filename]==want
        audit=load(ROOT/'qa'/('AUDIT_'+tag+'.json'))
        assert audit['status']=='PASS' and audit['records']==len(cases)*len(methods)*repetitions
        assert audit['bindings'][str(Path('inputs')/'holdout_48.json')]==generated['inputs_sha256']
        assert all(audit['bindings']['current_implementation_hash_matches'].values())
        assert audit['bindings'][str(Path('results')/tag/'records.jsonl')]==sha(ROOT/'results'/tag/'records.jsonl')
        runs[tag]=dict(expected_records=len(cases)*len(methods)*repetitions,metadata_sha256=sha(ROOT/'results'/tag/'run_metadata.json'),
                       certificate_audit_sha256=sha(ROOT/'qa'/('AUDIT_'+tag+'.json')),methods=methods,repetitions=repetitions)
    out=dict(status='PASS',input_sha256=generated['inputs_sha256'],
             frozen_files_verified=len(frozen['file_sha256']),public_input_fields=sorted(PUBLIC_FIELDS),
             separately_frozen_analysis_files_verified=len(analysis_frozen['files']),
             solver_local_import_closure=closure,reused_development_physical_cases=0,runs=runs,
             scope='Checks recorded hash binding, exact input schema, no old physical case reuse, static solver import closure, method/grid/budget/thread metadata and certified record counts. Does not independently prove historical chronology or measured wall-clock accuracy.',
             checker_sha256=sha(Path(__file__)))
    (ROOT/'qa/AUDIT_freeze_provenance.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
