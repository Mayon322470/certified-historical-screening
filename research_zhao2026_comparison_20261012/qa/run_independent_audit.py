"""Stream named-baseline certificates through independent rational auditors."""
from collections import Counter, defaultdict
from fractions import Fraction as Q
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import random
import time

import audit_comparison as audit
import audit_zhao_contract as zhao

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def reject(function):
    try:
        function()
    except (AssertionError,ValueError,KeyError,IndexError):
        return True
    return False


def check_quick(case,result):
    packet=[tuple(map(Q,v)) for v in case['packet']]
    H,rho,limit=(Q(case[k]) for k in ('H','rho','limit'))
    U=max(abs(Q(u)) for _,u in case['future'])
    Ap=max(abs(v) for v in packet[2])
    Abar=max(Ap,U)+rho*Q(case['prior'][1])
    upper=packet[0][1]+max(packet[1][1],Q(0))*H+Abar*H*H/2
    theta=sum(map(Q,case['prior']))/2
    for key,value in dict(A_packet=Ap,U=U,A_bar=Abar,H=H,rho=rho,limit=limit,
                          quick_upper=upper,safety_margin=limit-upper,midpoint_theta=theta).items():
        assert Q(result[key])==value
    bounds=[tuple(map(Q,v)) for v in result['midpoint_state_bounds']]
    fixed=[tuple(Q(int(v),2**96) for v in pair) for pair in result['midpoint_state_fixedpoint']]
    assert bounds==fixed
    for actual,outer in zip(audit.endpoint(case,theta),bounds):
        assert audit.ref.contained(actual,outer)
    margins=[(bounds[j][0]-packet[j][0],packet[j][1]-bounds[j][1]) for j in range(3)]
    assert [tuple(map(Q,p)) for p in result['midpoint_packet_margins']]==margins
    inside=all(l>0 and r>0 for l,r in margins)
    passes=upper<=limit
    assert result['midpoint_strictly_inside']==inside
    assert result['bound_proves_universal_safety']==passes
    assert result['status']==('QUICK_CERTIFIED_SAFE' if passes and inside else 'QUICK_NO_CERTIFICATE')


def mutation_checks(examples):
    result={}
    if 'ma' in examples:
        case,stage=examples['ma'];certificate=stage['frontend_certificates'][0]['certificate']
        bad=copy.deepcopy(certificate)
        bad['strips'][0]['remainder'][0]=str(Q(bad['strips'][0]['remainder'][0])+1)
        result['ma_signed_square']=reject(lambda:audit.check_contract(case,bad))
        bad=copy.deepcopy(certificate);bad['lp_radius']=str(Q(bad['lp_radius'])/2-Q(1,100))
        result['ma_LP_optimum']=reject(lambda:audit.check_contract(case,bad))
    if 'zhao' in examples:
        case,stage=examples['zhao'];certificate=stage['frontend_certificates'][0]['certificate']
        bad=copy.deepcopy(certificate)
        pair=bad['degree_remainders'][1]['derivative_component_intervals'][0]
        pair[0]=str(Q(pair[0])+10**6);pair[1]=str(Q(pair[1])+10**6)
        result['zhao_whole_cell_derivative']=reject(lambda:zhao.check_contract(case,bad))
        bad=copy.deepcopy(certificate);bad['initial_lift_center_active'][2]=str(Q(bad['initial_lift_center_active'][2])+1)
        result['zhao_squared_initial_center']=reject(lambda:zhao.check_contract(case,bad))
        bad=copy.deepcopy(certificate);bad['posterior_center'][1]=str(Q(bad['posterior_center'][1])+1)
        result['zhao_gain_center_reconstruction']=reject(lambda:zhao.check_contract(case,bad))
        bad=copy.deepcopy(certificate);bad['theorem3']['kappa']=str(Q(bad['theorem3']['kappa'])+1)
        result['zhao_kappa']=reject(lambda:zhao.check_contract(case,bad))
        bad=copy.deepcopy(certificate);bad['initial_active_indices'][1:]=list(reversed(bad['initial_active_indices'][1:]))
        result['zhao_ordered_lift_indices']=reject(lambda:zhao.check_contract(case,bad))
        bad=copy.deepcopy(certificate);bad['q_intersection'][0]=str(Q(bad['q_intersection'][0])+1)
        result['zhao_q_projection']=reject(lambda:zhao.check_contract(case,bad))
    if 'geometry' in examples:
        case,stage=examples['geometry'];cell=stage['cells'][0]
        bad=copy.deepcopy(cell);bad['max_node_upper']=str(Q(bad['max_node_upper'])-10**6)
        result['lowered_future_upper']=reject(lambda:audit.check_geometry(case,bad,nt=stage['nt']))
        bad=copy.deepcopy(stage);bad['cells'].pop();bad['retained']-=1
        result['omitted_retained_cell']=reject(lambda:audit.check_result(case,bad))
    assert all(result.values())
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    args=parser.parse_args();started=time.perf_counter()
    folder=ROOT/'results'/args.tag
    metadata=json.loads((folder/'run_metadata.json').read_text())
    settings=metadata['arguments'];inputpath=ROOT/'inputs'/settings['inputs']
    cases=json.loads(inputpath.read_text(encoding='utf-8-sig'))
    assert digest(inputpath)==metadata['input_sha256']
    assert (folder/'summary.json').exists(),'Timing producer has not finished'
    summary=json.loads((folder/'summary.json').read_text())
    methods=settings['methods'].split(',');grids=list(map(int,settings['grids'].split(',')))
    schedule=list(map(int,settings['nt_schedule'].split(','))) if settings.get('nt_schedule') else [settings['nt']]*len(grids)
    assert len(schedule)==len(grids)
    expected=[(rep,index,method) for rep in range(settings['reps']) for index in range(len(cases)) for method in methods]
    random.Random(712913).shuffle(expected)
    counts=Counter();unique=Counter();decisions=Counter();stage_cache={};duplicates=0;detailed=0
    method_status=defaultdict(Counter);method_elapsed=Counter();method_resolved=Counter()
    examples={};seen=0;errors=[]
    with (folder/'records.jsonl').open(encoding='utf-8') as records:
        for index,line in enumerate(records):
            assert index<len(expected)
            row=json.loads(line);seen+=1
            assert (row['rep'],row['case_index'],row['method'])==expected[index]
            case=cases[row['case_index']];result=row['result']
            assert row['name']==case['name'] and row['group']==case.get('group')
            assert row['method']==result['method']
            assert row['elapsed']>=0 and row['budget_seconds']==settings['budget']
            overshoot=max(0,row['elapsed']-settings['budget']) if settings['budget'] else 0
            assert math.isclose(row['overshoot_seconds'],overshoot,rel_tol=1e-12,abs_tol=1e-15)
            if result['status']=='ERROR':
                assert result=={'method':row['method'],'status':'ERROR'}
                assert not row['stages'] and row['quick'] is None
                assert row.get('error_traceback')
                assert row['resolved'] is False and row['resolved_within_budget'] is False
                assert row['future_vertices_total']==row['frontend_calls_total']==0
                errors.append(dict(name=row['name'],method=row['method'],rep=row['rep']))
            else:
                if settings['quick']:
                    check_quick(case,row['quick']);counts['quick_gates']+=1
                else:
                    assert row['quick'] is None
                if row['quick'] and row['quick']['status']=='QUICK_CERTIFIED_SAFE':
                    assert not row['stages'] and result['quick_resolved']
                    assert result['status']=='CERTIFIED_SAFE'
                    assert Q(result['upper'])==Q(row['quick']['quick_upper'])
                    assert result['nonempty_witness']=={'theta':row['quick']['midpoint_theta']}
                    for key,want in dict(historical_vertices=1,future_vertices=0,retained=0,deleted=0,
                                         frontend_calls=0,completed_cells=0,grid_complete=True,budget_hit=False).items():
                        assert result[key]==want
                else:
                    assert 1<=len(row['stages'])<=len(grids)
                    assert result==row['stages'][-1]
                    for level,stage in enumerate(row['stages']):
                        assert (stage['n'],stage['nt'],stage['hessian_pieces'])==(grids[level],schedule[level],settings['hessian_pieces'])
                        assert stage['method']==row['method'] and 'frontend_certificates' in stage
                        fingerprint=hashlib.sha256(json.dumps(stage,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                        key=row['case_index'],fingerprint
                        if key in stage_cache:
                            verified=stage_cache[key];duplicates+=1
                        else:
                            try:
                                verified=audit.check_result(case,stage)
                            except Exception as exc:
                                raise RuntimeError(f'{args.tag} row={index} name={case["name"]} method={row["method"]} rep={row["rep"]} stage={level}') from exc
                            stage_cache[key]=verified;unique.update(verified)
                        counts.update(verified);detailed+=1
                        if stage['frontend_certificates']:
                            family='zhao' if row['method'].startswith('zhao2t3') else 'ma' if row['method'].startswith('ma') else None
                            if family and family not in examples:
                                examples[family]=(case,stage)
                        if stage['cells'] and 'geometry' not in examples:
                            examples['geometry']=(case,stage)
                        if level<len(row['stages'])-1:
                            assert stage['status']=='UNKNOWN' or (stage['status']=='CERTIFIED_SAFE' and stage['nonempty_witness'] is None)
                resolved=(result['status'] in ('CERTIFIED_UNSAFE','INCONSISTENT') or
                          result['status']=='CERTIFIED_SAFE' and result.get('nonempty_witness') is not None)
                assert row['resolved']==resolved
                assert row['resolved_within_budget']==(resolved and (not settings['budget'] or row['elapsed']<=settings['budget']))
                assert row['future_vertices_total']==sum(s['future_vertices'] for s in row['stages'])
                assert row['frontend_calls_total']==sum(s['frontend_calls'] for s in row['stages'])
            decisions[result['status']]+=1;method_status[row['method']][result['status']]+=1
            method_elapsed[row['method']]+=row['elapsed'];method_resolved[row['method']]+=row['resolved']
            if seen%10==0 or seen==len(expected):
                print(f'audited {args.tag} {seen}/{len(expected)}',flush=True)
    assert seen==len(expected)
    for method in methods:
        assert summary[method]['statuses']==dict(method_status[method])
        assert summary[method]['resolved']==method_resolved[method]
        assert math.isclose(summary[method]['elapsed_total'],method_elapsed[method],rel_tol=1e-12)
    contract_cache_counts=dict(zhao.CACHE_COUNTS)
    mutations=mutation_checks(examples)
    bindings={str(path.relative_to(ROOT)):digest(path) for path in
              (inputpath,folder/'records.jsonl',folder/'run_metadata.json',folder/'summary.json')}
    bindings['current_implementation_hash_matches']={name:digest(ROOT/'implementation'/name)==want
                                                   for name,want in metadata['implementation_sha256'].items()}
    out=dict(status='PASS',tag=args.tag,records=seen,detailed_stage_records=detailed,decisions=dict(decisions),
             checked=dict(counts),unique_computations_checked=dict(unique),duplicate_full_stage_payloads_checked_by_hash=duplicates,
             zhao_contract_cache=contract_cache_counts,
             mutations_rejected=mutations,recorded_errors=errors,bindings=bindings,elapsed_seconds=time.perf_counter()-started,
             independent_reference='command-jump Laurent-exponential model; ordered symbolic Kronecker products and derivatives; rational 220-bit exponential; adaptive closed-interval derivative coverage; exact rational reconstruction of all gain centers/generators; future flow outward 208 bits',
             scope='Every recorded frontend contraction, structural deletion, common-backend geometry, future certificate, witness, stage schedule and summary. Fixed finite binary gains are certified for containment; their numerical optimality and wall-clock accuracy are not asserted.',
             trust_boundary='Uses SHA256-bound prior independent structural auditors and reused curvature inequalities evaluated through formulas; not a formal machine proof or a universal superiority claim.',
             checker_source_sha256={path.name:digest(path) for path in Path(__file__).parent.glob('*.py')})
    path=ROOT/'qa'/('AUDIT_'+args.tag+'.json')
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    main()
