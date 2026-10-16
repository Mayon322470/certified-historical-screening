"""Independent streaming recomputation of fixed-suite descriptive statistics."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics as st
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=20261012947
POLICIES=('public_envelope_plus_5percent_scale','compatible_reference_plus_1e-2_scale',
          'compatible_reference_plus_1e-4_scale','compatible_reference_minus_1e-3_scale')
CHECKS=Counter()
ALL_RUN_DECISIONS=defaultdict(set)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def same(actual,expected,label=''):
    CHECKS['scalar_or_structure_comparisons']+=1
    if isinstance(expected,float):
        assert actual is not None and math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-12),(label,actual,expected)
    else:
        assert actual==expected,(label,actual,expected)


def ci(values,functions,context,enabled):
    """One full case-index draw rather than producer's 500-row chunk loop."""
    if not enabled or not values:
        return {key:None for key in functions}
    seed_context=int.from_bytes(hashlib.sha256(context.encode()).digest()[:8],'little')
    random=np.random.default_rng(np.random.SeedSequence([SEED,seed_context]))
    indices=random.integers(0,len(values),size=(10000,len(values)))
    sample=np.array(values,dtype=float)[indices]
    result={}
    for key,function in functions.items():
        result[key]=list(map(float,np.quantile(function(sample),[0.025,0.975])))
        CHECKS['bootstrap_intervals_recomputed']+=1
    return result


def same_ci(actual,expected):
    if expected is None:
        same(actual,None)
    else:
        assert len(actual)==2
        for a,b in zip(actual,expected):
            same(a,b,'bootstrap endpoint')


def method_stats(claim,values,context,enabled):
    n=len(values);solved=sum(v['within_budget_all'] for v in values)
    expectations=dict(cases=n,statuses_rep0=dict(Counter(v['status'] for v in values)),
        resolved_all=sum(v['resolved_all'] for v in values),within_budget_all=solved,
        within_budget_rate=solved/n if n else None,sum_case_median_seconds=sum(v['median_seconds'] for v in values),
        quick_only=sum(v['quick_only'] for v in values),unstable_status_cases=[v['name'] for v in values if not v['status_stable']])
    for key,want in expectations.items():
        same(claim[key],want,key)
    intervals=ci([[int(v['within_budget_all']),v['median_seconds']] for v in values],
                 {'rate':lambda a:np.sum(a[:,:,0],axis=1)/n,'cost':lambda a:np.sum(a[:,:,1],axis=1)},context,enabled)
    same_ci(claim['within_budget_rate_ci95'],intervals['rate'])
    same_ci(claim['sum_case_median_seconds_ci95'],intervals['cost'])


def reasons(a,b):
    exclusions=[]
    if not a['within_budget_all'] or not b['within_budget_all']:
        exclusions.append('not_both_all_repeats_resolved_within_budget')
    if not a['status_stable'] or not b['status_stable']:
        exclusions.append('unstable_repeat_status')
    if a['status']!=b['status']:
        exclusions.append('different_decision')
    if a['case_has_conflicting_certificates'] or b['case_has_conflicting_certificates']:
        exclusions.append('case_has_conflicting_certificates')
    if a['quick_only'] and b['quick_only']:
        exclusions.append('common_quick')
    return exclusions


def paired_stats(claim,pairs,context,enabled,details=False):
    eligible=[(a,b) for a,b in pairs if not reasons(a,b)]
    ratios=[a['median_seconds']/b['median_seconds'] for a,b in eligible]
    slower=[a['name'] for a,b in eligible if b['median_seconds']>1.05*a['median_seconds']]
    ac=sum(a['median_seconds'] for a,b in pairs);bc=sum(b['median_seconds'] for a,b in pairs)
    expectations=dict(cases=len(pairs),baseline_within_budget_all=sum(a['within_budget_all'] for a,b in pairs),
        candidate_within_budget_all=sum(b['within_budget_all'] for a,b in pairs),
        added_solved=[a['name'] for a,b in pairs if not a['within_budget_all'] and b['within_budget_all']],
        lost_solved=[a['name'] for a,b in pairs if a['within_budget_all'] and not b['within_budget_all']],
        contradictory_resolved=[a['name'] for a,b in pairs if len({r['status'] for v in (a,b) for r in v['repeats'] if r['resolved']})>1],
        baseline_sum_case_median_seconds=ac,candidate_sum_case_median_seconds=bc,
        all_task_cost_ratio=ac/bc if pairs else None,
        within_budget_rate_difference=sum(int(b['within_budget_all'])-int(a['within_budget_all']) for a,b in pairs)/len(pairs) if pairs else None,
        common_solved_nonquick_cases=len(eligible),
        common_solved_geomean_speed_ratio=math.prod(ratios)**(1/len(ratios)) if ratios else None,
        common_solved_median_speed_ratio=st.median(ratios) if ratios else None,
        common_solved_min_speed_ratio=min(ratios) if ratios else None,
        common_solved_max_speed_ratio=max(ratios) if ratios else None,
        slower_by_over_5pct=slower,slower_by_over_5pct_count=len(slower),
        slower_by_over_5pct_fraction=len(slower)/len(eligible) if eligible else None,
        exclusion_reason_counts_nonexclusive=dict(Counter(r for a,b in pairs for r in reasons(a,b))),
        common_quick_cases=sum(a['quick_only'] and b['quick_only'] for a,b in pairs),
        asymmetric_quick_cases=[a['name'] for a,b in pairs if a['quick_only']!=b['quick_only']])
    for key,want in expectations.items():
        same(claim[key],want,key)
    all_intervals=ci([[a['median_seconds'],b['median_seconds'],int(a['within_budget_all']),int(b['within_budget_all'])] for a,b in pairs],
        {'cost':lambda a:np.sum(a[:,:,0],axis=1)/np.sum(a[:,:,1],axis=1),
         'difference':lambda a:np.sum(a[:,:,3]-a[:,:,2],axis=1)/len(pairs)},context+':all',enabled)
    speed_intervals=ci(ratios,{'geo':lambda a:np.exp(np.sum(np.log(a),axis=1)/len(ratios)),
        'median':lambda a:np.median(a,axis=1),'slow':lambda a:np.sum(a<1/1.05,axis=1)/len(ratios)},context+':eligible',enabled)
    for field,want in (('all_task_cost_ratio_ci95',all_intervals['cost']),
                       ('within_budget_rate_difference_ci95',all_intervals['difference']),
                       ('common_solved_geomean_speed_ratio_ci95',speed_intervals['geo']),
                       ('common_solved_median_speed_ratio_ci95',speed_intervals['median']),
                       ('slower_by_over_5pct_fraction_ci95',speed_intervals['slow'])):
        same_ci(claim[field],want)
    if details:
        same(claim['common_solved_details'],[dict(case_index=a['case_index'],name=a['name'],group=a['group'],
            family=a['family'],threshold_policy=a['threshold_policy'],baseline_s=a['median_seconds'],candidate_s=b['median_seconds'],
            speed_ratio=a['median_seconds']/b['median_seconds'],candidate_slower_by_over_5pct=b['median_seconds']>1.05*a['median_seconds'])
            for a,b in eligible])
        same(claim['excluded_case_details'],[dict(case_index=a['case_index'],name=a['name'],reasons=reasons(a,b),
            baseline_statuses=a['statuses'],candidate_statuses=b['statuses']) for a,b in pairs if reasons(a,b)])


def analyze(tag):
    folder=ROOT/'results'/tag;meta=read(folder/'run_metadata.json');settings=meta['arguments']
    cases=read(ROOT/'inputs'/settings['inputs']);methods=settings['methods'].split(',');repetitions=settings['reps']
    assert settings['inputs']=='holdout_48.json' and len(cases)==48
    reduced={};decisions=defaultdict(set);digest=hashlib.sha256()
    with (folder/'records.jsonl').open('rb') as source:
        for line_number,line in enumerate(source,1):
            digest.update(line);row=json.loads(line);i=row['case_index'];method=row['method'];rep=row['rep'];result=row['result']
            key=i,method,rep
            assert key not in reduced and 0<=i<48 and method in methods and 0<=rep<repetitions
            rotation=0 if i<32 else 2*((i-32)//4) if i<40 else 1 if i<44 else 3
            resolved=result['status'] in ('CERTIFIED_UNSAFE','INCONSISTENT') or result['status']=='CERTIFIED_SAFE' and result.get('nonempty_witness') is not None
            within=bool(resolved and row['elapsed']<=settings['budget'])
            same(row['resolved'],resolved);same(row['resolved_within_budget'],within)
            same(row['overshoot_seconds'],max(0.,row['elapsed']-settings['budget']))
            engines=Counter()
            if rep==0:
                for stage in row['stages']:
                    for item in stage['deletions']:
                        label=item['engine']+(':'+item['certificate']['kind'] if item['engine']=='sparse' else '')
                        engines[label]+=1
            group=cases[i]['group']
            reduced[key]=dict(case_index=i,name=cases[i]['name'],method=method,rep=rep,line_number=line_number,
                group=group,family='ordinary' if i<32 else 'structured',threshold_policy=POLICIES[(i+rotation)%4],
                status=result['status'],elapsed=row['elapsed'],resolved=resolved,resolved_within_budget=within,
                overshoot_seconds=row['overshoot_seconds'],quick_resolved=bool(result.get('quick_resolved',False)),
                final_n=result.get('n'),future_vertices=row['future_vertices_total'],frontend_calls=row['frontend_calls_total'],
                deletion_engines=dict(engines),error_traceback=row.get('error_traceback'))
            if resolved:
                decisions[i].add(result['status'])
                ALL_RUN_DECISIONS[i].add(result['status'])
    same(set(reduced),{(i,m,r) for i in range(48) for m in methods for r in range(repetitions)})
    data={}
    for i,method in sorted({(i,m) for i,m,r in reduced}):
        reps=[reduced[i,method,r] for r in range(repetitions)];first=reps[0]
        statuses=[r['status'] for r in reps]
        data[i,method]=dict(case_index=i,name=first['name'],method=method,group=first['group'],family=first['family'],
            threshold_policy=first['threshold_policy'],status=first['status'],statuses=statuses,status_stable=len(set(statuses))==1,
            resolved_all=all(r['resolved'] for r in reps),within_budget_all=all(r['resolved_within_budget'] for r in reps),
            median_seconds=st.median(r['elapsed'] for r in reps),min_seconds=min(r['elapsed'] for r in reps),max_seconds=max(r['elapsed'] for r in reps),
            quick_only=all(r['quick_resolved'] for r in reps),quick_stable=len({r['quick_resolved'] for r in reps})==1,
            final_n=first['final_n'],future_vertices=first['future_vertices'],frontend_calls=first['frontend_calls'],repeats=reps,
            case_has_conflicting_certificates=len(decisions[i])>1)
    same(read(folder/'per_case_summary.json'),list(data.values()),'all per-case and per-repeat values')
    report=read(folder/'paired_summary.json');enabled=repetitions==3
    same(report['run_metadata'],meta)
    same(report['analysis_source_sha256'],hashlib.sha256((ROOT/'analysis/analyze_comparison.py').read_bytes()).hexdigest())
    same(report['trials'],len(reduced));same(report['cases'],48);same(report['repeats_per_case_method'],repetitions)
    same(report['integrity']['records_sha256'],digest.hexdigest())
    same(report['bootstrap']['enabled'],enabled);same(report['bootstrap']['seed'],SEED);same(report['bootstrap']['samples'],10000)
    same(report['late_certificates'],[r for r in reduced.values() if r['resolved'] and not r['resolved_within_budget']])
    same(report['errors'],[r for r in reduced.values() if r['status']=='ERROR' or r['error_traceback']])
    same(report['unresolved_safe_without_nonempty_witness'],[r for r in reduced.values() if r['status']=='CERTIFIED_SAFE' and not r['resolved']])
    same(report['status_variation'],[dict(case_index=v['case_index'],name=v['name'],method=v['method'],statuses=v['statuses']) for v in data.values() if not v['status_stable']])
    same(report['quick_status_variation'],[dict(name=v['name'],method=v['method']) for v in data.values() if not v['quick_stable']])
    same({v['case_index'] for v in report['conflicting_certificates']},{i for i,statuses in decisions.items() if len(statuses)>1})
    for method in methods:
        claim=report['methods'][method];vals=[v for (i,m),v in data.items() if m==method];trials=[v for (i,m,r),v in reduced.items() if m==method]
        method_stats(claim,vals,'method:'+method,enabled)
        engines=Counter()
        for trial in trials:
            engines.update(trial['deletion_engines'])
        for key,want in dict(statuses_all_trials=dict(Counter(r['status'] for r in trials)),
            overrun_trials=sum(r['overshoot_seconds']>0 for r in trials),max_overshoot_seconds=max(r['overshoot_seconds'] for r in trials),
            late_certificate_trials=sum(r['resolved'] and not r['resolved_within_budget'] for r in trials),
            error_trials=sum(r['status']=='ERROR' for r in trials),deletion_engines_rep0=dict(engines),
            unknown_trial_cost_seconds=sum(r['elapsed'] for r in trials if r['status']=='UNKNOWN')).items():
            same(claim[key],want,key)
        for field,label in (('group','groups'),('family','families'),('threshold_policy','threshold_policies')):
            same(set(claim[label]),{v[field] for v in vals})
            for group in sorted(claim[label]):
                method_stats(claim[label][group],[v for v in vals if v[field]==group],f'method:{method}:{field}:{group}',enabled)
    expected_pairs=[('ma_strip','ma_strip','ma_strip_structured','within_method'),('zhao2t3','zhao2t3','zhao2t3_structured','within_method')]
    expected_pairs += [(z+'__vs__'+m,z,m,'cross_paper') for m in ('ma_strip_structured','ma_strip') for z in ('zhao1','zhao2','zhao2t3')]
    expected_pairs += [(a+'__vs__'+b,a,b,'zhao_order_and_tightening') for a,b in (('zhao1','zhao2'),('zhao1','zhao2t3'),('zhao2','zhao2t3'))]
    expected_pairs += [('zhao2t3_structured__vs__'+m,'zhao2t3_structured',m,'ablation') for m in ('zhao2t3_sturm','zhao2t3_sparse')]
    expected_pairs += [('zhao2t3_sturm__vs__zhao2t3_sparse','zhao2t3_sturm','zhao2t3_sparse','ablation')]
    expected_pairs=[p for p in expected_pairs if p[1] in methods and p[2] in methods]
    same(set(report['paired']),{p[0] for p in expected_pairs})
    for identifier,baseline,candidate,category in expected_pairs:
        claim=report['paired'][identifier]
        same((claim['baseline'],claim['candidate'],claim['category']),(baseline,candidate,category))
        pairs=[(data[i,baseline],data[i,candidate]) for i in range(48)]
        paired_stats(claim,pairs,'pair:'+identifier,enabled,True)
        for field,label in (('group','groups'),('family','families'),('threshold_policy','threshold_policies')):
            same(set(claim[label]),{a[field] for a,b in pairs})
            for group in sorted(claim[label]):
                paired_stats(claim[label][group],[(a,b) for a,b in pairs if a[field]==group],f'pair:{identifier}:{field}:{group}',enabled)
    return dict(records=len(reduced),cases=48,methods=len(methods),pairs=len(expected_pairs),records_sha256=digest.hexdigest(),
                report_sha256=hashlib.sha256((folder/'paired_summary.json').read_bytes()).hexdigest(),
                per_case_sha256=hashlib.sha256((folder/'per_case_summary.json').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tags',required=True);args=parser.parse_args()
    runs={tag:analyze(tag) for tag in args.tags.split(',')}
    conflicts={index:sorted(values) for index,values in ALL_RUN_DECISIONS.items() if len(values)>1}
    assert not conflicts,('Contradictory certified decisions across supplied runs',conflicts)
    result=dict(status='PASS',runs=runs,checks=dict(CHECKS),numpy_version=np.__version__,
        contradictory_certificates_across_supplied_runs=0,
        scope='Independent raw-record reduction, all method/pair/stratified arithmetic, eligibility, late/error retention and 10000-draw fixed-suite bootstrap endpoints; no producer analysis import.',
        limitations='Same documented NumPy RNG family is required to reproduce seeded bootstrap draws; intervals remain descriptive fixed-suite resampling, not population inference.',
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'qa/AUDIT_statistics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
