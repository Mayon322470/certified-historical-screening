"""Independent raw-record statistics audit. Execute after all timers finish.

Does not import production analysis. Bootstrap uses multiplicity-weighted order
statistics instead of constructing the producer's repeated sample list.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import csv,hashlib,json,math,random,statistics,sys,time

ROOT=Path(__file__).resolve().parent.parent
METHODS=('ma_rank1','ma_rank1_structured','zhao1_rank1','zhao2t3_rank1','zhao2t3_rank1_structured')
PAIRS=(('ma_rank1','ma_rank1_structured'),('zhao2t3_rank1','zhao2t3_rank1_structured'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def median(values):
    values=sorted(values);n=len(values)
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2


def close(actual,expected):
    if expected is None:assert actual is None or actual==''
    elif isinstance(expected,bool):assert actual==expected or actual==str(expected)
    elif isinstance(expected,int):assert str(actual)==str(expected)
    elif isinstance(expected,float):assert math.isfinite(float(actual)) and math.isclose(float(actual),expected,rel_tol=2e-12,abs_tol=2e-12),(actual,expected)
    else:assert actual==expected,(actual,expected)


def fields(actual,expected):
    for key,value in expected.items():assert key in actual,key;close(actual[key],value)


def group_indices(name,groups):
    if name=='all':return set(range(32))
    constraints=[part.split('=',1) for part in name.split(';')]
    assert all(len(pair)==2 and pair[0] in ('population','epsilon','threshold_policy') for pair in constraints)
    return {i for i,g in enumerate(groups) if all(str(g[key])==value for key,value in constraints)}


def weighted_median(sorted_logs,weights):
    count=sum(weights[t] for _,t in sorted_logs)
    if count==0:return None,0
    targets=((count-1)//2,count//2);answers=[];position=0
    for value,template in sorted_logs:
        width=weights[template]
        for target in targets:
            if position<=target<position+width:answers.append((target,value))
        position+=width
    assert len(answers)==2
    return (answers[0][1]+answers[1][1])/2,count


def percentile(sorted_values,fraction):
    position=F(len(sorted_values)-1)*fraction;low=position.numerator//position.denominator
    high=min(low+1,len(sorted_values)-1);alpha=float(position-low)
    return sorted_values[low]*(1-alpha)+sorted_values[high]*alpha


def bootstrap(rows,draws):
    logs=sorted((r['log_speedup'],r['template_id']) for r in rows)
    values=[];counts=[]
    for draw in draws:
        value,n=weighted_median(logs,draw)
        if n:values.append(value);counts.append(n)
    values.sort()
    return dict(eligible_cases=len(rows),eligible_templates=len({r['template_id'] for r in rows}),
                template_clusters_in_sampling_frame=16,bootstrap_repetitions=10000,
                nonempty_bootstrap_samples=len(values),empty_bootstrap_samples=10000-len(values),
                bootstrap_eligible_cases_min=min(counts) if counts else 0,
                bootstrap_eligible_cases_max=max(counts) if counts else 0,
                paired_log_median_speedup=math.exp(median([r['log_speedup'] for r in rows])) if rows else None,
                ci95_low=math.exp(percentile(values,F(1,40))) if values else None,
                ci95_high=math.exp(percentile(values,F(39,40))) if values else None,
                ci_conditioned_on_nonempty=len(values)!=10000)


def main():
    start=time.perf_counter();formal=ROOT/'results/formal_main'
    assert (formal/'RUN_COMPLETED.json').exists(),'Statistics audit cannot overlap timers'
    analysis=ROOT/'analysis';summary_path=analysis/'SUMMARY_STATISTICS.json'
    summary=json.loads(summary_path.read_text());groups_path=ROOT/'protocol/frozen_suite/evaluation_groups.json'
    groups=json.loads(groups_path.read_text());assert len(groups)==32
    for g in groups:
        g['population']='structural' if g['structured'] else 'ordinary';g['epsilon']=str(F(g['epsilon']))
    records_path=formal/'records.jsonl';runs={};sequence=[]
    work={m:Counter() for m in METHODS};case017=[]
    for line in records_path.open(encoding='utf-8'):
        original=json.loads(line);result=original['result'];i,m,rep=original['case_index'],original['method'],original['rep']
        key=(i,m,rep);assert key not in runs and 0<=i<32 and m in METHODS and rep in (0,1,2)
        elapsed=original['elapsed'];assert isinstance(elapsed,(float,int)) and math.isfinite(elapsed) and elapsed>=0
        status=result['status'];assert status in ('CERTIFIED_SAFE','CERTIFIED_UNSAFE','INCONSISTENT','UNKNOWN','ERROR')
        assert result['method']==m and original['name']==groups[i]['name'] and original['budget_seconds']==5.0
        solved=status in ('CERTIFIED_UNSAFE','INCONSISTENT') or status=='CERTIFIED_SAFE' and result.get('nonempty_witness') is not None
        within=solved and elapsed<=5.0
        assert original['resolved']==solved and original['resolved_within_budget']==within
        assert original['overshoot_seconds']==max(0,elapsed-5)
        row=dict(case_index=i,name=groups[i]['name'],method=m,rep=rep,status=status,valid=True,resolved=solved,
                 within_budget=within,quick_resolved=result.get('quick_resolved',False),elapsed=elapsed,
                 overshoot_seconds=max(0.0,elapsed-5),frontend_calls_total=original['frontend_calls_total'],
                 future_vertices_total=original['future_vertices_total'])
        runs[key]=row;sequence.append((rep,i,m))
        work[m].update(frontend_calls=original['frontend_calls_total'],future_vertices=original['future_vertices_total'],runs=1)
        removed=Counter(d['engine'] for stage in original['stages'] for d in stage['deletions'])
        for engine,count in removed.items():work[m]['delete_'+engine]+=count
        if original['name']=='case_017':
            case017.append(dict(method=m,rep=rep,status=status,elapsed=elapsed,
                frontend=original['frontend_calls_total'],vertices=original['future_vertices_total'],
                deletes=dict(removed),levels=[s['n'] for s in original['stages']]))
    jobs=[(rep,i,m) for rep in range(3) for i in range(32) for m in METHODS];random.Random(712913).shuffle(jobs)
    assert sequence==jobs and len(runs)==480
    cases={}
    for i,g in enumerate(groups):
        for m in METHODS:
            batch=[runs[i,m,r] for r in (0,1,2)];statuses={r['status'] for r in batch}
            success=all(r['within_budget'] for r in batch) and len(statuses)==1
            cases[i,m]=dict(case_index=i,name=g['name'],template_id=g['template_id'],population=g['population'],
                           epsilon=g['epsilon'],method=m,expected_repetitions=3,valid_repetitions=3,
                           within_budget_repetitions=sum(r['within_budget'] for r in batch),
                           late_resolved_repetitions=sum(r['resolved'] and not r['within_budget'] for r in batch),
                           success=success,conclusion=next(iter(statuses)) if success else None,
                           all_quick=all(r['quick_resolved'] for r in batch),median_elapsed=median([r['elapsed'] for r in batch]),
                           **{f'status_rep{r}':batch[r]['status'] for r in (0,1,2)})
    paired={}
    for base,candidate in PAIRS:
        for i,g in enumerate(groups):
            a,b=cases[i,base],cases[i,candidate];reasons=[]
            if not a['success']:reasons.append('baseline_not_stable_within_budget')
            if not b['success']:reasons.append('candidate_not_stable_within_budget')
            if a['success'] and b['success'] and a['conclusion']!=b['conclusion']:reasons.append('conclusion_mismatch')
            quick=a['all_quick'] and b['all_quick']
            if quick:reasons.append('common_quick')
            if a['median_elapsed']<=0 or b['median_elapsed']<=0:reasons.append('nonpositive_median_time')
            eligible=not reasons;ratio=a['median_elapsed']/b['median_elapsed'] if eligible else None
            paired[i,base]=dict(case_index=i,name=g['name'],template_id=g['template_id'],population=g['population'],epsilon=g['epsilon'],
                baseline=base,candidate=candidate,baseline_success=a['success'],candidate_success=b['success'],
                baseline_conclusion=a['conclusion'],candidate_conclusion=b['conclusion'],common_quick=quick,timing_eligible=eligible,
                exclusion_reasons=';'.join(reasons),new_success=not a['success'] and b['success'],lost_success=a['success'] and not b['success'],
                baseline_median=a['median_elapsed'],candidate_median=b['median_elapsed'],speedup_ratio=ratio,
                log_speedup=math.log(ratio) if eligible else None,
                candidate_slowdown_fraction=b['median_elapsed']/a['median_elapsed']-1 if eligible else None,
                slowdown_gt_5pct=eligible and b['median_elapsed']>1.05*a['median_elapsed'])
    def read_table(name):
        with (analysis/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    slots=read_table('run_slots');assert len(slots)==480
    for row in slots:fields(row,runs[int(row['case_index']),row['method'],int(row['rep'])])
    table=read_table('case_methods');assert len(table)==160
    for row in table:fields(row,cases[int(row['case_index']),row['method']])
    table=read_table('paired_cases');assert len(table)==64
    for row in table:fields(row,paired[int(row['case_index']),row['baseline']])
    slowdown=[row for row in paired.values() if row['slowdown_gt_5pct']]
    changes=[row for row in paired.values() if row['new_success'] or row['lost_success']]
    for name,expected in [('slowdowns_gt_5pct',slowdown),('success_changes',changes)]:
        table=read_table(name);assert len(table)==len(expected)
        assert {(int(r['case_index']),r['baseline']) for r in table}=={(r['case_index'],r['baseline']) for r in expected}
        for row in table:fields(row,paired[int(row['case_index']),row['baseline']])
        assert len(summary[name])==len(expected)
        for row in summary[name]:fields(row,paired[row['case_index'],row['baseline']])
    expected_groups={'all','population=ordinary','population=structural','epsilon=1/1000','epsilon=1/100'}
    expected_groups|={f'population={p};epsilon={e}' for p in ('ordinary','structural') for e in ('1/1000','1/100')}
    expected_groups|={f'threshold_policy={j}' for j in range(4)}
    assert {r['group'] for r in summary['method_groups']}==expected_groups
    method_expectations={}
    for row in summary['method_groups']:
        name,m=row['group'],row['method'];ids=group_indices(name,groups)
        batch=[runs[i,m,rep] for i in sorted(ids) for rep in (0,1,2)];counts=Counter(r['status'] for r in batch)
        want=dict(group=name,method=m,expected_cases=len(ids),successful_cases=sum(cases[i,m]['success'] for i in ids),
                  expected_runs=3*len(ids),observed_records=3*len(ids),resolved_within_budget_runs=sum(r['within_budget'] for r in batch),
                  late_resolved_runs=sum(r['resolved'] and not r['within_budget'] for r in batch),
                  total_observed_elapsed_seconds=math.fsum(r['elapsed'] for r in batch),invalid_elapsed_records=0,
                  **{s.lower()+'_runs':counts[s] for s in ('CERTIFIED_SAFE','CERTIFIED_UNSAFE','INCONSISTENT','UNKNOWN','ERROR','MISSING','INVALID')})
        for key in row:
            if 'median' in key and 'elapsed' in key:want[key]=median([r['elapsed'] for r in batch]) if batch else None
        fields(row,want);method_expectations[name,m]=want
    assert len(method_expectations)==len(expected_groups)*5
    for row in read_table('method_groups'):fields(row,method_expectations[row['group'],row['method']])
    random_source=random.Random(20261013947);draws=[Counter(random_source.randrange(16) for _ in range(16)) for _ in range(10000)]
    pair_expectations={}
    for row in summary['paired_groups']:
        name,base=row['group'],row['baseline'];ids=group_indices(name,groups)
        batch=[paired[i,base] for i in sorted(ids)];eligible=[r for r in batch if r['timing_eligible']]
        want=dict(group=name,baseline=base,candidate=dict(PAIRS)[base],expected_cases=len(ids),
                  baseline_successes=sum(r['baseline_success'] for r in batch),candidate_successes=sum(r['candidate_success'] for r in batch),
                  new_successes=sum(r['new_success'] for r in batch),lost_successes=sum(r['lost_success'] for r in batch),
                  common_quick_cases=sum(r['common_quick'] for r in batch),slowdowns_gt_5pct=sum(r['slowdown_gt_5pct'] for r in batch),
                  **bootstrap(eligible,draws))
        fields(row,want);pair_expectations[name,base]=want
    assert len(pair_expectations)==len(expected_groups)*2
    for row in read_table('paired_groups'):fields(row,pair_expectations[row['group'],row['baseline']])
    # Complete validated data must have no certificate conflicts or nesting violation.
    conclusions={i:{runs[i,m,r]['status'] for m in METHODS for r in (0,1,2) if runs[i,m,r]['resolved']} for i in range(32)}
    assert all(len(v)<=1 for v in conclusions.values())
    for tid in range(16):assert not ('CERTIFIED_UNSAFE' in conclusions[2*tid] and 'CERTIFIED_SAFE' in conclusions[2*tid+1])
    for name in ('conclusion_conflicts','nested_budget_contradictions','validation_issues'):
        assert not summary[name] and not read_table(name)
    fields(summary,dict(expected_cases=32,expected_methods=5,expected_repetitions=3,expected_runs=480,
                        observed_records=480,input_complete_and_valid=True,valid_runs=480,missing_runs=0,invalid_slots=0,
                        total_observed_elapsed_seconds=math.fsum(r['elapsed'] for r in runs.values()),
                        cost_records_with_valid_elapsed=480,cost_records_with_invalid_elapsed=0))
    assert summary['statistics_valid_for_reporting'] is True
    assert summary['source_sha256']['records']==sha(records_path) and summary['source_sha256']['evaluation_groups']==sha(groups_path)
    assert summary['source_sha256']['script']==sha(analysis/'analyze_comparison.py')
    diagnostic_path=analysis/'DIAGNOSTIC_WORK_COUNTS.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    assert diagnostic['scope']=='POSTHOC_DESCRIPTIVE_WORK_ONLY_NOT_COMPONENT_ABLATION'
    assert diagnostic['aggregate']=={m:dict(work[m]) for m in METHODS}
    assert diagnostic['new_case_017']==case017
    files=[Path(__file__),summary_path,records_path,groups_path,analysis/'analyze_comparison.py',analysis/'STATS_RULES_CN.txt']+list(analysis.glob('*.csv'))
    files.append(diagnostic_path)
    report=dict(status='PASS',scope='INDEPENDENT_RAW_RUN_PAIRED_CLUSTER_STATISTICS_AUDIT',records=480,case_methods=160,paired_cases=64,
                method_group_rows=len(method_expectations),paired_group_rows=len(pair_expectations),cluster_bootstrap_replicates_per_row=10000,
                weighted_order_statistic_reconstruction=True,slowdown_cases=len(slowdown),success_change_cases=len(changes),
                all_csv_tables_checked=True,source_sha256={str(p):sha(p) for p in files},elapsed_seconds=time.perf_counter()-start,
                descriptive_all_stage_work_counts_checked=True,all_stage_work_counts={m:dict(work[m]) for m in METHODS},
                producer_analysis_imported=False,timing_overlap=False,certificate_validity_is_separately_audited=True,
                theory_novelty_or_publication_readiness_claimed=False)
    with (ROOT/'audit/STATISTICS_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
