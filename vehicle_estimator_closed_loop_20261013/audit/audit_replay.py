"""Independent replay selection, exact-output pairing and cost arithmetic.

Reads existing 90 runs only; never reruns estimators or simulation controllers.
"""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'/'paired_replay'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readlines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def near(a,b):return abs(float(a)-float(b))<=1e-10*max(1.,abs(float(a)),abs(float(b)))


def main():
    plan=load(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json');registered=plan['paired_replay']
    queries=load(OUT/'QUERIES.json');execution=load(OUT/'EXECUTION.json');summary=load(OUT/'SUMMARY.json')
    runs=readlines(OUT/'runs.jsonl');checks=0
    def check(x,msg):
        nonlocal checks
        checks+=1
        if not x:raise AssertionError(msg)
    check(queries['registration']==registered,'registration identity')
    check(queries['plan_sha256']==execution['plan_sha256']==sha(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'),'frozen plan binding')
    check(execution['completed_runs']==execution['planned_runs']==len(runs)==90,'all 90 planned runs')
    check(execution['runs_sha256']==sha(OUT/'runs.jsonl') and execution['query_source_sha256']==sha(OUT/'QUERIES.json'),'replay byte hashes')
    selected=[]
    for scene in registered['source_scenes']:
        case=next(c for c in plan['cases'] if c['scene_id']==scene and c['method']=='exact_range')
        records=readlines(ROOT/'cosim_results'/case['id']/'decision.jsonl')
        eligible=sorted((r for r in records if r['estimate'] is not None),key=lambda r:r['time'])
        for i,record in enumerate(eligible[:3]):
            selected.append((scene+'__q'+f'{i:03d}',case['id'],record))
    check(len(selected)==len(queries['queries'])==6,'all first at most three eligible queries')
    for (qid,cid,record),query in zip(selected,queries['queries']):
        check(query['query_id']==qid and query['source_case_id']==cid,'selected query identity')
        check(query['selected_rows']==record['selected_rows'] and query['current_packet']==record['packet']
              and query['prior']==record['prior'] and near(query['simulated_time_s'],record['time']),'selected full public input identity')
    methods=plan['methods'];lookup={(r['query_id'],r['repetition'],r['method']):r for r in runs}
    check(len(lookup)==90,'unique query/repetition/method denominator')
    totals={}
    for method in methods:
        own=[r for r in runs if r['method']==method]
        statuses=Counter(r['estimate']['status'] for r in own)
        estimate=sum(r['measured_estimate_s'] for r in own)
        preparation=sum(r['preparation_s'] for r in own);guard=sum(r['guard_s'] for r in own)
        total=estimate+preparation+guard
        reference=summary['methods'][method]
        check(len(own)==reference['runs']==18,'method 18 runs '+method)
        check(near(estimate,reference['estimate_total_s']) and near(statistics.median(r['measured_estimate_s'] for r in own),reference['estimate_median_s']),'method estimate stats '+method)
        check(near(preparation,reference['preparation_total_s']) and near(guard,reference['guard_total_s']),'other costs '+method)
        check(statuses['OK']==reference['ok'] and statuses['UNKNOWN']==reference['unknown'] and statuses['ERROR']==reference['errors'],'statuses '+method)
        check(sum(r['measured_estimate_late'] for r in own)==reference['measured_late'],'measured lateness '+method)
        check(sum(r['estimate'].get('frontend_calls',0) for r in own)==reference['frontend_calls'],'frontend calls '+method)
        totals[method]={'estimate_total_s':estimate,'preparation_total_s':preparation,'guard_total_s':guard,
                        'measured_pipeline_total_s':total,'runs':18,'status_counts':dict(statuses)}
    for r in runs:
        check(near(r['estimate_finished_monotonic_s']-r['estimate_started_monotonic_s'],r['measured_estimate_s']),'monotonic estimate duration')
        check(r['measured_estimate_late']==(r['measured_estimate_s']>registered['deadline_s']),'deadline label')
        order=methods[r['repetition']%len(methods):]+methods[:r['repetition']%len(methods)]
        check(order[r['order_index']]==r['method'],'frozen rotated method order')
        check(all(r[k]>=0 for k in ('preparation_s','measured_estimate_s','guard_s')),'nonnegative complete costs')
    pairs=[]
    for baseline,structured in (('ma','ma_structured'),('zhao2t3','zhao2t3_structured')):
        raw=[];qualified=[];same_exact_intervals=0;same_guard=0
        for qid,_,_ in selected:
          for repetition in range(3):
            left=lookup[(qid,repetition,baseline)];right=lookup[(qid,repetition,structured)]
            ratio=left['measured_estimate_s']/right['measured_estimate_s'];raw.append(ratio)
            if left['estimate']['status']==right['estimate']['status']=='OK' and not (left['measured_estimate_late'] or right['measured_estimate_late']):
                qualified.append(ratio)
                same_exact_intervals+=tuple(map(F,left['estimate']['b_interval_rational']))==tuple(map(F,right['estimate']['b_interval_rational']))
                same_guard+=left['guard_status']==right['guard_status']
        reference=next(p for p in summary['pairs'] if p['baseline']==baseline)
        leftcost=totals[baseline]['estimate_total_s'];rightcost=totals[structured]['estimate_total_s']
        check(reference['complete_pair_runs']==len(raw) and reference['qualified_pairs']==len(qualified),'pair denominators '+baseline)
        check(near(reference['total_cost_speed_ratio'],leftcost/rightcost) and near(reference['full_workload_cost_reduction_fraction'],1-rightcost/leftcost),'all deadline workload estimate cost '+baseline)
        check(near(reference['median_all_pair_speed_ratio'],statistics.median(raw)) and reference['structured_slower_pairs']==sum(r<1 for r in raw),'pair median/slower counts '+baseline)
        if qualified:check(near(reference['median_qualified_speed_ratio'],statistics.median(qualified)),'qualified median '+baseline)
        else:check(reference['median_qualified_speed_ratio'] is None,'no invalid qualified speedup '+baseline)
        pairs.append({'baseline':baseline,'structured':structured,'all_pairs':len(raw),'qualified_pairs':len(qualified),
            'exact_same_rational_interval_pairs':same_exact_intervals,'same_guard_conclusion_pairs':same_guard,
            'all_estimate_workload_cost_reduction_fraction':1-rightcost/leftcost,
            'pipeline_including_preparation_and_guard_reduction_fraction':1-totals[structured]['measured_pipeline_total_s']/totals[baseline]['measured_pipeline_total_s'],
            'median_qualified_speed_ratio':statistics.median(qualified) if qualified else None,
            'structured_slower_all_pairs':sum(r<1 for r in raw)})
    cheapest=[]
    for qid,_,_ in selected:
      for repetition in range(3):
        matched=[lookup[(qid,repetition,m)] for m in methods]
        cheapest.append(min(matched,key=lambda r:r['preparation_s']+r['measured_estimate_s']+r['guard_s'])['method'])
    report={'status':'PASS_INDEPENDENT_IDENTICAL_QUERY_REPLAY_AUDIT','checks':checks,
        'full_runs':90,'query_count':6,'source_scenes':registered['source_scenes'],
        'audit_source_sha256':sha(__file__),'plan_sha256':sha(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'),
        'source_hashes':{p.name:sha(p) for p in (OUT/'QUERIES.json',OUT/'EXECUTION.json',OUT/'runs.jsonl',OUT/'SUMMARY.json')},
        'method_costs':totals,'pairs':pairs,'cheapest_complete_pipeline_run_counts':dict(Counter(cheapest)),
        'interpretation':'Six preregistered same-input queries, three repeats and five methods. Ma equal rational intervals/guard verdicts can qualify matched speed ratios; Zhao baseline all UNKNOWN precludes an equal-output speedup. All timed-out work retained. Preparation/guard counted separately and pipeline costs include both. No new closed-loop trajectory or population inference.'}
    path=ROOT/'audit'/'INDEPENDENT_REPLAY_AUDIT.json'
    if path.exists():raise FileExistsError('Preserve prior replay audit')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','checks','full_runs','query_count','cheapest_complete_pipeline_run_counts')}))


if __name__=='__main__':main()
