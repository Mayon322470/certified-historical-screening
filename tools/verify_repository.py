"""Read-only integrity and reported-number checks; no estimator/simulator rerun."""
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math,statistics
from repository_data import ROOT,PROVENANCE,STREAMS,blocks,lines,records,read_json
def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def near(a,b,tol=1e-7):assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=str);p.add_argument('--skip-decoded-hashes',action='store_true');a=p.parse_args()
    result={'status':'PASS','scope':'Read-only archived-data verification; no new experiment or timing run'}
    for e in PROVENANCE['entries']:
        path=ROOT/e['path'];assert path.is_file(),e['path'];assert hash_file(path)==e['sha256'],e['path']
        assert path.stat().st_size==e['bytes']
    if not a.skip_decoded_hashes:
        for rel,meta in STREAMS.items():
            h=hashlib.sha256();n=0
            for b in blocks(rel):h.update(b);n+=len(b)
            assert h.hexdigest()==meta['sha256'] and n==meta['bytes'],rel
    result['provenance_files_verified']=len(PROVENANCE['entries']);result['decoded_streams_verified']=0 if a.skip_decoded_hashes else len(STREAMS)
    A='research_zhao2026_comparison_20261012';B='research_task_face_budget_20261013';C='vehicle_estimator_closed_loop_20261013'
    groups=defaultdict(list);seen=set();n=0
    for r in records(A+'/results/holdout_main/records.jsonl'):
        n+=1;slot=(r['rep'],r['case_index'],r['method']);assert slot not in seen;seen.add(slot);groups[(r['case_index'],r['method'])].append(r)
    assert n==864 and len(groups)==48*6
    methods=defaultdict(lambda:{'stable_completed':0,'sum_case_median_s':0.0})
    for (case,m),rr in groups.items():
        assert len(rr)==3
        methods[m]['stable_completed']+=int(all(r['resolved_within_budget'] for r in rr) and len({r['result']['status'] for r in rr})==1)
        methods[m]['sum_case_median_s']+=statistics.median(r['elapsed'] for r in rr)
    expected={'ma_strip':(40,52.37258839793503),'ma_strip_structured':(42,38.011885),'zhao1':(34,72.207219),'zhao2':(31,102.885457),'zhao2t3':(30,121.438163),'zhao2t3_structured':(37,73.206513)}
    archived=read_json(A+'/results/holdout_main/paired_summary.json')
    for m,(count,cost) in expected.items():
        assert methods[m]['stable_completed']==count;near(methods[m]['sum_case_median_s'],cost,tol=1e-6);near(methods[m]['sum_case_median_s'],archived['methods'][m]['sum_case_median_seconds'])
    result['exact_model_48']={'runs':n,'methods':dict(methods),'cost_definition':'sum of 48 per-case medians, including unresolved/late work'}
    ab=Counter(r['method'] for r in records(A+'/results/holdout_ablation/records.jsonl'));assert sum(ab.values())==96;result['ablation_runs']=dict(ab)
    groups=defaultdict(list);seen=set();statuses=Counter();n=0
    for r in records(B+'/results/formal_main/records.jsonl'):
        n+=1;slot=(r['rep'],r['case_index'],r['method']);assert slot not in seen;seen.add(slot);groups[(r['case_index'],r['method'])].append(r);statuses[r['result']['status']]+=1
    assert n==480 and len(groups)==32*5
    methods=defaultdict(lambda:{'stable_completed':0,'total_all_96_runs_s':0.0})
    for (case,m),rr in groups.items():
        assert len(rr)==3;methods[m]['stable_completed']+=int(all(r['resolved_within_budget'] for r in rr) and len({r['result']['status'] for r in rr})==1);methods[m]['total_all_96_runs_s']+=sum(r['elapsed'] for r in rr)
    archived=read_json(B+'/analysis/SUMMARY_STATISTICS.json')
    for row in archived['method_groups']:
        if row['group']=='all':
            m=row['method'];near(methods[m]['total_all_96_runs_s'],row['total_observed_elapsed_seconds']);assert methods[m]['stable_completed']==row['successful_cases']
    assert statuses==Counter(CERTIFIED_SAFE=120,CERTIFIED_UNSAFE=291,UNKNOWN=69)
    result['common_response_32']={'runs':n,'methods':dict(methods),'statuses':dict(statuses),'cost_definition':'sum of all 96 runs per method'}
    plan=read_json(C+'/inputs/COSIM_ESTIMATOR_PLAN_FROZEN.json');assert len(plan['cases'])==20;native_rows=0;gap=math.inf;unknown=0;timeouts=0;full_cycle_over=0;cycle_max=0;tails=[]
    for case in plan['cases']:
        rel=C+'/cosim_results/'+case['id'];r=read_json(rel+'/result.json');assert r['status']=='SAMPLED_NO_STOPLINE_CROSSING';assert r['sampled_backup_membership']
        count=0;first_x=None;mingap=math.inf
        for row in csv.DictReader(lines(rel+'/native.csv')):
            x=float(row['x']);first_x=x if first_x is None else first_x;mingap=min(mingap,r['stopline_s_m']-(r['calibrated_initial_s_m']+x-first_x+r['invariant_body_radius_m']));count+=1
        assert count==r['native_rows']==24001;assert mingap>=0;native_rows+=count;gap=min(gap,mingap);unknown+=r['estimator_unknown_count'];timeouts+=r['timeout_count'];full_cycle_over+=r['full_cycle_wall_over10ms_count'];cycle_max=max(cycle_max,r['full_cycle_wall_max_s']);tails.append(r['stopped_final_tail_s'])
    assert native_rows==480020 and unknown==6 and timeouts==10 and full_cycle_over==4293;near(gap,16.092648540685524)
    result['vehicle_trajectories']={'trajectories':20,'native_rows':native_rows,'minimum_conservative_body_clearance_m':gap,'unknown_estimates':unknown,'timeout_fallbacks':timeouts,'full_cycles_over_10ms':full_cycle_over,'maximum_full_cycle_s':cycle_max,'stopped_tail_range_s':[min(tails),max(tails)],'scope':'finite sampled CARLA/CarSim trajectories'}
    rows=list(csv.DictReader((ROOT/C/'analysis/paired_replay/runs_summary.csv').open(newline='',encoding='utf-8')));assert len(rows)==90;costs=defaultdict(float);ok=Counter();counts=Counter()
    for r in rows:
        m=r['method'];costs[m]+=sum(float(r[k]) for k in ('measured_estimate_s','guard_s','preparation_s'));counts[m]+=1;ok[m]+=int(r['status']=='OK' and r['late']=='False')
    ledger=read_json(C+'/analysis/COMPLETE_COST_ACCOUNTING.json')
    for m,cost in costs.items():assert counts[m]==18;near(cost,ledger['methods'][m]['complete_query_total_s'])
    assert ok['zhao2t3']==0 and ok['zhao2t3_structured']==15
    result['identical_vehicle_queries']={'queries':6,'runs':90,'complete_workload_cost_s':dict(costs),'on_time_OK':dict(ok),'scope':'six identical input queries, five methods and three repetitions; not 90 trajectories'}
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.report:
        from pathlib import Path
        out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as f:f.write(text+'\n')
    print(text)
if __name__=='__main__':main()
