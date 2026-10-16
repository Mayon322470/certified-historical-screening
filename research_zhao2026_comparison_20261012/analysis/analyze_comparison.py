"""Streaming fixed-suite comparison. UNKNOWN time is cost, never solve speedup."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'case-paired-streaming-v2'
BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES = 20261012947, 10000
MAIN_METHODS = ('ma_strip', 'ma_strip_structured', 'zhao1', 'zhao2', 'zhao2t3', 'zhao2t3_structured')
METHODS = MAIN_METHODS + ('zhao2t3_sturm', 'zhao2t3_sparse')
POLICIES = ('public_envelope_plus_5percent_scale', 'compatible_reference_plus_1e-2_scale',
            'compatible_reference_plus_1e-4_scale', 'compatible_reference_minus_1e-3_scale')
PAIR_DEFINITIONS = [('ma_strip','ma_strip','ma_strip_structured','within_method'),
                    ('zhao2t3','zhao2t3','zhao2t3_structured','within_method')]
PAIR_DEFINITIONS += [(f'{z}__vs__{m}',z,m,'cross_paper') for m in ('ma_strip_structured','ma_strip')
                     for z in ('zhao1','zhao2','zhao2t3')]
PAIR_DEFINITIONS += [(f'{a}__vs__{b}',a,b,'zhao_order_and_tightening')
                     for a,b in (('zhao1','zhao2'),('zhao1','zhao2t3'),('zhao2','zhao2t3'))]
PAIR_DEFINITIONS += [(f'zhao2t3_structured__vs__{m}','zhao2t3_structured',m,'ablation')
                     for m in ('zhao2t3_sturm','zhao2t3_sparse')]
PAIR_DEFINITIONS += [('zhao2t3_sturm__vs__zhao2t3_sparse','zhao2t3_sturm','zhao2t3_sparse','ablation')]

class AnalysisError(ValueError):
    """Input is incomplete, duplicated, or inconsistent with the recorded run."""


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def stratification(cases, input_path):
    """Join public index design only; no private generation data is opened."""
    holdout = input_path.name == 'holdout_48.json'
    if holdout and len(cases) != 48:
        raise AnalysisError('The new frozen holdout must contain exactly 48 cases')
    out = []
    for i,case in enumerate(cases):
        group = case.get('group') or 'unspecified'
        family = ('ordinary' if group == 'holdout_random' else 'structured'
                  if group in ('holdout_four_hold','holdout_valley_velocity','holdout_valley_position') else 'unspecified')
        policy = 'not_applicable'
        if holdout:
            if i < 32:
                name,rotation,expected_group = f'holdout_random_{i:02d}_N{(1,2,4,8)[i//8]}',0,'holdout_random'
            elif i < 40:
                name,rotation,expected_group = f'holdout_four_hold_{(i-32)//4}_{i%4}',2*((i-32)//4),'holdout_four_hold'
            elif i < 44:
                name,rotation,expected_group = f'holdout_valley_velocity_{i-40}',1,'holdout_valley_velocity'
            else:
                name,rotation,expected_group = f'holdout_valley_position_{i-44}',3,'holdout_valley_position'
            if case['name'] != name or group != expected_group:
                raise AnalysisError(f'Public holdout design mismatch at index {i}')
            policy = POLICIES[(i+rotation)%4]
        out.append(dict(group=group,family=family,threshold_policy=policy))
    if holdout and Counter(v['threshold_policy'] for v in out) != Counter(dict.fromkeys(POLICIES,12)):
        raise AnalysisError('Unbalanced public threshold-policy design')
    return out, ('NEW_HOLDOUT_PROTOCOL_CN.txt: public index/family rotation; no private data read'
                 if holdout else 'Not the new holdout: threshold policy not inferred')


def reduce_row(row, line, case, strata, budget):
    result = row['result']
    elapsed = float(row['elapsed'])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise AnalysisError(f'Nonpositive/nonfinite elapsed on line {line}')
    if row['name'] != case['name'] or row.get('group') != case.get('group'):
        raise AnalysisError(f'Case identity mismatch on line {line}')
    if float(row['budget_seconds']) != budget:
        raise AnalysisError(f'Budget mismatch on line {line}')
    status = result['status']
    resolved = (status in ('CERTIFIED_UNSAFE','INCONSISTENT') or
                status == 'CERTIFIED_SAFE' and result.get('nonempty_witness') is not None)
    within = resolved and (not budget or elapsed <= budget)
    if row['resolved'] != resolved or row['resolved_within_budget'] != within:
        raise AnalysisError(f'Certificate/budget resolution flag mismatch on line {line}')
    overshoot = max(0.,elapsed-budget) if budget else 0.
    if not math.isclose(float(row['overshoot_seconds']),overshoot,abs_tol=1e-9):
        raise AnalysisError(f'Overshoot mismatch on line {line}')
    engines = Counter()
    if row['rep'] == 0:
        for stage in row.get('stages',[]):
            for deletion in stage.get('deletions',[]):
                label = deletion['engine']
                if label == 'sparse': label += ':'+deletion['certificate']['kind']
                engines[label] += 1
    return dict(case_index=row['case_index'],name=row['name'],method=row['method'],rep=row['rep'],
                line_number=line,**strata,status=status,elapsed=elapsed,resolved=resolved,
                resolved_within_budget=within,overshoot_seconds=overshoot,
                quick_resolved=bool(result.get('quick_resolved',False)),final_n=result.get('n'),
                future_vertices=row.get('future_vertices_total',0),frontend_calls=row.get('frontend_calls_total',0),
                deletion_engines=dict(engines),error_traceback=row.get('error_traceback'))


def stream_records(folder, root):
    metadata = json.loads((folder/'run_metadata.json').read_text(encoding='utf-8-sig'))
    args = metadata['arguments']
    methods,repeats = args['methods'].split(','),int(args['reps'])
    if len(methods) != len(set(methods)) or any(not m for m in methods):
        raise AnalysisError('Duplicate/empty method in metadata')
    if repeats not in (1,3):
        raise AnalysisError('Preregistered analysis supports 3-repeat main or 1-repeat auxiliary runs')
    input_path = root/'inputs'/args['inputs']
    raw_input = input_path.read_bytes()
    if hashlib.sha256(raw_input).hexdigest() != metadata['input_sha256']:
        raise AnalysisError('Input SHA256 differs from run metadata')
    cases = json.loads(raw_input.decode('utf-8-sig'))
    strata,source = stratification(cases,input_path)
    if input_path.name == 'holdout_48.json' and set(methods) == set(MAIN_METHODS) and repeats != 3:
        raise AnalysisError('Six-method new-holdout main run requires exactly 3 repeats')
    expected = {(i,m,r) for i in range(len(cases)) for m in methods for r in range(repeats)}
    reduced,digest,max_bytes,lines = {},hashlib.sha256(),0,0
    with (folder/'records.jsonl').open('rb') as handle:
        for lines,raw in enumerate(handle,1):
            digest.update(raw)
            max_bytes = max(max_bytes,len(raw))
            try: row = json.loads(raw)
            except (json.JSONDecodeError,UnicodeDecodeError) as exc:
                raise AnalysisError(f'Invalid/truncated JSONL line {lines}: {exc}') from exc
            key = row['case_index'],row['method'],row['rep']
            if key not in expected: raise AnalysisError(f'Unexpected case/method/repeat {key} on line {lines}')
            if key in reduced: raise AnalysisError(f'Duplicate case/method/repeat {key} on line {lines}')
            reduced[key] = reduce_row(row,lines,cases[key[0]],strata[key[0]],float(args['budget']))
            del row
    missing = sorted(expected-reduced.keys())
    if missing: raise AnalysisError(f'Incomplete Cartesian product: {len(missing)} missing; first 12: {missing[:12]}')
    if not (folder/'summary.json').is_file():
        raise AnalysisError('Runner completion marker summary.json absent; no final analysis written')
    completion = json.loads((folder/'summary.json').read_text(encoding='utf-8-sig'))
    for method in methods:
        if Counter(v['status'] for v in reduced.values() if v['method']==method) != Counter(completion.get(method,{}).get('statuses',{})):
            raise AnalysisError(f'Completion marker status counts disagree for {method}')
    audit = dict(complete=True,expected_records=len(expected),read_records=len(reduced),physical_lines=lines,
                 duplicates=0,missing_records=0,records_sha256=digest.hexdigest(),maximum_line_bytes=max_bytes,
                 raw_certificate_retention='One JSONL record at a time; retain reduced per-repeat statistics only',
                 stratification_source=source,input_sha256=metadata['input_sha256'])
    return metadata,methods,repeats,reduced,audit


class Bootstrap:
    """Resample cases after repeat reduction; never treat repeats as independent."""
    def __init__(self, enabled, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
        self.enabled,self.samples,self.seed = enabled,samples,seed

    def statistics(self, array, functions, key):
        values = np.asarray(array,dtype=float)
        if not len(values) or not self.enabled: return {name:None for name in functions}
        context = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,context]))
        results = {name:[] for name in functions}
        for start in range(0,self.samples,500):
            sample = values[rng.integers(0,len(values),size=(min(500,self.samples-start),len(values)))]
            for name,function in functions.items(): results[name].append(function(sample))
        return {name:[float(x) for x in np.percentile(np.concatenate(chunks),[2.5,97.5])]
                for name,chunks in results.items()}


def case_summaries(rows, repeats):
    grouped = defaultdict(list)
    decisions = defaultdict(set)
    for row in rows.values():
        grouped[row['case_index'],row['method']].append(row)
        if row['resolved']: decisions[row['case_index']].add(row['status'])
    data = {}
    for key,reps in sorted(grouped.items()):
        reps.sort(key=lambda r:r['rep'])
        if [r['rep'] for r in reps] != list(range(repeats)): raise AnalysisError(f'Repeat mismatch for {key}')
        first,statuses = reps[0],[r['status'] for r in reps]
        data[key] = dict(case_index=key[0],name=first['name'],method=key[1],group=first['group'],
            family=first['family'],threshold_policy=first['threshold_policy'],status=first['status'],statuses=statuses,
            status_stable=len(set(statuses))==1,resolved_all=all(r['resolved'] for r in reps),
            within_budget_all=all(r['resolved_within_budget'] for r in reps),
            median_seconds=st.median(r['elapsed'] for r in reps),min_seconds=min(r['elapsed'] for r in reps),
            max_seconds=max(r['elapsed'] for r in reps),quick_only=all(r['quick_resolved'] for r in reps),
            quick_stable=len({r['quick_resolved'] for r in reps})==1,final_n=first['final_n'],
            future_vertices=first['future_vertices'],frontend_calls=first['frontend_calls'],repeats=reps,
            case_has_conflicting_certificates=len(decisions[key[0]])>1)
    return data


def method_stats(vals, boot, key):
    n,solved = len(vals),sum(v['within_budget_all'] for v in vals)
    cis = boot.statistics([[int(v['within_budget_all']),v['median_seconds']] for v in vals],
        {'rate':lambda a:a[:,:,0].mean(axis=1),'cost':lambda a:a[:,:,1].sum(axis=1)},key)
    return dict(cases=n,statuses_rep0=dict(Counter(v['status'] for v in vals)),
        resolved_all=sum(v['resolved_all'] for v in vals),within_budget_all=solved,
        within_budget_rate=solved/n if n else None,within_budget_rate_ci95=cis['rate'],
        sum_case_median_seconds=sum(v['median_seconds'] for v in vals),sum_case_median_seconds_ci95=cis['cost'],
        quick_only=sum(v['quick_only'] for v in vals),unstable_status_cases=[v['name'] for v in vals if not v['status_stable']])


def eligibility(a,b):
    reasons = []
    if not (a['within_budget_all'] and b['within_budget_all']): reasons.append('not_both_all_repeats_resolved_within_budget')
    if not (a['status_stable'] and b['status_stable']): reasons.append('unstable_repeat_status')
    if a['status'] != b['status']: reasons.append('different_decision')
    if a['case_has_conflicting_certificates'] or b['case_has_conflicting_certificates']:
        reasons.append('case_has_conflicting_certificates')
    if a['quick_only'] and b['quick_only']: reasons.append('common_quick')
    return reasons


def paired_stats(pairs, boot, key, details=False):
    eligible = [(a,b) for a,b in pairs if not eligibility(a,b)]
    ratios = [a['median_seconds']/b['median_seconds'] for a,b in eligible]
    slower = [a['name'] for a,b in eligible if b['median_seconds']>1.05*a['median_seconds']]
    all_cis = boot.statistics([[a['median_seconds'],b['median_seconds'],int(a['within_budget_all']),int(b['within_budget_all'])]
                              for a,b in pairs],
        {'cost':lambda a:a[:,:,0].sum(axis=1)/a[:,:,1].sum(axis=1),
         'rate_difference':lambda a:(a[:,:,3]-a[:,:,2]).mean(axis=1)},key+':all')
    speed_cis = boot.statistics(ratios,{'geomean':lambda a:np.exp(np.log(a).mean(axis=1)),
        'median':lambda a:np.median(a,axis=1),'slower_fraction':lambda a:(a<1/1.05).mean(axis=1)},key+':eligible')
    ac,bc = sum(a['median_seconds'] for a,b in pairs),sum(b['median_seconds'] for a,b in pairs)
    report = dict(cases=len(pairs),baseline_within_budget_all=sum(a['within_budget_all'] for a,b in pairs),
        candidate_within_budget_all=sum(b['within_budget_all'] for a,b in pairs),
        added_solved=[a['name'] for a,b in pairs if not a['within_budget_all'] and b['within_budget_all']],
        lost_solved=[a['name'] for a,b in pairs if a['within_budget_all'] and not b['within_budget_all']],
        contradictory_resolved=[a['name'] for a,b in pairs if len({r['status'] for v in (a,b) for r in v['repeats'] if r['resolved']})>1],
        baseline_sum_case_median_seconds=ac,candidate_sum_case_median_seconds=bc,
        all_task_cost_ratio=ac/bc if pairs else None,all_task_cost_ratio_ci95=all_cis['cost'],
        within_budget_rate_difference=st.mean(int(b['within_budget_all'])-int(a['within_budget_all']) for a,b in pairs) if pairs else None,
        within_budget_rate_difference_ci95=all_cis['rate_difference'],common_solved_nonquick_cases=len(eligible),
        common_solved_geomean_speed_ratio=math.exp(st.mean(math.log(x) for x in ratios)) if ratios else None,
        common_solved_geomean_speed_ratio_ci95=speed_cis['geomean'],
        common_solved_median_speed_ratio=st.median(ratios) if ratios else None,
        common_solved_median_speed_ratio_ci95=speed_cis['median'],
        common_solved_min_speed_ratio=min(ratios) if ratios else None,common_solved_max_speed_ratio=max(ratios) if ratios else None,
        slower_by_over_5pct=slower,slower_by_over_5pct_count=len(slower),
        slower_by_over_5pct_fraction=len(slower)/len(eligible) if eligible else None,
        slower_by_over_5pct_fraction_ci95=speed_cis['slower_fraction'],
        exclusion_reason_counts_nonexclusive=dict(Counter(reason for a,b in pairs for reason in eligibility(a,b))),
        common_quick_cases=sum(a['quick_only'] and b['quick_only'] for a,b in pairs),
        asymmetric_quick_cases=[a['name'] for a,b in pairs if a['quick_only'] != b['quick_only']])
    if details:
        report['common_solved_details'] = [dict(case_index=a['case_index'],name=a['name'],group=a['group'],family=a['family'],
            threshold_policy=a['threshold_policy'],baseline_s=a['median_seconds'],candidate_s=b['median_seconds'],
            speed_ratio=a['median_seconds']/b['median_seconds'],candidate_slower_by_over_5pct=b['median_seconds']>1.05*a['median_seconds'])
            for a,b in eligible]
        report['excluded_case_details'] = [dict(case_index=a['case_index'],name=a['name'],reasons=eligibility(a,b),
            baseline_statuses=a['statuses'],candidate_statuses=b['statuses']) for a,b in pairs if eligibility(a,b)]
    return report


def build_report(tag, metadata, methods, repeats, rows, audit, data):
    boot = Bootstrap(repeats==3)
    report = dict(version=VERSION,tag=tag,trials=len(rows),cases=len({i for i,m in data}),repeats_per_case_method=repeats,
        main_methods=list(MAIN_METHODS),main_table_complete=all(m in methods for m in MAIN_METHODS),
        run_metadata=metadata,integrity=audit,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definitions=dict(case_time='Median of declared repeats (3 main; 1 auxiliary)',
            within_budget_all='Every declared repeat has a nonvacuous certificate and elapsed <= budget (or no budget)',
            speed_eligibility='Both all-repeat within-budget; both repeat statuses stable; same decision; not common quick; no conflicting certified decisions on that case',
            common_quick='Both methods resolve through shared quick gate in every repeat',
            speed_ratio='Baseline case-median seconds / candidate case-median seconds; >1 favors candidate',
            slowdown='Candidate case-median seconds >1.05 times baseline, on eligible solved subset',
            all_task_cost='Sum of case-median wall times, including UNKNOWN, late certificates and errors; not solve speedup',
            threshold_policy='Difficulty design only: numerical-reference margins are not safety truth labels',
            exclusions='Explicit and nonexclusive; no cases omitted from all-task costs or solve rates'),
        bootstrap=dict(enabled=boot.enabled,samples=boot.samples,seed=boot.seed,
            interval='Percentile [2.5,97.5]; paired case resampling with replacement after within-case median reduction',
            unit='Case, never individual repeat; eligible-time intervals resample fixed eligible subset',
            scope='Descriptive variability of this fixed suite; not population inference or repeat-level timing uncertainty',
            multiplicity='No hypothesis tests or multiplicity-adjusted inferential claims',
            rng='numpy default_rng SeedSequence([fixed_seed, SHA256(context) first 8 bytes little-endian])',
            disabled_reason=None if boot.enabled else 'Single-repeat auxiliary/development: confidence intervals suppressed'),methods={},paired={})
    report['status_variation'] = [dict(case_index=v['case_index'],name=v['name'],method=v['method'],statuses=v['statuses'])
                                  for v in data.values() if not v['status_stable']]
    report['quick_status_variation'] = [dict(name=v['name'],method=v['method']) for v in data.values() if not v['quick_stable']]
    report['conflicting_certificates'] = []
    for i in sorted({i for i,m in data}):
        certs = [r for r in rows.values() if r['case_index']==i and r['resolved']]
        if len({r['status'] for r in certs})>1:
            report['conflicting_certificates'].append(dict(case_index=i,name=certs[0]['name'],statuses=sorted({r['status'] for r in certs}),
                evidence=[{k:r[k] for k in ('method','rep','status','elapsed','line_number')} for r in certs]))
    report['late_certificates'] = [r for r in rows.values() if r['resolved'] and not r['resolved_within_budget']]
    report['errors'] = [r for r in rows.values() if r['status']=='ERROR' or r['error_traceback']]
    report['unresolved_safe_without_nonempty_witness'] = [r for r in rows.values() if r['status']=='CERTIFIED_SAFE' and not r['resolved']]
    for method in methods:
        vals = [v for (i,m),v in data.items() if m==method]
        trials = [r for r in rows.values() if r['method']==method]
        entry = method_stats(vals,boot,'method:'+method)
        engines = Counter()
        for r in trials: engines.update(r['deletion_engines'])
        entry.update(statuses_all_trials=dict(Counter(r['status'] for r in trials)),
            overrun_trials=sum(r['overshoot_seconds']>0 for r in trials),max_overshoot_seconds=max(r['overshoot_seconds'] for r in trials),
            late_certificate_trials=sum(r['resolved'] and not r['resolved_within_budget'] for r in trials),
            error_trials=sum(r['status']=='ERROR' for r in trials),deletion_engines_rep0=dict(engines),
            unknown_trial_cost_seconds=sum(r['elapsed'] for r in trials if r['status']=='UNKNOWN'))
        for field,label in (('group','groups'),('family','families'),('threshold_policy','threshold_policies')):
            entry[label] = {v:method_stats([a for a in vals if a[field]==v],boot,f'method:{method}:{field}:{v}')
                            for v in sorted({a[field] for a in vals})}
        report['methods'][method] = entry
    for pair_id,baseline,candidate,category in PAIR_DEFINITIONS:
        if baseline not in methods or candidate not in methods: continue
        pairs = [(data[i,baseline],data[i,candidate]) for i in sorted({i for i,m in data})]
        entry = paired_stats(pairs,boot,'pair:'+pair_id,details=True)
        entry.update(baseline=baseline,candidate=candidate,category=category)
        for field,label in (('group','groups'),('family','families'),('threshold_policy','threshold_policies')):
            entry[label] = {v:paired_stats([(a,b) for a,b in pairs if a[field]==v],boot,f'pair:{pair_id}:{field}:{v}')
                            for v in sorted({a[field] for a,b in pairs})}
        report['paired'][pair_id] = entry
    return report


def table_text(report):
    lines = ['Fixed-suite descriptive comparison','',
             'All-task cost includes UNKNOWN, late certificates and errors; it is not solve speedup.',
             'Bootstrap intervals describe case composition of this fixed suite, not a population.','',
             'method\tcases\tall-repeats-within-budget\trate\tsum-case-median-cost-s\tlate-trials\terror-trials']
    for method,v in report['methods'].items():
        lines.append(f"{method}\t{v['cases']}\t{v['within_budget_all']}\t{v['within_budget_rate']:.6f}\t"
                     f"{v['sum_case_median_seconds']:.6f}\t{v['late_certificate_trials']}\t{v['error_trials']}")
    lines += ['', 'baseline\tcandidate\teligible-n\tgeomean-time-ratio\tdescriptive-ci95\tmedian-time-ratio\t>5pct-slower-n']
    for p in report['paired'].values():
        lines.append('\t'.join(map(str,(p['baseline'],p['candidate'],p['common_solved_nonquick_cases'],
            p['common_solved_geomean_speed_ratio'],p['common_solved_geomean_speed_ratio_ci95'],
            p['common_solved_median_speed_ratio'],p['slower_by_over_5pct_count']))))
    return '\n'.join(lines)+'\n'


def analyze(tag, root=ROOT):
    folder = Path(root)/'results'/tag
    metadata,methods,repeats,rows,audit = stream_records(folder,Path(root))
    data = case_summaries(rows,repeats)
    report = build_report(tag,metadata,methods,repeats,rows,audit,data)
    dump(folder/'paired_summary.json',report)
    dump(folder/'per_case_summary.json',list(data.values()))
    (folder/'analysis_tables.txt').write_text(table_text(report),encoding='utf-8')
    print(table_text(report),end='')
    return report,data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    args = parser.parse_args()
    analyze(args.tag)
