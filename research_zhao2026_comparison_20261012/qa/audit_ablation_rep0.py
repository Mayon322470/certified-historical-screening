"""Independently check the single-repeat ablation view from audited reductions."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import audit_statistics as independent

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read = independent.read
    same = independent.same
    previous = read(ROOT / 'qa/AUDIT_statistics.json')
    same(previous['status'], 'PASS')
    same(previous['checker_sha256'], digest(Path(independent.__file__)))
    reduced = {}
    sources = {}
    for tag, chosen in [('holdout_main', {'zhao2t3_structured'}),
                        ('holdout_ablation', {'zhao2t3_sturm', 'zhao2t3_sparse'})]:
        folder = ROOT / 'results' / tag
        path = folder / 'per_case_summary.json'
        same(digest(path), previous['runs'][tag]['per_case_sha256'])
        sources[tag] = {'per_case_sha256': digest(path),
                        'records_sha256': previous['runs'][tag]['records_sha256']}
        for value in read(path):
            if value['method'] in chosen:
                row = dict(value['repeats'][0], source_tag=tag)
                same(row['rep'], 0)
                reduced[row['case_index'], row['method']] = row
    same(len(reduced), 144)
    decisions = defaultdict(set)
    for (index, method), row in reduced.items():
        if row['resolved']:
            decisions[index].add(row['status'])
    data = {}
    for key, row in sorted(reduced.items()):
        data[key] = dict(case_index=row['case_index'], name=row['name'], method=row['method'],
            group=row['group'], family=row['family'], threshold_policy=row['threshold_policy'],
            status=row['status'], statuses=[row['status']], status_stable=True,
            resolved_all=row['resolved'], within_budget_all=row['resolved_within_budget'],
            median_seconds=row['elapsed'], min_seconds=row['elapsed'], max_seconds=row['elapsed'],
            quick_only=row['quick_resolved'], quick_stable=True, final_n=row['final_n'],
            future_vertices=row['future_vertices'], frontend_calls=row['frontend_calls'],
            repeats=[row], case_has_conflicting_certificates=len(decisions[row['case_index']]) > 1)
    folder = ROOT / 'results/holdout_ablation'
    same(read(folder/'ablation_rep0_per_case.json'), list(data.values()))
    report = read(folder/'ablation_rep0_summary.json')
    same(report['trials'], 144); same(report['cases'], 48)
    same(report['repeats_per_case_method'], 1)
    same(report['bootstrap']['enabled'], False)
    same(report['main_table_complete'], False)
    same(report['run_metadata']['main_source'], read(ROOT/'results/holdout_main/run_metadata.json'))
    same(report['run_metadata']['auxiliary_source'], read(folder/'run_metadata.json'))
    same(report['analysis_source_sha256'], digest(ROOT/'analysis/analyze_comparison.py'))
    same(report['ablation_source_sha256'], digest(ROOT/'analysis/analyze_ablation.py'))
    same(report['method_sources'], {
        'zhao2t3_structured': {'tag': 'holdout_main', 'rep': 0},
        'zhao2t3_sturm': {'tag': 'holdout_ablation', 'rep': 0},
        'zhao2t3_sparse': {'tag': 'holdout_ablation', 'rep': 0}})
    # Original record order matters for preserved diagnostic lists.
    ordered = sorted((r for r in reduced.values() if r['source_tag'] == 'holdout_main'),
                     key=lambda r:r['line_number'])
    ordered += sorted((r for r in reduced.values() if r['source_tag'] == 'holdout_ablation'),
                      key=lambda r:r['line_number'])
    same(report['late_certificates'], [r for r in ordered if r['resolved'] and not r['resolved_within_budget']])
    same(report['errors'], [r for r in ordered if r['status'] == 'ERROR' or r['error_traceback']])
    same(report['status_variation'], [])
    same(report['quick_status_variation'], [])
    same(report['conflicting_certificates'], [])
    methods = ('zhao2t3_structured', 'zhao2t3_sturm', 'zhao2t3_sparse')
    for method in methods:
        claim = report['methods'][method]
        vals = [v for (i,m),v in data.items() if m == method]
        rows = [r for (i,m),r in reduced.items() if m == method]
        independent.method_stats(claim, vals, '', False)
        engines = Counter()
        for row in rows:
            engines.update(row['deletion_engines'])
        expected = dict(statuses_all_trials=dict(Counter(r['status'] for r in rows)),
            overrun_trials=sum(r['overshoot_seconds'] > 0 for r in rows),
            max_overshoot_seconds=max(r['overshoot_seconds'] for r in rows),
            late_certificate_trials=sum(r['resolved'] and not r['resolved_within_budget'] for r in rows),
            error_trials=sum(r['status'] == 'ERROR' for r in rows),
            deletion_engines_rep0=dict(engines),
            unknown_trial_cost_seconds=sum(r['elapsed'] for r in rows if r['status'] == 'UNKNOWN'))
        for field,want in expected.items():
            same(claim[field],want,field)
        for field,label in [('group','groups'), ('family','families'), ('threshold_policy','threshold_policies')]:
            same(set(claim[label]), {v[field] for v in vals})
            for group in claim[label]:
                independent.method_stats(claim[label][group], [v for v in vals if v[field] == group], '', False)
    pairs = [('zhao2t3_structured','zhao2t3_sturm'),
             ('zhao2t3_structured','zhao2t3_sparse'),
             ('zhao2t3_sturm','zhao2t3_sparse')]
    same(set(report['paired']), {a+'__vs__'+b for a,b in pairs})
    for a,b in pairs:
        claim=report['paired'][a+'__vs__'+b]
        same((claim['baseline'],claim['candidate'],claim['category']), (a,b,'ablation'))
        values=[(data[i,a],data[i,b]) for i in range(48)]
        independent.paired_stats(claim,values,'',False,True)
        for field,label in [('group','groups'), ('family','families'), ('threshold_policy','threshold_policies')]:
            same(set(claim[label]),{x[field] for x,y in values})
            for group in claim[label]:
                independent.paired_stats(claim[label][group],[(x,y) for x,y in values if x[field] == group],'',False)
    result=dict(status='PASS',selected_records=144,cases=48,methods=3,paired_comparisons=3,
        checks=dict(independent.CHECKS),source_bindings=sources,
        statistics_audit_sha256=digest(ROOT/'qa/AUDIT_statistics.json'),
        summary_sha256=digest(folder/'ablation_rep0_summary.json'),
        per_case_sha256=digest(folder/'ablation_rep0_per_case.json'),
        checker_sha256=digest(Path(__file__)),
        scope='Independent subset selection, per-case reconstruction, all method and paired stratified arithmetic from hash-bound independently audited raw-record reductions; no producer import.',
        limitations='One run per case/method across separate run tags. No confidence intervals, repeat-level timing inference, machine-state drift estimate, or universal ablation claim.')
    (ROOT/'qa/AUDIT_ablation_rep0.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
