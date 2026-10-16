"""Compare full structured rep0 with one-run ablations from a separate run tag.

This is an auxiliary single-run comparison, not a comparison between a three-run
median and a single timing. It does not create confidence intervals.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import analyze_comparison as ac

FULL = 'zhao2t3_structured'
ABLATIONS = ('zhao2t3_sturm', 'zhao2t3_sparse')
SETTINGS = ('inputs', 'grids', 'nt', 'nt_schedule', 'hessian_pieces', 'budget', 'quick')


def analyze(main_tag, auxiliary_tag, root=ac.ROOT):
    root = Path(root)
    mm, main_methods, main_reps, main_rows, main_audit = ac.stream_records(root/'results'/main_tag, root)
    am, aux_methods, aux_reps, aux_rows, aux_audit = ac.stream_records(root/'results'/auxiliary_tag, root)
    if FULL not in main_methods or main_reps != 3:
        raise ac.AnalysisError('Ablation reference must be full-structure rep0 from the complete three-repeat main run')
    if set(aux_methods) != set(ABLATIONS) or aux_reps != 1:
        raise ac.AnalysisError('Auxiliary run must contain exactly sturm and sparse, once per case')
    if mm['input_sha256'] != am['input_sha256']:
        raise ac.AnalysisError('Main and auxiliary input hashes differ')
    differing = [key for key in SETTINGS if mm['arguments'].get(key) != am['arguments'].get(key)]
    if differing:
        raise ac.AnalysisError(f'Main and auxiliary timed settings differ: {differing}')
    if mm.get('implementation_sha256') != am.get('implementation_sha256'):
        raise ac.AnalysisError('Main and auxiliary production implementation hashes differ')
    if mm.get('numerical_threads') != am.get('numerical_threads'):
        raise ac.AnalysisError('Main and auxiliary numerical thread settings differ')
    selected = {key:dict(row, source_tag=main_tag) for key,row in main_rows.items()
                if key[1] == FULL and key[2] == 0}
    selected.update({key:dict(row, source_tag=auxiliary_tag) for key,row in aux_rows.items()})
    data = ac.case_summaries(selected, 1)
    cases = sorted({key[0] for key in selected})
    methods = [FULL, *ABLATIONS]
    if set(selected) != {(i,m,0) for i in cases for m in methods}:
        raise ac.AnalysisError('Auxiliary three-column comparison has an incomplete case product')
    metadata = dict(arguments=dict(mm['arguments'], methods=','.join(methods), reps=1),
                    input_sha256=mm['input_sha256'], main_source=mm, auxiliary_source=am)
    audit = dict(complete=True, expected_records=3*len(cases), read_records=len(selected),
                 duplicates=0, missing_records=0, main_source=main_audit, auxiliary_source=aux_audit,
                 selection='Full structured main rep0 only; all auxiliary rows, each rep0')
    report = ac.build_report(auxiliary_tag+'_rep0_comparison', metadata, methods, 1, selected, audit, data)
    report['ablation_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['definitions']['case_time'] = 'One timed run: full structured main rep0; auxiliary sturm/sparse rep0'
    report['definitions']['within_budget_all'] = 'The single selected run resolves nonvacuously within its budget'
    report['definitions']['separate_run_tags'] = (
        'Fixed cases are paired across separate run tags. Execution-order or machine-state drift between '
        'tags is not estimated. No repeat-level or population confidence claim is made.')
    report['method_sources'] = {FULL:dict(tag=main_tag, rep=0),
                                **{m:dict(tag=auxiliary_tag, rep=0) for m in ABLATIONS}}
    destination = root/'results'/auxiliary_tag
    ac.dump(destination/'ablation_rep0_summary.json', report)
    ac.dump(destination/'ablation_rep0_per_case.json', list(data.values()))
    text = ('Auxiliary single-run comparison: full structured main rep0 / sturm / sparse.\n'
            'Separate run tags; no confidence intervals or repeat-level timing claims.\n\n'+ac.table_text(report))
    (destination/'ablation_rep0_tables.txt').write_text(text, encoding='utf-8')
    print(text, end='')
    return report, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main_tag')
    parser.add_argument('auxiliary_tag')
    args = parser.parse_args()
    analyze(args.main_tag, args.auxiliary_tag)
