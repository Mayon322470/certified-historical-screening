"""Synthetic cross-tag ablation checks. Run only after all timed runs finish."""
import io
import json
from collections import Counter
from contextlib import redirect_stdout
import unittest

import analyze_comparison as ac
import analyze_ablation as aa
from test_analyze_comparison import AnalysisContract


class AblationContract(unittest.TestCase):
    make_fixture = AnalysisContract.make_fixture
    write_rows = AnalysisContract.write_rows
    tearDown = AnalysisContract.tearDown

    def setUp(self):
        AnalysisContract.setUp(self)
        # Main full-structure rep0 deliberately differs from its 3-repeat median.
        for row in self.rows:
            if row['case_index'] == 1 and row['method'] == aa.FULL:
                row['elapsed'] = (1., 3., 5.)[row['rep']]
        self.write_rows()
        self.aux = self.root/'results'/'auxiliary'
        self.aux.mkdir()
        meta = json.loads((self.folder/'run_metadata.json').read_text(encoding='utf-8'))
        meta['arguments'].update(methods=','.join(aa.ABLATIONS), reps=1)
        ac.dump(self.aux/'run_metadata.json',meta)
        aux_rows = []
        for row in self.rows:
            if row['method'] != aa.FULL or row['rep'] != 0:
                continue
            for method,multiplier in zip(aa.ABLATIONS,(2.,1.5)):
                item = dict(row,method=method,result=dict(row['result']))
                item['elapsed'] *= multiplier
                item['resolved_within_budget'] = item['resolved'] and item['elapsed'] <= 5
                item['overshoot_seconds'] = max(0.,item['elapsed']-5)
                aux_rows.append(item)
        with (self.aux/'records.jsonl').open('w',encoding='utf-8') as handle:
            for row in aux_rows: handle.write(json.dumps(row)+'\n')
        ac.dump(self.aux/'summary.json',{m:dict(statuses=dict(Counter(r['result']['status'] for r in aux_rows if r['method']==m)))
                                       for m in aa.ABLATIONS})

    def test_rep0_reference_not_main_median(self):
        with redirect_stdout(io.StringIO()): report,data = aa.analyze('synthetic','auxiliary',self.root)
        self.assertEqual(data[1,aa.FULL]['median_seconds'],1.)
        self.assertEqual(data[1,'zhao2t3_sturm']['median_seconds'],2.)
        self.assertEqual(report['repeats_per_case_method'],1)
        self.assertEqual(len(report['paired']),3)
        self.assertFalse(report['bootstrap']['enabled'])
        for pair in report['paired'].values():
            self.assertIsNone(pair['all_task_cost_ratio_ci95'])
            self.assertIsNone(pair['common_solved_geomean_speed_ratio_ci95'])
        self.assertEqual(report['method_sources'][aa.FULL],dict(tag='synthetic',rep=0))

    def test_production_hash_difference_rejected(self):
        path = self.aux/'run_metadata.json'
        meta = json.loads(path.read_text(encoding='utf-8'))
        meta['implementation_sha256'] = {'unexpected.py':'changed'}
        ac.dump(path,meta)
        with self.assertRaisesRegex(ac.AnalysisError,'implementation hashes differ'):
            aa.analyze('synthetic','auxiliary',self.root)


if __name__ == '__main__':
    # The main-analyzer checks are run separately.
    names = ('test_rep0_reference_not_main_median','test_production_hash_difference_rejected')
    suite = unittest.TestSuite(AblationContract(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
