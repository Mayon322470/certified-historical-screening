"""Synthetic analysis-contract checks; no solver and no real holdout is read."""
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze_comparison as ac

HERE = Path(__file__).resolve().parent

class AnalysisContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_analysis_',dir=HERE)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(HERE)
        (self.root/'inputs').mkdir()
        self.folder = self.root/'results'/'synthetic'
        self.folder.mkdir(parents=True)
        self.make_fixture()

    def tearDown(self):
        assert self.root.is_relative_to(HERE) and self.root.name.startswith('synthetic_analysis_')
        self.temp.cleanup()

    def make_fixture(self,repeats=3):
        self.cases = [dict(name=f'case{i}',group='holdout_random' if i<5 else 'holdout_four_hold') for i in range(10)]
        path = self.root/'inputs'/'synthetic.json'
        ac.dump(path,self.cases)
        ac.dump(self.folder/'run_metadata.json',dict(arguments=dict(inputs='synthetic.json',methods=','.join(ac.MAIN_METHODS),
            reps=repeats,budget=5),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows = []
        for i,case in enumerate(self.cases):
            for method in ac.MAIN_METHODS:
                for rep in range(repeats):
                    status,t,quick,witness = 'CERTIFIED_SAFE',2.,i==0,dict(theta='1')
                    if i==0: t=.1
                    if i==1 and method=='ma_strip': t=[3.,4.,5.][rep]
                    if i==2 and method in ('ma_strip','ma_strip_structured'): t=1. if method=='ma_strip' else 1.2
                    if i==3 and method=='ma_strip': status='UNKNOWN'
                    if i==4 and method=='ma_strip': t=6.
                    if i==5 and method=='ma_strip_structured' and rep==2: status='ERROR'
                    if i==6 and method=='ma_strip_structured' and rep==2: status='CERTIFIED_UNSAFE'
                    if i==7 and method=='ma_strip_structured': quick=True
                    if i==8: witness=None
                    if i==9 and method=='ma_strip': t=[4.9,5.1,4.9][rep]
                    result = dict(status=status,quick_resolved=quick,nonempty_witness=witness,n=8)
                    resolved = status in ('CERTIFIED_UNSAFE','INCONSISTENT') or status=='CERTIFIED_SAFE' and witness is not None
                    rows.append(dict(case_index=i,**case,method=method,rep=rep,result=result,elapsed=t,
                        resolved=resolved,resolved_within_budget=resolved and t<=5,budget_seconds=5,overshoot_seconds=max(0,t-5),
                        stages=[dict(deletions=[dict(engine='sparse',certificate=dict(kind='positive',payload='exact '*1000))])],
                        frontend_calls_total=3,future_vertices_total=4,error_traceback='synthetic traceback' if status=='ERROR' else None))
        self.rows = rows
        self.write_rows()

    def write_rows(self):
        with (self.folder/'records.jsonl').open('w',encoding='utf-8') as handle:
            for row in self.rows: handle.write(json.dumps(row)+'\n')
        ac.dump(self.folder/'summary.json',{m:dict(statuses=dict(Counter(r['result']['status'] for r in self.rows if r['method']==m)))
                                          for m in ac.MAIN_METHODS})

    def analyze(self):
        with redirect_stdout(io.StringIO()): return ac.analyze('synthetic',self.root)

    def test_reduction_eligibility_and_anomalies(self):
        report,data = self.analyze()
        self.assertEqual(report['trials'],180)
        self.assertEqual(len(report['paired']),11)
        p = report['paired']['ma_strip']
        self.assertEqual([v['name'] for v in p['common_solved_details']],['case1','case2','case7'])
        self.assertEqual(data[1,'ma_strip']['median_seconds'],4.)
        self.assertAlmostEqual(p['common_solved_geomean_speed_ratio'],(2/1.2)**(1/3))
        self.assertEqual(p['common_solved_median_speed_ratio'],1.)
        self.assertEqual(p['slower_by_over_5pct'],['case2'])
        self.assertEqual(p['asymmetric_quick_cases'],['case7'])
        self.assertEqual(p['common_quick_cases'],1)
        self.assertFalse(data[9,'ma_strip']['within_budget_all'])
        self.assertLess(data[9,'ma_strip']['median_seconds'],5)
        self.assertEqual(len(report['late_certificates']),4)
        self.assertEqual(len(report['errors']),1)
        self.assertEqual(len(report['conflicting_certificates']),1)
        self.assertEqual(len(report['unresolved_safe_without_nonempty_witness']),18)
        self.assertEqual(report['methods']['ma_strip']['deletion_engines_rep0'],{'sparse:positive':10})
        self.assertNotIn('stages',data[1,'ma_strip']['repeats'][0])
        self.assertNotIn('case6',[v['name'] for v in report['paired']['zhao1__vs__zhao2']['common_solved_details']])
        self.assertEqual(report['bootstrap']['seed'],20261012947)
        self.assertEqual(report['bootstrap']['samples'],10000)
        self.assertIsNotNone(p['common_solved_geomean_speed_ratio_ci95'])

    def test_streaming_and_reproducible_bootstrap(self):
        original = Path.read_text
        def guard(path,*args,**kwargs):
            if path.name=='records.jsonl': raise AssertionError('Whole-file JSONL read forbidden')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):
            first,_ = self.analyze()
            second,_ = self.analyze()
        self.assertEqual(first,second)
        self.assertEqual(ac.Bootstrap(True).statistics([2.,2.],{'mean':lambda a:a.mean(axis=1)},'constant'),{'mean':[2.,2.]})

    def test_missing_repeat_rejected(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ac.AnalysisError,'Incomplete Cartesian'): self.analyze()
        self.assertFalse((self.folder/'paired_summary.json').exists())

    def test_duplicate_repeat_rejected(self):
        self.rows.append(self.rows[0])
        self.write_rows()
        with self.assertRaisesRegex(ac.AnalysisError,'Duplicate case'): self.analyze()

    def test_unexpected_repeat_rejected(self):
        self.rows[0]['rep']=3
        self.write_rows()
        with self.assertRaisesRegex(ac.AnalysisError,'Unexpected case'): self.analyze()

    def test_truncated_json_rejected(self):
        with (self.folder/'records.jsonl').open('a',encoding='utf-8') as handle: handle.write('{"unfinished":')
        with self.assertRaisesRegex(ac.AnalysisError,'truncated JSONL'): self.analyze()

    def test_completion_marker_required(self):
        (self.folder/'summary.json').unlink()
        with self.assertRaisesRegex(ac.AnalysisError,'completion marker'): self.analyze()

    def test_input_hash_required(self):
        with (self.root/'inputs'/'synthetic.json').open('a') as handle: handle.write(' ')
        with self.assertRaisesRegex(ac.AnalysisError,'SHA256'): self.analyze()

    def test_single_repeat_has_no_ci(self):
        self.make_fixture(repeats=1)
        report,_ = self.analyze()
        self.assertFalse(report['bootstrap']['enabled'])
        self.assertIsNone(report['paired']['ma_strip']['all_task_cost_ratio_ci95'])
        self.assertIsNone(report['paired']['ma_strip']['common_solved_geomean_speed_ratio_ci95'])

    def test_public_threshold_assignment_only(self):
        cases=[]
        for i in range(48):
            if i<32: name,group=f'holdout_random_{i:02d}_N{(1,2,4,8)[i//8]}','holdout_random'
            elif i<40: name,group=f'holdout_four_hold_{(i-32)//4}_{i%4}','holdout_four_hold'
            elif i<44: name,group=f'holdout_valley_velocity_{i-40}','holdout_valley_velocity'
            else: name,group=f'holdout_valley_position_{i-44}','holdout_valley_position'
            cases.append(dict(name=name,group=group))
        strata,_=ac.stratification(cases,Path('holdout_48.json'))
        self.assertEqual(Counter(v['threshold_policy'] for v in strata),dict.fromkeys(ac.POLICIES,12))
        self.assertEqual(Counter(v['threshold_policy'] for v in strata if v['family']=='ordinary'),dict.fromkeys(ac.POLICIES,8))
        self.assertEqual(Counter(v['threshold_policy'] for v in strata if v['family']=='structured'),dict.fromkeys(ac.POLICIES,4))

if __name__=='__main__': unittest.main(verbosity=2)
