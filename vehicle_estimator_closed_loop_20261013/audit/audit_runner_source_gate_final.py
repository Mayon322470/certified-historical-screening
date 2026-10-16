"""Read-only pre-freeze gate: source/adapter algebra and declared 20-case protocol.

AST/source assertions enforce a narrow reviewed interface; they are not a
general-purpose proof that arbitrary Python has no information side channels.
"""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runner=ROOT/'integration'/'run_estimator_closed_loop.py'
    sensor=ROOT/'source'/'sensor_estimators.py'
    protocol=ROOT/'inputs'/'PROTOCOL_DRAFT.json'
    text=runner.read_text(encoding='utf-8')
    tree=ast.parse(text)
    plan=json.loads(protocol.read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json').read_text(encoding='utf-8'))
    for filename,digest in frozen['source_sha256'].items():
        assert sha(filename)==digest, 'Frozen final source drift: '+filename
    assert frozen['protocol_sha256']==sha(protocol)
    checks=0
    methods={'ma','ma_structured','zhao2t3','zhao2t3_structured','exact_range'}
    assert set(plan['methods'])==methods
    assert len(plan['cases'])==20
    assert len({c['id'] for c in plan['cases']})==20
    checks+=3
    groups=defaultdict(list)
    for case in plan['cases']:
        groups[case['scene_id']].append(case)
    assert len(groups)==4
    checks+=1
    for cases in groups.values():
        assert {c['method'] for c in cases}==methods
        reference={k:v for k,v in cases[0].items() if k not in ('id','method')}
        assert all({k:v for k,v in c.items() if k not in ('id','method')}==reference for c in cases)
        checks+=2
    assert plan['compute_budget_s']==plan['response_delay_s']==plan['decision_period_s']==.1
    assert plan['native_dt_s']==.0005 and plan['tracker_dt_s']==.01
    assert plan['max_observation_age_s']==.02
    assert plan['emergency_command']==[0.,2.5,0.]
    assert plan['measurement_pad_m']==.005
    assert plan['checks']['waiting_acceleration_upper_mps2']==2.5
    assert '0.2s' in plan['availability_model'] and 'No wall-clock' in plan['availability_model']
    checks+=7
    callargs=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if isinstance(node.func.value,ast.Name) and node.func.value.id=='estimators':
                if node.func.attr=='estimate':
                    args=[ast.unparse(a) for a in node.args]
                    assert args==["case['method']","rows","plan['prior']","deadline"]
                    callargs.append(args)
                    checks+=1
                if node.func.attr=='guard_bounds':
                    args=[ast.unparse(a) for a in node.args]
                    assert args in (["latest_packet","plan['prior']"],
                                    ["latest_packet","estimate['b_interval']"])
                    checks+=1
    assert len(callargs)==1
    checks+=1
    assert "history_packets[max(0, n-3+j)]['rows'][j]" in text
    assert "history_packets.append(packet)" in text
    assert "if packet is not None:" in text
    assert "age > plan['max_observation_age_s']" in text
    assert "pending_at = t+plan['response_delay_s']" in text
    assert "decision_status not in ('ACCEPT_NORMAL', 'QUICK_PRIOR_ACCEPT')" in text
    assert "measured > plan['compute_budget_s']" in text
    assert "fault['kind'] == 'drop'" in text and "fault['kind'] == 'late'" in text
    assert "for case in plan['cases']:" in text
    assert "if RESULTS.exists():" in text
    assert "No overwrite or retry" in text
    assert "native_truth_in_estimator=False" in text
    checks+=12
    component=json.loads((ROOT/'audit'/'ESTIMATOR_COMPONENT_AUDIT.json').read_text(encoding='utf-8'))
    assert component['status']=='PASS_INDEPENDENT_OBSERVATION_ADAPTER_COMPONENT_AUDIT'
    assert component['estimator_source_sha256']==sha(sensor)
    checks+=2
    report={'status':'PASS_INDEPENDENT_FINAL_FROZEN_INTERFACE_GATE', 'checks':checks,
            'source_sha256':{str(p):sha(p) for p in (runner,sensor,protocol,Path(__file__))},
            'planned_cases':20,'same_scene_parameters_across_methods':True,
            'estimate_call_arguments':callargs,
            'qualitative_source_review':[
                'Live clearance and native truth only update audit/diagnostic logs; command veto uses packet-derived state bounds.',
                'Initial s0/stopline and invariant body radius are declared fixed calibration; true initial speed and b do not enter estimator arguments.',
                'Common quick is prior-only and identical for all five methods.',
                'Three histories use fixed latest-packet indexing; drop keeps original latest stamp and fails age before estimation.',
                'Normal request follows the public timeline; veto schedules brake at fixed 0.1s window end. Guard reserves next decision plus response; this is a modeled deadline watchdog, not a concurrent real-time implementation.',
                'All 20 planned cases execute without result-driven selection/retry; partial failures remain in execution records.',
                'Quadratic observation kernel has independent rational point/deletion tests. CarSim waiting/braking membership still requires new sampled logs.'
            ],
            'scope':'Source and frozen-protocol preparation only. No simulator, new trajectory, sampled membership or all-domain proof.'}
    report.update(plan_sha256=sha(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'), original_source_gate_preserved_sha256=sha(ROOT/'audit'/'RUNNER_SOURCE_GATE.json'), revision_note='Original pre-freeze gate/source record is preserved. Final runner adds freeze checks/document metadata and final protocol adds replay metadata. Re-running the original gate rejected its existing output. This final audit binds the frozen final source and protocol before new simulations.')
    output=ROOT/'audit'/'RUNNER_SOURCE_GATE_FINAL.json'
    if output.exists():raise FileExistsError('Preserve prior source gate')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':checks,'planned_cases':20}))


if __name__=='__main__':main()
