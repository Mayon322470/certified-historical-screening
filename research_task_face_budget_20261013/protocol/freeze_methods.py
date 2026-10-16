"""One-shot method/analysis/protocol registration before new input generation."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (ROOT/'protocol'/'frozen_suite').exists(), 'Formal inputs must not precede registration'
    path = ROOT/'protocol'/'METHODS_FROZEN.json'
    assert not path.exists(), 'Frozen registration cannot be overwritten'
    required = [ROOT/'analysis'/'analyze_comparison.py', ROOT/'analysis'/'STATS_RULES_CN.txt']
    required += [ROOT/'audit'/p for p in ('RANK1_CONTRACT_AUDIT.json',
                                         'COUPLING_AND_SERIALIZATION_AUDIT.json',
                                         'BACKEND_DEVELOPMENT_AUDIT.json')]
    for f in required:
        assert f.is_file(), ('Missing prerequisite', str(f))
    for f in required[2:]:
        assert json.loads(f.read_text(encoding='utf-8'))['status'] == 'PASS', str(f)
    sources = sorted((ROOT/'implementation').glob('*.py'))
    sources += sorted((ROOT/'theory').glob('*.py'))
    sources += [ROOT/'theory'/'SHARED_TAU_RESPONSE_INTERFACE_CN.txt']
    sources += sorted((ROOT/'protocol').glob('*.py'))
    sources += [ROOT/'protocol'/'FINITE_COMPARISON_PROTOCOL_CN.txt']
    sources += [ROOT/'protocol'/'PROTOCOL_DESIGN.json']
    sources += required[:2]
    record = dict(status='METHODS_FROZEN', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        generator_sha256=sha(ROOT/'protocol'/'generate_taskface_suite.py'),
        protocol_sha256=sha(ROOT/'protocol'/'FINITE_COMPARISON_PROTOCOL_CN.txt'),
        source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in sources},
        prefreeze_validation_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in required[2:]},
        configuration=dict(methods=['ma_rank1', 'ma_rank1_structured', 'zhao1_rank1',
                                   'zhao2t3_rank1', 'zhao2t3_rank1_structured'],
                           grids=[8,32,128,512], time_grids=[32,64,128,256], budget_seconds=5.0,
                           repeats=3, order_seed=712913, interval_bits=96, numerical_threads=1,
                           derivative_subdivisions=1, quick_gate=True, collect_every_repeat=True),
        statistics=dict(template_clusters=16, case_count=32, expected_records=480,
                        bootstrap_replicates=10000, bootstrap_seed=20261013947,
                        threshold_for_listed_slowdown=1.05),
        formal_data_generated_before_freeze=False, formal_solver_called_before_freeze=False,
        physical_budget_qualification_claimed=False, python=sys.version)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=record['status'], path=str(path), sha256=sha(path),
                         pinned_sources=len(sources)), indent=2))


if __name__ == '__main__':
    main()
