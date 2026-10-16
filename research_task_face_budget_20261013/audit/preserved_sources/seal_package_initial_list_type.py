"""Create the final byte manifest once, after reports and status backup exist."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import verify_package

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT/'PACKAGE_MANIFEST.json'
    assert not path.exists(), 'Sealed manifest cannot be overwritten'
    dependencies = {}
    # Audit provenance includes the independently reused read-only old modules.
    for report in (ROOT/'audit').glob('*.json'):
        data = json.loads(report.read_text(encoding='utf-8'))
        source_hashes = data.get('source_sha256', {})
        # Some small self-tests store one script digest as a string. It is not
        # a path-to-digest dependency map; the report and source remain pinned
        # by the complete local manifest.
        if not isinstance(source_hashes, dict):
            source_hashes = {}
        for name, expected in source_hashes.items():
            dep = Path(name)
            if dep.is_absolute() and not dep.is_relative_to(ROOT):
                assert dep.is_file()
                actual = hashlib.sha256(dep.read_bytes()).hexdigest()
                assert actual == expected, ('External source drift', name)
                if str(dep) in dependencies:
                    assert dependencies[str(dep)] == expected
                dependencies[str(dep)] = expected
    files = {f.relative_to(ROOT).as_posix():dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)
             for f in sorted(ROOT.rglob('*')) if f.is_file() and f != path}
    record = dict(status='SEALED', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  files=files, external_dependencies=dependencies, source_exemptions=0)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(verify_package.verify(), indent=2))


if __name__ == '__main__':
    main()
