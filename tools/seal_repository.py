"""Regenerate SHA256SUMS after author-approved repository metadata edits."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts or p.name=='SHA256SUMS':continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    rows.append(h.hexdigest()+'  '+p.relative_to(ROOT).as_posix())
(ROOT/'SHA256SUMS').write_text('\n'.join(rows)+'\n',encoding='utf-8')
print('Sealed',len(rows),'files')
