"""Read original byte streams from the lossless public archive."""
from pathlib import Path
import gzip,json
ROOT=Path(__file__).resolve().parents[1]
PROVENANCE=json.loads((ROOT/'PROVENANCE.json').read_text(encoding='utf-8'))
STREAMS={r['source']:r for r in PROVENANCE['compressed_streams']}
def blocks(relative):
    relative=str(relative).replace('\\','/')
    if relative in STREAMS:
        for part in STREAMS[relative]['parts']:
            with gzip.open(ROOT/part,'rb') as f:
                yield from iter(lambda:f.read(1024*1024),b'')
    else:
        with (ROOT/relative).open('rb') as f:yield from iter(lambda:f.read(1024*1024),b'')
def lines(relative):
    carry=b''
    for block in blocks(relative):
        complete=carry+block;rows=complete.split(b'\n');carry=rows.pop()
        for row in rows:yield (row+b'\n').decode('utf-8')
    if carry:yield carry.decode('utf-8')
def records(relative):
    for line in lines(relative):
        if line.strip():yield json.loads(line)
def read_json(relative):return json.loads((ROOT/relative).read_text(encoding='utf-8-sig'))
