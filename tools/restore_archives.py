"""Restore original raw streams in a NEW output directory (about 2.2 GB)."""
import argparse,hashlib
from pathlib import Path
from repository_data import STREAMS,blocks
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
for rel,meta in STREAMS.items():
    out=a.output/rel;out.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
    with out.open('xb') as f:
        for b in blocks(rel):f.write(b);h.update(b);n+=len(b)
    assert h.hexdigest()==meta['sha256'] and n==meta['bytes'],rel
print('Restored and verified',len(STREAMS),'original byte streams to',a.output)
