"""Freeze diagnostic cases without consulting benchmark outcomes."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,random
import certified_core as cc
from polynomial_contract import encode

ROOT=Path(__file__).resolve().parent.parent
SEED=20261012

def parse(c):
    d=dict(c)
    for k in ('initial','prior'):
        d[k]=tuple(map(F,c[k]))
    for k in ('past','packet','future'):
        d[k]=[tuple(map(F,v)) for v in c[k]]
    for k in ('H','rho','limit'):
        d[k]=F(c[k])
    return d

def fixtures():
    cases=[parse(c) for c in json.loads((ROOT/'inputs/previous_fixtures.json').read_text())]
    rng=random.Random(SEED)
    # Twelve varied cases, predetermined independently of algorithm outcomes.
    for j in range(12):
        N=(1,2,4,8)[j%4]
        past=[(F(rng.randint(1,5),rng.choice((3,5,7,11))),F(rng.randint(-8,8),4)) for _ in range(N)]
        initial=(F(0),F(1),F(rng.randint(-4,4),4))
        truth=cc.endpoint(F(3,2),initial,past)
        radii=(F(1,1000),F(1,20),F(1))[j//4]
        packet=[]
        for st in truth:
            center=F(round(st.midpoint()*10**6),10**6)
            packet.append((center-radii,center+radii))
        rho=F(j%3,100);H=F(1)
        # A broad predeclared position bound focuses these cases on screening cost.
        limit=packet[0][1]+abs(packet[1][1])+max(abs(x) for x in packet[2])+2
        cases.append(dict(name=f'varied_rational_{j:02d}_N{N}',group='varied_rational',
            initial=initial,past=past,prior=(F(1,2),F(3)),packet=packet,
            future=[(F(1,3),F(-1,2)),(F(2,3),F(1,2))],H=H,rho=rho,limit=limit))
    # Six nonaligned stationary-valley cases. Same explicit factorized history,
    # but ordinary velocity/position packet bounds remove exact polynomial alignment.
    r0=F(51341711903259202687,10**20);r1=F(60653065971263342360,10**20);k=F(1,10000)
    a3=F(1);a2=-(2*r1+r0);a1=r1*r1+2*r0*r1;a0=-r0*r1*r1
    qs=[k*z for z in (-a3,a3-a2,a2-a1,a1-a0,a0)]
    xs=F(35,2)+sum(q*(F(9,2)-j) for j,q in enumerate(qs))
    for kind in ('velocity','position'):
        for exponent in (10,16,22):
            eps=F(1,10**exponent)
            packet=[(F(33,2),F(37,2)),(F(6)+eps,F(8)),(F(0),F(2))] if kind=='velocity' else [
                (xs-1,xs-eps),(F(4),F(8)),(F(0),F(2))]
            cases.append(dict(name=f'nonaligned_{kind}_eps_1e-{exponent}',group='nonaligned_valley',
                initial=(F(0),F(1),F(1)),past=[(F(1),1+q) for q in qs],prior=(F(1),F(3)),
                packet=packet,future=[(F(1),F(0))],H=F(1),rho=F(0),limit=F(23907,1000)))
    return cases

def main():
    cases=fixtures()
    data=(json.dumps(encode(cases),indent=2)+'\n').encode()
    output=ROOT/'inputs/fixtures.json'
    if output.exists():
        raise FileExistsError('Frozen fixtures already exist; preserve them')
    output.write_bytes(data)
    meta=dict(seed=SEED,cases=len(cases),sha256=hashlib.sha256(data).hexdigest(),
        generation='25 prior + 12 predetermined varied rational + 6 nonaligned convex-valley cases; no benchmark-outcome selection',
        primary_grid=[32,64],long_horizon_time_grid=[32,256],
        long_workload_case_indices=[0,8,16,18,25,28,33,36,37,40],repetitions=3)
    (ROOT/'inputs/FROZEN_INPUTS.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
