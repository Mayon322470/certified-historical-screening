"""Declared digital observation model; no learned CarSim dynamics.

Only Sensor sees geometry/speed truth. The estimators receive three original
quadratic observation packets with one common unknown odometer/speed bias.
Ma/Zhao are independent observation-task adapters, not the authors' software.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from fractions import Fraction as Q
from pathlib import Path
import math
import sys
import time

OLD=Path(__file__).resolve().parents[2]/'research_task_face_budget_20261013'/'implementation'
sys.path.insert(0,str(OLD))
import certified_core as cc
import zhao_carleman as zhao
import rank1_ma as ma

METHODS=('ma','ma_structured','zhao2t3','zhao2t3_structured','exact_range')
PRIOR=(-.8,.8)
GRID=8
ODOM_PAD=.02
SPEED_PAD=.01
RANGE_PAD=.04
NUMERIC_RANGE_PAD=.000001

def rational(x):
    return x if isinstance(x,Q) else Q.from_float(float(x))

def wire(x):
    if isinstance(x,Q):return str(x)
    if isinstance(x,dict):return {str(k):wire(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [wire(v) for v in x]
    return x

class Sensor:
    """Reproducible bounded digital telemetry, not camera/radar qualification.

    CARLA supplies live along-route CG position. Ranges are ideal transverse-
    compensated lane ranges to three declared landmarks; not rendered pixels.
    The common low-level tracker has its own native telemetry, unavailable to
    the high-level guard and estimators.
    """
    def __init__(self,seed,bias,s0,landmarks):
        self.seed=int(seed);self.bias=float(bias);self.s0=float(s0)
        self.landmarks=[dict(row) for row in landmarks]
        if not PRIOR[0]<=self.bias<=PRIOR[1] or len(landmarks)!=3:
            raise ValueError('Bias prior and three landmarks required')
        self.metadata=dict(seed=self.seed,bias_mps=self.bias,s0_m=self.s0,
            landmarks=self.landmarks,odom_error_bound_m=ODOM_PAD,
            speed_error_bound_mps=SPEED_PAD,range_error_bound_m=RANGE_PAD,
            numerical_range_padding_m=NUMERIC_RANGE_PAD,
            sensor_role='Ideal digitally corrupted CARLA lane geometry and native speed; not rendered perception')

    def _noise(self,t,channel,bound):
        # A fixed phase and analytic bounded waveform; never use future states.
        phase=((self.seed*17+channel*31)%997)/997.*2*math.pi
        return .8*bound*math.sin(1.37*float(t)+phase)

    def observe(self,obs,native_speed,time):
        t=float(time);s=float(obs['cg_route_s'])
        odom=s-self.s0+self.bias*t+self._noise(t,0,ODOM_PAD)
        speed=float(native_speed)+self.bias+self._noise(t,1,SPEED_PAD)
        rows=[]
        total=RANGE_PAD+ODOM_PAD+NUMERIC_RANGE_PAD
        for j,landmark in enumerate(self.landmarks):
            c2=float(landmark['lateral'])**2
            r=math.sqrt((s-float(landmark['s']))**2+c2)+self._noise(t,j+2,RANGE_PAD)
            rows.append(dict(stamp_s=t,a=self.s0+odom-float(landmark['s']),c2=c2,
                packet=[max(0.,r-total)**2,(r+total)**2],landmark_id=j,
                measured_range_m=r))
        return dict(stamp_s=t,odom_m=odom,speed_mps=speed,s0_m=self.s0,rows=rows)

def guard_bounds(packet,b_interval):
    if b_interval is None:return dict(valid=False,reason='NO_BIAS_OUTER')
    lo,hi=map(float,b_interval);t=float(packet['stamp_s'])
    vals=[lo,hi,t,packet['odom_m'],packet['speed_mps'],packet['s0_m']]
    if not all(math.isfinite(float(v)) for v in vals) or lo>hi or t<0:
        return dict(valid=False,reason='INVALID_BIAS_OR_PACKET')
    # Tiny final floating padding is separate from the declared sensor errors.
    p=1e-8
    return dict(valid=True,reason='COMMON_BIAS_OUTER',
        s_lower=packet['s0_m']+packet['odom_m']-hi*t-ODOM_PAD-p,
        s_upper=packet['s0_m']+packet['odom_m']-lo*t+ODOM_PAD+p,
        v_lower=packet['speed_mps']-hi-SPEED_PAD-p,
        v_upper=packet['speed_mps']-lo+SPEED_PAD+p)

def quadratic(row,b):
    a,t,c=map(rational,(row['a'],row['stamp_s'],row['c2']))
    return (a-t*b)**2+c

def quadratic_range(row,lo,hi):
    a,t,c=map(rational,(row['a'],row['stamp_s'],row['c2']))
    ends=[(a-t*lo)**2+c,(a-t*hi)**2+c]
    lower=min(ends)
    if t and lo<=a/t<=hi:lower=c
    return lower,max(ends)

def jet(rows,center,lo,hi):
    out=[]
    for degree in range(4):
        values=[]
        for row in rows:
            a,t,c=map(rational,(row['a'],row['stamp_s'],row['c2']))
            if degree==0:
                l,h=quadratic_range(row,lo,hi)
            elif degree==1:
                l,h=sorted((-2*t*(a-t*lo),-2*t*(a-t*hi)))
            elif degree==2:l=h=2*t*t
            else:l=h=Q(0)
            values.append(cc.IV(cc.IV.of(l).lo,cc.IV.of(h).hi))
        lead=(lo-center,hi-center) if degree==0 else (Q(1),Q(1)) if degree==1 else (Q(0),Q(0))
        out.append([cc.IV(cc.IV.of(lead[0]).lo,cc.IV.of(lead[1]).hi),*values])
    return out

def ma_contract(rows,lo,hi):
    center=(lo+hi)/2;R=(hi-lo)/2
    slopes=[];widths=[];predicted=[];measure=[];strips=[]
    for row in rows:
        a,t=map(rational,(row['a'],row['stamp_s']))
        value=quadratic(row,center);C=-2*t*(a-t*center)
        remainder=t*t*R*R
        low,high=map(rational,row['packet'])
        slopes.append(C);widths.append((high-low+remainder)/2)
        predicted.append(value+remainder/2);measure.append((low+high)/2)
        strips.append(dict(C=C,low=low-value-remainder+C*center,
                           high=high-value+C*center))
    N,postR,gain=ma.optimal_gain(R,slopes,widths,[Q(0)]*3,Q(0))
    postC=center+sum((n*(y-h) for n,y,h in zip(N,measure,predicted)),Q(0))
    retained=(max(lo,postC-postR),min(hi,postC+postR))
    projected,project=ma.project_strips(strips,[Q(0)]*3,Q(0),(lo,hi))
    if projected is not None:
        retained=(max(retained[0],projected[0]),min(retained[1],projected[1]))
    rejected=projected is None or retained[0]>retained[1]
    return None if rejected else retained,dict(kind='ma2026_scalar_quadratic_observation',
        gain=wire(gain),posterior_center=str(postC),posterior_radius=str(postR),
        strips=wire(strips),projection=wire(project),strengthened_all_strips=True,
        retained=wire(retained),rejected=rejected)

def zhao_contract(rows,lo,hi):
    center=(lo+hi)/2;R=(hi-lo)/2
    packets=[list(map(rational,row['packet'])) for row in rows]
    base,tight,cert=zhao.from_jets(jet(rows,center,center,center),
        [jet(rows,center,lo,hi)],R,packets,nu=2,tighten=True,collect=True)
    retained=(max(lo,center+base[0]),min(hi,center+base[1]))
    if tight is not None:
        retained=(max(retained[0],center+tight[0]),min(retained[1],center+tight[1]))
    rejected=retained[0]>retained[1]
    cert.update(b_input=wire((lo,hi)),b_center=str(center),
                b_retained=wire(retained),rejected=rejected)
    return None if rejected else retained,wire(cert)

def sqrt_bounds(q):
    """Exact rational enclosure: lower^2<=q<=upper^2, width 1e-12."""
    q=rational(q)
    if q<0:raise ValueError('negative radicand')
    scale=10**12
    n=math.isqrt((q.numerator*scale*scale)//q.denominator)
    return Q(n,scale),Q(n+1,scale)

def intersect_union(left,right):
    result=[]
    for a,b in left:
        for c,d in right:
            l,h=max(a,c),min(b,d)
            if l<=h:result.append((l,h))
    result.sort()
    merged=[]
    for l,h in result:
        if merged and l<=merged[-1][1]:merged[-1]=(merged[-1][0],max(h,merged[-1][1]))
        else:merged.append((l,h))
    return merged

def exact_ranges(rows,prior):
    domains=[prior];cert=[]
    for row in rows:
        a,t,c=map(rational,(row['a'],row['stamp_s'],row['c2']))
        low,high=map(rational,row['packet'])
        if t==0:
            if not low<=a*a+c<=high:domains=[]
            cert.append(dict(zero_stamp=True));continue
        if high<c:domains=[];break
        upper=sqrt_bounds(high-c)[1]
        lower=sqrt_bounds(max(Q(0),low-c))[0]
        intervals=[((a-upper)/t,(a-lower)/t),((a+lower)/t,(a+upper)/t)]
        domains=intersect_union(domains,intervals)
        cert.append(dict(sqrt_lower=str(lower),sqrt_upper=str(upper),
                         inverse_intervals=wire(intervals)))
        if not domains:break
    return domains,cert

def estimate(method,rows,prior=PRIOR,deadline=None):
    started=time.perf_counter()
    if method not in METHODS or len(rows)!=3:raise ValueError('Frozen method and three observations required')
    lo,hi=map(rational,prior)
    if not lo<hi or any(float(row['stamp_s'])<0 for row in rows):raise ValueError('Invalid prior/history')
    result=dict(method=method,status='STARTED',b_interval=None,cells_completed=0,
                frontend_calls=0,deleted=0,grid_cells=GRID,certificates=[])
    def finish(status,interval):
        result.update(status=status,b_interval=list(map(float,interval)) if interval else None,
            elapsed_s=time.perf_counter()-started,
            b_interval_rational=wire(interval) if interval else None)
        if deadline is not None and time.perf_counter()>deadline:
            result.update(status='UNKNOWN',b_interval=None,deadline_hit=True)
        else:result['deadline_hit']=False
        return result
    if all(rational(row['stamp_s'])==0 for row in rows):
        compatible=all(rational(row['packet'][0])<=quadratic(row,Q(0))<=rational(row['packet'][1]) for row in rows)
        return finish('OK' if compatible else 'INCONSISTENT',(lo,hi) if compatible else None)
    if method=='exact_range':
        domains,cert=exact_ranges(rows,(lo,hi))
        result.update(certificates=cert,inverse_components=wire(domains),cells_completed=GRID)
        return finish('OK' if domains else 'INCONSISTENT',(domains[0][0],domains[-1][1]) if domains else None)
    kept=[]
    for i in range(GRID):
        if deadline is not None and time.perf_counter()>=deadline:return finish('UNKNOWN',None)
        l=lo+(hi-lo)*Q(i,GRID);h=lo+(hi-lo)*Q(i+1,GRID)
        rejected=None
        if method.endswith('_structured'):
            for j,row in enumerate(rows):
                lower,upper=quadratic_range(row,l,h)
                p,q=map(rational,row['packet'])
                if upper<p or lower>q:
                    rejected=dict(cell=i,stage='quadratic_history_face',row=j,
                        b_input=wire((l,h)),quadratic_range=wire((lower,upper)),packet=wire((p,q)))
                    break
        if rejected is not None:
            result['deleted']+=1;result['certificates'].append(rejected)
        else:
            interval,cert=(ma_contract(rows,l,h) if method.startswith('ma') else zhao_contract(rows,l,h))
            result['frontend_calls']+=1
            result['certificates'].append(dict(cell=i,stage='literature_frontend',certificate=cert))
            if interval is None:result['deleted']+=1
            else:kept.append(interval)
        result['cells_completed']+=1
    return finish('OK' if kept else 'INCONSISTENT',
                  (min(r[0] for r in kept),max(r[1] for r in kept)) if kept else None)
