"""Rational-input, outward fixed-point certificates; no floating LP or simulator.

All interval endpoints are integers divided by 2**BITS. Exponential bounds
come from alternating rational Taylor sums on [0, 1/2] and interval squaring.
"""
from fractions import Fraction as F
from functools import lru_cache
from dataclasses import dataclass

BITS = 96
S = 1 << BITS
EXP_CERTIFICATES = {}

def ceildiv(n, d):
    return -((-n) // d)

@dataclass(frozen=True)
class IV:
    lo: int
    hi: int
    def __post_init__(self):
        assert self.lo <= self.hi
    @staticmethod
    def of(x):
        if isinstance(x, IV): return x
        x = F(x)
        return IV((x.numerator*S)//x.denominator,
                  ceildiv(x.numerator*S,x.denominator))
    def __add__(self, y):
        y = IV.of(y); return IV(self.lo+y.lo,self.hi+y.hi)
    __radd__ = __add__
    def __neg__(self): return IV(-self.hi,-self.lo)
    def __sub__(self,y): return self+-IV.of(y)
    def __rsub__(self,y): return IV.of(y)+-self
    def __mul__(self,y):
        y=IV.of(y)
        p=[self.lo*y.lo,self.lo*y.hi,self.hi*y.lo,self.hi*y.hi]
        return IV(min(p)//S,ceildiv(max(p),S))
    __rmul__=__mul__
    def __truediv__(self,y):
        y=IV.of(y); assert not y.lo<=0<=y.hi
        p=[F(a,b) for a in (self.lo,self.hi) for b in (y.lo,y.hi)]
        l=min(p);u=max(p)
        return IV(l.numerator*S//l.denominator,ceildiv(u.numerator*S,u.denominator))
    def lower(self): return F(self.lo,S)
    def upper(self): return F(self.hi,S)
    def midpoint(self): return F(self.lo+self.hi,2*S)
    def radius(self): return F(self.hi-self.lo,2*S)
    def serialized(self): return [str(self.lo),str(self.hi)]

@lru_cache(maxsize=200000)
def exp_neg(x):
    x=F(x); assert x>=0
    original=x
    if not x:
        ans=IV.of(1);EXP_CERTIFICATES[str(original)]=ans.serialized();return ans
    k=0
    while x>F(1,2): x/=2;k+=1
    total=F(1);term=F(1);n=0
    while True:
        previous=total;n+=1;term=term*(-x)/n;total+=term
        if abs(term)<F(1,1<<(BITS+8)):
            ans=IV(IV.of(min(total,previous)).lo, IV.of(max(total,previous)).hi)
            break
    for _ in range(k): ans=ans*ans
    EXP_CERTIFICATES[str(original)]=ans.serialized()
    return ans

def flow(theta,state,dt,u,rho=F(0)):
    """Exact constant-command/disturbance flow, enclosed with directed rounding."""
    theta,dt,u,rho=map(F,(theta,dt,u,rho))
    if dt==0: return tuple(map(IV.of,state))
    x,v,a=map(IV.of,state)
    e=exp_neg(dt/theta);one=1-e;eq=IV.of(u+theta*rho)
    delta=a-eq
    return (x+v*dt+eq*(dt*dt/2)+delta*(IV.of(theta*dt)-theta*theta*one),
            v+eq*dt+delta*theta*one, eq+delta*e)

def endpoint(theta,initial,segments):
    state=tuple(map(IV.of,initial))
    for dt,u in segments: state=flow(theta,state,dt,u)
    return state

def timeline(theta,state,segments,times,rho=F(0)):
    """Propagate through both command-change and requested evaluation times."""
    out=[];clock=F(0);j=0;end=F(segments[0][0]);state=tuple(map(IV.of,state))
    for t in times:
        while clock<t:
            while clock==end and j+1<len(segments):
                j+=1;end+=F(segments[j][0])
            stop=min(t,end)
            assert stop>clock, "future input record shorter than query"
            state=flow(theta,state,stop-clock,F(segments[j][1]),rho)
            clock=stop
        out.append(state)
    return out

def b_kernel(theta,s):
    return IV.of(theta*s)-theta*theta*(1-exp_neg(s/theta))

def history_bounds(case):
    tm=F(case['prior'][0]);T=sum(t for t,u in case['past'])
    accelerations=[case['initial'][2]]+[u for t,u in case['past']]
    A0=max(map(abs,accelerations));D=max(accelerations)-min(accelerations);e=min(F(1),T/tm)
    La=D*e/tm;Ma=2*D*e*(1+e)/(tm*tm)
    return dict(T=T,A0=A0,La=La,M=(T*T*Ma/2,T*Ma,Ma))

def clip_affine(interval,l,r,y0,y1,lower,upper):
    """Exact projection of lower <= endpoint chord <= upper."""
    a,b=interval
    slope=(y1-y0)/(r-l)
    if slope==0:
        return (a,b) if lower<=y0<=upper else None
    c0=l+(lower-y0)/slope;c1=l+(upper-y0)/slope
    a=max(a,min(c0,c1));b=min(b,max(c0,c1))
    return None if a>b else (a,b)

def make_cell(l,r,left,right,packet,M):
    width=r-l
    mids0=[v.midpoint() for v in left];mids1=[v.midpoint() for v in right]
    radii=[M[i]*width*width/8+max(left[i].radius(),right[i].radius()) for i in range(3)]
    projection=(l,r)
    for i in range(3):
        projection=clip_affine(projection,l,r,mids0[i],mids1[i],packet[i][0]-radii[i],packet[i][1]+radii[i])
        if projection is None: return None
    return dict(l=l,r=r,gamma=projection,mids0=mids0,mids1=mids1,radii=radii)

def linear_value(theta,l,r,x0,x1):
    return ((r-theta)*x0+(theta-l)*x1)/(r-l)

def chord_upper(cell,left_query,right_query,s,bounds,U,rho):
    l=cell['l'];r=cell['r'];w=r-l
    # Safe rational bounds B1<=s, B2<=2, b<=s^2/2 are deliberately used.
    mx,mv,ma=bounds['M'];La=bounds['La'];A0=bounds['A0'];bmax=s*s/2
    Mpsi=mx+s*mv+bmax*ma+2*s*La+2*(A0+U+rho*s)
    high=max(linear_value(t,l,r,left_query.upper(),right_query.upper()) for t in cell['gamma'])
    rx,rv,ra=cell['radii']
    return high+Mpsi*w*w/8+2*rx+2*s*rv+2*bmax*ra

def box_bounds(cell,packet):
    l,r=cell['l'],cell['r'];a,b=cell['gamma'];result=[]
    for i in range(3):
        vals=[linear_value(t,l,r,cell['mids0'][i],cell['mids1'][i]) for t in (a,b)]
        result.append((max(packet[i][0],min(vals)-cell['radii'][i]),
                       min(packet[i][1],max(vals)+cell['radii'][i])))
    return result

def certify(case,n,nt,method='chord',collect=False):
    """Finite, sound SAFE/UNSAFE/INCONSISTENT/UNKNOWN; no claim of completeness at budget."""
    assert method in ('chord','box')
    bounds=history_bounds(case);lo,hi=case['prior'];H=case['H'];rho=case['rho']
    times=[H*F(j,nt) for j in range(nt+1)]
    grid=[lo+(hi-lo)*F(j,n) for j in range(n+1)]
    U=max(abs(u) for t,u in case['future'])
    states=[endpoint(t,case['initial'],case['past']) for t in grid]
    query=[timeline(t,st,case['future'],times,rho) for t,st in zip(grid,states)]
    # A strict feasible witness with a strict position violation is certified unsafe.
    witness=None
    for t,st,series in zip(grid,states,query):
        if all(case['packet'][i][0]<=st[i].lower() and st[i].upper()<=case['packet'][i][1] for i in range(3)):
            for s,y in zip(times,series):
                if y[0].lower()>case['limit']:
                    witness=dict(theta=str(t),time=str(s),position_lower=str(y[0].lower()))
                    break
        if witness: break
    retained=[];sample_upper=[];cells=[]
    max_acc=F(0)
    for i in range(n):
        cell=make_cell(grid[i],grid[i+1],states[i],states[i+1],case['packet'],bounds['M'])
        if cell is None: continue
        retained.append(i);boxes=box_bounds(cell,case['packet'])
        assert all(a<=b for a,b in boxes)
        max_acc=max(max_acc,abs(boxes[2][0]),abs(boxes[2][1]))
        if method=='chord':
            ubs=[chord_upper(cell,query[i][j][0],query[i+1][j][0],s,bounds,U,rho) for j,s in enumerate(times)]
        else:
            a,b=cell['gamma'];zero=(F(0),F(0),F(0))
            ga=timeline(a,zero,case['future'],times,rho);gb=timeline(b,zero,case['future'],times,rho)
            ubs=[]
            for j,s in enumerate(times):
                be=[b_kernel(a,s),b_kernel(b,s)];acc=boxes[2][1]
                ba=max(v.upper()*acc for v in be) if acc>=0 else max(v.lower()*acc for v in be)
                forced=max(ga[j][0].upper(),gb[j][0].upper())+2*(U+rho*s)*(b-a)**2/8
                ubs.append(boxes[0][1]+s*boxes[1][1]+ba+forced)
        sample_upper.append(max(ubs))
        if collect:
            cells.append(dict(index=i,interval=[str(cell['l']),str(cell['r'])],projection=list(map(str,cell['gamma'])),
                              radii=list(map(str,cell['radii'])),max_node_upper=str(max(ubs))))
    if not retained:
        status='INCONSISTENT';upper=None;allowance=F(0)
    else:
        Af=max(max_acc,U)+rho*hi
        allowance=Af*(H/nt)**2/8
        upper=max(sample_upper)+allowance
        status='CERTIFIED_SAFE' if upper<=case['limit'] else ('CERTIFIED_UNSAFE' if witness else 'UNKNOWN')
    assert not(status=='CERTIFIED_SAFE' and witness is not None)
    result=dict(status=status,n=n,nt=nt,method=method,retained=len(retained),
                upper=str(upper) if upper is not None else None,limit=str(case['limit']),
                time_allowance=str(allowance),witness=witness)
    if collect: result['cells']=cells
    return result
