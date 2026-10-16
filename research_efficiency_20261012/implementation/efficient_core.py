"""Same-grid baseline and filters with a shared, packet-based time allowance."""
from fractions import Fraction as F
import certified_core as cc
import polynomial_contract as pc
import sparse_faces as sf

METHODS = ('eager_chord','lazy_chord','interval_faces','curvature_faces','sturm_faces','hybrid_faces')

def certify(case,n,nt,method,collect=True):
    if method not in METHODS:
        raise ValueError(method)
    case=dict(case)
    for key in ('initial','prior'):
        case[key]=tuple(map(F,case[key]))
    for key in ('past','future','packet'):
        case[key]=[tuple(map(F,item)) for item in case[key]]
    for key in ('H','rho','limit'):
        case[key]=F(case[key])
    sparse = sf.prepare(case) if method in ('interval_faces','curvature_faces','hybrid_faces') else None
    polynomial = pc.prepare(case,'aligned_sturm') if method in ('sturm_faces','hybrid_faces') else None
    bounds=cc.history_bounds(case);lo,hi=case['prior'];H=case['H'];rho=case['rho']
    times=[H*F(j,nt) for j in range(nt+1)]
    grid=[lo+(hi-lo)*F(j,n) for j in range(n+1)]
    U=max(abs(u) for dt,u in case['future'])
    A_packet=max(abs(c) for c in case['packet'][2])
    allowance=(max(A_packet,U)+rho*hi)*(H/nt)**2/8
    states={};queries={};deleted=[];cells={}
    def state(i):
        if i not in states:
            states[i]=cc.endpoint(grid[i],case['initial'],case['past'])
        return states[i]
    def query(i):
        if i not in queries:
            queries[i]=cc.timeline(grid[i],state(i),case['future'],times,rho)
        return queries[i]
    if method=='eager_chord':
        for i in range(n+1):
            query(i)
    for i in range(n):
        rejection=None
        if polynomial is not None:
            cert=pc.reject_cell(polynomial,grid[i],grid[i+1])
            if cert is not None:
                rejection=dict(engine='sturm',certificate=cert)
        if rejection is None and sparse is not None:
            cert=sf.reject_cell(sparse,grid[i],grid[i+1],method!='interval_faces')
            if cert is not None:
                rejection=dict(engine='sparse',certificate=cert)
        if rejection is not None:
            deleted.append(dict(index=i,**rejection))
            continue
        cell=cc.make_cell(grid[i],grid[i+1],state(i),state(i+1),case['packet'],bounds['M'])
        if cell is not None:
            cells[i]=cell
    needed=sorted({k for i in cells for k in (i,i+1)})
    for i in needed:
        query(i)
    witness=None
    for i in needed:
        st=state(i)
        if all(case['packet'][k][0]<=st[k].lower() and st[k].upper()<=case['packet'][k][1]
               for k in range(3)):
            for s,y in zip(times,query(i)):
                if y[0].lower()>case['limit']:
                    witness=dict(theta=grid[i],time=s,position_lower=y[0].lower())
                    break
        if witness:
            break
    records=[]
    for i,cell in cells.items():
        node=max(cc.chord_upper(cell,query(i)[j][0],query(i+1)[j][0],s,bounds,U,rho)
                 for j,s in enumerate(times))
        records.append(dict(index=i,interval=[cell['l'],cell['r']],projection=cell['gamma'],
                            radii=cell['radii'],max_node_upper=node))
    upper=max((c['max_node_upper'] for c in records),default=None)
    if upper is None:
        status='INCONSISTENT';actual_allowance=F(0)
    else:
        actual_allowance=allowance;upper+=allowance
        status='CERTIFIED_SAFE' if upper<=case['limit'] else ('CERTIFIED_UNSAFE' if witness else 'UNKNOWN')
    if status=='CERTIFIED_SAFE' and witness:
        raise ValueError('Conflicting safe and unsafe certificates')
    result=dict(status=status,method=method,n=n,nt=nt,upper=upper,limit=case['limit'],
                time_allowance=actual_allowance,witness=witness,retained=len(cells),
                historical_vertices=len(states),future_vertices=len(queries),
                requested_time_intervals=nt,filtered_cells=len(deleted),
                sparse_terms=sparse['sparse_terms'] if sparse else None,
                sturm_status=polynomial['status'] if polynomial else None)
    if collect:
        result.update(cells=records,deletions=deleted,sparse_model=sparse,sturm_model=polynomial)
    return result
