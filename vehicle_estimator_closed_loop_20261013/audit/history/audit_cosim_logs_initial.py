"""Independent row-by-row audit of completed new CARLA/CarSim cases.

Only CSV/JSON/math are loaded. No simulator or estimator source is imported.
Execution failures and sampled membership failures remain separate from log
integrity. Use --case to review a completed case during a longer owned run.
"""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def tick(t):return round(float(t)*1000000)
def near(a,b,tol=1e-8):return abs(float(a)-float(b))<=tol
def vector(text):return json.loads(text)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))


def reserve(v,period,response,pad):
    v=max(0.,float(v));A=2.5;delay=.3+response
    next_v=v+A*period
    return v*period+.5*A*period**2+next_v*delay+.5*A*delay**2+(next_v+A*delay)**2/4+.25+pad


def prefix(v,t):
    v=max(0.,v);first=min(t,.3);out=v*first+1.25*first*first
    if t>.3:
        peak=v+.75;s=min(t-.3,peak/2)
        out+=peak*s-s*s
    return out+.25


def qbounds(row,lo,hi):
    a,t,c=map(F.from_float,map(float,(row['a'],row['stamp_s'],row['c2'])))
    lo,hi=F(lo),F(hi)
    zl,zh=sorted((a-t*lo,a-t*hi))
    minimum=F(0) if zl<=0<=zh else min(zl*zl,zh*zh)
    return minimum+c,max(zl*zl,zh*zh)+c


def audit_case(case,plan):
    out=ROOT/'cosim_results'/case['id'];result=load(out/'result.json')
    if result['status']=='STARTED':raise RuntimeError('Case is still running: '+case['id'])
    errors=[];checks=0
    def check(cond,label):
        nonlocal checks
        checks+=1
        if not cond and len(errors)<60:errors.append(label)
    check(result['id']==case['id'] and result['method']==case['method'],'case identity')
    check(result['plan_sha256']==sha(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'),'plan hash')
    check(not result['continuous_certificate'] and not result['real_time_wall_clock_guarantee'],'evidence scope flags')
    data={name:rows(out/(name+'.csv')) for name in ('native','policy','sensor','timing')}
    decisions=[json.loads(line) for line in (out/'decision.jsonl').read_text(encoding='utf-8').splitlines() if line]
    for name in data:
        check(len(data[name])==result[name+'_rows'],name+' row count')
        check(sha(out/(name+'.csv'))==result[name+'_sha256'],name+' hash')
    check(sha(out/'decision.jsonl')==result['decision_sha256'],'decision hash')
    if result['status']=='FAILED':
        return dict(id=case['id'],method=case['method'],audit_status='PASS_RETAINED_EXECUTION_FAILURE' if not errors else 'FAIL_LOG_INTEGRITY',
                    result_status=result['status'],checks=checks,errors=errors,counts={k:len(v) for k,v in data.items()})
    native,policy,sensors,timing=(data[name] for name in ('native','policy','sensor','timing'))
    world=rows(out/'scene'/'world.csv');scene=load(out/'scene'/'session.json');meta=load(out/'sensor_metadata.json')
    check(sha(out/'scene'/'world.csv')==scene['world_csv_sha256'],'world session hash')
    check(scene['owned_server_stopped'] and scene['cleanup_complete'] and not scene['cleanup_errors'],'owned server cleanup')
    check(len(world)==result['world_rows'],'world row count')
    check(len(native)==round(plan['duration_s']/plan['native_dt_s'])+1,'full native denominator')
    check(len(policy)==len(timing)==round(plan['duration_s']/plan['tracker_dt_s']),'full policy/timing denominator')
    check(len(sensors)==len(decisions)==round(plan['duration_s']/plan['decision_period_s']),'full decision denominator')
    wmap={tick(r['native_stamp_s']):r for r in world};pmap={tick(r['time']):r for r in policy}
    origin=scene['road_origin_xyz'];angle=math.radians(scene['route_heading_deg']);e=[math.cos(angle),math.sin(angle),0.]
    line=float(result['stopline_s_m']);s0=float(result['calibrated_initial_s_m']);radius=float(result['invariant_body_radius_m'])
    min_world=math.inf;max_radius=0.
    for i,row in enumerate(world):
        check(near(row['native_stamp_s'],i*plan['tracker_dt_s'],1e-8),'world cadence '+str(i))
        check(abs(float(row['relative_world_stamp_s'])-float(row['native_stamp_s']))<=2e-6,'world/native clock '+str(i))
        if i:check(int(row['frame'])==int(world[i-1]['frame'])+1,'world frame '+str(i))
        verts=vector(row['bbox_world_vertices_json']);cg=vector(row['actual_cg_world_xyz_json'])
        matrix=vector(row['actual_matrix_json']);offset=[-1.015,0.,.54]
        reconstructed_cg=[matrix[j][3]+sum(matrix[j][k]*offset[k] for k in range(3)) for j in range(3)]
        check(all(near(reconstructed_cg[j],cg[j],2e-9) for j in range(3)),'world actual O-to-CG '+str(i))
        front=max(dot([p[j]-origin[j] for j in range(3)],e) for p in verts)
        cg_s=dot([cg[j]-origin[j] for j in range(3)],e)
        actual_radius=max(norm([p[j]-cg[j] for j in range(3)]) for p in verts)
        max_radius=max(max_radius,actual_radius)
        gap=line-front;min_world=min(min_world,gap)
        check(len(verts)==8 and near(front,row['front_support'],2e-9),'world bbox projection '+str(i))
        check(near(cg_s,row['cg_route_s'],2e-9) and near(gap,row['clearance'],2e-9),'world CG/gap '+str(i))
        check(actual_radius<=radius+2e-6,'invariant sphere '+str(i))
        check(float(row['pose_translation_error_m'])<=1e-4 and float(row['cg_publication_error_m'])<=1e-4,'pose publication '+str(i))
    check(near(min_world,result['minimum_world_clearance_m'],2e-9),'world minimum result')
    history=[];bias=float(case['bias']);sensor_violations=0;estimate_violations=0
    max_odom=max_range=max_speed=0.;structured_deletions=frontend_calls=quick_count=0
    first_veto=None;fault_effective=0;max_guard=0.;total_guard=0.;timeouts=unknowns=0
    for i,(sr,decision) in enumerate(zip(sensors,decisions)):
        t=float(sr['time']);packet=json.loads(sr['packet_json']);selected=json.loads(sr['selected_rows_json'])
        check(near(t,i*plan['decision_period_s'],1e-8),'sensor/decision cadence '+str(i))
        check(near(t,decision['time']) and packet==decision['packet'] and selected==decision['selected_rows'],'sensor/decision identity '+str(i))
        if sr['available_new_packet']=='True':history.append(packet)
        expected=[history[max(0,len(history)-3+j)]['rows'][j] for j in range(3)] if history else None
        check(selected==expected,'fixed past history selection '+str(i))
        if packet:
            stamp=float(packet['stamp_s']);truthw=wmap[tick(stamp)];truthp=pmap[tick(stamp)]
            s=float(truthw['cg_route_s']);v=float(truthp['audit_v'])
            eo=float(packet['odom_m'])-(s-s0+bias*stamp)
            ev=float(packet['speed_mps'])-(v+bias)
            max_odom=max(max_odom,abs(eo));max_speed=max(max_speed,abs(ev))
            valid=abs(eo)<=.02+1e-9 and abs(ev)<=.01+1e-9
            check(valid,'digital odo/speed membership '+str(i));sensor_violations+=int(not valid)
            check(stamp<=t+1e-9 and near(t-stamp,decision['observation_age_s'],1e-9),'packet stamp/age '+str(i))
            for j,row in enumerate(packet['rows']):
                land=meta['landmarks'][j];actual_range=math.hypot(s-float(land['s']),float(land['lateral']))
                er=float(row['measured_range_m'])-actual_range;max_range=max(max_range,abs(er))
                check(abs(er)<=.04+1e-9,'digital range membership '+str(i)+'/'+str(j))
                total=.060001;reading=float(row['measured_range_m'])
                check(near(row['packet'][0],max(0.,reading-total)**2,1e-9) and near(row['packet'][1],(reading+total)**2,1e-9),'square packet '+str(i)+'/'+str(j))
                check(near(row['a'],s0+float(packet['odom_m'])-float(land['s']),1e-9) and near(row['c2'],float(land['lateral'])**2,1e-9),'quadratic coefficients '+str(i)+'/'+str(j))
                h=(float(row['a'])-float(row['stamp_s'])*bias)**2+float(row['c2'])
                check(float(row['packet'][0])-1e-8<=h<=float(row['packet'][1])+1e-8,'true common b packet '+str(i)+'/'+str(j))
        status=decision['status'];estimate=decision['estimate'];bounds=decision['bounds']
        check(not selected or all(float(row['stamp_s'])<=t+1e-9 for row in selected),'no future history '+str(i))
        if estimate:
            frontend_calls+=estimate.get('frontend_calls',0)
            unknowns+=int(estimate['status']=='UNKNOWN')
            if estimate.get('b_interval'):
                blo,bhi=map(float,estimate['b_interval']);contained=blo-1e-12<=bias<=bhi+1e-12
                estimate_violations+=int(not contained);check(contained,'true b estimator outer '+str(i))
                rational_interval=estimate.get('b_interval_rational')
                if rational_interval:
                    check(near(blo,float(F(rational_interval[0])),1e-14) and near(bhi,float(F(rational_interval[1])),1e-14),'wire interval matches exact rational '+str(i))
            for cert in estimate.get('certificates',[]):
                if cert.get('stage')=='quadratic_history_face':
                    structured_deletions+=1;lo,hi=cert['b_input'];bound=qbounds(selected[cert['row']],lo,hi)
                    check(tuple(map(F,cert['quadratic_range']))==bound,'independent structural extrema '+str(i))
                    p,q=map(F,cert['packet']);check(bound[1]<p or bound[0]>q,'structural rejection inequality '+str(i))
        if bounds:
            bint=estimate.get('b_interval') if estimate and estimate.get('b_interval') else plan['prior']
            blo,bhi=map(float,bint);stamp=float(packet['stamp_s'])
            su=s0+float(packet['odom_m'])-blo*stamp+.02+1e-8
            vu=float(packet['speed_mps'])-blo+.01+1e-8
            check(near(su,bounds['s_upper'],1e-9) and near(vu,bounds['v_upper'],1e-9),'state upper reconstruction '+str(i))
            true_s=float(wmap[tick(stamp)]['cg_route_s']);true_v=float(pmap[tick(stamp)]['audit_v'])
            check(bounds['s_lower']-1e-7<=true_s<=bounds['s_upper']+1e-7 and bounds['v_lower']-1e-7<=true_v<=bounds['v_upper']+1e-7,'state outer truth '+str(i))
            required=reserve(vu,plan['decision_period_s'],plan['response_delay_s'],plan['measurement_pad_m'])+radius
            margin=line-su-required
            check(near(margin,decision['guard_margin_m'],2e-8),'full guard face '+str(i))
            if status in ('ACCEPT_NORMAL','QUICK_PRIOR_ACCEPT'):check(margin>=0,'accept margin '+str(i))
        if status=='QUICK_PRIOR_ACCEPT':quick_count+=1;check(estimate is None,'quick avoids estimator '+str(i))
        if status in ('ACCEPT_NORMAL','QUICK_PRIOR_ACCEPT'):
            check(float(decision['measured_compute_s'])<=plan['compute_budget_s'],'no late acceptance '+str(i))
            check(decision['observation_age_s']<=plan['max_observation_age_s'],'no stale acceptance '+str(i))
        if decision['modeled_pending_apply_s'] is not None and first_veto is None:
            first_veto=(t,float(decision['modeled_pending_apply_s']),status)
            check(near(first_veto[1]-first_veto[0],plan['response_delay_s'],1e-9),'modeled veto availability')
        if status=='FORCED_COMPUTE_TIMEOUT':
            fault_effective+=1;check(decision['fault_injected'] and near(decision['reported_compute_s'],case['fault']['reported_elapsed_s']),'forced late report')
        if status=='STALE_OBSERVATION' and decision['fault_injected']:fault_effective+=1
        if status!='BACKUP_ALREADY_LATCHED':
            max_guard=max(max_guard,float(decision['measured_compute_s']));total_guard+=float(decision['measured_compute_s']);timeouts+=int('TIMEOUT' in status)
    check(estimate_violations==result['audit_true_bias_exclusion_count'],'result true-b exclusions')
    check(quick_count==result.get('quick_prior_accept_count',quick_count),'result quick count')
    check(timeouts==result['timeout_count'] and unknowns==result['estimator_unknown_count'],'result timeout/unknown')
    check(near(max_guard,result['guard_measured_compute_max_s'],1e-9) and near(total_guard,result['guard_measured_compute_total_s'],1e-9),'guard cost totals')
    max_prefix=-math.inf;min_native_gap=math.inf;waiting_a_max=-math.inf;native_violations=0
    x_initial=float(native[0]['x']);emergency=result['emergency_applied_s'];stop_tail=0.
    # Each accepted information set must contain the entire declared waiting
    # window. Compare all native samples, including any intermediate emergency
    # phase, against the same upper waiting flow; no continuous claim follows.
    waiting_flow_points=waiting_flow_origins=waiting_position_violations=waiting_speed_violations=0
    waiting_position_max_excess=waiting_speed_max_excess=-math.inf
    waiting_span=plan['decision_period_s']+plan['response_delay_s']
    for decision in decisions:
        if decision['status'] not in ('ACCEPT_NORMAL','QUICK_PRIOR_ACCEPT'):continue
        waiting_flow_origins+=1;t0=float(decision['time']);bounds=decision['bounds']
        first_index=round(t0/plan['native_dt_s'])
        last_index=min(len(native)-1,round((t0+waiting_span)/plan['native_dt_s']))
        for nrow in native[first_index:last_index+1]:
            dt=float(nrow['time'])-t0
            s_native=s0+float(nrow['x'])-x_initial
            s_wait_upper=float(bounds['s_upper'])+float(bounds['v_upper'])*dt+1.25*dt*dt+plan['measurement_pad_m']
            v_wait_upper=float(bounds['v_upper'])+2.5*dt
            x_excess=s_native-s_wait_upper;v_excess=float(nrow['vx'])-v_wait_upper
            waiting_flow_points+=1
            waiting_position_max_excess=max(waiting_position_max_excess,x_excess)
            waiting_speed_max_excess=max(waiting_speed_max_excess,v_excess)
            waiting_position_violations+=int(x_excess>1e-8)
            waiting_speed_violations+=int(v_excess>1e-7)
    for i,row in enumerate(native):
        t=float(row['time']);check(near(t,i*plan['native_dt_s'],1e-7),'native cadence '+str(i))
        if i:check(near(row['state_time_before'],float(native[i-1]['time']),1e-7),'native previous time '+str(i))
        if i:
            physical_policy=pmap[tick(row['command_set_at_s'])]
            check(near(row['throttle_fraction'],physical_policy['throttle'],1e-9)
                  and near(row['brake_mpa'],physical_policy['brake'],1e-9)
                  and near(row['steer_deg'],physical_policy['steer'],1e-9),'native physical command '+str(i))
        if tick(t) in pmap:
            prow=pmap[tick(t)]
            check(near(prow['audit_x'],row['x'],1e-9) and near(prow['audit_v'],row['vx'],1e-9),'policy/native state time '+str(i))
        gap=line-(s0+float(row['x'])-x_initial+radius);min_native_gap=min(min_native_gap,gap)
        native_violations+=int(gap<-1e-8)
        if row['phase']=='normal':waiting_a_max=max(waiting_a_max,float(row['ax']))
        else:
            dt=t-float(emergency);upper=float(row['emergency_origin_x'])+prefix(float(row['emergency_origin_v']),dt)
            check(near(upper,row['position_envelope_upper'],2e-8),'emergency prefix formula '+str(i))
            max_prefix=max(max_prefix,float(row['x'])-upper)
            policyrow=pmap[tick(row['command_set_at_s'])]
            check(near(policyrow['throttle'],0.) and near(policyrow['brake'],2.5) and near(policyrow['steer'],0.),'direct physical fallback '+str(i))
    for row in reversed(native[1:]):
        if abs(float(row['vx']))>.05:break
        stop_tail+=plan['native_dt_s']
    check(near(max_prefix,result['max_emergency_prefix_excess_m'],2e-8),'backup prefix reported max')
    check((max_prefix<=1e-8)==result['sampled_backup_membership'],'backup membership label')
    check(near(waiting_a_max,result['normal_acceleration_max_mps2'],1e-9),'waiting acceleration result')
    check(near(stop_tail,result['stopped_final_tail_s'],1e-7),'complete stopped tail')
    check(first_veto is not None and near(emergency,first_veto[1],1e-8),'first veto/brake application')
    emergency_origins=[(float(row['emergency_origin_x']),float(row['emergency_origin_v'])) for row in native if row['phase']=='emergency']
    brake_origin_v=emergency_origins[0][1] if emergency_origins else None
    check(not emergency_origins or all(near(x,emergency_origins[0][0],1e-9) and near(v,brake_origin_v,1e-9) for x,v in emergency_origins),'single latched emergency origin')
    max_loop=0.;over=0
    for i,row in enumerate(timing):
        keys=('cycle_started_monotonic_s','observe_finished_monotonic_s','control_finished_monotonic_s','native_started_monotonic_s','native_finished_monotonic_s','cycle_finished_monotonic_s')
        stamps=[float(row[k]) for k in keys];check(all(a<=b for a,b in zip(stamps,stamps[1:])),'timing order '+str(i))
        full=stamps[-1]-stamps[0];max_loop=max(max_loop,full);over+=int(full>.01)
        check(near(full,row['full_cycle_wall_s'],1e-8),'timing full loop '+str(i))
        check(near(stamps[1]-stamps[0],row['observe_wall_s'],1e-8),'timing observe '+str(i))
        check(near(stamps[2]-stamps[1],row['sensor_estimator_guard_and_tracker_wall_s'],1e-8),'timing control '+str(i))
        check(near(stamps[4]-stamps[3],row['native_integrate_and_log_wall_s'],1e-8),'timing native '+str(i))
        check(near(stamps[5]-stamps[4],row['scene_publish_and_tick_wall_s'],1e-8),'timing publish '+str(i))
    check(near(max_loop,result['full_cycle_wall_max_s'],1e-8) and over==result['full_cycle_wall_over10ms_count'],'full loop result')
    return dict(id=case['id'],method=case['method'],audit_status='PASS_INDEPENDENT_LOG_INTEGRITY' if not errors else 'FAIL_LOG_INTEGRITY',
                result_status=result['status'],checks=checks,errors=errors,
                counts={**{k:len(v) for k,v in data.items()},'world':len(world),'decision':len(decisions)},
                min_world_gap_m=min_world,min_native_conservative_gap_m=min_native_gap,native_conservative_violations=native_violations,
                sensor_contract_violation_count=sensor_violations,estimate_true_b_exclusion_count=estimate_violations,
                max_odom_error_m=max_odom,max_range_error_m=max_range,max_speed_error_mps=max_speed,
                sampled_waiting_acceleration_upper_pass=waiting_a_max<=2.5+1e-7,waiting_acceleration_max_mps2=waiting_a_max,
                sampled_waiting_flow_points=waiting_flow_points,sampled_waiting_flow_origins=waiting_flow_origins,
                sampled_waiting_flow_includes_emergency_samples=True,sampled_waiting_flow_horizon_s=waiting_span,
                sampled_waiting_position_max_excess_m=waiting_position_max_excess if waiting_flow_points else None,
                sampled_waiting_speed_max_excess_mps=waiting_speed_max_excess if waiting_flow_points else None,
                sampled_waiting_position_violation_count=waiting_position_violations,sampled_waiting_speed_violation_count=waiting_speed_violations,
                sampled_waiting_flow_pass=waiting_position_violations==waiting_speed_violations==0,
                sampled_backup_prefix_pass=max_prefix<=1e-8,backup_prefix_max_excess_m=max_prefix,
                actual_brake_origin_speed_mps=brake_origin_v,
                actual_brake_origin_in_declared_sampled_envelope_speed_range=brake_origin_v is not None and 0<=brake_origin_v<=18.,
                brake_origin_scope='Speed within 0..18 is a sampled envelope check, not proved forward-closed physical domain; developer/held-out braking examples remain finite.',
                final_stopped_tail_s=stop_tail,final_stopped_tail_pass=stop_tail>=3.-1e-7,
                fault_effective_count=fault_effective,structured_deletions=structured_deletions,frontend_calls=frontend_calls,
                max_bbox_radius_m=max_radius,full_loop_max_s=max_loop,full_loop_over10ms_count=over,
                guard_cost_total_s=total_guard,quick_prior_accepts=quick_count,timeout_count=timeouts,unknown_count=unknowns)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case');ap.add_argument('--output',type=Path);args=ap.parse_args()
    plan=load(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json')
    for filename,digest in plan['source_sha256'].items():
        if sha(filename)!=digest:raise RuntimeError('Frozen source changed: '+filename)
    cases=[c for c in plan['cases'] if args.case is None or c['id']==args.case]
    if not cases:raise ValueError('Case not in frozen full denominator')
    records=[audit_case(c,plan) for c in cases]
    status='PASS_INDEPENDENT_COMPLETED_COSIM_LOG_AUDIT' if all(r['audit_status'].startswith('PASS') for r in records) else 'FAIL_INDEPENDENT_COSIM_LOG_AUDIT'
    report=dict(status=status,planned=len(plan['cases']),audited=len(records),full_denominator_audit=len(records)==len(plan['cases']),
                audit_source_sha256=sha(__file__),plan_sha256=sha(ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'),records=records,
                scope='Completed logs, digital sensor/method truth containment, native/world sampled faces, fixed-window veto and timing integrity. No simulator, continuous all-domain qualification or real-time wall-clock guarantee.')
    output=args.output or ROOT/'audit'/('INDEPENDENT_COSIM_ESTIMATOR_AUDIT.json' if args.case is None else args.case+'_AUDIT.json')
    if output.exists():raise FileExistsError('Preserve existing independent audit')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'audited':len(records),'checks':sum(r['checks'] for r in records),'errors':sum(len(r['errors']) for r in records)}))


if __name__=='__main__':main()
