"""Owned CARLA/CarSim estimator comparison with a shared position backup.

No import or --freeze operation starts CARLA or a native solver.  Only --run
enters those contexts, after byte-verifying the root-authored frozen protocol.
Current native/world truth is restricted to sensor generation, the common low
level actuator tracker and audit logs; the guard consumes sanitized packets.
Normal requests follow a public fixed timeline during a modeled computation
window. The query determines only a brake veto at that window's end. A previous
accepted query covers the next decision plus response window; no claim is made
that a newly accepted normal command was available at the current simulated
instant, or that the complete CARLA loop meets a wall-clock real-time deadline.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OLD_POSITION = ROOT.parent/'vehicle_position_contract_20261013'
OLD_VEHICLE = ROOT.parent/'vehicle_validation_20261012'
BASE = Path(r'E:\codex_project\CarlaCarSimValidation')
PROTOCOL = ROOT/'inputs'/'PROTOCOL_DRAFT.json'
PLAN = ROOT/'inputs'/'COSIM_ESTIMATOR_PLAN_FROZEN.json'
RESULTS = ROOT/'cosim_results'
EXPECTED_METHODS = ('ma', 'ma_structured', 'zhao2t3', 'zhao2t3_structured', 'exact_range')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def utc():
    return datetime.now(timezone.utc).isoformat()


def build_plan(protocol_path=PROTOCOL):
    """Freeze an existing, reviewable protocol; never choose cases here."""
    protocol_path = Path(protocol_path).resolve()
    plan = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    for key in ('cases', 'duration_s', 'native_dt_s', 'tracker_dt_s',
                'decision_period_s', 'compute_budget_s', 'response_delay_s',
                'max_observation_age_s', 'measurement_pad_m', 'prior',
                'landmark_offsets', 'emergency_command', 'raw_request'):
        if key not in plan:
            raise ValueError('Protocol must explicitly declare '+key)
    if len(plan['prior']) != 2 or not plan['prior'][0] < plan['prior'][1]:
        raise ValueError('Invalid common bias prior')
    if not 0 < plan['compute_budget_s'] <= plan['response_delay_s']:
        raise ValueError('Compute budget must fit within modeled response window')
    if not math.isclose(plan['native_dt_s'], .0005, abs_tol=1e-15):
        raise ValueError('Native backend fixed at 0.5 ms')
    if not math.isclose(plan['tracker_dt_s'], .01, abs_tol=1e-15):
        raise ValueError('CARLA publication fixed at 10 ms')
    ratio = plan['decision_period_s']/plan['tracker_dt_s']
    if ratio < 1 or abs(ratio-round(ratio)) > 1e-10:
        raise ValueError('Decision cadence must be an integer number of tracker ticks')
    duration_ticks = plan['duration_s']/plan['tracker_dt_s']
    if duration_ticks <= 0 or abs(duration_ticks-round(duration_ticks)) > 1e-10:
        raise ValueError('Duration must finish at a publication endpoint')
    ids = [case['id'] for case in plan['cases']]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Case identifiers must be unique and nonempty')
    for case in plan['cases']:
        if case['method'] not in EXPECTED_METHODS:
            raise ValueError('Unrecognized method '+case['method'])
        for key in ('speed_mps', 'bias', 'seed', 'initial_front_clearance_m'):
            if key not in case:
                raise ValueError('Case must explicitly declare '+key)
        if Path(case['id']).name != case['id'] or case['id'] in ('.', '..'):
            raise ValueError('Unsafe case identifier')
        if case.get('fault'):
            fault = case['fault']
            fault_tick = fault['at_s']/plan['decision_period_s']
            if fault['kind'] not in ('drop', 'late') or abs(fault_tick-round(fault_tick)) > 1e-10:
                raise ValueError('Fault must be a declared event on a decision tick')
    if len(plan['landmark_offsets']) != 3:
        raise ValueError('Three predeclared range landmarks are required')
    gates = [
        (ROOT/'audit'/'ESTIMATOR_COMPONENT_AUDIT.json',
         'PASS_INDEPENDENT_OBSERVATION_ADAPTER_COMPONENT_AUDIT'),
        (ROOT/'audit'/'OBSERVATION_MATH_CHECKS.json',
         'PASS_INDEPENDENT_RATIONAL_OBSERVATION_COMPONENT_CHECKS'),
        (ROOT/'audit'/'RUNNER_SMOKE_CHECKS.json',
         'PASS_PURE_FAKE_BACKEND_RUNNER_CHECKS'),
        (OLD_POSITION/'audit'/'INDEPENDENT_POSITION_VALIDATION_AUDIT.json',
         'PASS_INDEPENDENT_AUDIT_OF_NEW_POSITION_VALIDATION'),
    ]
    gate_records = {}
    for path, expected in gates:
        gate = json.loads(path.read_text(encoding='utf-8'))
        if gate['status'] != expected:
            raise RuntimeError('Pre-execution component gate failed '+str(path))
        gate_records[str(path)] = dict(status=gate['status'], sha256=sha(path))
    component = json.loads(gates[0][0].read_text(encoding='utf-8'))
    if component['estimator_source_sha256'] != sha(ROOT/'source'/'sensor_estimators.py'):
        raise RuntimeError('Estimator source changed after independent component gate')
    if component['audit_source_sha256'] != sha(ROOT/'audit'/'audit_estimator_components.py'):
        raise RuntimeError('Independent component audit source changed after gate')
    observation = json.loads(gates[1][0].read_text(encoding='utf-8'))
    if observation['source_sha256'] != sha(ROOT/'audit'/'reference_bias_math.py'):
        raise RuntimeError('Observation math audit source changed after gate')
    smoke = json.loads(gates[2][0].read_text(encoding='utf-8'))
    if smoke['runner_source_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Runner changed after fake-backend decision checks')
    if smoke['smoke_source_sha256'] != sha(ROOT/'integration'/'runner_smoke_checks.py'):
        raise RuntimeError('Fake-backend check source changed after gate')
    files = [Path(__file__), ROOT/'source'/'sensor_estimators.py',
             OLD_POSITION/'source'/'position_backup.py',
             OLD_POSITION/'integration'/'coupling_session_v3.py',
             OLD_VEHICLE/'source'/'cg_vehicle.py', OLD_VEHICLE/'source'/'tracker.py',
             OLD_VEHICLE/'inputs'/'actuator_model_v2.json', BASE/'validate.py',
             BASE/'validation_config.json', protocol_path,
             ROOT/'audit'/'audit_estimator_components.py',
             ROOT/'audit'/'reference_bias_math.py',
             ROOT/'integration'/'runner_smoke_checks.py']
    files += [path for path, _ in gates]
    files += [Path(p) for p in plan.get('additional_frozen_files', [])]
    backend = load('estimator_freeze_readonly_cg', OLD_VEHICLE/'source'/'cg_vehicle.py')
    files += [backend.DLL, backend.ARCHIVE]
    config = json.loads((BASE/'validation_config.json').read_text(encoding='utf-8-sig'))
    executable = Path(config['carla_root'])/config['carla_executable_relative']
    plan.update(status='FROZEN_BEFORE_NEW_CLOSED_LOOP_TRAJECTORIES', created_utc=utc(),
                protocol_path=str(protocol_path), protocol_sha256=sha(protocol_path),
                pre_execution_gates=gate_records,
                source_sha256={str(p.resolve()): sha(p) for p in files},
                carla_executable=dict(path=str(executable), bytes=executable.stat().st_size),
                no_retry_or_overwrite=True, all_failures_retained=True)
    return plan


class CsvLog:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('x', newline='', encoding='utf-8')
        self.writer = None
        self.rows = 0

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row), extrasaction='raise')
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows += 1

    def close(self):
        self.file.close()


def selected_history_rows(history_packets):
    """Same oldest/middle/newest staggered landmark history for all methods."""
    n = len(history_packets)
    return [history_packets[max(0, n-3+j)]['rows'][j] for j in range(3)]


def finite_bounds(bounds):
    return (isinstance(bounds, dict) and bounds.get('valid', True)
            and all(k in bounds and math.isfinite(float(bounds[k]))
                    for k in ('s_lower', 's_upper', 'v_lower', 'v_upper'))
            and bounds['s_lower'] <= bounds['s_upper']
            and bounds['v_lower'] <= bounds['v_upper'])


def run_case(case, plan, Vehicle, Tracker, CouplingSession, estimators, backup):
    out = RESULTS/case['id']
    out.mkdir(parents=True, exist_ok=False)
    save(out/'case.json', dict(case, plan_sha256=sha(PLAN), created_utc=utc()))
    result = dict(id=case['id'], method=case['method'], status='STARTED',
                  plan_sha256=sha(PLAN), started_utc=utc(),
                  continuous_certificate=False, real_time_wall_clock_guarantee=False,
                  sensor_perception=False, native_truth_in_estimator=False,
                  availability_model=plan.get('availability_model',
                      'Public normal requests continue during response window; query is a future brake veto'),
                  guard_deadline_scope='Sensor packet generation, common prior quick, estimator, state bounds, position guard; excludes log serialization and CARLA scene operation, both included in full cycle timing')
    save(out/'result.json', result)
    native, policy, sensors, timing = [CsvLog(out/(name+'.csv'))
                                     for name in ('native', 'policy', 'sensor', 'timing')]
    decisions = (out/'decision.jsonl').open('x', encoding='utf-8')
    started = time.perf_counter()
    emergency_at = pending_at = None
    emergency_reason = pending_reason = None
    emergency_state = None
    first_crossing = None
    max_prefix_excess = -math.inf
    normal_a_max = -math.inf
    clearance_min = math.inf
    last_margin = None
    latest_packet = None
    history_packets = []
    world_rows = decision_calls = timeout_count = unknown_count = quick_count = 0
    true_bias_exclusions = 0
    loop_times, guard_times, elapsed_reported = [], [], []
    speeds, stopped_tail = [], 0.
    fault_seen = False
    try:
        with CouplingSession(out/'scene', case['initial_front_clearance_m']) as scene:
          with Vehicle(out/'solver', case['speed_mps'], plan['duration_s']+.01) as solver:
            state = solver.sample((0., 0., 0.))
            save(out/'solver_metadata.json', solver.metadata)
            obs = scene.anchor(state)
            world_rows = 1
            import numpy as np
            cg = np.array(json.loads(obs['actual_cg_world_xyz_json']))
            vertices = np.array(json.loads(obs['bbox_world_vertices_json']))
            radius = float(max(np.linalg.norm(point-cg) for point in vertices))+.001
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError('Invalid initial invariant body radius')
            # These are fixed initial map calibration/vehicle shape data.
            s0, line_s = float(obs['cg_route_s']), float(scene.metadata['stopline_s_m'])
            landmarks = [dict(s=s0+item['s'], lateral=item['lateral'])
                         for item in plan['landmark_offsets']]
            sensor = estimators.Sensor(case['seed'], case['bias'], s0, landmarks)
            save(out/'sensor_metadata.json', sensor.metadata)
            result.update(invariant_body_radius_m=radius, calibrated_initial_s_m=s0,
                          stopline_s_m=line_s, landmarks=landmarks)
            tracker = Tracker()
            native.write(dict(state, phase='normal', raw_request=0., command_set_at_s=0.,
                              state_time_before=0., modeled_pending_apply_s=-1.,
                              emergency_origin_x=0., emergency_origin_v=0., position_envelope_upper=0.))
            decision_every = round(plan['decision_period_s']/plan['tracker_dt_s'])
            for k in range(round(plan['duration_s']/plan['tracker_dt_s'])):
                t = k*plan['tracker_dt_s']
                cycle_start = time.perf_counter()
                before = solver.state.copy()
                if abs(before['time']-t) > 1e-8:
                    raise RuntimeError('Native state is not current policy endpoint')
                obs = scene.observe()
                observe_finished = time.perf_counter()
                # Ground truth is logged/audited here, never used by guard.
                clearance = float(obs['clearance'])
                clearance_min = min(clearance_min, clearance)
                if clearance < -1e-8 and first_crossing is None:
                    first_crossing = t
                if pending_at is not None and t+1e-9 >= pending_at and emergency_at is None:
                    emergency_at, emergency_reason = t, pending_reason
                decision_status = 'HELD'
                measured = reported = 0.
                bounds = None
                estimate = None
                if k % decision_every == 0:
                    guard_start = time.perf_counter()
                    rows = None
                    sensor_error = None
                    prior_bounds = None
                    prior_margin = None
                    fault = case.get('fault', {})
                    inject = (not fault_seen and fault and abs(t-float(fault['at_s'])) < 1e-9)
                    try:
                        packet = sensor.observe(obs, before['vx'], t)
                        if inject and fault['kind'] == 'drop':
                            packet = None
                        if packet is not None:
                            latest_packet = packet
                            history_packets.append(packet)
                        if latest_packet is not None:
                            rows = selected_history_rows(history_packets)
                    except Exception as exc:
                        sensor_error = repr(exc)
                    can_decide = emergency_at is None and pending_at is None
                    age = (t-float(latest_packet['stamp_s'])) if latest_packet else None
                    if not can_decide:
                        decision_status = 'BACKUP_ALREADY_LATCHED'
                    elif sensor_error:
                        decision_status = 'SENSOR_ERROR'
                    elif latest_packet is None:
                        decision_status = 'MISSING_OBSERVATION'
                    elif age < -1e-9 or age > plan['max_observation_age_s']:
                        decision_status = 'STALE_OBSERVATION'
                    else:
                        deadline = guard_start+plan['compute_budget_s']
                        try:
                            prior_bounds = estimators.guard_bounds(latest_packet, plan['prior'])
                            if not finite_bounds(prior_bounds):
                                decision_status = 'INVALID_PRIOR_STATE_BOUNDS'
                            else:
                                prior_required = backup.next_period_required(
                                    prior_bounds['v_upper'], plan['decision_period_s'],
                                    measurement_pad=plan['measurement_pad_m'],
                                    deadline_delay=plan['response_delay_s'])+radius
                                prior_margin = line_s-prior_bounds['s_upper']-prior_required
                                if prior_margin >= 0:
                                    bounds = prior_bounds
                                    last_margin = prior_margin
                                    decision_status = 'QUICK_PRIOR_ACCEPT'
                                    quick_count += 1
                                else:
                                    decision_calls += 1
                                    estimate = estimators.estimate(case['method'], rows, plan['prior'], deadline)
                                    if estimate['status'] != 'OK' or estimate.get('b_interval') is None:
                                        decision_status = 'ESTIMATOR_'+estimate['status']
                                        if estimate['status'] == 'UNKNOWN':
                                            unknown_count += 1
                                    else:
                                        bounds = estimators.guard_bounds(latest_packet, estimate['b_interval'])
                                        if not finite_bounds(bounds):
                                            decision_status = 'INVALID_STATE_BOUNDS'
                                        else:
                                            required = backup.next_period_required(
                                                bounds['v_upper'], plan['decision_period_s'],
                                                measurement_pad=plan['measurement_pad_m'],
                                                deadline_delay=plan['response_delay_s'])+radius
                                            last_margin = line_s-bounds['s_upper']-required
                                            decision_status = 'ACCEPT_NORMAL' if last_margin >= 0 else 'BACKUP_MARGIN'
                        except Exception as exc:
                            estimate = dict(status='ERROR', error=repr(exc), b_interval=None)
                            decision_status = 'ESTIMATOR_ERROR'
                    measured = time.perf_counter()-guard_start
                    reported = measured
                    if inject and fault['kind'] == 'late' and can_decide:
                        reported = float(fault['reported_elapsed_s'])
                        decision_status = 'FORCED_COMPUTE_TIMEOUT'
                    elif can_decide and measured > plan['compute_budget_s']:
                        decision_status = 'MEASURED_GUARD_TIMEOUT'
                    if inject:
                        fault_seen = True
                    if can_decide:
                        guard_times.append(measured)
                        elapsed_reported.append(reported)
                        if 'TIMEOUT' in decision_status:
                            timeout_count += 1
                        if decision_status not in ('ACCEPT_NORMAL', 'QUICK_PRIOR_ACCEPT'):
                            pending_at = t+plan['response_delay_s']
                            pending_reason = decision_status
                    contained = None
                    if estimate and estimate.get('b_interval'):
                        lo, hi = estimate['b_interval']
                        contained = bool(lo-1e-12 <= case['bias'] <= hi+1e-12)
                        true_bias_exclusions += int(not contained)
                    record = dict(time=t, method=case['method'], packet=latest_packet,
                                  selected_rows=rows, prior=list(plan['prior']), estimate=estimate,
                                  prior_bounds=prior_bounds, prior_margin_m=prior_margin,
                                  bounds=bounds, status=decision_status, guard_margin_m=last_margin,
                                  observation_age_s=age, sensor_error=sensor_error,
                                  measured_compute_s=measured, reported_compute_s=reported,
                                  modeled_pending_apply_s=pending_at,
                                  audit_true_bias_in_estimate=contained, fault_injected=bool(inject))
                    decisions.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    decisions.flush()
                    sensors.write(dict(time=t, stamp_s=latest_packet['stamp_s'] if latest_packet else '',
                                       available_new_packet=packet is not None if not sensor_error else False,
                                       sensor_error=sensor_error or '',
                                       packet_json=json.dumps(latest_packet, ensure_ascii=False, allow_nan=False),
                                       selected_rows_json=json.dumps(rows, ensure_ascii=False, allow_nan=False),
                                       audit_true_s_m=obs['cg_route_s'], audit_true_speed_mps=before['vx'],
                                       audit_true_bias=case['bias']))
                if emergency_at is not None:
                    if emergency_state is None:
                        emergency_state = dict(time=t, x=before['x'], v=before['vx'])
                    command, raw, phase = tuple(plan['emergency_command']), -3., 'emergency'
                else:
                    raw_spec = plan['raw_request']
                    raw = raw_spec['before'] if t < raw_spec['switch_s'] else raw_spec['after']
                    command, _ = tracker.command(raw, before, plan['tracker_dt_s'])
                    phase = 'normal'
                control_finished = time.perf_counter()
                policy.write(dict(time=t, audit_state_time=before['time'], audit_x=before['x'],
                                  audit_v=before['vx'], audit_a=before['ax'],
                                  audit_current_carla_clearance_m=clearance, invariant_body_radius_m=radius,
                                  observation_stamp_s=latest_packet['stamp_s'] if latest_packet else '',
                                  estimated_s_upper_m=bounds['s_upper'] if bounds else '',
                                  estimated_v_upper_mps=bounds['v_upper'] if bounds else '',
                                  guard_margin_m=last_margin if last_margin is not None else '',
                                  decision_status=decision_status, guard_measured_compute_s=measured,
                                  guard_reported_compute_s=reported, raw_request=raw,
                                  throttle=command[0], brake=command[1], steer=command[2], phase=phase,
                                  modeled_pending_apply_s=pending_at if pending_at is not None else -1.,
                                  emergency_reason=emergency_reason or ''))
                native_started = time.perf_counter()
                for _ in range(round(plan['tracker_dt_s']/plan['native_dt_s'])):
                    prev = solver.last_t
                    row = solver.sample(command)
                    if phase == 'normal':
                        normal_a_max = max(normal_a_max, row['ax'])
                    upper = 0.
                    if emergency_state:
                        upper = emergency_state['x']+backup.prefix_upper(
                            max(0., emergency_state['v']), row['time']-emergency_at)
                        max_prefix_excess = max(max_prefix_excess, row['x']-upper)
                    row.update(phase=phase, raw_request=raw, command_set_at_s=t,
                               state_time_before=prev, modeled_pending_apply_s=pending_at if pending_at is not None else -1.,
                               emergency_origin_x=emergency_state['x'] if emergency_state else 0.,
                               emergency_origin_v=emergency_state['v'] if emergency_state else 0.,
                               position_envelope_upper=upper)
                    native.write(row)
                    speeds.append(row['vx'])
                    if row['stop_requested']:
                        raise RuntimeError('Unexpected native stop')
                native_finished = time.perf_counter()
                obs = scene.publish(row)
                world_rows += 1
                cycle_finished = time.perf_counter()
                loop_times.append(cycle_finished-cycle_start)
                timing.write(dict(time=t, cycle_started_monotonic_s=cycle_start,
                                  observe_finished_monotonic_s=observe_finished,
                                  control_finished_monotonic_s=control_finished,
                                  native_started_monotonic_s=native_started,
                                  native_finished_monotonic_s=native_finished,
                                  cycle_finished_monotonic_s=cycle_finished,
                                  observe_wall_s=observe_finished-cycle_start,
                                  sensor_estimator_guard_and_tracker_wall_s=control_finished-observe_finished,
                                  native_integrate_and_log_wall_s=native_finished-native_started,
                                  scene_publish_and_tick_wall_s=cycle_finished-native_finished,
                                  full_cycle_wall_s=cycle_finished-cycle_start,
                                  guard_measured_compute_s=measured, guard_reported_compute_s=reported))
                timing.file.flush()
            clearance_min = min(clearance_min, float(obs['clearance']))
            for speed in reversed(speeds):
                if abs(speed) > .05:
                    break
                stopped_tail += plan['native_dt_s']
            result.update(status='SAMPLED_NO_STOPLINE_CROSSING' if clearance_min >= -1e-8 else 'SAMPLED_STOPLINE_VIOLATION',
                          minimum_world_clearance_m=clearance_min, first_world_crossing_s=first_crossing,
                          emergency_applied_s=emergency_at, emergency_reason=emergency_reason,
                          fault_injected=fault_seen, max_emergency_prefix_excess_m=max_prefix_excess if math.isfinite(max_prefix_excess) else None,
                          sampled_backup_membership=emergency_state is not None and max_prefix_excess <= 1e-8,
                          normal_acceleration_max_mps2=normal_a_max if math.isfinite(normal_a_max) else None,
                          stopped_final_tail_s=stopped_tail, final_speed_mps=speeds[-1],
                          estimator_calls=decision_calls, estimator_unknown_count=unknown_count,
                          quick_prior_accept_count=quick_count,
                          timeout_count=timeout_count, audit_true_bias_exclusion_count=true_bias_exclusions,
                          guard_measured_compute_max_s=max(guard_times) if guard_times else None,
                          guard_reported_compute_max_s=max(elapsed_reported) if elapsed_reported else None,
                          guard_measured_compute_total_s=sum(guard_times),
                          full_cycle_wall_max_s=max(loop_times),
                          full_cycle_wall_over10ms_count=sum(v > .01 for v in loop_times),
                          wall_elapsed_is_not_simulated_delay=True)
    except Exception as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        for log in (native, policy, sensors, timing):
            log.close()
        decisions.close()
        result.update(finished_utc=utc(), wall_s=time.perf_counter()-started,
                      native_rows=native.rows, policy_rows=policy.rows,
                      sensor_rows=sensors.rows, timing_rows=timing.rows, world_rows=world_rows,
                      native_sha256=sha(native.path), policy_sha256=sha(policy.path),
                      sensor_sha256=sha(sensors.path), timing_sha256=sha(timing.path),
                      decision_sha256=sha(out/'decision.jsonl'))
        save(out/'result.json', result)
    print(json.dumps({k: result[k] for k in ('id', 'method', 'status', 'minimum_world_clearance_m',
                                            'emergency_applied_s', 'wall_s') if k in result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.freeze:
        if PLAN.exists():
            raise FileExistsError('Preserve existing frozen protocol')
        save(PLAN, build_plan(args.protocol))
        print('COSIM_ESTIMATOR_PLAN_FROZEN '+sha(PLAN))
        return
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    for filename, digest in plan['source_sha256'].items():
        if sha(filename) != digest:
            raise RuntimeError('Frozen source drift '+filename)
    if RESULTS.exists():
        raise FileExistsError('No overwrite or retry of retained runs')
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(OLD_POSITION/'integration'))
    from validate import exclusive_lock, check_no_server
    import psutil
    estimators = load('closed_loop_sensor_estimators', ROOT/'source'/'sensor_estimators.py')
    backup = load('closed_loop_readonly_backup', OLD_POSITION/'source'/'position_backup.py')
    if tuple(estimators.METHODS) != EXPECTED_METHODS:
        raise ValueError('Frozen method inventory differs')
    with exclusive_lock(BASE):
        check_no_server(psutil, [2100, 2101, 2102])
        Vehicle = load('closed_loop_readonly_cg', OLD_VEHICLE/'source'/'cg_vehicle.py').TelemetryVehicle
        Tracker = load('closed_loop_readonly_tracker', OLD_VEHICLE/'source'/'tracker.py').DynamicTracker
        from coupling_session_v3 import CouplingSession
        RESULTS.mkdir(exist_ok=False)
        records = []
        save(RESULTS/'EXECUTION.json', dict(planned=len(plan['cases']), executed=0, records=[]))
        for case in plan['cases']:
            records.append(run_case(case, plan, Vehicle, Tracker, CouplingSession, estimators, backup))
            save(RESULTS/'EXECUTION.json', dict(planned=len(plan['cases']), executed=len(records), records=records))


if __name__ == '__main__':
    main()
