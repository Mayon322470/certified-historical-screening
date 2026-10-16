"""Owned CARLA scene for a native-CarSim static-stopline experiment.

The caller holds CarlaCarSimValidation/simulation.lock for this entire session.
This module never advances a solver or chooses controls. CARLA actor physics is
disabled: the scene supplies geometric feedback, not road forces/contact impulses.
Construction/import does not start a server; only entering the context does.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import carla
import numpy as np
import psutil

BASE = Path(r'E:\codex_project\CarlaCarSimValidation')
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
from validate import check_no_server

WORLD_DT = .01
CG_BODY_NATIVE_M = np.array([-1.015, 0., .54])
REFLECT = np.diag([1., -1., 1.])
EXPECTED_MAP = 'Town10HD_Opt'


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_rotation(state):
    """Native right-handed body-to-world Rz(yaw) Ry(pitch) Rx(roll)."""
    y, p, r = (float(state[k]) for k in ('yaw', 'pitch', 'roll'))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _rotation_from_matrix(matrix):
    # CARLA Euler conventions checked in the old full-pose coupling qualification.
    p = math.asin(float(np.clip(matrix[2, 0], -1., 1.)))
    y = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    r = math.atan2(float(-matrix[2, 1]), float(matrix[2, 2]))
    return carla.Rotation(pitch=math.degrees(p), yaw=math.degrees(y), roll=math.degrees(r))


def _angle_difference(a, b):
    return abs((float(a)-float(b)+180.) % 360. - 180.)


class CouplingSession:
    """Context for one owned server/ego and one fixed geometric stopline.

    Required outer contract: exclusive simulation.lock already held. Calls are
    sequential. anchor() takes solver state at t=0; publish() takes each next
    native endpoint exactly 10 ms later. observe() never ticks the world.
    """
    def __init__(self, out, stopline_distance_m):
        self.out = Path(out).resolve()
        self.distance = float(stopline_distance_m)
        if not math.isfinite(self.distance) or self.distance <= 0:
            raise ValueError('Stopline initial front clearance must be positive and finite')
        self.process = self.world = self.client = self.old_settings = self.ego = self.stdout = None
        self.actors = []
        self.world_file = self.writer = None
        self._anchored = self._closed = False
        self._line_s = None
        self._last_state = self._last_expected = None
        self._base_elapsed = self._last_frame = None
        self._last_observation = None
        self.metadata = dict(status='CREATED', source_sha256=_sha(__file__),
            stopline_initial_front_clearance_m=self.distance, master_dt_s=WORLD_DT,
            lock_ownership='Caller must hold simulation.lock throughout context',
            route='External native CarSim -> disabled-physics CARLA pose; live CARLA stopline geometry feedback',
            dynamics_authority='CarSim', carla_road_force_feedback=False, collision_impulse_feedback=False,
            perception_sensor=False, realtime_concurrent_controller=False,
            geometry_scope='Declared CARLA proxy envelope; not a validated CarSim body collision silhouette',
            continuous_physical_certificate=False, expected_current_map_suffix=EXPECTED_MAP,
            corridor_rule=dict(step_m=2.,length_m=max(35.,self.distance+30.),
                max_heading_difference_deg=1.,max_lateral_deviation_m=.10,
                max_elevation_difference_m=.10,min_lane_width_m=2.8,exclude_junctions=False),
            road_scope='Straight geometric corridor, including straight continuation across empty intersections; no traffic-rule or crossing-traffic safety claim',
            initial_bbox_bottom_road_clearance_m=.02, cg_body_native_m=CG_BODY_NATIVE_M.tolist(),
            cg_offset_provenance='Expected source-backend offset; caller/independent audit must verify actual solver echo',
            source_mu_override=False, ground_contact_claim=False)

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.metadata.update(status='STARTING', started_utc=_stamp())
        _save(self.out/'session.json', self.metadata)
        try:
            self._start()
            return self
        except Exception as exc:
            self.metadata.update(status='FAILED', error=repr(exc))
            self.close()
            raise

    def _start(self):
        check_no_server(psutil, [2100, 2101, 2102])
        config_path = BASE/'validation_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        exe = Path(config['carla_root'])/config['carla_executable_relative']
        if not exe.is_file():
            raise FileNotFoundError(exe)
        userdir = self.out/'unreal_user'
        userdir.mkdir()
        cmd = [str(exe), '-RenderOffScreen', '-quality-level=Low', '-ResX=640', '-ResY=360',
               '-unattended', '-nosound', '-carla-rpc-port=2100', '-carla-streaming-port=2101',
               '-abslog='+str(self.out/'server.log'), '-UserDir='+str(userdir), '-stdout', '-FullStdOutLogOutput']
        self.stdout = (self.out/'server_stdout.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen(cmd, cwd=self.out, stdout=self.stdout,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.metadata.update(owned_server_pid=self.process.pid, server_command=cmd,
            config_sha256=_sha(config_path), executable_path=str(exe))
        _save(self.out/'session.json', self.metadata)
        deadline = time.monotonic()+210.
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Owned CARLA exited during startup: '+str(self.process.returncode))
            try:
                client = carla.Client('127.0.0.1', 2100)
                client.set_timeout(3.)
                version = client.get_server_version()
                world = client.get_world()
                self.client, self.world = client, world
                break
            except (RuntimeError, UnicodeError) as exc:
                last_error = repr(exc)
                time.sleep(1.)
        if self.world is None:
            raise RuntimeError('Owned CARLA startup timeout: '+str(last_error))
        if version != '0.10.0' or self.client.get_client_version() != version:
            raise RuntimeError('CARLA client/server version mismatch')
        self.client.set_timeout(20.)
        map_name = self.world.get_map().name
        if map_name.split('/')[-1] != EXPECTED_MAP:
            raise RuntimeError('Current map differs; no automatic load_world: '+map_name)
        self.old_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = WORLD_DT
        settings.substepping = True
        settings.max_substeps = 1
        settings.max_substep_delta_time = WORLD_DT
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.metadata.update(status='SERVER_READY', carla_version=version, map=map_name)
        _save(self.out/'session.json', self.metadata)
        print('CARLA ready for owned stopline scene', flush=True)

    def _choose_corridor(self):
        road_map = self.world.get_map()
        rule = self.metadata['corridor_rule']
        rejected = []
        starts = [road_map.get_waypoint(spawn.location) for spawn in road_map.get_spawn_points()]
        starts += sorted(road_map.generate_waypoints(2.),key=lambda v:(v.road_id,v.lane_id,v.s))
        for index, wp in enumerate(starts):
            if wp is None:
                rejected.append(dict(spawn_index=index, reason='no_waypoint'))
                continue
            heading = float(wp.transform.rotation.yaw)
            angle = math.radians(heading)
            e = np.array([math.cos(angle), math.sin(angle), 0.])
            lateral = np.array([-e[1], e[0], 0.])
            origin = np.array([wp.transform.location.x, wp.transform.location.y, wp.transform.location.z])
            route = []
            reason = None
            current = wp
            for k in range(math.ceil(rule['length_m']/rule['step_m'])+1):
                p = np.array([current.transform.location.x, current.transform.location.y, current.transform.location.z])
                if rule['exclude_junctions'] and current.is_junction:
                    reason = 'junction'; break
                if current.lane_width < rule['min_lane_width_m']:
                    reason = 'narrow_lane'; break
                if _angle_difference(current.transform.rotation.yaw, heading) > rule['max_heading_difference_deg']:
                    reason = 'heading_change'; break
                if abs(float(np.dot(p-origin, lateral))) > rule['max_lateral_deviation_m']:
                    reason = 'lateral_deviation'; break
                if abs(float(p[2]-origin[2])) > rule['max_elevation_difference_m']:
                    reason = 'elevation_change'; break
                route.append(dict(road_id=current.road_id, lane_id=current.lane_id, s=current.s,
                    xyz=p.tolist(), yaw_deg=current.transform.rotation.yaw, lane_width_m=current.lane_width))
                if k == math.ceil(rule['length_m']/rule['step_m']):
                    break
                candidates = sorted(current.next(rule['step_m']), key=lambda v:(v.road_id,v.lane_id,v.s))
                straight = []
                for candidate in candidates:
                    cp=np.array([candidate.transform.location.x,candidate.transform.location.y,candidate.transform.location.z])
                    if (candidate.lane_width>=rule['min_lane_width_m']
                        and _angle_difference(candidate.transform.rotation.yaw,heading)<=rule['max_heading_difference_deg']
                        and abs(float(np.dot(cp-origin,lateral)))<=rule['max_lateral_deviation_m']
                        and abs(float(cp[2]-origin[2]))<=rule['max_elevation_difference_m']):
                        straight.append(candidate)
                if not straight:
                    reason = 'no_geometrically_straight_continuation'; break
                current = straight[0]
            if reason is not None:
                rejected.append(dict(spawn_index=index, reason=reason, qualified_waypoints=len(route)))
                continue
            self.metadata.update(corridor_selected_spawn_index=index, corridor_rejected=rejected,
                route_waypoints=route, route_heading_deg=heading, road_origin_xyz=origin.tolist())
            return origin, heading
        self.metadata['corridor_rejected'] = rejected
        raise RuntimeError('No qualifying straight/flat corridor on current map')

    def _validate_state(self, state):
        fields = ('time','x','y','z','vx','xo','yo','zo','yaw','pitch','roll')
        if any(k not in state or not math.isfinite(float(state[k])) for k in fields):
            raise ValueError('Native pose missing or nonfinite')
        reconstructed = np.array([state[k] for k in ('xo','yo','zo')],dtype=float)+native_rotation(state)@CG_BODY_NATIVE_M
        error = float(np.max(abs(reconstructed-np.array([state[k] for k in ('x','y','z')],dtype=float))))
        if error > 1e-6:
            raise RuntimeError('Native O-to-CG reconstruction failed: '+str(error))
        return error

    def _pose(self, state):
        native_o = np.array([state[k] for k in ('xo','yo','zo')],dtype=float)
        p = self._anchor_cg + self._world_from_native@(native_o-self._initial_cg)
        r = self._route_rotation@REFLECT@native_rotation(state)@REFLECT
        return carla.Transform(carla.Location(x=float(p[0]),y=float(p[1]),z=float(p[2])), _rotation_from_matrix(r))

    def anchor(self, initial_native):
        if self._anchored or self.world is None:
            raise RuntimeError('anchor requires ready, unanchored session')
        if abs(float(initial_native['time'])) > 1e-10:
            raise ValueError('Initial native anchor must be t=0')
        self._validate_state(initial_native)
        road_origin, heading = self._choose_corridor()
        a = math.radians(heading)
        self._route_rotation = np.array([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]])
        self._world_from_native = self._route_rotation@REFLECT
        self._forward = self._route_rotation[:,0]
        self._lateral = self._route_rotation[:,1]
        self._road_origin = road_origin
        self._anchor_cg = road_origin.copy()
        self._initial_cg = np.array([initial_native[k] for k in ('x','y','z')],dtype=float)
        bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz')
        bp.set_attribute('role_name','paper1_stopline_owned_ego')
        # The actor is spawned close to this lane; physics is disabled before the
        # first tick and the final initial pose is then set in one synchronous batch.
        provisional = self._pose(initial_native)
        provisional.location.z += 1.
        self.ego = self.world.spawn_actor(bp,provisional)
        self.actors.append(self.ego)
        self.ego.set_simulate_physics(False)
        initial_pose = self._pose(initial_native)
        lowest = min(v.z for v in self.ego.bounding_box.get_world_vertices(initial_pose))
        self._anchor_cg[2] += road_origin[2]+.02-lowest
        initial_pose = self._pose(initial_native)
        replies = self.client.apply_batch_sync([carla.command.SetSimulatePhysics(self.ego.id,False),
            carla.command.ApplyTransform(self.ego.id,initial_pose)],False)
        if any(r.error for r in replies):
            raise RuntimeError('Initial actor pose publication failed')
        frame = self.world.tick()
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError('Initial snapshot frame mismatch')
        self._base_elapsed = snapshot.timestamp.elapsed_seconds
        self._last_frame = frame
        self._last_state = dict(initial_native)
        self._last_expected = initial_pose
        self._anchored = True
        initial_geometry = self._observe(snapshot)
        self._line_s = initial_geometry['front_support']+self.distance
        line_center = self._road_origin+self._forward*self._line_s
        left, right = line_center-2.*self._lateral, line_center+2.*self._lateral
        left[2] = right[2] = self._road_origin[2]+.05
        self.world.debug.draw_line(carla.Location(*map(float,left)),carla.Location(*map(float,right)),
            thickness=.12,color=carla.Color(255,40,40),life_time=120.,persistent_lines=False)
        bbox = self.ego.bounding_box
        self.metadata.update(status='ANCHORED', ego_actor_id=self.ego.id, proxy_blueprint=bp.id,
            initial_native_cg_xyz=self._initial_cg.tolist(), map_anchor_cg_xyz=self._anchor_cg.tolist(),
            fixed_vertical_translation_m=float(self._anchor_cg[2]-self._road_origin[2]),
            initial_front_support_m=initial_geometry['front_support'], stopline_s_m=self._line_s,
            stopline_world_xyz=line_center.tolist(), stopline_forward_normal=self._forward.tolist(),
            stopline_draw_endpoints_xyz=[left.tolist(),right.tolist()], baseline_world_frame=frame,
            baseline_world_elapsed_s=self._base_elapsed,
            bbox_location_xyz=[bbox.location.x,bbox.location.y,bbox.location.z],
            bbox_extent_xyz=[bbox.extent.x,bbox.extent.y,bbox.extent.z],
            bbox_rotation_degrees=[bbox.rotation.pitch,bbox.rotation.yaw,bbox.rotation.roll])
        _save(self.out/'session.json',self.metadata)
        obs = self._observe(snapshot)
        self._record(obs)
        return obs

    def _observe(self, snapshot):
        actor_state = snapshot.find(self.ego.id)
        if actor_state is None:
            raise RuntimeError('Owned ego absent from snapshot')
        actual = actor_state.get_transform()
        matrix = np.array(actual.get_matrix(),dtype=float)
        expected = np.array(self._last_expected.get_matrix(),dtype=float)
        points = np.array([[v.x,v.y,v.z] for v in self.ego.bounding_box.get_world_vertices(actual)],dtype=float)
        longitudinal = (points-self._road_origin)@self._forward
        lateral = (points-self._road_origin)@self._lateral
        actual_cg = matrix[:3,3]+matrix[:3,:3]@(REFLECT@CG_BODY_NATIVE_M)
        native_cg = np.array([self._last_state[k] for k in ('x','y','z')],dtype=float)
        expected_cg = self._anchor_cg+self._world_from_native@(native_cg-self._initial_cg)
        front, rear = float(max(longitudinal)),float(min(longitudinal))
        t_native = float(self._last_state['time'])
        relative = float(snapshot.timestamp.elapsed_seconds-self._base_elapsed)
        clock_error = abs(relative-t_native)
        translation_error = float(np.max(abs(matrix[:3,3]-expected[:3,3])))
        rotation_error = float(np.max(abs(matrix[:3,:3]-expected[:3,:3])))
        if clock_error > 2e-6:
            raise RuntimeError('World/native clock mismatch: '+str(clock_error))
        if translation_error > .0001 or rotation_error > 2e-6:
            raise RuntimeError('Applied ego pose mismatch: '+str((translation_error,rotation_error)))
        cg_s = float(np.dot(actual_cg-self._road_origin,self._forward))
        return dict(frame=snapshot.frame,native_stamp_s=t_native,world_stamp_s=float(snapshot.timestamp.elapsed_seconds),
            relative_world_stamp_s=relative,clock_error_s=clock_error,
            native_x=float(self._last_state['x']),native_vx=float(self._last_state['vx']),
            cg_route_s=cg_s,front_support=front,rear_support=rear,
            clearance=None if self._line_s is None else float(self._line_s-front),
            front_offset_from_cg_m=front-cg_s,lateral_support_min_m=float(min(lateral)),
            lateral_support_max_m=float(max(lateral)),bbox_bottom_z_m=float(min(points[:,2])),
            road_anchor_z_m=float(self._road_origin[2]),cg_reconstruction_error_m=self._validate_state(self._last_state),
            cg_publication_error_m=float(np.max(abs(actual_cg-expected_cg))),
            pose_translation_error_m=translation_error,pose_rotation_max_entry_error=rotation_error,
            bbox_world_vertices_json=json.dumps(points.tolist()),actual_cg_world_xyz_json=json.dumps(actual_cg.tolist()),
            expected_matrix_json=json.dumps(expected.tolist()),actual_matrix_json=json.dumps(matrix.tolist()))

    def _record(self, observation):
        if self.world_file is None:
            self.world_file = (self.out/'world.csv').open('w',newline='',encoding='utf-8')
            self.writer = csv.DictWriter(self.world_file,fieldnames=list(observation))
            self.writer.writeheader()
        self.writer.writerow(observation)
        self.world_file.flush()
        self._last_observation = dict(observation)

    def observe(self):
        if not self._anchored:
            raise RuntimeError('observe before anchor')
        snapshot = self.world.get_snapshot()
        if snapshot.frame != self._last_frame:
            raise RuntimeError('World advanced outside owned publish hook')
        return self._observe(snapshot)

    def publish(self, native_state):
        if not self._anchored:
            raise RuntimeError('publish before anchor')
        self._validate_state(native_state)
        if abs(float(native_state['time'])-float(self._last_state['time'])-WORLD_DT)>1e-9:
            raise ValueError('publish requires exactly the next 10 ms native endpoint')
        expected = self._pose(native_state)
        replies = self.client.apply_batch_sync([carla.command.ApplyTransform(self.ego.id,expected)],False)
        if any(r.error for r in replies):
            raise RuntimeError('Owned ego publication failed')
        frame = self.world.tick()
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame or frame != self._last_frame+1:
            raise RuntimeError('World publish frame discontinuity')
        self._last_state,self._last_expected = dict(native_state),expected
        self._last_frame = frame
        observation = self._observe(snapshot)
        self._record(observation)
        return observation

    def close(self):
        if self._closed:
            return
        self._closed = True
        cleanup_errors = []
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except Exception as exc:
                cleanup_errors.append('owned actor: '+repr(exc))
        self.actors.clear()
        if self.world is not None and self.old_settings is not None:
            try:
                self.world.apply_settings(self.old_settings)
            except Exception as exc:
                cleanup_errors.append('restore settings: '+repr(exc))
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    owned = psutil.Process(self.process.pid)
                    children = owned.children(recursive=True)
                    targets = [owned,*children]
                    for process in targets:
                        try:
                            process.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    _,alive = psutil.wait_procs(targets,timeout=8.)
                    for process in alive:
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    self.process.wait(timeout=10.)
                except Exception as exc:
                    cleanup_errors.append('owned process: '+repr(exc))
            self.metadata['owned_server_stopped'] = self.process.poll() is not None
        if self.world_file:
            self.world_file.close()
            self.metadata['world_csv_sha256'] = _sha(self.out/'world.csv')
        if self.stdout:
            self.stdout.close()
        self.metadata.update(finished_utc=_stamp(),cleanup_errors=cleanup_errors,
            cleanup_complete=not cleanup_errors and self.metadata.get('owned_server_stopped',self.process is None),
            last_complete_observation=self._last_observation)
        if self.out.exists():
            _save(self.out/'session.json',self.metadata)

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            self.metadata['status'] = 'SESSION_COMPLETED' if self._anchored else 'SESSION_WITHOUT_TRAJECTORY'
        else:
            self.metadata.update(status='FAILED',error=repr(exc))
        self.close()
        return False
