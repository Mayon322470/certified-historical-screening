"""Pure fake-backend checks of runner decisions; never a vehicle experiment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner_smoke_target', HERE/'run_estimator_closed_loop.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Vehicle:
    def __init__(self, out, speed, duration):
        self.last_t = 0.
        self.k = 0
        self.state = dict(time=0., x=0., vx=speed, ax=0.)
        self.metadata = {'fake_backend': True}
        self.speed = speed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def sample(self, command):
        t = self.k*.0005
        self.k += 1
        self.last_t = t
        self.state = dict(time=t, x=t*self.speed, vx=self.speed, ax=0.)
        return dict(self.state, stop_requested=False)


class Tracker:
    def command(self, raw, before, dt):
        return (0., 0., 0.), {}


class Scene:
    def __init__(self, out, distance):
        self.metadata = {'stopline_s_m': distance}
        self.distance = distance
        self.state = dict(time=0., x=0., vx=1.)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def observe(self):
        return dict(cg_route_s=self.state['x'], clearance=self.distance-self.state['x'],
                    actual_cg_world_xyz_json='[0,0,0]',
                    bbox_world_vertices_json='[[1,0,0],[-1,0,0]]')

    def anchor(self, state):
        self.state = state
        return self.observe()

    def publish(self, state):
        self.state = state
        return self.observe()


class Estimators:
    force_query = False
    unknown = False
    seen_rows = []

    class Sensor:
        def __init__(self, seed, bias, s0, landmarks):
            self.metadata = {'fake_sensor': True}

        def observe(self, obs, native_speed, time):
            # Truth is intentionally incompatible with the guard measurement.
            return dict(stamp_s=time, odom_m=0., speed_mps=0., s0_m=0.,
                        rows=[dict(stamp_s=time, landmark_id=j) for j in range(3)])

    @staticmethod
    def guard_bounds(packet, interval):
        prior = interval == [-.8, .8]
        upper = 100. if prior and Estimators.force_query else 0.
        return dict(valid=True, s_lower=upper, s_upper=upper, v_lower=0., v_upper=0.)

    @staticmethod
    def estimate(method, rows, prior, deadline):
        assert deadline > time.perf_counter()
        Estimators.seen_rows.append(rows)
        return dict(status='UNKNOWN' if Estimators.unknown else 'OK',
                    b_interval=None if Estimators.unknown else [-.01, .01], elapsed_s=0.,
                    cells_completed=8, frontend_calls=8, deleted=0)


class Backup:
    @staticmethod
    def next_period_required(v_upper, decision_period, measurement_pad, deadline_delay):
        assert v_upper == 0., 'Guard received native speed rather than sanitized measurement'
        return 1.

    @staticmethod
    def prefix_upper(v, t):
        return 100.


def run_checks():
    checks = []
    with tempfile.TemporaryDirectory(prefix='paper1_runner_smoke_') as temporary:
        runner.RESULTS = Path(temporary)/'fake_results'
        runner.PLAN = Path(temporary)/'fake_plan.json'
        runner.PLAN.write_text('{}', encoding='utf-8')
        plan = dict(duration_s=.4, tracker_dt_s=.01, native_dt_s=.0005,
                    decision_period_s=.1, compute_budget_s=.1, response_delay_s=.1,
                    max_observation_age_s=.02, measurement_pad_m=.005, prior=[-.8, .8],
                    landmark_offsets=[dict(s=s, lateral=3) for s in (12, 26, 46)],
                    emergency_command=[0., 2.5, 0.], raw_request=dict(before=0., after=.75, switch_s=1.))
        for label, fault, query, unknown, expected, brake_time in (
            ('quick', None, False, False, 'QUICK_PRIOR_ACCEPT', None),
            ('query', None, True, False, 'ACCEPT_NORMAL', None),
            ('stale', dict(kind='drop', at_s=.1), False, False, 'STALE_OBSERVATION', .2),
            ('late', dict(kind='late', at_s=.1, reported_elapsed_s=.15), False, False, 'FORCED_COMPUTE_TIMEOUT', .2),
            ('unknown', None, True, True, 'ESTIMATOR_UNKNOWN', .1),
        ):
            Estimators.force_query, Estimators.unknown = query, unknown
            case = dict(id=label, method='ma', speed_mps=1., bias=0., seed=1,
                        initial_front_clearance_m=40.)
            if fault:
                case['fault'] = fault
            result = runner.run_case(case, plan, Vehicle, Tracker, Scene, Estimators, Backup)
            assert result['status'] == 'SAMPLED_NO_STOPLINE_CROSSING', result
            assert result['emergency_applied_s'] == brake_time, result
            records = [json.loads(line) for line in (runner.RESULTS/label/'decision.jsonl').read_text().splitlines()]
            assert any(record['status'] == expected for record in records)
            if label == 'query':
                assert [r['stamp_s'] for r in records[2]['selected_rows']] == [0., .1, .2]
            if label == 'quick':
                assert result['estimator_calls'] == 0 and result['quick_prior_accept_count'] == 4
            checks.append(dict(case=label, status='PASS', expected_status=expected,
                               expected_emergency_s=brake_time))
    return dict(status='PASS_PURE_FAKE_BACKEND_RUNNER_CHECKS',
                runner_source_sha256=runner.sha(HERE/'run_estimator_closed_loop.py'),
                smoke_source_sha256=runner.sha(Path(__file__)),
                native_solver_started=False, carla_started=False,
                checks=checks, fake_logs_discarded=True)


if __name__ == '__main__':
    result = run_checks()
    target = HERE.parent/'audit'/'RUNNER_SMOKE_CHECKS.json'
    runner.save(target, result)
    print(json.dumps(result))
