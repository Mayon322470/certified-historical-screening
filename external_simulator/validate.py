"""Owned-server CARLA capability probe and gated native CarSim evidence capture.

Default mode only checks prerequisites. Negative mode deliberately uses a missing
.sim to detect false-positive RPC success; it is NEVER a CarSim simulation result.
Native mode needs a compatible build, real .sim, solver DLL and solver outputs.
It collects evidence for review, never infers a valid license from API success.
Requires existing carla + psutil; does not install packages or edit CARLA.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    s = p.stat()
    return {"path": str(p), "exists": True, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "sha256": sha256(p)}


def preflight(c):
    problems = []
    if not (Path(c["carla_root"]) / c["carla_executable_relative"]).is_file():
        problems.append("CARLA_EXECUTABLE_MISSING")
    for field in ("carsim_simfile", "carsim_solver_dll", "compatible_build_evidence"):
        if not c.get(field) or not Path(c[field]).is_file():
            problems.append(field.upper() + "_MISSING")
    if not c.get("carsim_output_files"):
        problems.append("CARSIM_OUTPUT_PATHS_NOT_CONFIGURED")
    if c.get("license_status") != "SOLVER_CHECKOUT_VERIFIED":
        problems.append("CARSIM_SOLVER_LICENSE_NOT_VERIFIED")
    dt = c["world_step_s"]
    if dt <= 0 or dt > 0.1:
        problems.append("INVALID_WORLD_STEP")
    for period in c["controller_periods_s"]:
        if dt <= 0 or not math.isclose(period / dt, round(period / dt), abs_tol=1e-8):
            problems.append("CONTROLLER_PERIOD_NOT_INTEGER_WORLD_STEPS")
    internal = c["carsim_expected_internal_step_s"]
    if internal <= 0 or dt <= 0 or not math.isclose(dt / internal, round(dt / internal), abs_tol=1e-8):
        problems.append("SOLVER_STEP_NOT_INTEGER_DIVISOR")
    return problems


@contextlib.contextmanager
def exclusive_lock(root):
    import msvcrt
    p = root / "simulation.lock"
    with p.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def solver_modules(proc):
    return sorted({m.path for m in proc.memory_maps()
                   if any(s in Path(m.path).name.lower() for s in ("carsim", "vehiclesim", "vs_solver"))})


def check_no_server(psutil, ports):
    existing = [p.info for p in psutil.process_iter(["pid", "name"])
                if (p.info["name"] or "").lower().startswith(("carlaunreal", "carlaue4"))]
    if existing:
        raise RuntimeError("Existing CARLA server detected; refusing another instance: " + str(existing))
    for port in ports:
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))


def capture(c, mode, out, result):
    import carla
    import psutil
    root = Path(c["carla_root"])
    exe = root / c["carla_executable_relative"]
    port = c["rpc_port"]
    check_no_server(psutil, [port, port + 1, port + 2])
    if c["host"] != "127.0.0.1":
        raise ValueError("This runner owns and controls a local server only")
    log = out / "server.log"
    userdir = out / "unreal_user"
    userdir.mkdir()
    command = [str(exe), "-RenderOffScreen", "-quality-level=Low", "-ResX=640", "-ResY=360",
               "-unattended", "-nosound", "-carla-rpc-port=" + str(port),
               "-carla-streaming-port=" + str(port + 1), "-abslog=" + str(log),
               "-UserDir=" + str(userdir), "-stdout", "-FullStdOutLogOutput"]
    result.update(command=command, client_version=carla.Client(c["host"], port).get_client_version(),
                  api_path=carla.__file__, python=sys.version,
                  executable=fingerprint(exe), server_log=str(log))
    process = None
    world = None
    original = None
    vehicle = None
    stdout = (out / "server_stdout.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=str(out), stdout=stdout, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        owned = psutil.Process(process.pid)
        result["server_pid"] = process.pid
        client = carla.Client(c["host"], port)
        client.set_timeout(3.0)
        deadline = time.monotonic() + 150
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Owned CARLA server exited with " + str(process.returncode))
            try:
                # A client that first connected before startup can retain a
                # localized rpclib error on Windows. Recreate it for each try.
                client = carla.Client(c["host"], port)
                client.set_timeout(3.0)
                result["server_version"] = client.get_server_version()
                world = client.get_world()
                print("CARLA RPC ready: " + result["server_version"], flush=True)
                break
            except (RuntimeError, UnicodeError) as e:
                last_error = str(e)
                time.sleep(1)
        if world is None:
            raise RuntimeError("CARLA startup timeout: " + last_error)
        if result["server_version"] != result["client_version"]:
            raise RuntimeError("Client/server version mismatch")
        if result["server_version"] != c["expected_carla_version"]:
            raise RuntimeError("Unexpected CARLA server version")
        client.set_timeout(20.0)
        result["map"] = world.get_map().name
        print("Map: " + result["map"], flush=True)
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = c["world_step_s"]
        settings.substepping = True
        settings.max_substep_delta_time = c["world_step_s"]
        settings.max_substeps = 1
        world.apply_settings(settings)
        result["settings"] = {"synchronous_mode": True, "fixed_delta_seconds": c["world_step_s"],
                              "note": "CARLA world step; CarSim internal step needs separate solver evidence"}
        blueprints = world.get_blueprint_library()
        choices = list(blueprints.filter(c["blueprint_preference"]))
        if not choices and mode == "native":
            raise RuntimeError("Configured vehicle blueprint absent; geometry must be selected explicitly")
        choices = choices or list(blueprints.filter("vehicle.*"))
        if not choices:
            raise RuntimeError("No vehicle blueprint")
        bp = sorted(choices, key=lambda b: b.id)[0]
        bp.set_attribute("role_name", "codex_carsim_validation")
        for transform in world.get_map().get_spawn_points():
            transform.location.z += 0.5
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Unable to spawn validation vehicle")
        result["actor"] = {"id": vehicle.id, "blueprint": bp.id}
        for _ in range(10):
            world.tick()
        result["solver_modules_before"] = solver_modules(owned)
        print("Vehicle ready; checking native CarSim activation", flush=True)
        before = [fingerprint(p) for p in c["carsim_output_files"]] if mode == "native" else []
        sim = Path(c["carsim_simfile"]) if mode == "native" else out / "DELIBERATELY_MISSING.sim"
        if mode == "negative" and sim.exists():
            raise RuntimeError("Negative probe requires absent simfile")
        result["simfile"] = fingerprint(sim)
        try:
            returned = vehicle.enable_carsim(str(sim))
            result["enable_carsim"] = {"returned": repr(returned), "exception": None}
        except RuntimeError as e:
            result["enable_carsim"] = {"exception": str(e)}
            if mode == "native":
                raise
        world.tick()
        result["solver_modules_after"] = solver_modules(owned)
        if mode == "negative":
            result["status"] = "NEGATIVE_CAPABILITY_PROBE_COMPLETE"
            result["interpretation"] = "Absent .sim plus RPC return is not evidence of CarSim activation. No genuine co-simulation attempted."
            return
        dll = Path(c["carsim_solver_dll"]).resolve()
        if not any(Path(p).resolve() == dll for p in result["solver_modules_after"]):
            raise RuntimeError("Configured CarSim solver DLL not loaded in CARLA process; refusing default-physics fallback")
        vehicle.use_carsim_road(c["road_source"] == "CARSIM")
        result["solver_dll"] = fingerprint(dll)
        fields = ["frame", "time_s", "delta_s", "throttle", "brake", "steer", "x_m", "y_m", "z_m", "yaw_deg",
                  "vx_mps", "vy_mps", "vz_mps", "ax_mps2", "ay_mps2", "az_mps2"]
        dt = c["world_step_s"]
        previous = world.get_snapshot()
        result["frame_start"] = previous.frame
        with (out / "controls_states.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for k in range(round(c["smoke_duration_s"] / dt)):
                t = k * dt
                throttle, brake = (0.2, 0.0) if t < 1.5 else (0.0, 0.25)
                steer = 0.02 if 0.75 <= t < 1.25 else 0.0
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
                frame = world.tick()
                snap = world.get_snapshot()
                actual_dt = snap.timestamp.elapsed_seconds - previous.timestamp.elapsed_seconds
                if frame != previous.frame + 1 or snap.frame != frame or not math.isclose(actual_dt, dt, abs_tol=1e-6):
                    raise RuntimeError("Frame ownership or fixed-step synchronization failure")
                tr, v, a = vehicle.get_transform(), vehicle.get_velocity(), vehicle.get_acceleration()
                vals = [frame, snap.timestamp.elapsed_seconds, actual_dt, throttle, brake, steer,
                        tr.location.x, tr.location.y, tr.location.z, tr.rotation.yaw,
                        v.x, v.y, v.z, a.x, a.y, a.z]
                if not all(math.isfinite(x) for x in vals):
                    raise RuntimeError("Non-finite state")
                writer.writerow(dict(zip(fields, vals)))
                previous = snap
        result["frame_end"] = previous.frame
        vehicle.destroy()
        vehicle = None
        world.tick()
        after = [fingerprint(p) for p in c["carsim_output_files"]]
        result.update(solver_outputs_before=before, solver_outputs_after=after)
        if not all(p["exists"] and p["bytes"] > 0 and
                   (not old["exists"] or p["sha256"] != old["sha256"] or p["mtime_ns"] != old["mtime_ns"])
                   for old, p in zip(before, after)):
            raise RuntimeError("CarSim outputs did not update during this run")
        result["status"] = "NATIVE_EVIDENCE_CAPTURED_REQUIRES_SOLVER_TRACE_REVIEW"
        result["remaining_verification"] = ["Inspect solver logs for valid license checkout and absence of fallback",
            "Verify solver time, control imports and state exports against controls_states.csv",
            "Check CarSim output coordinates and vehicle size alignment",
            "Confirm no switch to default physics on collision"]
    finally:
        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception as e:
                result["actor_cleanup_error"] = str(e)
        if world is not None and original is not None:
            try:
                world.apply_settings(original)
            except Exception as e:
                result["settings_restore_error"] = str(e)
        if process is not None and process.poll() is None:
            # Only the process created above and its own descendants are stopped.
            children = owned.children(recursive=True)
            owned.terminate()
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs([owned] + children, timeout=8)
            for p in alive:
                p.kill()
            result["owned_server_stopped"] = process.poll() is not None or not owned.is_running()
        stdout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("validation_config.json"))
    parser.add_argument("--mode", choices=["preflight", "negative", "native"], default="preflight")
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8-sig"))
    base = Path(__file__).resolve().parent
    out = base / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.mode)
    out.mkdir(parents=True)
    result = {"mode": args.mode, "run_utc": datetime.now(timezone.utc).isoformat(),
              "is_verified_carla_carsim_result": False, "config": fingerprint(args.config),
              "status": "STARTING", "prerequisite_blocks": preflight(c)}
    code = 0
    try:
        if args.mode == "preflight":
            result["status"] = "BLOCKED" if result["prerequisite_blocks"] else "READY_FOR_NATIVE_ATTEMPT"
            code = 2 if result["prerequisite_blocks"] else 0
        elif args.mode == "native" and result["prerequisite_blocks"]:
            result["status"] = "BLOCKED_BEFORE_SERVER_LAUNCH"
            code = 2
        else:
            with exclusive_lock(base):
                capture(c, args.mode, out, result)
    except Exception as e:
        result.update(status="FAILED", error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        code = 1
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "result": str(out / "result.json"),
                          "verified_cosimulation": False, "error": result.get("error")}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
