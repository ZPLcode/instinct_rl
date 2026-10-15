"""Single-robot parkour inference using native CPU MuJoCo physics."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np
import torch

from .observations import History, ObservationBuilder
from .policy import load_policy
from .scene import build_scene


@contextlib.contextmanager
def passive_viewer(model, data, key_callback):
    """Wait for the viewer's render thread before Python tears down GLFW."""
    import glfw
    from mujoco import viewer as viewer_module

    if not glfw.init():
        raise ValueError(
            "Cannot connect to the desktop display; run from a desktop terminal or use --headless"
        )
    before = set(threading.enumerate())
    viewer = viewer_module.launch_passive(model, data, key_callback=key_callback)
    # MuJoCo 3.11's Handle.close() requests exit but does not join its daemon.
    # The private target identifies only this viewer on our pinned MuJoCo 3.11.
    # That thread also joins the model-load thread before destroying the viewer.
    threads = [
        thread
        for thread in threading.enumerate()
        if thread not in before
        and getattr(thread, "_target", None) is viewer_module._launch_internal
    ]
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                warnings.warn("MuJoCo viewer cleanup timed out", RuntimeWarning)


def robot_control_key(key, backend):
    """Leave the native viewer's display shortcuts available to its UI."""
    if backend == "control":
        return key
    import glfw

    return {
        glfw.KEY_UP: ord("W"),
        glfw.KEY_DOWN: ord("S"),
        glfw.KEY_LEFT: ord("A"),
        glfw.KEY_RIGHT: ord("D"),
        glfw.KEY_F8: ord("X"),
        glfw.KEY_F9: ord("R"),
        glfw.KEY_F10: ord("T"),
        glfw.KEY_F11: ord("C"),
        glfw.KEY_F12: ord("F"),
        glfw.KEY_SPACE: 32,
    }.get(key)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--load-run", help="Run directory or name; defaults to latest g1_parkour run"
    )
    source = result.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", help="Checkpoint path, filename, or iteration (default: latest)"
    )
    source.add_argument(
        "--onnx", type=Path, help="Exported ONNX with matching JSON metadata"
    )
    result.add_argument(
        "--export-onnx",
        type=Path,
        help="Export complete depth encoder + actor before running",
    )
    result.add_argument("--export-only", action="store_true")
    result.add_argument(
        "--terrain",
        choices=("flat", "stairs", "boxes", "gap", "rough", "slope", "parkour"),
        default="parkour",
    )
    result.add_argument(
        "--terrain-geometry",
        choices=("boxes", "heightfield"),
        default="boxes",
        help="Primitive vertical faces, or a sampled heightfield like the training surface",
    )
    result.add_argument(
        "--scene-xml",
        type=Path,
        help="Static terrain XML; trained robot is attached automatically",
    )
    result.add_argument(
        "--step-height", type=float, default=0.10, help="Base obstacle height in metres"
    )
    result.add_argument(
        "--step-width", type=float, default=0.3, help="Stair tread width in metres"
    )
    result.add_argument(
        "--stair-steps",
        type=int,
        default=6,
        help="Steps on each side of the stair platform (1-20)",
    )
    result.add_argument(
        "--roughness",
        type=float,
        default=0.05,
        help="Rough-ground peak height in metres",
    )
    result.add_argument(
        "--slope-angle", type=float, default=10.0, help="Ramp angle in degrees"
    )
    result.add_argument(
        "--duration", type=float, default=60.0, help="Total simulated seconds"
    )
    result.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Forward target speed for target following (m/s)",
    )
    result.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="World target; defaults to course end",
    )
    result.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        help="Fixed body-frame command, replacing target following",
    )
    result.add_argument(
        "--headless", action="store_true", help="No viewer; runs as fast as possible"
    )
    result.add_argument(
        "--viewer",
        choices=("control", "mujoco"),
        default="mujoco",
        help="Native MuJoCo viewer uses arrows/F8-F12; control window uses WASD",
    )
    result.add_argument(
        "--realtime", action="store_true", help="Pace headless rollout to real time too"
    )
    result.add_argument(
        "--auto-reset",
        action="store_true",
        help="Reset after a fall instead of stopping",
    )
    result.add_argument(
        "--observation-noise",
        action="store_true",
        help="Apply saved proprioceptive training noise",
    )
    result.add_argument(
        "--action-delay",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Physics-step action delay (default: 0 for Isaac implicit PD)",
    )
    result.add_argument(
        "--depth-delay",
        type=int,
        choices=(0, 1),
        default=1,
        help="Depth delay in 20 ms control steps",
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--threads", type=int, default=1, help="PyTorch CPU inference threads"
    )
    result.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/parkour_sim2sim/latest_report.json"),
    )
    result.add_argument(
        "--trace", type=Path, help="Optional compressed NumPy trace (.npz)"
    )
    result.add_argument(
        "--video", type=Path, help="Optional MP4; needs a MuJoCo GL rendering context"
    )
    result.add_argument(
        "--show-depth",
        action="store_true",
        help="Overlay the latest depth frame actually fed to the policy in the viewer/video",
    )
    result.add_argument(
        "--show-camera",
        action="store_true",
        help="Show mounted camera RGB, raw depth, actual policy depth and viewing frustum",
    )
    return result


def policy_depth_panel(
    observation: np.ndarray, segments: dict[str, tuple[int, ...]]
) -> np.ndarray:
    """Create an RGB panel from the policy's delayed depth, with fixed [0, 1] scale.

    This reads the exact observation used for the current action. The sensor's
    immediate frame can be newer, so it must not be used for this display.
    """
    import cv2

    offset = 0
    for name, shape in segments.items():
        size = math.prod(shape)
        if name == "depth_image":
            depth = observation[offset : offset + size].reshape(shape)[-1]
            break
        offset += size
    else:
        raise ValueError("--show-depth requires a depth_image observation")
    gray = np.rint(np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
    width = 256
    height = round(width * gray.shape[0] / gray.shape[1])
    scaled = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
    panel = np.full((height + 62, width, 3), 24, dtype=np.uint8)
    panel[26 : 26 + height] = scaled[..., None]
    panel[height + 31 : height + 39] = np.arange(256, dtype=np.uint8)[None, :, None]
    cv2.putText(
        panel,
        "Policy depth (action input)",
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "0 = near                 1 = far",
        (7, height + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def target_command(data, scene, target: np.ndarray, speed: float) -> np.ndarray:
    """Heading controller producing the same body-frame velocity command type."""
    delta = target - data.xpos[scene.root_body_id, :2]
    distance = float(np.linalg.norm(delta))
    if distance < 0.4:
        return np.zeros(3, dtype=np.float32)
    rotation = data.xmat[scene.root_body_id].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    error = math.atan2(
        math.sin(math.atan2(delta[1], delta[0]) - yaw),
        math.cos(math.atan2(delta[1], delta[0]) - yaw),
    )
    body_dx = math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1]
    return np.array(
        [np.clip(2.0 * body_dx, 0.0, speed), 0.0, np.clip(2.0 * error, -1.0, 1.0)],
        dtype=np.float32,
    )


def run(args: argparse.Namespace) -> dict:
    if args.duration <= 0 or not np.isfinite(args.duration):
        raise ValueError("--duration must be positive and finite")
    if args.threads < 1 or not 0 <= args.speed <= 0.8:
        raise ValueError(
            "Use positive --threads and --speed within the trained range [0, 0.8]"
        )
    for name in ("command", "target"):
        if (
            getattr(args, name) is not None
            and not np.isfinite(getattr(args, name)).all()
        ):
            raise ValueError(f"--{name} must be finite")
    if args.export_only and args.export_onnx is None:
        raise ValueError("--export-only requires --export-onnx")
    torch.set_num_threads(args.threads)
    policy = load_policy(
        load_run=args.load_run, checkpoint=args.checkpoint, onnx=args.onnx
    )
    if not policy.env_config:
        raise ValueError("The policy must include its saved environment configuration")
    if args.action_delay is None:
        args.action_delay = 0
    args.show_depth = args.show_depth or args.show_camera
    scene = build_scene(
        args.terrain,
        args.scene_xml,
        env_config=policy.env_config,
        step_height=args.step_height,
        terrain_geometry=args.terrain_geometry,
        step_width=args.step_width,
        stair_steps=args.stair_steps,
        roughness=args.roughness,
        slope_angle=args.slope_angle,
        seed=args.seed,
    )
    if policy.num_actions != len(scene.joint_names):
        raise ValueError("Policy action count does not match robot joints")
    metadata = {
        "joint_names": list(scene.joint_names),
        "default_joint_pos": scene.default_joint_pos.tolist(),
        "action_scale": scene.action_scale.tolist(),
        "physics_dt": scene.physics_dt,
        "control_dt": scene.control_dt,
    }
    if args.onnx:
        if any(key not in policy.metadata for key in metadata):
            raise ValueError(
                "ONNX is missing robot mappings; export it with instinct-sim2sim --export-onnx"
            )
        if policy.metadata["joint_names"] != metadata["joint_names"]:
            raise ValueError("ONNX joint order differs from the native robot")
        for key in ("default_joint_pos", "action_scale", "physics_dt", "control_dt"):
            saved, actual = np.asarray(policy.metadata[key]), np.asarray(metadata[key])
            if saved.shape != actual.shape or not np.allclose(
                saved, actual, rtol=1e-8, atol=1e-10
            ):
                raise ValueError(f"ONNX {key} differs from the native scene")
    if args.export_onnx:
        path = policy.export_onnx(args.export_onnx, metadata_extra=metadata)
        print(f"Exported full policy and metadata: {path}", flush=True)
        if args.export_only:
            return {"export": str(path)}
    print(f"Policy: {args.onnx or policy.metadata['checkpoint']}", flush=True)
    print(
        f"Native MuJoCo CPU | {args.terrain} | physics {1 / scene.physics_dt:g} Hz | "
        f"policy {1 / scene.control_dt:g} Hz | observation {policy.observation_size} -> {policy.num_actions}",
        flush=True,
    )
    data = mujoco.MjData(scene.model)
    scene.reset(data)
    target = np.asarray(
        args.target if args.target is not None else scene.target_xy, dtype=float
    )
    controls = {
        "paused": False,
        "reset": False,
        "show_camera": args.show_camera,
        "follow": True,
        "manual": None
        if args.command is None
        else np.array(args.command, dtype=np.float32),
    }

    def command():
        return (
            controls["manual"].copy()
            if controls["manual"] is not None
            else target_command(data, scene, target, args.speed)
        )

    def on_key(key):
        key = robot_control_key(key, args.viewer)
        if key == 32:
            controls["paused"] = not controls["paused"]
        elif key == ord("C"):
            controls["show_camera"] = not controls["show_camera"]
        elif key == ord("R"):
            controls["reset"] = True
        elif key == ord("F"):
            controls["follow"] = not controls["follow"]
        elif key == ord("T"):
            controls["manual"] = None
        elif key in (ord("W"), ord("S"), ord("A"), ord("D"), ord("X")):
            value = command()
            if key == ord("W"):
                value[0] = min(0.8, value[0] + 0.1)
            elif key == ord("S"):
                value[0] = max(0.0, value[0] - 0.1)
            elif key == ord("A"):
                value[2] = min(1.0, value[2] + 0.2)
            elif key == ord("D"):
                value[2] = max(-1.0, value[2] - 0.2)
            else:
                value[:] = 0
            controls["manual"] = value

    builder = ObservationBuilder(
        scene,
        policy.env_config,
        noise=args.observation_noise,
        seed=args.seed,
        depth_delay=args.depth_delay,
        group_name=policy.metadata.get("policy_observation_group", "policy"),
    )
    observation = builder.reset(data, command())
    if observation.size != policy.observation_size:
        raise ValueError(
            f"Native observation has {observation.size} values; policy expects {policy.observation_size}"
        )
    delayed_targets = History(args.action_delay + 1)
    # Training reset writes zero absolute position targets before the first action.
    delayed_targets.append(np.zeros(policy.num_actions, dtype=np.float32))
    samples = {
        key: []
        for key in ("time", "root_pos", "qpos", "qvel", "action", "command", "episode")
    }
    start_position = data.xpos[scene.root_body_id].copy()
    min_height, max_tilt = float(start_position[2]), 0.0
    episodes, falls, steps, traveled = 1, 0, 0, 0.0
    previous_position = start_position.copy()
    tracking_errors = []
    reason, finite = "duration", True
    terminations = policy.env_config.get("terminations", {})
    min_root_height = (
        terminations.get("root_height", {}).get("params", {}).get("minimum_height", 0.5)
    )
    max_root_tilt = (
        terminations.get("bad_orientation", {})
        .get("params", {})
        .get("limit_angle", 1.0)
    )
    started = time.monotonic()
    final_position = start_position.copy()
    with contextlib.ExitStack() as stack:
        viewer = None
        if not args.headless:

            def status_text():
                value = command()
                return (
                    f"{'MANUAL' if controls['manual'] is not None else 'AUTO'} | "
                    f"vx={value[0]:.2f} m/s yaw={value[2]:.2f} rad/s\n"
                    f"{'PAUSED' if controls['paused'] else 'RUNNING'} | "
                    f"follow={'ON' if getattr(viewer, 'follow', controls['follow']) else 'OFF'}"
                )

            if args.viewer == "control":
                from .control_viewer import ControlViewer

                viewer = stack.enter_context(ControlViewer(scene.model, data, on_key))
                viewer.status_provider = status_text
            else:
                viewer = stack.enter_context(passive_viewer(scene.model, data, on_key))
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                4.0,
                135,
                -20,
            )
            if args.viewer == "control":
                print(
                    "Keys: W/S speed, A/D turn, X stop, T auto, R reset, Space pause, C camera, F follow, Esc quit.",
                    flush=True,
                )
            else:
                print(
                    "Native MuJoCo keys: Up/Down speed, Left/Right turn, F8 stop, F9 reset, "
                    "F10 auto, F11 camera, F12 follow, Space pause. WASD retains viewer shortcuts.",
                    flush=True,
                )

            def sync_viewer():
                if args.viewer == "mujoco":
                    help_text = (
                        "Up/Down speed | Left/Right turn\n"
                        "F8 stop | F9 reset | F10 auto\n"
                        "F11 camera | F12 follow | Space pause\n"
                    )
                    viewer.set_texts(
                        (
                            mujoco.mjtFont.mjFONT_NORMAL,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            help_text + status_text(),
                            "",
                        )
                    )
                viewer.sync()

        camera_views = None
        if args.show_camera and (viewer or args.video):
            from .camera_visualization import CameraViews, camera_frustum

            camera_views = CameraViews(scene.model, builder.camera)
            stack.callback(camera_views.close)
        renderer, writer, camera = None, None, None
        if args.video:
            import cv2

            args.video.parent.mkdir(parents=True, exist_ok=True)
            video_width, video_height = (960, 720) if args.show_camera else (640, 480)
            renderer = mujoco.Renderer(
                scene.model, height=video_height, width=video_width
            )
            stack.callback(renderer.close)
            writer = cv2.VideoWriter(
                str(args.video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (video_width, video_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video output: {args.video}")
            stack.callback(writer.release)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 4.0, 135, -20
        try:
            while steps < math.ceil(args.duration / scene.control_dt):
                tick = time.monotonic()
                if viewer and not viewer.is_running():
                    reason = "viewer_closed"
                    break
                if controls["reset"]:
                    scene.reset(data)
                    observation = builder.reset(data, command())
                    delayed_targets.clear()
                    delayed_targets.append(
                        np.zeros(policy.num_actions, dtype=np.float32)
                    )
                    previous_position = data.xpos[scene.root_body_id].copy()
                    episodes += 1
                    controls["reset"] = False
                if controls["paused"]:
                    if viewer:
                        sync_viewer()
                    time.sleep(0.02)
                    continue
                # Commands sampled after physics enter the next policy observation.
                depth_panel = (
                    policy_depth_panel(observation, policy.observation_segments)
                    if args.show_depth
                    else None
                )
                action = policy(observation)
                targets = scene.default_joint_pos + scene.action_scale * action
                applied_command = command()
                for _ in range(scene.decimation):
                    data.ctrl[scene.position_actuator_ids] = delayed_targets.append(
                        targets
                    )[0]
                    data.ctrl[scene.velocity_actuator_ids] = 0.0
                    mujoco.mj_step(scene.model, data)
                mujoco.mj_forward(scene.model, data)
                steps += 1
                finite = bool(
                    np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                )
                if not finite:
                    reason = "nonfinite_state"
                    break
                final_position = data.xpos[scene.root_body_id].copy()
                rotation = data.xmat[scene.root_body_id].reshape(3, 3)
                tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
                min_height = min(min_height, float(final_position[2]))
                max_tilt = max(max_tilt, tilt)
                traveled += float(
                    np.linalg.norm(final_position[:2] - previous_position[:2])
                )
                velocity = rotation.T @ (
                    (final_position - previous_position) / scene.control_dt
                )
                tracking_errors.append(
                    float(np.linalg.norm(velocity[:2] - applied_command[:2]))
                )
                previous_position = final_position.copy()
                observation = builder.compute(data, command(), action)
                if camera_views and controls["show_camera"]:
                    depth_panel = camera_views.panel(data, depth_panel)
                if args.trace:
                    for key, value in (
                        ("time", steps * scene.control_dt),
                        ("root_pos", final_position),
                        ("qpos", data.qpos.copy()),
                        ("qvel", data.qvel.copy()),
                        ("action", action),
                        ("command", applied_command),
                        ("episode", episodes),
                    ):
                        samples[key].append(value)
                if viewer:
                    with viewer.lock():
                        if getattr(viewer, "follow", controls["follow"]):
                            viewer.cam.lookat[:] = final_position
                        if camera_views:
                            viewer.user_scn.ngeom = 0
                            if controls["show_camera"]:
                                camera_frustum(viewer.user_scn, builder.camera, data)
                    if depth_panel is not None:
                        viewport = viewer.viewport
                        height, width = depth_panel.shape[:2]
                        rect = mujoco.MjrRect(
                            viewport.left + max(0, viewport.width - width - 12),
                            viewport.bottom + 12,
                            width,
                            height,
                        )
                        # MuJoCo 3.11 Handle.set_images flips top-first RGB for OpenGL.
                        # Passing the panel unchanged here avoids a second vertical flip.
                        viewer.set_images((rect, depth_panel))
                    sync_viewer()
                if renderer and (steps - 1) % 2 == 0:
                    camera.lookat[:] = final_position
                    renderer.update_scene(data, camera=camera)
                    if camera_views and controls["show_camera"]:
                        camera_frustum(renderer.scene, builder.camera, data)
                    frame = renderer.render()
                    if depth_panel is not None:
                        height, width = depth_panel.shape[:2]
                        frame[-height - 12 : -12, -width - 12 : -12] = depth_panel
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if steps % 250 == 0:
                    print(
                        f"t={steps * scene.control_dt:.1f}s x={final_position[0]:.2f} "
                        f"y={final_position[1]:.2f} z={final_position[2]:.2f} falls={falls}",
                        flush=True,
                    )
                if tilt > max_root_tilt or final_position[2] < min_root_height:
                    falls += 1
                    if args.auto_reset:
                        controls["reset"] = True
                    else:
                        reason = "fall"
                        break
                if args.realtime or viewer:
                    time.sleep(max(0, scene.control_dt - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            reason = "interrupted"
    elapsed = time.monotonic() - started
    report = {
        "backend": "native MuJoCo CPU",
        "checkpoint": policy.metadata.get("checkpoint"),
        "onnx": str(args.onnx) if args.onnx else None,
        "terrain": scene.metadata,
        "control_steps": steps,
        "simulated_seconds": steps * scene.control_dt,
        "wall_seconds": elapsed,
        "realtime_factor": steps * scene.control_dt / max(elapsed, 1e-9),
        "termination": reason,
        "finite": finite,
        "falls": falls,
        "episodes": episodes,
        "initial_root_position": start_position.tolist(),
        "final_root_position": final_position.tolist(),
        "displacement_x": float(final_position[0] - start_position[0]),
        "path_length_xy": traveled,
        "min_root_height": min_height,
        "max_tilt_radians": max_tilt,
        "mean_xy_velocity_error": float(np.mean(tracking_errors))
        if tracking_errors
        else None,
        "target_xy": target.tolist(),
        "distance_to_target": float(np.linalg.norm(final_position[:2] - target)),
        "observation_noise": args.observation_noise,
        "action_delay_physics_steps": args.action_delay,
        "depth_delay_control_steps": args.depth_delay,
        "seed": args.seed,
        **metadata,
        "show_depth": args.show_depth,
        "show_camera": args.show_camera,
        "training_backend": policy.metadata.get("training_backend"),
        "viewer_backend": None if args.headless else args.viewer,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("wb") as stream:
            np.savez_compressed(
                stream, **{key: np.asarray(value) for key, value in samples.items()}
            )
    print(
        f"Finished: {reason}; {steps * scene.control_dt:.2f}s, x={final_position[0]:.2f}m, "
        f"falls={falls}. Report: {args.report}",
        flush=True,
    )
    return report


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        report = run(args)
    except (ValueError, FileNotFoundError) as error:
        argument_parser.error(str(error))
    if report.get("termination") in ("fall", "nonfinite_state"):
        raise SystemExit(2)
