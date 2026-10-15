"""Controlled depth ablations and foot/obstacle contact measurements.

Run with ``python -m parkour_mujoco.diagnose_vision``.
Every condition starts from the same physical state. The flat condition hides
obstacles only from the depth camera; the robot still collides with them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .observations import History, ObservationBuilder
from .policy import load_policy
from .run import target_command
from .scene import build_scene


def side_contact_forces(scene, data) -> np.ndarray:
    """Per-foot contact force on primitive vertical obstacle faces, in newtons.

    This is a contact proxy, not a fall detector: contact normal must be within
    30 degrees of horizontal. Ground/support contacts are excluded.
    """
    forces = np.zeros(2)
    for contact_id, contact in enumerate(data.contact):
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        names = (scene.model.geom(geom1).name, scene.model.geom(geom2).name)
        foot, obstacle = (
            (geom1, geom2) if names[1].startswith("terrain_") else (geom2, geom1)
        )
        if not scene.model.geom(obstacle).name.startswith(
            ("terrain_step_", "terrain_box_")
        ):
            continue
        body_name = scene.model.body(scene.model.geom_bodyid[foot]).name
        if body_name not in (
            "robot/left_ankle_roll_link",
            "robot/right_ankle_roll_link",
        ):
            continue
        if abs(float(contact.frame[2])) >= 0.5:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(scene.model, data, contact_id, force)
        forces[0 if "left_" in body_name else 1] += np.linalg.norm(force[:3])
    return forces


def evaluate(
    policy,
    terrain: str,
    mode: str,
    duration: float,
    step_height: float,
    speed: float,
    output: Path,
    initial_x: float = 0.0,
    terrain_geometry: str = "boxes",
) -> dict:
    scene = build_scene(
        terrain,
        env_config=policy.env_config,
        step_height=step_height,
        terrain_geometry=terrain_geometry,
    )
    data = mujoco.MjData(scene.model)
    scene.reset(data)
    data.qpos[0] += initial_x
    mujoco.mj_forward(scene.model, data)
    builder = ObservationBuilder(scene, policy.env_config)
    command = lambda: target_command(data, scene, scene.target_xy, speed)
    observation = builder.reset(data, command())
    depth_size = int(np.prod(policy.observation_segments["depth_image"]))
    names = list(policy.observation_segments)
    if names[-1] != "depth_image":
        raise ValueError(
            "Diagnostic expects depth_image last in the policy observation"
        )
    flat_scene, flat_data, flat_camera, flat_history = None, None, None, None
    if mode == "flat":
        flat_scene = build_scene("flat", env_config=policy.env_config)
        flat_data = mujoco.MjData(flat_scene.model)
        flat_camera = ObservationBuilder(flat_scene, policy.env_config).camera
        flat_history = History(builder.depth_history.length)
    frozen = observation[-depth_size:].copy()

    def intervene(obs):
        obs = obs.copy()
        if mode == "flat":
            flat_data.qpos[:] = data.qpos
            flat_data.qvel[:] = data.qvel
            mujoco.mj_forward(flat_scene.model, flat_data)
            history = flat_history.append(flat_camera.capture(flat_data))
            obs[-depth_size:] = history[-1 - builder.depth_offsets].ravel()
        elif mode == "frozen":
            obs[-depth_size:] = frozen
        elif mode == "far":
            obs[-depth_size:] = 1.0
        return obs

    observation = intervene(observation)
    delay = History(2)
    delay.append(np.zeros(policy.num_actions))
    record = {
        key: []
        for key in (
            "time",
            "qpos",
            "qvel",
            "root_pos",
            "actions",
            "depth",
            "side_force",
        )
    }
    previous_contact = np.zeros(2, dtype=bool)
    events = []
    contact_steps = 0
    total_impulse = 0.0
    termination = "duration"
    max_tilt = 0.0
    for step in range(math.ceil(duration / scene.control_dt)):
        used_depth = (
            observation[-depth_size:]
            .reshape(policy.observation_segments["depth_image"])[-1]
            .copy()
        )
        action = policy(observation)
        targets = scene.default_joint_pos + scene.action_scale * action
        max_forces = np.zeros(2)
        for _ in range(scene.decimation):
            data.ctrl[scene.position_actuator_ids] = delay.append(targets)[0]
            data.ctrl[scene.velocity_actuator_ids] = 0.0
            mujoco.mj_step(scene.model, data)
            forces = side_contact_forces(scene, data)
            total_impulse += float(forces.sum() * scene.physics_dt)
            max_forces = np.maximum(max_forces, forces)
        mujoco.mj_forward(scene.model, data)
        pos = data.xpos[scene.root_body_id].copy()
        contacts = max_forces > 10.0
        contact_steps += int(contacts.any())
        for foot in np.flatnonzero(contacts & ~previous_contact):
            events.append(
                {
                    "time": (step + 1) * scene.control_dt,
                    "foot": ("left", "right")[foot],
                    "root_x": float(pos[0]),
                    "force_n": float(max_forces[foot]),
                }
            )
        previous_contact = contacts
        for key, value in (
            ("time", (step + 1) * scene.control_dt),
            ("qpos", data.qpos.copy()),
            ("qvel", data.qvel.copy()),
            ("root_pos", pos),
            ("actions", action),
            ("depth", used_depth),
            ("side_force", max_forces),
        ):
            record[key].append(value)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        tilt = math.acos(
            float(np.clip(data.xmat[scene.root_body_id].reshape(3, 3)[2, 2], -1, 1))
        )
        max_tilt = max(max_tilt, tilt)
        if not finite or tilt > 1.0 or pos[2] < 0.5:
            termination = "fall" if finite else "nonfinite"
            break
        observation = intervene(builder.compute(data, command(), action))
    stem = f"{terrain}_{mode}_x{initial_x:g}"
    arrays = {key: np.asarray(value) for key, value in record.items()}
    np.savez_compressed(output / f"{stem}.npz", **arrays)
    result = {
        "mode": mode,
        "terrain": terrain,
        "terrain_geometry": terrain_geometry,
        "initial_x": initial_x,
        "step_height": step_height,
        "speed_command": speed,
        "checkpoint": policy.metadata["checkpoint"],
        "duration": float(arrays["time"][-1]),
        "termination": termination,
        "final_root_position": pos.tolist(),
        "max_root_x": float(arrays["root_pos"][:, 0].max()),
        "distance_to_target": float(np.linalg.norm(pos[:2] - scene.target_xy)),
        "max_tilt_rad": max_tilt,
        "side_contact_control_steps": contact_steps
        if terrain_geometry == "boxes"
        else None,
        "side_contact_seconds": contact_steps * scene.control_dt
        if terrain_geometry == "boxes"
        else None,
        "side_contact_force_threshold_n": 10,
        "side_contact_impulse_ns": total_impulse
        if terrain_geometry == "boxes"
        else None,
        "side_contact_events": events if terrain_geometry == "boxes" else None,
        "contact_metric_note": "Foot contacts >10 N on primitive obstacle faces with |normal_z| < 0.5. "
        "Not a loss-of-balance count; unavailable for heightfields, whose faces are sloped.",
    }
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "mode",
                    "initial_x",
                    "termination",
                    "max_root_x",
                    "side_contact_control_steps",
                    "side_contact_impulse_ns",
                )
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="27000")
    parser.add_argument("--load-run", default="2026-10-04_00-31-53_5060ti_2048")
    parser.add_argument(
        "--terrain", choices=("stairs", "boxes", "parkour"), default="parkour"
    )
    parser.add_argument(
        "--terrain-geometry", choices=("boxes", "heightfield"), default="boxes"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("real", "flat", "frozen", "far"),
        default=["real", "flat", "frozen", "far"],
    )
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--step-height", type=float, default=0.10)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--initial-x", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/parkour_sim2sim/vision_ablation")
    )
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    if not 0 <= args.speed <= 0.8 or not np.isfinite(args.initial_x).all():
        parser.error("--speed must be in [0, .8], --initial-x must be finite")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    policy = load_policy(load_run=args.load_run, checkpoint=args.checkpoint)
    results = []
    for initial_x in args.initial_x:
        for mode in args.modes:
            results.append(
                evaluate(
                    policy,
                    args.terrain,
                    mode,
                    args.duration,
                    args.step_height,
                    args.speed,
                    args.output,
                    initial_x,
                    args.terrain_geometry,
                )
            )
            (args.output / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
