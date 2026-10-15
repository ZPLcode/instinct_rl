"""Bounded native Isaac checkpoint evaluation, with paired initial-state evidence.

Run with the IsaacLab Python environment. No training or reward-preset import occurs.
The analysis helpers also run on CPU: --analyze-only EXISTING_OUTPUT_DIRECTORY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
# Fixed tile levels for repeatable terrain comparisons. Disabling
# curriculum alone would skip the initial reset's level decrement and change them.
REFERENCE_LEVELS = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
EVALUATION_SCHEMA_VERSION = 2
RESET_PROTOCOL = (
    "native_reset_then_refresh_target_command_and_rebuild_observation_history_v2"
)


def paired_reset(env):
    """Discard stale target-derived commands before the first policy action.

    CommandTerm.reset resamples its target but does not call _update_command.
    Consequently native reset's first observation can contain the previous
    episode's velocity command, repeated across its eight history entries.
    Keep the native scene/sensor/action reset, refresh only this derived command,
    then rebuild observation terms and histories using the normal wrapper layout.
    This performs no physics step and does not resample the target.
    """
    env.reset()
    raw = env.unwrapped
    raw.command_manager.get_term("base_velocity")._update_command()
    raw.observation_manager.reset()
    raw.obs_buf = raw.observation_manager.compute(update_history=True)
    obs_pack = env._flatten_all_obs_groups(raw.obs_buf)
    return obs_pack["policy"], {"observations": obs_pack}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def digest_arrays(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def capsule_geometry(urdf):
    """Isaac imports these seven cylinders as capsules with the same axial length."""
    from scipy.spatial.transform import Rotation

    model = ET.parse(urdf).getroot()
    result = []
    for name in FEET:
        capsules = []
        for collision in model.find(f"link[@name='{name}']").findall("collision"):
            cylinder = collision.find("geometry/cylinder")
            if cylinder is None:
                raise ValueError(f"Unexpected non-cylinder shoe collision: {name}")
            origin = collision.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            axis = Rotation.from_euler("xyz", rpy).apply(
                [0, 0, float(cylinder.get("length")) / 2]
            )
            capsules.append(
                dict(
                    endpoints=[(xyz - axis).tolist(), (xyz + axis).tolist()],
                    radius=float(cylinder.get("radius")),
                )
            )
        result.append(capsules)
    return result


def gap_geometry(vertices, faces, origin):
    local = vertices - origin
    good = (np.abs(local[:, 0]) < 3.8) & (np.abs(local[:, 1]) < 0.8)
    triangles = local[faces[np.all(good[faces], axis=1)]]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(lengths[:, None], 1e-12)
    vertical = (
        (lengths > 1e-9)
        & (np.abs(normal[:, 0]) > 0.99)
        & (triangles[:, :, 0].mean(1) > 0)
    )
    walls = np.unique(np.round(triangles[vertical, :, 0], 6))
    if len(walls) != 2:
        raise ValueError(f"Square-gap tile requires two east faces; got {walls}")
    return dict(
        near_x=float(walls[0]),
        far_x=float(walls[1]),
        width=float(walls[1] - walls[0]),
        top_z=float(triangles[:, :, 2].max()),
        bottom_z=float(triangles[:, :, 2].min()),
    )


def gap_result(a, ids, env_id, manifest, root, force):
    from scipy.spatial.transform import Rotation

    g = manifest["square_gap_geometry"]
    origin = np.asarray(manifest["env_origins"][env_id])
    foot = a["foot_pos_w"][ids, env_id] - origin
    quat = a["foot_quat_w"][ids, env_id].reshape(-1, 4)
    rot = (
        Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix().reshape(len(ids), 2, 3, 3)
    )
    capsules = manifest["shoe_capsules"]
    endpoints = np.array([[c["endpoints"] for c in cs] for cs in capsules])
    radius = np.array([[c["radius"] for c in cs] for cs in capsules])
    centers = foot[:, :, None, None, :] + np.einsum("tfij,fcqj->tfcqi", rot, endpoints)
    shoe_min = (centers - radius[None, :, :, None, None]).min(axis=(2, 3))
    near, far, top, bottom = (g[k] for k in ["near_x", "far_x", "top_z", "bottom_z"])
    enter, leave = np.flatnonzero(root[:, 0] >= near), np.flatnonzero(root[:, 0] >= far)
    corridor = bool(
        len(enter)
        and len(leave)
        and leave[0] >= enter[0]
        and np.max(np.abs(root[enter[0] : leave[0] + 1, 1])) <= 0.75
    )
    support = (
        (force[:, :, 2] > 20)
        & (shoe_min[:, :, 0] > far + 0.005)
        & (np.abs(shoe_min[:, :, 2] - top) < 0.08)
    )
    first_support = [
        int(k[0]) if len(k := np.flatnonzero(support[:, j])) else None for j in range(2)
    ]
    inside = (
        (foot[:, :, 0] >= near - 0.03)
        & (foot[:, :, 0] <= far + 0.03)
        & (np.abs(foot[:, :, 1]) <= 0.75)
    )
    bottom_support = (
        inside & (force[:, :, 2] > 20) & (shoe_min[:, :, 2] <= bottom + 0.08)
    )
    low_root = (
        (root[:, 0] >= near - 0.15)
        & (root[:, 0] <= far + 0.15)
        & (np.abs(root[:, 1]) <= 0.75)
        & (root[:, 2] - top < 0.5)
    )
    fell = bool(np.any(bottom_support) or np.any(low_root))
    return dict(
        east_corridor_crossed=corridor,
        both_feet_far_support=all(x is not None for x in first_support),
        first_far_support_s=[
            float(a["t"][ids[x]]) if x is not None else None for x in first_support
        ],
        root_cross_time_s=float(a["t"][ids[leave[0]]]) if len(leave) else None,
        pit_floor_support=bool(np.any(bottom_support)),
        fell_into_gap=fell,
        deep_foot_intrusion=bool(np.any(inside & (shoe_min[:, :, 2] < top - 0.15))),
        first_pit_fall_s=float(
            a["t"][ids[np.flatnonzero(np.any(bottom_support, axis=1) | low_root)[0]]]
        )
        if fell
        else None,
    )


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text())
    a = np.load(output / "trace.npz")
    dt = manifest["physics_dt"]
    records = []
    for trial in np.unique(a["trial"]):
        for env_id in range(10):
            ids = np.flatnonzero((a["trial"] == trial) & (a["episode"][:, env_id] == 0))
            if not len(ids):
                continue
            t = a["t"][ids]
            origin = np.asarray(manifest["env_origins"][env_id])
            root_w = a["root_pos_w"][ids, env_id]
            root = root_w - origin
            force = a["contact_force_w"][ids, env_id]
            horizontal = np.linalg.norm(force[:, :, :2], axis=-1)
            hit = (horizontal > 20) & (horizontal > 1.5 * np.maximum(force[:, :, 2], 0))
            events = []
            for foot in range(2):
                indices = np.flatnonzero(hit[:, foot])
                for chunk in np.split(
                    indices, np.flatnonzero(np.diff(indices) > 1) + 1
                ):
                    if not len(chunk):
                        continue
                    peak = int(chunk[np.argmax(horizontal[chunk, foot])])
                    events.append(
                        dict(
                            foot=FEET[foot],
                            start_s=float(t[chunk[0]]),
                            end_s=float(t[chunk[-1]]),
                            duration_s=float(len(chunk) * dt),
                            peak_horizontal_N=float(horizontal[peak, foot]),
                            horizontal_impulse_Ns=float(
                                horizontal[chunk, foot].sum() * dt
                            ),
                            force_at_peak_w=force[peak, foot].tolist(),
                            foot_pos_at_peak_w=a["foot_pos_w"][
                                ids[peak], env_id, foot
                            ].tolist(),
                        )
                    )
            events.sort(key=lambda event: event["start_s"])
            distance = np.linalg.norm(
                root_w[:, :2] - a["target_pos_w"][ids, env_id, :2], axis=1
            )
            height = root_w[:, 2] - min(float(origin[2]), 0.0)
            tilt = np.arccos(np.clip(-a["projected_gravity_b"][ids, env_id, 2], -1, 1))
            upright = (height >= 0.5) & (tilt < 0.6)
            stable_arrival = None
            hold = int(np.ceil(0.5 / dt))
            for i in np.flatnonzero(distance <= 0.4):
                if i + hold < len(ids) and np.all(upright[i : i + hold + 1]):
                    stable_arrival = float(t[i + hold])
                    break
            resets = [
                event
                for event in manifest["resets"]
                if event["trial"] == int(trial)
                and event["env"] == env_id
                and event["episode"] == 0
            ]
            bad_reset = any(
                any(
                    reason not in ["time_out", "dataset_exhausted"]
                    for reason in ev["reasons"]
                )
                for ev in resets
            )
            rec = dict(
                trial=int(trial),
                seed=manifest["trial_seeds"][int(trial)],
                env=env_id,
                terrain=manifest["column_terrain_names"][env_id],
                level=manifest["terrain_levels"][env_id],
                first_episode_duration_s=float(t[-1]),
                events=events,
                event_count=len(events),
                horizontal_impulse_Ns=float(horizontal[hit].sum() * dt),
                maximum_horizontal_force_N=float(horizontal.max()),
                final_goal_distance_m=float(distance[-1]),
                minimum_goal_distance_m=float(distance.min()),
                goal_reached=bool(np.any(distance <= 0.4)),
                stable_goal_time_s=stable_arrival,
                low_root=bool(np.any(height < 0.5)),
                bad_orientation=bool(np.any(tilt > 1.0)),
                fell=bool(np.any(height < 0.5) or np.any(tilt > 1.0) or bad_reset),
                minimum_root_height_m=float(height.min()),
                maximum_tilt_rad=float(tilt.max()),
                final_root_local=root[-1].tolist(),
                resets=resets,
            )
            if env_id == manifest["gap_env"]:
                gap = gap_result(a, ids, env_id, manifest, root, force)
                completed = (
                    gap["east_corridor_crossed"]
                    and gap["both_feet_far_support"]
                    and stable_arrival is not None
                )
                gap["completed"] = completed
                if completed:
                    gap["outcome"] = "direct_cross"
                    if gap["fell_into_gap"]:
                        gap["outcome"] = (
                            "fell_then_recovered"
                            if gap["first_pit_fall_s"] <= stable_arrival
                            else "crossed_then_fell"
                        )
                else:
                    gap["outcome"] = (
                        "fell_in_gap" if gap["fell_into_gap"] else "not_completed"
                    )
                rec["gap"] = gap
            records.append(rec)
    result = dict(
        checkpoint=manifest["checkpoint"],
        terrain_sha256=manifest["terrain_sha256"],
        contact_criterion="Per-foot H > 20 N and H > 1.5 * max(Fz, 0). Event = contiguous 200 Hz samples. Impulse = integral of horizontal resultant magnitude during qualifying samples. This flags side-force candidates, not geometric proof for every event.",
        fall_criterion="Any root_z - min(origin_z,0) < 0.5 m, tilt > 1 rad, or non-time-out failure. PLAY low-height termination remains disabled; this is an independent monitor.",
        goal_criterion="Goal distance <=0.4 m, followed by 0.5 s upright (height>=0.5 m and tilt<0.6 rad). Only episode 0 per trial.",
        gap_criterion="Actual imported mesh east crossing within |local y|<=0.75; each entire shoe beyond far edge by 5 mm with upward normal force>20 N and sole within 8 cm of shore. Pit support means sole within 8 cm of pit floor with upward force>20 N, or low root while over pit.",
        trials=records,
    )
    for env_id, key in [(5, "stairs_env5"), (manifest["gap_env"], "square_gaps")]:
        selected = [r for r in records if r["env"] == env_id]
        result[key] = dict(
            attempts=len(selected),
            total_events=sum(r["event_count"] for r in selected),
            total_horizontal_impulse_Ns=sum(
                r["horizontal_impulse_Ns"] for r in selected
            ),
            stable_goal_count=sum(
                r["stable_goal_time_s"] is not None for r in selected
            ),
            fell_count=sum(r["fell"] for r in selected),
            goal_distances_m=[r["final_goal_distance_m"] for r in selected],
        )
        if key == "square_gaps":
            result[key]["outcomes"] = [r["gap"]["outcome"] for r in selected]
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(args, app):
    import copy

    import gymnasium as gym
    import instinctlab.tasks  # noqa: F401
    import isaaclab.sim as sim_utils
    import torch
    from instinct_rl.runners import OnPolicyRunner
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    def cpu(value):
        return value.detach().cpu().numpy().copy()

    started = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite previous evaluation: {output}")
    task = "Instinct-Parkour-Target-Amp-G1-Play-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=10)
    cfg.seed = args.construction_seed
    cfg.episode_length_s = args.duration
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.terrain_generator = copy.deepcopy(
        cfg.scene.terrain.terrain_generator
    )
    gen = cfg.scene.terrain.terrain_generator
    gen.seed = args.terrain_seed
    gen.num_rows, gen.num_cols = 4, 10
    gen.curriculum = True
    cfg.scene.leg_volume_points.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.48, 0.55, 0.62), roughness=0.8
    )
    cfg.scene.sky_light.spawn.texture_file = None
    if args.upstairs_speed is not None:
        cfg.commands.base_velocity.velocity_ranges = copy.deepcopy(
            cfg.commands.base_velocity.velocity_ranges
        )
        cfg.commands.base_velocity.velocity_ranges["pyramid_stairs_inv"][
            "lin_vel_x"
        ] = (args.upstairs_speed, args.upstairs_speed)
    cfg.viewer.env_index = args.video_env
    cfg.viewer.eye = (3.5, 3.0, 2.2)
    cfg.viewer.lookat = (0.3, 0.0, 0.2)
    cfg.viewer.resolution = (960, 720)
    physics_config = json_value(
        dict(
            robot=cfg.scene.robot.to_dict(),
            sim=cfg.sim.to_dict(),
            observations=cfg.observations.to_dict(),
            actions=cfg.actions.to_dict(),
            commands=cfg.commands.to_dict(),
            terrain=gen.to_dict(),
            reset_events=cfg.events.to_dict(),
            decimation=cfg.decimation,
            evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
            reset_protocol=RESET_PROTOCOL,
        )
    )
    config_json = json.dumps(physics_config, sort_keys=True, indent=2)
    (output / "evaluation_config.json").write_text(config_json + "\n")
    manifest = dict(
        task=task,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_digest(args.checkpoint),
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        reset_protocol=RESET_PROTOCOL,
        training_started=False,
        construction_seed=args.construction_seed,
        terrain_seed=args.terrain_seed,
        trial_seeds=args.trial_seeds,
        duration_s=args.duration,
        num_envs=10,
        status="starting",
        trials=[],
        resets=[],
        config_sha256=hashlib.sha256(config_json.encode()).hexdigest(),
        force_semantics="World sum of contact normal forces on each ankle link, excluding tangential friction.",
        trial_reset_semantics="Reseed; native scene/sensor/action/command reset; refresh target-derived velocity command without advancing time; reset observation terms and histories and recompute all groups; flatten with native wrapper. No physics warm-up. Freeze reference terrain levels. Record exact initial arrays and hashes. This changes the first observation from legacy v1 evaluation and requires new baselines. Later RNG streams can diverge if policies trigger different automatic resets.",
        capture_semantics="After existing scene.update each physics step, before termination/reset; no extra sensor/observation update.",
        foot_names=FEET,
        video=bool(args.video),
        video_env=args.video_env,
        upstairs_speed=args.upstairs_speed,
    )
    env = writer = None
    update = termination_compute = None
    fields = {
        key: []
        for key in [
            "trial",
            "t",
            "episode",
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "projected_gravity_b",
            "foot_pos_w",
            "foot_quat_w",
            "contact_force_w",
            "joint_pos",
            "joint_vel",
            "joint_target",
            "target_pos_w",
            "command",
            "max_command_b",
        ]
    }
    initial_states = {}
    capture = False
    trial = tick = 0
    episodes = np.zeros(10, dtype=np.int32)

    def save(status):
        manifest.update(status=status, elapsed_wall_s=time.monotonic() - started)
        (output / "manifest.json").write_text(
            json.dumps(json_value(manifest), indent=2) + "\n"
        )

    try:
        meshes = []
        importer = cfg.scene.terrain.class_type
        original_import = importer.import_mesh

        def capture_mesh(self, name, mesh):
            result = original_import(self, name, mesh)
            meshes.append(mesh.copy())
            return result

        importer.import_mesh = capture_mesh
        try:
            env = InstinctRlVecEnvWrapper(
                gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None)
            )
        finally:
            importer.import_mesh = original_import
        raw = env.unwrapped
        robot, terrain = raw.scene["robot"], raw.scene.terrain
        contact = raw.scene["contact_forces"]
        command = raw.command_manager.get_term("base_velocity")
        assert tuple(terrain.terrain_origins.shape[:2]) == (4, 10)
        assert cpu(terrain.terrain_types).tolist() == list(range(10))
        # Freeze the recorded tiles before any evaluated reset (including env5 level1).
        terrain.terrain_levels[:] = torch.tensor(args.terrain_levels, device=raw.device)
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        body_ids = [robot.body_names.index(name) for name in FEET]
        contact_ids = [contact.body_names.index(name) for name in FEET]
        assert contact.cfg.history_length > 0, (
            "Contact sensor must refresh every physics step."
        )
        assert cfg.scene.robot.spawn.replace_cylinders_with_capsules
        assert len(meshes) == 1
        vertices, faces = np.asarray(meshes[0].vertices), np.asarray(meshes[0].faces)
        origins = cpu(raw.scene.env_origins)
        mesh_data = dict(
            vertices=vertices,
            faces=faces,
            env_origins=origins,
            terrain_origins=cpu(terrain.terrain_origins),
        )
        np.savez_compressed(output / "terrain_mesh.npz", **mesh_data)
        proportions = np.array([x.proportion for x in gen.sub_terrains.values()])
        proportions /= proportions.sum()
        names = list(gen.sub_terrains)
        column_names = [
            names[int(np.flatnonzero(i / 10 + 0.001 < np.cumsum(proportions))[0])]
            for i in range(10)
        ]
        gap_env = column_names.index("square_gaps")
        manifest.update(
            physics_dt=raw.physics_dt,
            control_dt=raw.step_dt,
            env_origins=origins.tolist(),
            terrain_levels=args.terrain_levels,
            terrain_types=cpu(terrain.terrain_types).tolist(),
            column_terrain_names=column_names,
            gap_env=gap_env,
            terrain_sha256=digest_arrays(mesh_data),
            square_gap_geometry=gap_geometry(vertices, faces, origins[gap_env]),
            shoe_capsules=capsule_geometry(cfg.scene.robot.spawn.asset_path),
            joint_names=robot.joint_names,
        )
        agent_cfg = load_cfg_from_registry(task, "instinct_rl_cfg_entry_point")
        agent_cfg.device = args.device
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=args.device
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        if args.video:
            import cv2

            writer = cv2.VideoWriter(
                str(output / "env5_trial0.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                cfg.viewer.resolution,
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")
            raw.render()
            for _ in range(8):
                raw.sim.render()
        update = raw.scene.update
        termination_compute = raw.termination_manager.compute

        def record_update(dt):
            nonlocal tick
            update(dt)
            if not capture:
                return
            tick += 1
            fields["trial"].append(trial)
            fields["t"].append(tick * raw.physics_dt)
            fields["episode"].append(episodes.copy())
            for key, value in [
                ("root_pos_w", robot.data.root_pos_w),
                ("root_quat_w", robot.data.root_quat_w),
                ("root_lin_vel_w", robot.data.root_lin_vel_w),
                ("projected_gravity_b", robot.data.projected_gravity_b),
                ("foot_pos_w", robot.data.body_pos_w[:, body_ids]),
                ("foot_quat_w", robot.data.body_quat_w[:, body_ids]),
                ("contact_force_w", contact._data.net_forces_w[:, contact_ids]),
                ("joint_pos", robot.data.joint_pos),
                ("joint_vel", robot.data.joint_vel),
                ("joint_target", robot.data.joint_pos_target),
                ("target_pos_w", command.pos_command_w),
                ("command", command.command),
                ("max_command_b", command.max_command_b),
            ]:
                fields[key].append(cpu(value))

        def record_termination():
            value = termination_compute()
            if capture:
                for env_id in torch.where(value)[0].tolist():
                    reasons = [
                        name
                        for name in raw.termination_manager.active_terms
                        if raw.termination_manager.get_term(name)[env_id].item()
                    ]
                    manifest["resets"].append(
                        dict(
                            trial=trial,
                            t=tick * raw.physics_dt,
                            env=env_id,
                            episode=int(episodes[env_id]),
                            reasons=reasons,
                        )
                    )
            return value

        raw.scene.update = record_update
        raw.termination_manager.compute = record_termination
        for trial, seed in enumerate(args.trial_seeds):
            capture = False
            tick = 0
            episodes[:] = 0
            env.seed(seed)
            with torch.inference_mode():
                obs, _ = paired_reset(env)
                runner.alg.actor_critic.reset()
            assert tuple(obs.shape) == (10, 5376), obs.shape
            initial = dict(
                root_state=cpu(robot.data.root_state_w),
                joint_pos=cpu(robot.data.joint_pos),
                joint_vel=cpu(robot.data.joint_vel),
                target_pos=cpu(command.pos_command_w),
                command=cpu(command.command),
                max_command=cpu(command.max_command_b),
                action=cpu(raw.action_manager.action),
                previous_action=cpu(raw.action_manager.prev_action),
                terrain_levels=cpu(terrain.terrain_levels),
                env_origins=cpu(raw.scene.env_origins),
            )
            manifest["trials"].append(
                dict(
                    trial=trial,
                    seed=seed,
                    initial_state_sha256=digest_arrays(initial),
                    initial_observation_sha256=digest_arrays(dict(policy=cpu(obs))),
                    root_pose=cpu(robot.data.root_pose_w).tolist(),
                    command=initial["command"].tolist(),
                    max_command=initial["max_command"].tolist(),
                    target_pos=initial["target_pos"].tolist(),
                )
            )
            initial["policy_observation"] = cpu(obs)
            for key, value in initial.items():
                initial_states[f"trial{trial}_{key}"] = value
            save("running")
            print(
                "EVAL_TRIAL_READY",
                trial,
                seed,
                manifest["trials"][-1]["initial_state_sha256"],
                flush=True,
            )
            for step in range(round(args.duration / raw.step_dt)):
                capture = True
                with torch.inference_mode():
                    action = policy(obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite policy actions")
                    obs, _, dones, _ = env.step(action)
                    runner.alg.actor_critic.reset(dones)
                capture = False
                episodes += cpu(dones).astype(np.int32)
                if writer is not None and trial == 0 and (step + 1) % 2 == 0:
                    frame = cv2.cvtColor(np.asarray(raw.render()), cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        frame,
                        f"{args.checkpoint.name} | Isaac env{args.video_env} | t={(step + 1) * raw.step_dt:.2f}s",
                        (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (245, 245, 245),
                        1,
                        cv2.LINE_AA,
                    )
                    writer.write(frame)
            if writer is not None:
                writer.release()
                writer = None
            assert tick == round(args.duration / raw.physics_dt), tick
            print("EVAL_TRIAL_DONE", trial, episodes.tolist(), flush=True)
        arrays = {key: np.asarray(value) for key, value in fields.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(output / "trace.npz", **arrays)
        np.savez_compressed(output / "initial_states.npz", **initial_states)
        save("complete")
        summary = summarize(output)
        print(
            "EVAL_DONE",
            json.dumps({k: summary[k] for k in ["stairs_env5", "square_gaps"]}),
            flush=True,
        )
    except BaseException as exc:
        manifest["error"] = repr(exc)
        save("failed")
        raise
    finally:
        capture = False
        if writer is not None:
            writer.release()
        if env is not None:
            if update is not None:
                raw.scene.update = update
            if termination_compute is not None:
                raw.termination_manager.compute = termination_compute
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze-only", type=Path)
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--construction-seed", type=int, default=42)
    parser.add_argument("--terrain-seed", type=int, default=0)
    parser.add_argument(
        "--terrain-levels", type=int, nargs=10, default=REFERENCE_LEVELS
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--upstairs-speed",
        type=float,
        help="Override only ordinary uphill forward command (m/s).",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-env", type=int, default=5)
    if "--analyze-only" in sys.argv:
        args = parser.parse_args()
        print(json.dumps(summarize(args.analyze_only), indent=2))
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required")
    if (
        args.duration <= 0
        or not 0 <= args.video_env < 10
        or any(x < 0 or x > 3 for x in args.terrain_levels)
    ):
        parser.error("Invalid duration, video env, or terrain levels")
    if args.upstairs_speed is not None and (
        not np.isfinite(args.upstairs_speed) or args.upstairs_speed <= 0
    ):
        parser.error("Upstairs speed must be finite and positive")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    args.headless = True
    args.enable_cameras = args.video
    launcher = AppLauncher(args)
    try:
        run(args, launcher.app)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
