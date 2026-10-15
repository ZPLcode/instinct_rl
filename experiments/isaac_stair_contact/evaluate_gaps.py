"""Paired native Isaac gap inference: three widths, three reseeded 12 s trials.

Run with the isaaclab interpreter; rendering is optional (--video). The policy's
native depth observations remain enabled even when display rendering is disabled.
Use analyze_gaps.py, then compare_gaps.py, for CPU analysis of the saved outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from evaluate import (
    EVALUATION_SCHEMA_VERSION,
    RESET_PROTOCOL,
    digest_arrays,
    file_digest,
    json_value,
    paired_reset,
)


def run(args, app):
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
    task = "Instinct-Parkour-Target-Amp-G1-Play-v0"
    widths, seeds = [0.2, 0.4, 0.6], [42, 43, 44]
    cfg = parse_env_cfg(task, device=args.device, num_envs=3)
    cfg.seed = 42
    cfg.episode_length_s = 12.0
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 0
    gen = copy.deepcopy(cfg.scene.terrain.terrain_generator)
    square = gen.sub_terrains["square_gaps"]
    gen.num_rows, gen.num_cols, gen.seed = 1, 3, 0
    gen.sub_terrains = {}
    velocity = copy.deepcopy(cfg.commands.base_velocity.velocity_ranges["square_gaps"])
    cfg.commands.base_velocity.velocity_ranges = {}
    cfg.commands.base_velocity.random_velocity_terrain = []
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    for width in widths:
        name = f"square_gaps_{round(width * 100)}cm"
        tile = copy.deepcopy(square)
        tile.gap_distance_range = (width, width)
        tile.proportion = 1.0
        gen.sub_terrains[name] = tile
        cfg.commands.base_velocity.velocity_ranges[name] = copy.deepcopy(velocity)
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.leg_volume_points.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.26, 0.32, 0.38), roughness=0.9
    )
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.sky_light.spawn.intensity = 650.0
    if args.video:
        import cv2

        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (2.1, -5.2, 3.0)
        cfg.viewer.lookat = (1.25, 0.0, 0.3)
        cfg.viewer.resolution = (960, 720)
    config = json_value(
        dict(
            robot=cfg.scene.robot.to_dict(),
            sim=cfg.sim.to_dict(),
            observations=cfg.observations.to_dict(),
            actions=cfg.actions.to_dict(),
            commands=cfg.commands.to_dict(),
            terrain=gen.to_dict(),
            reset_events=cfg.events.to_dict(),
            decimation=cfg.decimation,
            episode_length_s=cfg.episode_length_s,
            evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
            reset_protocol=RESET_PROTOCOL,
        )
    )
    config_json = json.dumps(config, sort_keys=True, indent=2)
    (args.output / "evaluation_config.json").write_text(config_json + "\n")
    agent_cfg = load_cfg_from_registry(task, "instinct_rl_cfg_entry_point")
    agent_cfg.device = args.device
    report = dict(
        task=task,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_digest(args.checkpoint),
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        reset_protocol=RESET_PROTOCOL,
        config_sha256=hashlib.sha256(config_json.encode()).hexdigest(),
        training_started=False,
        configured_widths_m=widths,
        num_envs=3,
        trials_per_width=3,
        trial_duration_s=12.0,
        seed=42,
        terrain_seed=gen.seed,
        trial_seeds=seeds,
        trials=[],
        video=bool(args.video),
        changes="Square-gaps-only scene, fixed widths, disabled curriculum and standing commands; native physics/actions/observations/speed ranges. Each trial reseeded; reset target-derived command refreshed and observation history rebuilt without physics warm-up. RGB rendering optional.",
        capture_semantics="After native scene.update on every physics step, before termination/reset. Original contact normal force read without an extra sensor update.",
        trial_reset_semantics="Shared evaluator v2 paired_reset: reseed 42/43/44, native reset, refresh target-derived command, reset observation terms/history, recompute and flatten; actor state reset. No extra physics step. Later RNG may diverge after policy-dependent automatic resets.",
        status="starting",
    )
    fields = {
        key: []
        for key in [
            "trial",
            "t",
            "episode",
            "root_pos_w",
            "projected_gravity_b",
            "foot_pos_w",
            "foot_quat_w",
            "contact_force_w",
            "target_pos_w",
            "command",
            "max_command_b",
        ]
    }
    initial_states, events = {}, []
    env = writer = None
    original_update = original_termination = None
    frames = tick = trial = 0
    capture = False
    episodes = np.zeros(3, dtype=np.int32)

    def save_report(status):
        report.update(
            status=status,
            video_frames=frames,
            resets=events,
            elapsed_wall_s=time.monotonic() - started,
        )
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "report.json")

    try:
        save_report("starting")
        meshes = []
        importer = cfg.scene.terrain.class_type
        original_import = importer.import_mesh

        def capture_mesh(self, name, mesh):
            value = original_import(self, name, mesh)
            meshes.append(mesh.copy())
            return value

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
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        body_ids = [robot.body_names.index(name) for name in feet]
        contact_ids = [contact.body_names.index(name) for name in feet]
        command = raw.command_manager.get_term("base_velocity")
        assert len(meshes) == 1
        mesh_data = dict(
            vertices=np.asarray(meshes[0].vertices),
            faces=np.asarray(meshes[0].faces),
            env_origins=cpu(raw.scene.env_origins),
            terrain_origins=cpu(terrain.terrain_origins),
        )
        np.savez_compressed(args.output / "terrain_mesh.npz", **mesh_data)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=args.device
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        report.update(
            env_origins=cpu(raw.scene.env_origins).tolist(),
            terrain_sha256=digest_arrays(mesh_data),
            terrain_types=cpu(terrain.terrain_types).tolist(),
            terrain_levels=cpu(terrain.terrain_levels).tolist(),
            foot_names=feet,
            control_dt=raw.step_dt,
            physics_dt=raw.physics_dt,
            observation_shape=[3, env.num_obs],
        )
        assert report["terrain_types"] == [0, 1, 2]
        assert report["terrain_levels"] == [0, 0, 0]
        assert env.num_obs == 5376
        if args.video:
            raw.render()
            for _ in range(8):
                raw.sim.render()
            writer = cv2.VideoWriter(
                str(args.output / "square_gaps_raw.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                cfg.viewer.resolution,
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot create gap video writer")
        original_update, original_termination = (
            raw.scene.update,
            raw.termination_manager.compute,
        )

        def record_update(dt):
            nonlocal tick
            original_update(dt)
            if not capture:
                return
            tick += 1
            fields["trial"].append(trial)
            fields["t"].append(tick * raw.physics_dt)
            fields["episode"].append(episodes.copy())
            for key, value in [
                ("root_pos_w", robot.data.root_pos_w),
                ("projected_gravity_b", robot.data.projected_gravity_b),
                ("foot_pos_w", robot.data.body_pos_w[:, body_ids]),
                ("foot_quat_w", robot.data.body_quat_w[:, body_ids]),
                ("contact_force_w", contact._data.net_forces_w[:, contact_ids]),
                ("target_pos_w", command.pos_command_w),
                ("command", command.command),
                ("max_command_b", command.max_command_b),
            ]:
                fields[key].append(cpu(value))

        def record_termination():
            value = original_termination()
            if capture:
                for env_id in torch.where(value)[0].tolist():
                    reasons = [
                        name
                        for name in raw.termination_manager.active_terms
                        if raw.termination_manager.get_term(name)[env_id].item()
                    ]
                    events.append(
                        dict(
                            trial=trial,
                            t=tick * raw.physics_dt,
                            env=env_id,
                            episode=int(episodes[env_id]),
                            reasons=reasons,
                        )
                    )
            return value

        raw.scene.update, raw.termination_manager.compute = (
            record_update,
            record_termination,
        )
        for trial, seed in enumerate(seeds):
            capture = False
            episodes[:] = 0
            tick = 0
            env.seed(seed)
            with torch.inference_mode():
                obs, _ = paired_reset(env)
                runner.alg.actor_critic.reset()
            assert tuple(obs.shape) == (3, 5376) and torch.isfinite(obs).all()
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
            report["trials"].append(
                dict(
                    trial=trial,
                    seed=seed,
                    initial_state_sha256=digest_arrays(initial),
                    initial_observation_sha256=digest_arrays(dict(policy=cpu(obs))),
                    command=initial["command"].tolist(),
                    max_command=initial["max_command"].tolist(),
                    target_pos=initial["target_pos"].tolist(),
                )
            )
            initial["policy_observation"] = cpu(obs)
            for key, value in initial.items():
                initial_states[f"trial{trial}_{key}"] = value
            np.savez_compressed(args.output / "initial_states.npz", **initial_states)
            save_report("running")
            print(
                "GAP_TRIAL_READY",
                trial,
                seed,
                report["trials"][-1]["initial_state_sha256"],
                flush=True,
            )
            if args.video:
                raw.viewport_camera_controller.set_view_env_index(trial)
                for _ in range(4):
                    raw.sim.render()
            for step in range(round(12.0 / raw.step_dt)):
                capture = True
                with torch.inference_mode():
                    action = policy(obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite policy action")
                    obs, _, dones, _ = env.step(action)
                    runner.alg.actor_critic.reset(dones)
                capture = False
                episodes += cpu(dones).astype(np.int32)
                if args.video and (step + 1) % 2 == 0:
                    frame = cv2.cvtColor(np.asarray(raw.render()), cv2.COLOR_RGB2BGR)
                    labels = [
                        f"Isaac | {args.checkpoint.stem} | square_gaps",
                        f"Configured {widths[trial] * 100:.0f} cm | mesh opening ~{widths[trial] * 100 + 5:.0f} cm",
                        f"Trial {trial + 1}, seed {seed} | t={(step + 1) * raw.step_dt:.2f}s",
                    ]
                    cv2.rectangle(frame, (0, 0), (960, 90), (25, 25, 25), -1)
                    for i, label in enumerate(labels):
                        cv2.putText(
                            frame,
                            label,
                            (15, 25 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (245, 245, 245),
                            1,
                            cv2.LINE_AA,
                        )
                    writer.write(frame)
                    frames += 1
            assert tick == round(12.0 / raw.physics_dt)
            arrays = {key: np.asarray(value) for key, value in fields.items()}
            assert all(np.isfinite(value).all() for value in arrays.values())
            np.savez_compressed(args.output / "trace.npz", **arrays)
            print("GAP_TRIAL_DONE", trial, episodes.tolist(), flush=True)
        if writer is not None:
            writer.release()
            writer = None
        save_report("complete")
        print("GAP_EVALUATION_DONE", frames, time.monotonic() - started, flush=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        save_report("failed")
        raise
    finally:
        capture = False
        if writer is not None:
            writer.release()
        if env is not None:
            if original_update is not None:
                raw.scene.update = original_update
            if original_termination is not None:
                raw.termination_manager.compute = original_termination
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Refusing to overwrite a nonempty evaluation output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.headless, args.enable_cameras = True, args.video
    launcher = AppLauncher(args)
    try:
        run(args, launcher.app)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
