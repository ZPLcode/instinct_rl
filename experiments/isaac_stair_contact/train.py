"""Resume Isaac parkour training with stair contact and landing rewards.

Run with a Python environment containing Isaac Lab and InstinctLab.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--save-interval", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--side-weight", type=float, default=-0.5)
parser.add_argument(
    "--landing-weight",
    type=float,
    default=0.0,
    help="Ordinary-stair reciprocal landing bonus per event; zero disables it.",
)
parser.add_argument(
    "--catchup-cost",
    type=float,
    default=0.0,
    help="Actual reward deducted per verified interior catch-up; requires positive landing-weight.",
)
parser.add_argument("--upstairs-proportion", type=float, default=None)
parser.add_argument("--upstairs-height-range", type=float, nargs=2, default=None)
parser.add_argument("--capped-airtime-weight", type=float, default=None)
parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Override episode duration; omission keeps the task default (20 s).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.iterations <= 0 or args.num_envs <= 0 or args.save_interval <= 0:
    parser.error("iterations, num-envs and save-interval must be positive")
if args.side_weight >= 0 or not 0 < args.learning_rate <= 1e-3:
    parser.error("side-weight must be negative and learning-rate in (0, 1e-3]")
if not 0 <= args.landing_weight <= 5.0:
    parser.error("landing-weight must be in [0, 5]")
if not 0 <= args.catchup_cost <= 1.0 or (
    args.catchup_cost and args.landing_weight == 0
):
    parser.error(
        "catchup-cost must be in [0, 1] and requires positive landing-weight when enabled"
    )
if args.upstairs_proportion is not None and not 0 < args.upstairs_proportion <= 1:
    parser.error("upstairs-proportion must be in (0, 1]")
if (
    args.upstairs_height_range is not None
    and not 0 < args.upstairs_height_range[0] <= args.upstairs_height_range[1] <= 0.23
):
    parser.error("upstairs-height-range must satisfy 0 < min <= max <= 0.23 m")
if args.capped_airtime_weight is not None and not 0 < args.capped_airtime_weight <= 2.0:
    parser.error("capped-airtime-weight must be in (0, 2]")
if args.episode_length_s is not None and not 0 < args.episode_length_s < float("inf"):
    parser.error("episode-length-s must be finite and positive")
args.checkpoint = args.checkpoint.resolve(strict=True)
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "status.json").exists():
    parser.error("Use a new output directory for each experiment")
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import instinctlab.tasks  # noqa: F401
import isaaclab.sim as sim_utils
import torch
from config import apply_contact_rewards, apply_landing_reward
from instinct_rl.runners import OnPolicyRunner
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def archive_sources():
    """Capture the exact implementation and starting policy before any training."""

    def sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    archived = args.output / "source"
    archived.mkdir(exist_ok=True)
    sources = {}
    for name in ["train.py", "config.py", "rewards.py", "stair_landing.py"]:
        source = Path(__file__).resolve().parent / name
        if not source.exists():
            continue
        target = archived / name
        shutil.copy2(source, target)
        sources[name] = dict(
            original_path=str(source), archived_path=str(target), sha256=sha256(target)
        )
    provenance = dict(
        source_files=sources,
        source_checkpoint=dict(
            path=str(args.checkpoint), sha256=sha256(args.checkpoint)
        ),
        init_at_random_ep_len=False,
    )
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main():
    started = time.monotonic()
    task = "Instinct-Parkour-Target-Amp-G1-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    cfg.scene.terrain.terrain_generator = copy.deepcopy(
        cfg.scene.terrain.terrain_generator
    )
    cfg.scene.leg_volume_points.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.26, 0.32, 0.38), roughness=0.9
    )
    baseline_cfg = cfg.to_dict()
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    upstairs = cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"]
    if args.upstairs_proportion is not None:
        upstairs.proportion = args.upstairs_proportion
    if args.upstairs_height_range is not None:
        upstairs.step_height_range = tuple(args.upstairs_height_range)
    apply_contact_rewards(cfg, args.side_weight)
    if args.landing_weight:
        apply_landing_reward(cfg, args.landing_weight, args.catchup_cost)
    if args.capped_airtime_weight is not None:
        from instinctlab.tasks.locomotion.mdp.rewards import (
            feet_air_time_positive_biped,
        )

        airtime = cfg.rewards.rewards.feet_air_time
        airtime.func, airtime.weight = (
            feet_air_time_positive_biped,
            args.capped_airtime_weight,
        )
        airtime.params.pop("vel_threshold")
        airtime.params["threshold"] = 0.5
    agent_cfg = load_cfg_from_registry(task, "instinct_rl_cfg_entry_point")
    agent_cfg.seed = 42
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.run_name = (
        "isaac_stair_landing" if args.landing_weight else "isaac_stair_contact"
    )
    agent_cfg.resume = True
    agent_cfg.load_run = str(args.checkpoint.parent)
    agent_cfg.load_checkpoint = args.checkpoint.name
    agent_dict = agent_cfg.to_dict()
    report = dict(
        task=task,
        source_checkpoint=str(args.checkpoint),
        num_envs=args.num_envs,
        additional_iterations=args.iterations,
        save_interval=args.save_interval,
        episode_length_s=cfg.episode_length_s,
        learning_rate=args.learning_rate,
        schedule="fixed",
        side_weight=args.side_weight,
        landing_weight=args.landing_weight,
        catchup_cost_per_event=args.catchup_cost,
        upstairs_proportion=args.upstairs_proportion,
        upstairs_height_range=args.upstairs_height_range,
        capped_airtime_weight=args.capped_airtime_weight,
        optimizer="PPO Adam moments cleared; loaded AMP discriminator and optimizer retained",
        reward_changes=[
            "feet_side_collision",
            "feet_flat_ori support gate",
            "feet_at_plane support gate",
        ],
        observation_changes=False,
        actuator_changes=False,
        terrain_changes=args.upstairs_proportion is not None
        or args.upstairs_height_range is not None,
        status="starting",
    )
    if args.landing_weight:
        report["reward_changes"].append("ordinary-stair reciprocal landing event")
    if args.capped_airtime_weight is not None:
        report["reward_changes"].append(
            "existing biped airtime: cap 0.5 s, linear command >0.1 m/s, no yaw-only gate"
        )
    if args.catchup_cost:
        report["reward_changes"].append("ordinary-stair interior catch-up landing cost")
    env = runner = None

    def write_status(status):
        report.update(status=status, elapsed_wall_s=time.monotonic() - started)
        if env is not None and args.landing_weight:
            term = env.unwrapped.reward_manager.get_term_cfg(
                "stair_reciprocal_landing", "rewards"
            ).func
            report["landing_diagnostics"] = term.diagnostics()
        temporary = args.output / "status.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    try:
        archive_sources()
        write_status("starting")
        dump_yaml(str(args.output / "params" / "baseline_env.yaml"), baseline_cfg)
        dump_yaml(str(args.output / "params" / "env.yaml"), cfg)
        dump_yaml(str(args.output / "params" / "agent.yaml"), agent_dict)
        env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg, render_mode=None))
        assert env.num_obs == 5376, env.num_obs
        side_sensor = env.unwrapped.scene["stair_contact_forces"]
        assert len(side_sensor.body_names) == 2, side_sensor.body_names
        assert side_sensor.data.net_forces_w_history.shape == (
            args.num_envs,
            cfg.decimation,
            2,
            3,
        )
        runner = OnPolicyRunner(
            env, copy.deepcopy(agent_dict), log_dir=str(args.output), device=args.device
        )
        runner.load(str(args.checkpoint))
        runner.alg.optimizer.state.clear()
        runner.alg.learning_rate = args.learning_rate
        runner.alg.schedule = "fixed"
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.learning_rate
        initial_iteration = runner.current_learning_iteration
        report.update(
            start_iteration=initial_iteration,
            target_iteration=initial_iteration + args.iterations,
            reward_sensor_bodies=side_sensor.body_names,
        )
        original_save = runner.save

        def atomic_save(path, infos=None):
            temporary = str(path) + ".tmp"
            original_save(temporary, infos=infos)
            os.replace(temporary, path)
            report["last_checkpoint"] = str(path)
            report["last_saved_iteration"] = runner.current_learning_iteration
            write_status("training")

        runner.save = atomic_save
        runner.save(str(args.output / f"warmstart_{initial_iteration}.pt"))
        print("CONTACT_TRAIN_READY " + json.dumps(report), flush=True)
        write_status("training")
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        report["final_iteration"] = runner.current_learning_iteration
        assert runner.current_learning_iteration == initial_iteration + args.iterations
        # Finite checkpoint validation also covers the AMP discriminator tensors.
        final = torch.load(
            report["last_checkpoint"], map_location="cpu", weights_only=False
        )
        nonfinite = []

        def check(value, prefix=""):
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all()
            ):
                nonfinite.append(prefix)
            elif isinstance(value, dict):
                for name, item in value.items():
                    check(item, prefix + "/" + str(name))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    check(item, prefix + "/" + str(index))

        check(final)
        if nonfinite:
            raise RuntimeError(f"Nonfinite checkpoint tensors: {nonfinite}")
        report["finite_checkpoint"] = True
        write_status("complete")
        print("CONTACT_TRAIN_DONE " + json.dumps(report), flush=True)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_status("failed")
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        if env is not None:
            env.close()


try:
    main()
finally:
    app.close()
