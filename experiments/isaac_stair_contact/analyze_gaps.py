"""Offline native Isaac square-gap evaluation using saved meshes and first episodes only."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("MPLCONFIGDIR", "/tmp/isaac_square_gaps_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_dir", type=Path, help="Completed square_gaps.py output directory"
)
parser.add_argument(
    "--shoe-geometry",
    type=Path,
    default=Path(__file__).resolve().parent / "assets/shoe_geometry.json",
    help="Verified native USD capsule geometry JSON",
)
args = parser.parse_args()
P = args.input_dir.expanduser().resolve(strict=True)
A = np.load(P / "trace.npz")
mesh = np.load(P / "terrain_mesh.npz")
report = json.loads((P / "report.json").read_text())
G = json.loads(args.shoe_geometry.expanduser().resolve(strict=True).read_text())
feet = report["foot_names"]
caps = [[c for c in G["capsules"] if "/" + name + "/" in c["path"]] for name in feet]
endpoints = np.array([[c["endpoints_in_ankle_link"] for c in cs] for cs in caps])
radii = np.array([[c["radius"] for c in cs] for cs in caps])
v, f, origins = mesh["vertices"], mesh["faces"], mesh["env_origins"]
geometry = []
for e, origin in enumerate(origins):
    local = v - origin
    selected = (np.abs(local[:, 0]) < 3.8) & (np.abs(local[:, 1]) < 0.8)
    tri = local[f[np.all(selected[f], axis=1)]]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ns = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(ns[:, None], 1e-12)
    vertical = (ns > 1e-9) & (np.abs(normals[:, 0]) > 0.99) & (tri[:, :, 0].mean(1) > 0)
    walls = np.unique(np.round(tri[vertical, :, 0], 6))
    assert len(walls) == 2, walls
    ground = float(np.max(tri[:, :, 2]))
    bottom = float(np.min(tri[:, :, 2]))
    geometry.append(
        dict(
            env=e,
            configured_width_m=report["configured_widths_m"][e],
            actual_width_m=float(walls[1] - walls[0]),
            near_x_m=float(walls[0]),
            far_x_m=float(walls[1]),
            ground_z_m=ground,
            bottom_z_m=bottom,
            depth_m=ground - bottom,
            origin_w=origin.tolist(),
        )
    )

criteria = dict(
    episode="Only episode 0 for every trial/env; resets never joined across episodes.",
    east_crossing="Root crosses actual near and far +X mesh planes in order, with |local root y| <= 0.75 m throughout that crossing.",
    far_support="For each foot at least one sample with world normal Fz > 20 N, every shoe capsule beyond far plane by 5 mm, and minimum shoe z within 8 cm of the upper ground. The two feet need not contact simultaneously.",
    goal="Root horizontal distance to recorded target <= 0.4 m at least once. From an arrival sample, upright root (tilt < 0.6 rad, height >= 0.5 m above upper ground) on far side for at least 0.5 s; foot support evidence must exist by then.",
    fell_into_gap="Either foot supports on pit bottom: normal Fz > 20 N, ankle x within gap +/-3 cm, |local y| <= 0.75 m, and minimum shoe z <= bottom+8 cm; OR root is within gap +/-15 cm, |local y| <=0.75 m and height above upper ground <0.5 m.",
    deep_foot_intrusion="Separate caution flag when an ankle lies horizontally inside the gap corridor and the minimum shoe z falls >15 cm below upper ground. This alone is not called falling or bottom support.",
    failures="Non-time-out reset, incomplete crossing, side-route crossing, or unrecovered pit fall. A time-out alone is not a failure if success was already established.",
    geometry_source="Imported native Isaac triangle mesh; shoe capsule axes/radii read from the actual native USD of the same shoe asset. No physics simulation in this analysis.",
    sample_limit="Three random trials per width at the original random commanded speed range; descriptive small-sample results, not a general reliability estimate.",
)
results = []
plotting = []
for tr in sorted(np.unique(A["trial"]).tolist()):
    for e, g in enumerate(geometry):
        mask = (A["trial"] == tr) & (A["episode"][:, e] == 0)
        ids = np.flatnonzero(mask)
        t = A["t"][ids]
        dt = float(np.median(np.diff(t)))
        root = A["root_pos_w"][ids, e] - origins[e]
        foot = A["foot_pos_w"][ids, e] - origins[e]
        quat = A["foot_quat_w"][ids, e]
        rot = (
            Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]])
            .as_matrix()
            .reshape(len(t), 2, 3, 3)
        )
        centers = foot[:, :, None, None, :] + np.einsum(
            "tfij,fcqj->tfcqi", rot, endpoints
        )
        lo = centers - radii[None, :, :, None, None]
        hi = centers + radii[None, :, :, None, None]
        shoe_min = lo.min(axis=(2, 3))
        shoe_max = hi.max(axis=(2, 3))
        force = A["contact_force_w"][ids, e]
        grav = A["projected_gravity_b"][ids, e]
        tilt = np.arccos(np.clip(-grav[:, 2], -1, 1))
        goal = np.linalg.norm(
            A["root_pos_w"][ids, e, :2] - A["target_pos_w"][ids, e, :2], axis=-1
        )
        near, far, ground, bottom = (
            g["near_x_m"],
            g["far_x_m"],
            g["ground_z_m"],
            g["bottom_z_m"],
        )
        enter = np.flatnonzero(root[:, 0] >= near)
        exit_ = np.flatnonzero(root[:, 0] >= far)
        enter_i = int(enter[0]) if len(enter) else None
        exit_i = int(exit_[0]) if len(exit_) else None
        corridor = bool(
            enter_i is not None
            and exit_i is not None
            and exit_i >= enter_i
            and np.max(np.abs(root[enter_i : exit_i + 1, 1])) <= 0.75
        )
        support = (
            (force[:, :, 2] > 20)
            & (shoe_min[:, :, 0] > far + 0.005)
            & (np.abs(shoe_min[:, :, 2] - ground) < 0.08)
        )
        first_support = [
            int(k[0]) if len(k := np.flatnonzero(support[:, j])) else None
            for j in range(2)
        ]
        inside = (
            (foot[:, :, 0] >= near - 0.03)
            & (foot[:, :, 0] <= far + 0.03)
            & (np.abs(foot[:, :, 1]) <= 0.75)
        )
        bottom_support = (
            inside & (force[:, :, 2] > 20) & (shoe_min[:, :, 2] <= bottom + 0.08)
        )
        root_low = (
            (root[:, 0] >= near - 0.15)
            & (root[:, 0] <= far + 0.15)
            & (np.abs(root[:, 1]) <= 0.75)
            & (root[:, 2] - ground < 0.5)
        )
        fell = np.any(bottom_support, axis=1) | root_low
        deep = inside & (shoe_min[:, :, 2] < ground - 0.15)
        upright = (tilt < 0.6) & (root[:, 2] - ground >= 0.5) & (root[:, 0] >= far)
        arrival = np.flatnonzero(goal <= 0.4)
        success_i = None
        hold_samples = int(np.ceil(0.5 / dt))
        if corridor and all(k is not None for k in first_support):
            first_supported = max(first_support)
            for k in arrival:
                end = k + hold_samples
                if (
                    end < len(t)
                    and end >= first_supported
                    and np.all(upright[k : end + 1])
                ):
                    success_i = int(end)
                    break
        if success_i is not None:
            fell_before = bool(np.any(fell[: success_i + 1]))
            deep_before = bool(np.any(deep[: success_i + 1]))
            status = (
                "fell_then_recovered_to_goal"
                if fell_before
                else (
                    "crossed_with_deep_foot_intrusion"
                    if deep_before
                    else "direct_cross_success"
                )
            )
        elif np.any(fell):
            status = "fell_not_recovered_to_goal"
        elif exit_i is not None and not corridor:
            status = "side_route_not_counted"
        elif corridor:
            status = "crossed_but_goal_not_completed"
        else:
            status = "stopped_or_failed_before_crossing"
        resets = [
            x
            for x in report["resets"]
            if x["trial"] == tr and x["env"] == e and x["episode"] == 0
        ]
        non_timeout = any(any(r != "time_out" for r in ev["reasons"]) for ev in resets)
        if non_timeout and success_i is None and not np.any(fell):
            status = "failed_termination"
        cross_mask = (root[:, 0] >= near - 0.25) & (root[:, 0] <= far + 0.25)
        gap_foot = (foot[:, :, 0] >= near) & (foot[:, :, 0] <= far)
        gap_min = [
            float(shoe_min[:, j, 2][gap_foot[:, j]].min())
            if np.any(gap_foot[:, j])
            else None
            for j in range(2)
        ]
        result = dict(
            trial=int(tr),
            env=e,
            configured_width_m=g["configured_width_m"],
            actual_width_m=g["actual_width_m"],
            status=status,
            success=success_i is not None,
            first_episode_samples=len(ids),
            first_episode_duration_s=float(t[-1]),
            root_enter_gap_s=float(t[enter_i]) if enter_i is not None else None,
            root_exit_gap_s=float(t[exit_i]) if exit_i is not None else None,
            east_corridor_crossing=corridor,
            max_crossing_lateral_m=float(np.max(np.abs(root[enter_i : exit_i + 1, 1])))
            if corridor
            else None,
            first_far_shore_support_s=[
                float(t[k]) if k is not None else None for k in first_support
            ],
            target_arrival_s=float(t[arrival[0]]) if len(arrival) else None,
            success_confirmed_s=float(t[success_i]) if success_i is not None else None,
            minimum_target_distance_m=float(goal.min()),
            final_target_distance_m=float(goal[-1]),
            final_root_local_m=root[-1].tolist(),
            minimum_crossing_root_height_m=float(np.min(root[cross_mask, 2] - ground))
            if np.any(cross_mask)
            else None,
            maximum_crossing_tilt_rad=float(tilt[cross_mask].max())
            if np.any(cross_mask)
            else None,
            minimum_shoe_z_while_ankle_over_gap_m=gap_min,
            any_deep_foot_intrusion=bool(np.any(deep)),
            any_pit_floor_support=bool(np.any(bottom_support)),
            any_pit_root_low=bool(np.any(root_low)),
            first_pit_fall_s=float(t[np.flatnonzero(fell)[0]])
            if np.any(fell)
            else None,
            sampled_speed_limits_mps=np.unique(
                np.round(A["max_command_b"][ids, e, 0], 6)
            ).tolist(),
            resets=resets,
        )
        results.append(result)
        plotting.append((tr, e, t, root, foot, shoe_min, result))
summary = []
for e, g in enumerate(geometry):
    cases = [r for r in results if r["env"] == e]
    summary.append(
        dict(
            env=e,
            configured_width_m=g["configured_width_m"],
            actual_width_m=g["actual_width_m"],
            attempts=len(cases),
            successes=sum(r["success"] for r in cases),
            direct_cross_successes=sum(
                r["status"] == "direct_cross_success" for r in cases
            ),
            fell_then_recovered=sum(
                r["status"] == "fell_then_recovered_to_goal" for r in cases
            ),
            statuses=[r["status"] for r in cases],
        )
    )
output = dict(criteria=criteria, geometry=geometry, summary=summary, attempts=results)
(P / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
fig, axes = plt.subplots(
    3, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True
)
for tr, e, t, root, foot, shoe_min, r in plotting:
    ax = axes[tr, e]
    g = geometry[e]
    ax.plot(foot[:, 0, 0], shoe_min[:, 0, 2], lw=1.1, label="Left shoe")
    ax.plot(foot[:, 1, 0], shoe_min[:, 1, 2], lw=1.1, label="Right shoe")
    ax.plot(
        [-0.3, g["near_x_m"], g["near_x_m"], g["far_x_m"], g["far_x_m"], 3.9],
        [
            g["ground_z_m"],
            g["ground_z_m"],
            g["bottom_z_m"],
            g["bottom_z_m"],
            g["ground_z_m"],
            g["ground_z_m"],
        ],
        color="k",
        lw=1.3,
        label="Actual mesh profile",
    )
    ax.axhline(g["ground_z_m"] - 0.15, color="gray", ls=":", lw=0.8)
    ax.set_title(
        f"Mesh gap {g['actual_width_m'] * 100:.0f} cm | trial {tr + 1}\n{r['status'].replace('_', ' ')}",
        fontsize=10,
    )
    ax.set_xlim(-0.3, 3.9)
    ax.set_ylim(-0.7, 0.65)
    ax.grid(alpha=0.2)
    if e == 0:
        ax.set_ylabel("Minimum shoe z above platform (m)")
    if tr == 2:
        ax.set_xlabel("Ankle x relative to spawn (m)")
axes[0, 0].legend(fontsize=8)
fig.suptitle(
    f"Native Isaac {Path(report['checkpoint']).name}: first episode only; recorded shoe poses"
)
fig.savefig(P / "trajectories.png", dpi=160)
print(
    json.dumps(
        dict(
            geometry=geometry,
            summary=summary,
            attempts=[
                {
                    k: r[k]
                    for k in [
                        "trial",
                        "env",
                        "status",
                        "success_confirmed_s",
                        "minimum_crossing_root_height_m",
                        "minimum_shoe_z_while_ankle_over_gap_m",
                        "any_pit_floor_support",
                        "any_pit_root_low",
                        "sampled_speed_limits_mps",
                    ]
                }
                for r in results
            ],
        ),
        indent=2,
    )
)
