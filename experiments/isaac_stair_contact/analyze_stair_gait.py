"""CPU analysis of stair landings using native forces, poses and imported mesh.

Usage: python analyze_stair_gait.py EVALUATION_DIRECTORY [--envs 2 3 5 6]
The step numbers are recovered from the actual mesh, not inferred from foot z.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


class Surface:
    def __init__(self, vertices, faces, origin):
        # Crop to the east corridor, retaining triangles that intersect it.
        triangle = vertices[faces] - origin
        lo, hi = triangle.min(1), triangle.max(1)
        select = (
            (hi[:, 0] >= -0.5)
            & (lo[:, 0] <= 3.9)
            & (hi[:, 1] >= -0.8)
            & (lo[:, 1] <= 0.8)
        )
        triangle = triangle[select]
        a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
        det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
            a[:, 1] - c[:, 1]
        )
        select = np.abs(det) > 1e-10
        self.triangle, self.det = triangle[select], det[select]
        self.lo, self.hi = (
            self.triangle[:, :, :2].min(1),
            self.triangle[:, :, :2].max(1),
        )

    def height(self, xy):
        indices = np.flatnonzero(
            np.all((xy >= self.lo - 1e-8) & (xy <= self.hi + 1e-8), axis=1)
        )
        if not len(indices):
            return float("nan")
        a, b, c = (self.triangle[indices, i] for i in range(3))
        x, y = xy
        u = (
            (b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])
        ) / self.det[indices]
        v = (
            (c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])
        ) / self.det[indices]
        inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
        values = (u * a[:, 2] + v * b[:, 2] + (1 - u - v) * c[:, 2])[inside]
        return float(values.max()) if len(values) else float("nan")

    def profile(self):
        x = np.arange(0.005, 3.806, 0.01)
        # Median over the corridor suppresses the 5 mm perlin height roughness.
        z = np.array(
            [np.median([self.height([q, y]) for y in [-0.20, 0.0, 0.20]]) for q in x]
        )
        if not np.isfinite(z).all():
            raise ValueError(
                "Stair profile contains missing mesh intersections; cannot define tread boundaries."
            )
        candidates = np.flatnonzero(np.abs(np.diff(z)) > 0.045)
        clusters = np.split(candidates, np.flatnonzero(np.diff(candidates) > 3) + 1)
        edges = [int(g[np.argmax(np.abs(np.diff(z)[g]))]) for g in clusters if len(g)]
        edge_x = [(x[i] + x[i + 1]) / 2 for i in edges]
        jumps = [z[i + 1] - z[i] for i in edges]
        direction = (
            "ascending"
            if jumps and min(jumps) > 0
            else "descending"
            if jumps and max(jumps) < 0
            else "mixed_or_flat"
        )
        return dict(
            x=x.tolist(),
            z=z.tolist(),
            riser_x=edge_x,
            riser_height_m=jumps,
            direction=direction,
        )


def sole_center(foot_pos, foot_quat, capsules):
    from scipy.spatial.transform import Rotation

    endpoints = np.array([c["endpoints"] for c in capsules]).reshape(-1, 3)
    local_center = (endpoints.min(0) + endpoints.max(0)) / 2
    rotation = Rotation.from_quat(foot_quat[:, [1, 2, 3, 0]])
    return foot_pos + rotation.apply(
        np.broadcast_to(local_center, foot_pos.shape).copy()
    )


def trial_landings(arrays, ids, env, origin, profile, surface, capsules, dt):
    t = arrays["t"][ids]
    pos = arrays["foot_pos_w"][ids, env] - origin
    quat = arrays["foot_quat_w"][ids, env]
    centers = np.stack(
        [sole_center(pos[:, foot], quat[:, foot], capsules[foot]) for foot in range(2)],
        axis=1,
    )
    force = arrays["contact_force_w"][ids, env]
    h = np.linalg.norm(force[:, :, :2], axis=-1)
    supports = (force[:, :, 2] > 20) & (h <= 1.5 * force[:, :, 2])
    air_min, hold_min = round(0.05 / dt), round(0.05 / dt)
    landings = []
    for foot in range(2):
        mask = supports[:, foot].copy()
        # Bridge at most 10 ms of contact flicker, never a genuine swing.
        for start, end in runs(~mask):
            if start > 0 and end < len(mask) and end - start <= round(0.01 / dt):
                mask[start:end] = True
        previous_end = 0
        for start, end in runs(mask):
            air = start - previous_end
            previous_end = end
            if end - start < hold_min or (start > 0 and air < air_min):
                continue
            sample = slice(start, start + hold_min)
            center = np.median(centers[sample, foot], axis=0)
            terrain_z = surface.height(center[:2])
            geometry_available = bool(
                np.isfinite(terrain_z) and np.isfinite(center).all()
            )
            in_corridor = bool(abs(center[1]) < 0.65)
            step = (
                int(np.searchsorted(profile["riser_x"], center[0]))
                if geometry_available and in_corridor
                else None
            )
            edge_distance = min(
                [abs(center[0] - x) for x in profile["riser_x"]], default=100.0
            )
            landings.append(
                dict(
                    foot=["left", "right"][foot],
                    foot_id=foot,
                    start_s=float(t[start]),
                    confirmation_s=float(t[start + hold_min - 1]),
                    duration_s=float((end - start) * dt),
                    preceding_air_s=float(air * dt),
                    center_local_m=center.tolist(),
                    foot_link_local_m=np.median(pos[sample, foot], axis=0).tolist(),
                    median_upward_force_N=float(np.median(force[sample, foot, 2])),
                    terrain_z_m=terrain_z if geometry_available else None,
                    tread=step,
                    geometry_available=geometry_available,
                    in_stair_corridor=in_corridor,
                    ambiguous_geometry=not geometry_available,
                    edge_distance_m=float(edge_distance),
                    ambiguous_edge=bool(edge_distance < 0.025),
                    initial_support=bool(start < round(0.2 / dt)),
                )
            )
    landings.sort(key=lambda r: (r["start_s"], r["foot_id"]))
    last = [None, None]
    last_progress_foot = None
    interior_progress = []
    for event in landings:
        foot, step = event["foot_id"], event["tread"]
        own, other = last[foot], last[1 - foot]
        event["previous_own_tread"] = None if own is None else own["tread"]
        event["previous_other_tread"] = None if other is None else other["tread"]
        event["classification"] = "initial_or_flat"
        if step is None:
            event["classification"] = (
                "unavailable_geometry"
                if not event["geometry_available"]
                else "outside_stair_corridor"
            )
            continue
        interior = 0 < step < len(profile["riser_x"])
        progressing = own is not None and step > own["tread"]
        if (
            interior
            and progressing
            and not event["ambiguous_edge"]
            and other is not None
        ):
            delta_own, delta_other = step - own["tread"], step - other["tread"]
            if delta_other == 0:
                label = "step_to_same_tread"
            elif delta_other == 1 and delta_own == 2 and last_progress_foot != foot:
                label = "reciprocal_next_tread"
            elif delta_other == 1 and delta_own == 1:
                label = (
                    "opening_step"
                    if own["tread"] == other["tread"] == 0
                    else "lead_after_same_tread"
                )
            elif delta_other > 1:
                label = "skips_tread_ahead_of_other"
            else:
                label = "other_progress"
            event["classification"] = label
            interior_progress.append(event)
            last_progress_foot = foot
        # Do not let ambiguous contacts at a vertical edge redefine tread history.
        if not event["ambiguous_edge"]:
            last[foot] = event
    eligible = [r for r in interior_progress if r["classification"] != "opening_step"]
    counts = {
        label: sum(r["classification"] == label for r in eligible)
        for label in [
            "reciprocal_next_tread",
            "step_to_same_tread",
            "lead_after_same_tread",
            "skips_tread_ahead_of_other",
            "other_progress",
        ]
    }
    return dict(
        landings=landings,
        interior_progress_landings=len(interior_progress),
        eligible_landings=len(eligible),
        unavailable_geometry_landings=sum(
            not r["geometry_available"] for r in landings
        ),
        outside_corridor_landings=sum(not r["in_stair_corridor"] for r in landings),
        classifications=counts,
        reciprocal_fraction=counts["reciprocal_next_tread"] / len(eligible)
        if eligible
        else None,
        foot_center_local=centers,
        support_mask=supports,
    )


def analyze(output, envs):
    manifest = json.loads((output / "manifest.json").read_text())
    arrays, mesh = np.load(output / "trace.npz"), np.load(output / "terrain_mesh.npz")
    result = dict(
        checkpoint=manifest["checkpoint"],
        evaluation_schema_version=manifest.get("evaluation_schema_version", 1),
        criterion="Mesh tread boundaries: median east corridor height jump >4.5 cm. Landing: >=50 ms upward support Fz>20 N and H<=1.5Fz, after >=50 ms without such support; <=10 ms contact flicker bridged. Classify first 50 ms median shoe capsule center by actual mesh x boundary; omit centers within 2.5 cm of risers from progression counts. Missing mesh intersections and centers outside |local y|<0.65 m receive null tread and do not update landing history or counts. Omit initial and final platforms. Reciprocal requires own foot advances two treads and lands one tread beyond other foot with opposite preceding progress foot; opening ascent step excluded. Thus simple alternating swing with both feet on each tread is not rewarded by this metric.",
        limitations="Tread assignment uses east-corridor mesh geometry and shoe center, not measured contact point. Appropriate to these straight fixed +x stair trials. Missing/brief supports and edge-ambiguous contacts are retained in the trace but not eligible; always inspect landing lists and side-impact/fall/completion results. Descending terrain is reported separately, not counted as ascending evidence.",
        profiles={},
        trials=[],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for env in envs:
        origin = np.asarray(manifest["env_origins"][env])
        surface = Surface(mesh["vertices"], mesh["faces"], origin)
        profile = surface.profile()
        result["profiles"][str(env)] = profile
        trial_ids = np.unique(arrays["trial"])
        fig, axs = plt.subplots(
            len(trial_ids), 2, figsize=(13, 3.2 * len(trial_ids)), squeeze=False
        )
        for row, trial in enumerate(trial_ids):
            ids = np.flatnonzero(
                (arrays["trial"] == trial) & (arrays["episode"][:, env] == 0)
            )
            record = trial_landings(
                arrays,
                ids,
                env,
                origin,
                profile,
                surface,
                manifest["shoe_capsules"],
                manifest["physics_dt"],
            )
            centers, supports = (
                record.pop("foot_center_local"),
                record.pop("support_mask"),
            )
            record.update(
                env=env,
                trial=int(trial),
                terrain=manifest["column_terrain_names"][env],
                direction=profile["direction"],
            )
            result["trials"].append(record)
            ax, phase = axs[row]
            t = arrays["t"][ids]
            for foot, color in [(0, "tab:blue"), (1, "tab:orange")]:
                name = ["Left", "Right"][foot]
                ax.plot(t, centers[:, foot, 0], color=color, label=name, linewidth=1)
                phase.plot(
                    centers[:, foot, 0],
                    centers[:, foot, 2],
                    color=color,
                    linewidth=1,
                    label=name,
                )
                for event in record["landings"]:
                    if event["foot_id"] == foot:
                        x, y, z = event["center_local_m"]
                        ax.scatter(event["start_s"], x, color=color, s=20)
                        ax.text(
                            event["start_s"],
                            x + 0.025,
                            str(event["tread"]) if event["tread"] is not None else "?",
                            fontsize=7,
                            color=color,
                        )
            for boundary in profile["riser_x"]:
                ax.axhline(boundary, color=".65", linewidth=0.5)
            phase.plot(profile["x"], profile["z"], color="black", label="mesh surface")
            ax.set(
                xlabel="Time (s)",
                ylabel="Shoe center forward x (m)",
                title=f"Trial {trial}: reciprocal {record['classifications']['reciprocal_next_tread']}/{record['eligible_landings']}",
            )
            phase.set(
                xlabel="Forward x (m)",
                ylabel="Shoe center / terrain z (m)",
                title=f"{profile['direction']}, step labels on left",
            )
            ax.legend()
            phase.legend()
            ax.grid(alpha=0.2)
            phase.grid(alpha=0.2)
        fig.suptitle(
            f"{Path(manifest['checkpoint']).stem}: env {env} / {manifest['column_terrain_names'][env]}"
        )
        fig.tight_layout()
        fig.savefig(output / f"stair_gait_env{env}.png", dpi=140)
        plt.close(fig)
    ascending = [r for r in result["trials"] if r["direction"] == "ascending"]
    ordinary_names = {"pyramid_stairs", "pyramid_stairs_inv"}
    for key, selected in [
        (
            "ordinary_ascending_aggregate",
            [r for r in ascending if r["terrain"] in ordinary_names],
        ),
        (
            "high_ascending_aggregate",
            [r for r in ascending if r["terrain"] not in ordinary_names],
        ),
        ("ascending_aggregate", ascending),
    ]:
        count = sum(r["eligible_landings"] for r in selected)
        reciprocal = sum(
            r["classifications"]["reciprocal_next_tread"] for r in selected
        )
        result[key] = dict(
            sampled_attempts=len(selected),
            attempts_with_eligible_landings=sum(
                r["eligible_landings"] > 0 for r in selected
            ),
            eligible_landings=count,
            reciprocal_landings=reciprocal,
            reciprocal_fraction=reciprocal / count if count else None,
            step_to_landings=sum(
                r["classifications"]["step_to_same_tread"] for r in selected
            ),
            note="Use ordinary_ascending_aggregate as the primary gait result. A single high riser has no interior-tread alternation opportunity; zero eligible landings means not applicable, not a failure.",
        )
    (output / "stair_gait.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--envs", type=int, nargs="+", default=[2, 3, 5, 6])
    args = parser.parse_args()
    result = analyze(args.output, args.envs)
    print(
        json.dumps(
            dict(
                ordinary_ascending_aggregate=result["ordinary_ascending_aggregate"],
                high_ascending_aggregate=result["high_ascending_aggregate"],
                trials=[
                    {
                        key: r[key]
                        for key in [
                            "env",
                            "trial",
                            "direction",
                            "eligible_landings",
                            "classifications",
                        ]
                    }
                    for r in result["trials"]
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
