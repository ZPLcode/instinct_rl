"""Compare two evaluate.py outputs; refuse an unpaired comparison by default.

CPU only. Usage: python compare.py BASELINE_OUTPUT FINETUNED_OUTPUT --output comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(baseline: Path, candidate: Path):
    old = json.loads((baseline / "manifest.json").read_text())
    new = json.loads((candidate / "manifest.json").read_text())
    old_summary = json.loads((baseline / "summary.json").read_text())
    new_summary = json.loads((candidate / "summary.json").read_text())
    mismatches = []
    for key, legacy in [
        ("evaluation_schema_version", 1),
        ("reset_protocol", "legacy_native_reset_v1"),
    ]:
        a, b = old.get(key, legacy), new.get(key, legacy)
        if a != b:
            mismatches.append(
                dict(
                    field=key,
                    baseline=a,
                    candidate=b,
                    reason="Reset protocols differ; rerun both checkpoints with the same evaluator.",
                )
            )
    for key in [
        "status",
        "terrain_sha256",
        "config_sha256",
        "trial_seeds",
        "duration_s",
        "construction_seed",
        "terrain_seed",
        "terrain_levels",
        "physics_dt",
        "control_dt",
    ]:
        if old[key] != new[key]:
            mismatches.append(dict(field=key, baseline=old[key], candidate=new[key]))
    if old["status"] != "complete" or new["status"] != "complete":
        mismatches.append(
            dict(field="completion", reason="Both evaluations must be complete.")
        )
    old_states, new_states = (
        np.load(baseline / "initial_states.npz"),
        np.load(candidate / "initial_states.npz"),
    )
    checks = []
    if set(old_states.files) != set(new_states.files):
        mismatches.append(
            dict(field="initial_state_fields", reason="State keys differ.")
        )
    for key in sorted(set(old_states.files) & set(new_states.files)):
        a, b = old_states[key], new_states[key]
        tolerance = 1e-5 if key.endswith("policy_observation") else 1e-6
        same = a.shape == b.shape and np.allclose(
            a, b, rtol=0, atol=tolerance, equal_nan=False
        )
        maximum = float(np.max(np.abs(a - b))) if a.shape == b.shape else None
        checks.append(
            dict(
                field=key,
                matches=bool(same),
                max_absolute_difference=maximum,
                tolerance=tolerance,
            )
        )
        if not same:
            mismatches.append(checks[-1])
    pairs = []
    old_trials = {(r["trial"], r["env"]): r for r in old_summary["trials"]}
    new_trials = {(r["trial"], r["env"]): r for r in new_summary["trials"]}
    if old_trials.keys() != new_trials.keys():
        mismatches.append(dict(field="trials", reason="Trial/env keys differ."))
    for trial, env in sorted(old_trials.keys() & new_trials.keys()):
        a, b = old_trials[(trial, env)], new_trials[(trial, env)]
        record = dict(trial=trial, env=env, terrain=a["terrain"])
        for key in [
            "event_count",
            "horizontal_impulse_Ns",
            "maximum_horizontal_force_N",
            "final_goal_distance_m",
            "minimum_root_height_m",
            "maximum_tilt_rad",
        ]:
            record[key] = dict(baseline=a[key], candidate=b[key], delta=b[key] - a[key])
        for key in ["fell", "goal_reached", "stable_goal_time_s"]:
            record[key] = dict(baseline=a[key], candidate=b[key])
        if "gap" in a:
            record["gap_outcome"] = dict(
                baseline=a["gap"]["outcome"], candidate=b["gap"]["outcome"]
            )
        pairs.append(record)
    aggregate = {}
    for key in ["stairs_env5", "square_gaps"]:
        a, b = old_summary[key], new_summary[key]
        aggregate[key] = dict(
            baseline=a,
            candidate=b,
            horizontal_impulse_change_Ns=b["total_horizontal_impulse_Ns"]
            - a["total_horizontal_impulse_Ns"],
            event_count_change=b["total_events"] - a["total_events"],
            stable_goal_count_change=b["stable_goal_count"] - a["stable_goal_count"],
            fall_count_change=b["fell_count"] - a["fell_count"],
        )
    return dict(
        paired_conditions_match=not mismatches,
        mismatches=mismatches,
        initial_array_checks=checks,
        baseline_checkpoint=old["checkpoint"],
        candidate_checkpoint=new["checkpoint"],
        aggregate=aggregate,
        trials=pairs,
        interpretation="Assess contact impulse together with goal completion, falls, and gap retention. Lower side-force cost caused by standing still is not improvement. These are three paired initial conditions per tile, not a reliability estimate. Sensor side-force candidates can also include rough-surface or self contacts; use saved meshes/poses to diagnose individual events.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["paired_conditions_match", "mismatches", "aggregate"]
            },
            indent=2,
        )
    )
    if not result["paired_conditions_match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
