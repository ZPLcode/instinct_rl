"""CPU comparison of analyze_gaps.py outputs with strict v2 pairing checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_gaps(baseline, candidate):
    reports = [
        json.loads((p / "report.json").read_text()) for p in [baseline, candidate]
    ]
    analyses = [
        json.loads((p / "analysis.json").read_text()) for p in [baseline, candidate]
    ]
    old, new = reports
    mismatches, checks = [], []
    for label, report in [("baseline", old), ("candidate", new)]:
        if (
            report.get("evaluation_schema_version") != 2
            or report["status"] != "complete"
        ):
            mismatches.append(
                dict(
                    field=label, reason="Requires complete v2 evaluate_gaps.py output."
                )
            )
    for key in [
        "evaluation_schema_version",
        "reset_protocol",
        "config_sha256",
        "terrain_sha256",
        "trial_seeds",
        "configured_widths_m",
        "num_envs",
        "trials_per_width",
        "trial_duration_s",
        "seed",
        "terrain_seed",
        "terrain_types",
        "terrain_levels",
        "physics_dt",
        "control_dt",
        "observation_shape",
    ]:
        if key not in old or key not in new or old.get(key) != new.get(key):
            mismatches.append(
                dict(field=key, baseline=old.get(key), candidate=new.get(key))
            )
    a, b = analyses
    for key in ["geometry", "criteria"]:
        if a[key] != b[key]:
            mismatches.append(
                dict(
                    field="analysis_" + key,
                    reason="Actual geometry or outcome criteria differ.",
                )
            )
    state_paths = [p / "initial_states.npz" for p in [baseline, candidate]]
    if not all(p.exists() for p in state_paths):
        mismatches.append(
            dict(field="initial_states", reason="Missing recorded reset state arrays.")
        )
    else:
        x, y = (np.load(p) for p in state_paths)
        if set(x.files) != set(y.files):
            mismatches.append(
                dict(field="initial_state_fields", reason="Reset array keys differ.")
            )
        for key in sorted(set(x.files) & set(y.files)):
            tolerance = 1e-5 if key.endswith("policy_observation") else 1e-6
            same = x[key].shape == y[key].shape and np.allclose(
                x[key], y[key], atol=tolerance, rtol=0, equal_nan=False
            )
            record = dict(
                field=key,
                matches=bool(same),
                tolerance=tolerance,
                max_absolute_difference=float(np.abs(x[key] - y[key]).max())
                if x[key].shape == y[key].shape
                else None,
            )
            checks.append(record)
            if not same:
                mismatches.append(record)
    old_attempts, new_attempts = [
        {(r["trial"], r["env"]): r for r in data["attempts"]} for data in analyses
    ]
    if old_attempts.keys() != new_attempts.keys():
        mismatches.append(dict(field="attempts", reason="Trial/env sets differ."))
    trials = []
    for key in sorted(old_attempts.keys() & new_attempts.keys()):
        x, y = old_attempts[key], new_attempts[key]
        record = dict(trial=key[0], env=key[1], actual_width_m=x["actual_width_m"])
        for field in [
            "status",
            "success",
            "success_confirmed_s",
            "minimum_crossing_root_height_m",
            "maximum_crossing_tilt_rad",
            "minimum_shoe_z_while_ankle_over_gap_m",
            "any_deep_foot_intrusion",
            "any_pit_floor_support",
            "any_pit_root_low",
            "resets",
        ]:
            record[field] = dict(baseline=x[field], candidate=y[field])
        trials.append(record)
    return dict(
        paired_conditions_match=not mismatches,
        mismatches=mismatches,
        initial_array_checks=checks,
        baseline_checkpoint=old["checkpoint"],
        candidate_checkpoint=new["checkpoint"],
        baseline_summary=a["summary"],
        candidate_summary=b["summary"],
        trials=trials,
        interpretation="Three reseeded paired starts per gap width. Check deep foot intrusion and later failure resets in addition to reaching the target. No general reliability estimate; later stochastic events can diverge after policy-dependent automatic resets.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing comparison")
    result = compare_gaps(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "paired_conditions_match",
                    "mismatches",
                    "baseline_summary",
                    "candidate_summary",
                ]
            },
            indent=2,
        )
    )
    if not result["paired_conditions_match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
