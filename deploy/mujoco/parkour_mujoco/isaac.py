"""Translate saved Isaac parkour settings without importing Isaac or ROS.

Joint order follows instinct_onboard's G1_29Dof_TorsoBase.sim_joint_names.
Actions remain unscaled until the MuJoCo runner applies the saved joint PD.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

G1_ISAAC_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


class _SavedLoader(yaml.SafeLoader):
    pass


_SavedLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node),
)
_SavedLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: {"slice": loader.construct_sequence(node)},
)


def read_saved_yaml(path: str | Path) -> dict:
    """Allow Isaac's tuple/slice data tags, never arbitrary Python constructors."""
    return yaml.load(Path(path).read_text(), Loader=_SavedLoader)


def _value(value, name: str, default=0.0):
    if value is None:
        return default
    if not isinstance(value, dict):
        return float(value)
    matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one Isaac actuator value for {name}, got {matches}")
    return float(matches[0])


def normalize_env_config(config: dict) -> dict:
    """Adapt the supported torso-base shoe asset to the existing native runner."""
    if "entities" in config["scene"]:
        return config
    config = copy.deepcopy(config)
    robot = config["scene"]["robot"]
    if (
        Path(robot["spawn"]["asset_path"]).name
        != "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    ):
        raise ValueError("Isaac sim2sim requires the trained G1 torso-base shoe URDF")
    groups = []
    for name in G1_ISAAC_JOINT_NAMES:
        matched = [
            g
            for g in robot["actuators"].values()
            if any(re.fullmatch(p, name) for p in g["joint_names_expr"])
        ]
        if len(matched) != 1:
            raise ValueError(f"Missing or overlapping Isaac actuator for {name}")
        group = matched[0]
        if not group["class_type"].endswith(":ImplicitActuator"):
            raise ValueError("Only saved Isaac implicit PD actuators are supported")
        groups.append(
            dict(
                target_names_expr=[re.escape(name)],
                transmission_type="joint",
                stiffness=_value(group["stiffness"], name),
                damping=_value(group["damping"], name),
                armature=_value(group.get("armature"), name),
                effort_limit=_value(
                    group.get("effort_limit_sim") or group.get("effort_limit"), name
                ),
                delay_min_lag=0,
                delay_max_lag=0,
            )
        )
    native_robot = dict(
        spec_fn="parkour_mujoco:g1_with_shoes",
        init_state=robot["init_state"],
        articulation={"actuators": groups},
        policy_joint_names=list(G1_ISAAC_JOINT_NAMES),
    )
    config["scene"]["entities"] = {"robot": native_robot}
    for name in ("policy", "critic"):
        source = config["observations"][name]
        source["terms"] = {
            key: value
            for key, value in source.items()
            if isinstance(value, dict) and "func" in value
        }
    sensor = copy.deepcopy(config["scene"]["camera"])
    pattern = sensor["pattern_cfg"]
    sensor.update(
        name="camera",
        frame={"type": "body", "entity": "robot", "name": "torso_link"},
        pattern={"height": pattern["height"], "width": pattern["width"]},
        pixel_center_offset=0.0,
        exclude_parent_body=False,
    )
    for key in (
        "focal_length",
        "horizontal_aperture",
        "vertical_aperture",
        "horizontal_aperture_offset",
        "vertical_aperture_offset",
    ):
        sensor[key] = pattern[key]
    # This G1 camera includes all link visuals, including fixed geometry merged into the torso.
    sensor["include_body_names"] = [
        m.group(1)
        for item in sensor["mesh_prim_paths"]
        if isinstance(item, dict)
        and (m := re.search(r"/Robot/([^/]+)/visuals", item["prim_expr"]))
    ]
    sensor.pop("mesh_prim_paths")
    config["scene"]["sensors"] = [sensor]
    config["actions"]["joint_pos"]["actuator_names"] = [".*"]
    config["sim"]["mujoco"] = dict(
        timestep=config["sim"]["dt"],
        gravity=config["sim"]["gravity"],
        integrator="implicitfast",
        solver="newton",
        iterations=20,
        cone="pyramidal",
        impratio=1.0,
    )
    config["source_training_backend"] = "Isaac/PhysX"
    return config
