"""Isaac parkour reward configuration for side contacts and reciprocal landings."""

from __future__ import annotations

import copy

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from rewards import feet_at_plane, feet_orientation_contact, feet_side_collision


def apply_contact_rewards(cfg, side_weight: float = -0.5):
    """Change only the two stance regularizers and add a side-collision cost.

    A separate two-foot sensor covers all physics samples in one control step.
    The original contact sensor and its consumers retain their original history.
    Neither the actor nor the critic receives additional observations.
    """
    cfg.scene.stair_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        update_period=cfg.sim.dt,
        history_length=cfg.decimation,
        track_air_time=False,
    )
    sensor = SceneEntityCfg("stair_contact_forces", body_names=".*_ankle_roll_link")
    terms = cfg.rewards.rewards
    terms.feet_side_collision = RewardTermCfg(
        func=feet_side_collision,
        weight=side_weight,
        params=dict(
            sensor_cfg=copy.deepcopy(sensor), min_force=20.0, horizontal_ratio=1.5
        ),
    )
    terms.feet_flat_ori.func = feet_orientation_contact
    terms.feet_flat_ori.params["sensor_cfg"] = copy.deepcopy(sensor)
    terms.feet_flat_ori.params.update(min_force=20.0, horizontal_ratio=1.5)
    terms.feet_at_plane.func = feet_at_plane
    terms.feet_at_plane.params["contact_sensor_cfg"] = copy.deepcopy(sensor)
    terms.feet_at_plane.params.update(min_force=20.0, horizontal_ratio=1.5)
    return cfg


def apply_landing_reward(cfg, landing_weight: float = 0.1, catchup_cost: float = 0.0):
    """Add an event bonus for reciprocal landings on ordinary ascending/descending stairs.

    The callable returns events/step_dt so the manager's dt scaling gives a
    fixed bonus per landing. catchup_cost is the actual deduction per event,
    normalized below so it is independent of landing_weight.
    Terrain geometry is privileged reward information;
    policy observations and the training terrain mixture are unchanged.
    """
    from stair_landing import ReciprocalStairLanding

    if catchup_cost and landing_weight <= 0:
        raise ValueError("A catch-up cost requires positive landing_weight")

    cfg.rewards.rewards.stair_reciprocal_landing = RewardTermCfg(
        func=ReciprocalStairLanding,
        weight=landing_weight,
        params=dict(
            sensor_cfg=SceneEntityCfg(
                "stair_contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            asset_cfg=SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            left_height_scanner_cfg=SceneEntityCfg("left_height_scanner"),
            right_height_scanner_cfg=SceneEntityCfg("right_height_scanner"),
            command_name="base_velocity",
            catchup_cost=catchup_cost / landing_weight if catchup_cost else 0.0,
            min_swing_time=0.08,
            stable_support_time=0.04,
            min_forward=0.12,
            max_forward=0.5,
            min_rise=0.025,
            max_rise=0.26,
            min_force=20.0,
            horizontal_ratio=1.5,
        ),
    )
    return cfg
