"""Isaac parkour contact experiment, with no Isaac/Omniverse import requirement.

Only the side-impact cost and the contact gates of the two original foot
regularizers change. Force histories use Isaac's [env, time, body, xyz] layout.
The wrappers accept Isaac SceneEntityCfg and environment objects by interface.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def foot_contact_masks(
    forces_history: torch.Tensor,
    min_force: float = 20.0,
    horizontal_ratio: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-foot ``(side_impact, support)`` masks of shape [env, body].

    Classify each physics frame before reducing history: vertical support in
    one frame must not hide a horizontal impact in another. A side impact in
    the window vetoes support for that foot, so a previous stance sample does
    not activate the "flat and near ground" costs while the toe hits a riser.
    This conservative veto lasts only until that sample leaves the window;
    it does not alter the contact sensor, air-time rewards or other terms.
    """
    if (
        forces_history.ndim != 4
        or forces_history.shape[-1] != 3
        or forces_history.shape[1] == 0
    ):
        raise ValueError("Expected non-empty force history shaped [env, time, body, 3]")
    if not math.isfinite(min_force) or min_force < 0:
        raise ValueError("min_force must be finite and non-negative")
    if not math.isfinite(horizontal_ratio) or horizontal_ratio <= 0:
        raise ValueError("horizontal_ratio must be finite and positive")

    horizontal = torch.linalg.vector_norm(forces_history[..., :2], dim=-1)
    vertical = forces_history[..., 2]
    side_frames = (horizontal > min_force) & (
        horizontal > horizontal_ratio * vertical.clamp_min(0)
    )
    support_frames = (vertical > min_force) & (
        horizontal <= horizontal_ratio * vertical
    )
    side_impact = side_frames.any(dim=1)
    support = support_frames.any(dim=1) & ~side_impact
    return side_impact, support


def _selected_force_history(env: Any, sensor_cfg: Any) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """IsaacLab's unit-quaternion inverse rotation, in scalar-first order."""
    xyz = quat[..., 1:]
    twice_cross = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return (
        vec - quat[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)
    )


def feet_side_collision(
    env: Any,
    sensor_cfg: Any,
    min_force: float = 20.0,
    horizontal_ratio: float = 1.5,
) -> torch.Tensor:
    """Mean side-only horizontal force / 100 N; configure a negative weight.

    History spans one control step. Reward-manager dt scaling makes the cost
    proportional to side impulse, without including valid support frames.
    """
    forces = _selected_force_history(env, sensor_cfg)
    foot_contact_masks(
        forces, min_force, horizontal_ratio
    )  # Preserve input validation.
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    side = (horizontal > min_force) & (
        horizontal > horizontal_ratio * forces[..., 2].clamp_min(0)
    )
    return (horizontal * side).mean(dim=1).sum(dim=-1) / 100.0


def feet_orientation_contact(
    env: Any,
    sensor_cfg: Any,
    asset_cfg: Any,
    min_force: float = 20.0,
    horizontal_ratio: float = 1.5,
) -> torch.Tensor:
    """Original Isaac foot-orientation cost, gated by support instead of any contact."""
    asset = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = _quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = _quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    _, is_contact = foot_contact_masks(
        _selected_force_history(env, sensor_cfg), min_force, horizontal_ratio
    )

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5
        * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5
        * is_contact[:, 1]
    )


def feet_at_plane(
    env: Any,
    contact_sensor_cfg: Any,
    left_height_scanner_cfg: Any,
    right_height_scanner_cfg: Any,
    asset_cfg: Any,
    height_offset: float = 0.035,
    min_force: float = 20.0,
    horizontal_ratio: float = 1.5,
) -> torch.Tensor:
    """Original Isaac foot-height cost, gated by support instead of any contact."""
    asset = env.scene[asset_cfg.name]
    _, is_contact = foot_contact_masks(
        _selected_force_history(env, contact_sensor_cfg),
        min_force,
        horizontal_ratio,
    )
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(
        torch.isinf(right_sensor_data), 0.0, right_sensor_data
    )
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(
            left_height.unsqueeze(-1) - left_sensor_data - height_offset,
            min=0.0,
            max=0.3,
        )
        * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(
            right_height.unsqueeze(-1) - right_sensor_data - height_offset,
            min=0.0,
            max=0.3,
        )
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)
