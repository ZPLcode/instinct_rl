"""Opt-in reciprocal stair landing bonus, with a CPU-testable state machine.

The bonus is a single event, not a continuous reward for holding the feet apart.
A rear foot must leave the preceding terrace, swing, pass the other foot's established
foothold, and acquire stable support on the next terrace up or down. An optional
cost penalizes interior catch-up landings; initial steps and reset spawns remain free.

Terrain labels and the existing privileged foot scanners are reward inputs only;
this term does not add observations or change any actor input.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    from isaaclab.managers import ManagerTermBase
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab":
        raise

    class ManagerTermBase:  # CPU-only tests; Isaac uses its actual manager base.
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env


def nominal_stair_terraces(
    foot_xy: torch.Tensor,
    origin_xy: torch.Tensor,
    first_boundary: torch.Tensor,
    step_width: torch.Tensor,
    num_steps: torch.Tensor,
    inverted: torch.Tensor,
    boundary_margin: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed nominal terrace index and validity, independently of Perlin noise.

    Square-pyramid stairs have nested square contours. A higher signed index is
    always a higher terrace, including inverted pyramids. ``first_boundary`` is
    the outermost nominal riser radius; points near any riser are withheld from
    reward because height-field mesh conversion may move a vertex by one cell.
    """
    radius = (foot_xy - origin_xy[:, None, :]).abs().amax(dim=-1)
    coordinate = (first_boundary[:, None] - radius) / step_width[:, None]
    nesting = (torch.floor(coordinate).long() + 1).clamp_min(0)
    nesting = torch.minimum(nesting, num_steps[:, None])
    nearest = coordinate.round().clamp_min(0)
    nearest = torch.minimum(nearest, (num_steps - 1).clamp_min(0)[:, None])
    edge_distance = (coordinate - nearest).abs() * step_width[:, None]
    valid = torch.isfinite(radius) & (edge_distance > boundary_margin)
    signed = torch.where(inverted[:, None], -nesting, nesting)
    return signed, valid


class LandingTracker:
    """Vectorized state and reset handling; all inputs have an environment axis.

    ``support`` means current upward support, with no recent side impact.
    ``airborne`` is actual low contact force, not merely absence of support.
    ``terrace`` and ``ground_height`` increase along the course, after the adapter
    normalizes descent. Noisy heights cannot invent integer terrace progress.
    ``active`` combines the upright, moving-command and not-at-goal conditions.
    Lifetime counters intentionally survive episode resets for experiment audits.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        min_swing_time: float = 0.08,
        stable_support_time: float = 0.04,
        min_forward: float = 0.12,
        max_forward: float = 0.5,
        min_rise: float = 0.025,
        max_rise: float = 0.26,
        catchup_cost: float = 0.0,
    ):
        if not (
            0 < min_swing_time
            and 0 < stable_support_time
            and 0 < min_forward < max_forward
            and 0 < min_rise < max_rise
        ):
            raise ValueError("Landing thresholds must be positive and ordered")
        if not math.isfinite(catchup_cost) or catchup_cost < 0:
            raise ValueError("catchup_cost must be finite and nonnegative")
        self.catchup_cost = catchup_cost  # Normalized by the native reward term weight.
        self.min_swing_time, self.stable_support_time = (
            min_swing_time,
            stable_support_time,
        )
        self.min_forward, self.max_forward = min_forward, max_forward
        self.min_rise, self.max_rise = min_rise, max_rise
        self.air_time = torch.zeros(num_envs, 2, device=device)
        self.support_time = torch.zeros_like(self.air_time)
        self.previous_airborne = torch.zeros(
            num_envs, 2, dtype=torch.bool, device=device
        )
        self.seen_stance = torch.zeros_like(self.previous_airborne)
        self.candidate = torch.zeros_like(self.previous_airborne)
        self.stance_pos = torch.zeros(num_envs, 2, 3, device=device)
        self.stance_height = torch.zeros_like(self.air_time)
        self.stance_terrace = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self.takeoff_pos = torch.zeros_like(self.stance_pos)
        self.opposite_pos = torch.zeros_like(self.stance_pos)
        self.opposite_height = torch.zeros_like(self.air_time)
        self.opposite_terrace = torch.zeros_like(self.stance_terrace)
        self.swing_direction = torch.zeros(num_envs, 2, 2, device=device)
        self.highest_terrace = torch.full(
            (num_envs,), -1_000_000, dtype=torch.long, device=device
        )
        self.last_events = torch.zeros_like(self.previous_airborne)
        self.counts = {
            name: torch.zeros(num_envs, dtype=torch.long, device=device)
            for name in (
                "stable_landings",
                "ordinary_stable_landings",
                "nonordinary_stable_landings",
                "eligible_landings",
                "rewarded_landings",
                "nonordinary_rewarded_landings",
                "penalized_catchup_landings",
                "nonordinary_penalized_catchup_landings",
            )
        }

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "air_time",
            "support_time",
            "previous_airborne",
            "seen_stance",
            "candidate",
            "stance_pos",
            "stance_height",
            "stance_terrace",
            "takeoff_pos",
            "opposite_pos",
            "opposite_height",
            "opposite_terrace",
            "swing_direction",
            "last_events",
        ):
            getattr(self, name)[ids] = 0
        self.highest_terrace[ids] = -1_000_000

    def update(
        self,
        *,
        foot_pos: torch.Tensor,
        ground_height: torch.Tensor,
        terrace: torch.Tensor,
        terrain_valid: torch.Tensor,
        support: torch.Tensor,
        airborne: torch.Tensor,
        ordinary: torch.Tensor,
        direction: torch.Tensor,
        active: torch.Tensor,
        dt: float,
        top_terrace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        if self.catchup_cost and top_terrace is None:
            raise ValueError("Catch-up cost requires the signed top terrace")
        swap = [1, 0]
        beginning_swing = airborne & ~self.previous_airborne
        opposite_stance_pos = self.stance_pos[:, swap]
        rear_distance = (
            (self.stance_pos[..., :2] - opposite_stance_pos[..., :2])
            * direction[:, None]
        ).sum(-1)
        # Require the other foot to have a real established foothold. The foot
        # lifting must be on the immediately preceding terrace, not beside it.
        can_start = (
            self.seen_stance
            & self.seen_stance[:, swap]
            & support[:, swap]
            & terrain_valid[:, swap]
            & (self.support_time[:, swap] >= self.stable_support_time - 1e-7)
            & (terrace[:, swap] == self.stance_terrace[:, swap])
            & (self.stance_terrace == self.stance_terrace[:, swap] - 1)
            & (rear_distance < -self.min_forward)
            & ordinary[:, None]
            & active[:, None]
        )
        self.candidate = torch.where(beginning_swing, can_start, self.candidate)
        self.takeoff_pos = torch.where(
            beginning_swing[..., None], self.stance_pos, self.takeoff_pos
        )
        self.opposite_pos = torch.where(
            beginning_swing[..., None], opposite_stance_pos, self.opposite_pos
        )
        self.opposite_height = torch.where(
            beginning_swing, self.stance_height[:, swap], self.opposite_height
        )
        self.opposite_terrace = torch.where(
            beginning_swing, self.stance_terrace[:, swap], self.opposite_terrace
        )
        self.swing_direction = torch.where(
            beginning_swing[..., None], direction[:, None], self.swing_direction
        )
        self.air_time = torch.where(
            beginning_swing, torch.zeros_like(self.air_time), self.air_time
        )
        self.air_time += airborne * dt
        previous_support_time = self.support_time.clone()
        self.support_time = torch.where(
            support, self.support_time + dt, torch.zeros_like(self.support_time)
        )
        stable = self.support_time >= self.stable_support_time - 1e-7
        new_stable = stable & (previous_support_time < self.stable_support_time - 1e-7)
        # A reset/init settles into stance without being a landing event.
        landing = (
            new_stable
            & self.seen_stance
            & (self.air_time >= self.min_swing_time - 1e-7)
        )
        forward = (
            (foot_pos[..., :2] - self.opposite_pos[..., :2]) * self.swing_direction
        ).sum(-1)
        own_progress = (
            (foot_pos[..., :2] - self.takeoff_pos[..., :2]) * self.swing_direction
        ).sum(-1)
        rise = ground_height - self.opposite_height
        eligible = landing & ordinary[:, None] & active[:, None] & terrain_valid
        event = (
            eligible
            & self.candidate
            & (terrace == self.opposite_terrace + 1)
            & (terrace > self.highest_terrace[:, None])
            & (forward >= self.min_forward)
            & (forward <= self.max_forward)
            & (own_progress >= 2 * self.min_forward)
            & (rise >= self.min_rise)
            & (rise <= self.max_rise)
        )
        catchup = torch.zeros_like(event)
        if self.catchup_cost:
            # Last verified opposite foothold may be airborne; top and below-frontier landings stay free.
            catchup = (
                eligible
                & (self.stance_terrace < terrace)
                & self.seen_stance[:, swap]
                & (terrace == self.stance_terrace[:, swap])
                & (own_progress >= self.min_forward)
                & (terrace == self.highest_terrace[:, None])
                & (terrace < top_terrace[:, None])
            )
        self.last_events.copy_(event)
        self.counts["stable_landings"] += landing.sum(-1)
        self.counts["ordinary_stable_landings"] += (landing & ordinary[:, None]).sum(-1)
        self.counts["nonordinary_stable_landings"] += (
            landing & ~ordinary[:, None]
        ).sum(-1)
        self.counts["eligible_landings"] += eligible.sum(-1)
        self.counts["rewarded_landings"] += event.sum(-1)
        self.counts["nonordinary_rewarded_landings"] += (
            event & ~ordinary[:, None]
        ).sum(-1)
        self.counts["penalized_catchup_landings"] += catchup.sum(-1)
        self.counts["nonordinary_penalized_catchup_landings"] += (
            catchup & ~ordinary[:, None]
        ).sum(-1)
        valid_stance = stable & terrain_valid & torch.isfinite(ground_height)
        # An invalid new foothold must invalidate the preceding valid one.
        # Otherwise a later swing could reuse a stale terrace across a riser.
        self.seen_stance = torch.where(stable, valid_stance, self.seen_stance)
        self.stance_pos = torch.where(
            valid_stance[..., None], foot_pos, self.stance_pos
        )
        self.stance_height = torch.where(
            valid_stance, ground_height, self.stance_height
        )
        self.stance_terrace = torch.where(valid_stance, terrace, self.stance_terrace)
        current_high = torch.where(valid_stance, terrace, -1_000_000).amax(-1)
        self.highest_terrace = torch.maximum(self.highest_terrace, current_high)
        self.candidate &= ~new_stable
        self.air_time = torch.where(
            new_stable, torch.zeros_like(self.air_time), self.air_time
        )
        self.previous_airborne.copy_(airborne)
        return event.sum(-1).float() - self.catchup_cost * catchup.sum(-1)

    def diagnostics(self) -> dict[str, Any]:
        return {
            **{key: int(value.sum().item()) for key, value in self.counts.items()},
            "per_environment": {
                key: value.detach().cpu().tolist() for key, value in self.counts.items()
            },
        }


def _column_geometry(generator_cfg, device):
    """Match Isaac's deterministic curriculum column selection exactly."""
    if not generator_cfg.curriculum:
        raise ValueError(
            "Reciprocal stair labels require deterministic curriculum terrain columns"
        )
    items = list(generator_cfg.sub_terrains.items())
    total = sum(float(cfg.proportion) for _, cfg in items)
    if total <= 0:
        raise ValueError("Terrain proportions must sum to a positive value")
    cumulative = []
    running = 0.0
    for _, cfg in items:
        running += float(cfg.proportion) / total
        cumulative.append(running)
    result = {
        key: []
        for key in ("ordinary", "inverted", "first_boundary", "step_width", "num_steps")
    }
    for column in range(generator_cfg.num_cols):
        selected = next(
            i
            for i, limit in enumerate(cumulative)
            if column / generator_cfg.num_cols + 0.001 < limit
        )
        name, cfg = items[selected]
        ordinary = name in ("pyramid_stairs", "pyramid_stairs_inv")
        step_width = float(getattr(cfg, "step_width", 0.3))
        if ordinary and not math.isclose(step_width, 0.3, abs_tol=1e-6):
            raise ValueError(
                f"Ordinary stair terrain {name} has unsupported width {step_width}"
            )
        hscale = float(generator_cfg.horizontal_scale)
        size = generator_cfg.size
        if ordinary and not math.isclose(size[0], size[1], abs_tol=1e-6):
            raise ValueError(
                "Nominal reciprocal stair labels currently require square tiles"
            )
        width_pixels = int(size[0] / hscale) + 1
        border_pixels = int(float(getattr(cfg, "border_width", 0.0)) / hscale) + 1
        inner_pixels = width_pixels - 2 * border_pixels
        step_pixels = round(step_width / hscale)
        platform_pixels = round(float(getattr(cfg, "platform_width", 2.5)) / hscale)
        count = max(0, math.ceil((inner_pixels - platform_pixels) / (2 * step_pixels)))
        inverted = bool(getattr(cfg, "inverted", False))
        # At a vertical riser Isaac moves the low vertex onto the high vertex.
        # An inverted pyramid's high outer tread therefore extends one cell
        # farther inward than the raw height-field terrace.
        first = (
            size[0] / 2
            - (border_pixels + step_pixels) * hscale
            + (hscale if inverted else 0.0)
        )
        result["ordinary"].append(ordinary)
        result["inverted"].append(inverted)
        result["first_boundary"].append(first)
        result["step_width"].append(step_pixels * hscale)
        result["num_steps"].append(count)
    return {key: torch.tensor(value, device=device) for key, value in result.items()}


class ReciprocalStairLanding(ManagerTermBase):
    """Isaac adapter. Return events / control dt for a weight-sized event bonus."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = (
            "min_swing_time",
            "stable_support_time",
            "min_forward",
            "max_forward",
            "min_rise",
            "max_rise",
            "catchup_cost",
        )
        self.tracker = LandingTracker(
            env.num_envs,
            env.device,
            **{name: cfg.params[name] for name in names if name in cfg.params},
        )
        self.geometry = _column_geometry(
            env.scene["terrain"].cfg.terrain_generator, env.device
        )

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)

    def diagnostics(self):
        return self.tracker.diagnostics()

    def __call__(
        self,
        env,
        sensor_cfg,
        asset_cfg,
        left_height_scanner_cfg,
        right_height_scanner_cfg,
        command_name="base_velocity",
        min_swing_time=0.08,
        stable_support_time=0.04,
        min_forward=0.12,
        max_forward=0.5,
        min_rise=0.025,
        max_rise=0.26,
        min_force=20.0,
        horizontal_ratio=1.5,
        catchup_cost=0.0,
    ):
        robot = env.scene[asset_cfg.name]
        sensor = env.scene.sensors[sensor_cfg.name]
        foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]
        forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
        history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        if foot_pos.shape[1] != 2 or forces.shape[1] != 2:
            raise ValueError(
                "Reciprocal landing requires exactly left and right feet, in that order"
            )
        horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
        history_horizontal = torch.linalg.vector_norm(history[..., :2], dim=-1)
        side = (
            (history_horizontal > min_force)
            & (history_horizontal > horizontal_ratio * history[..., 2].clamp_min(0))
        ).any(1)
        support = (
            (forces[..., 2] > min_force)
            & (horizontal <= horizontal_ratio * forces[..., 2])
            & ~side
        )
        airborne = (torch.linalg.vector_norm(history, dim=-1) < min_force).all(1)
        hits = [
            env.scene[cfg.name].data.ray_hits_w[..., 2]
            for cfg in (left_height_scanner_cfg, right_height_scanner_cfg)
        ]
        finite = torch.stack([torch.isfinite(value).all(-1) for value in hits], dim=-1)
        ground_height = torch.stack([value.mean(-1) for value in hits], dim=-1)
        # A pair spanning a riser is not a verified foothold. On a noisy tread
        # this is deliberately conservative; nominal geometry is a second gate.
        terrain = env.scene["terrain"]
        columns = terrain.terrain_types
        geometry = {key: value[columns] for key, value in self.geometry.items()}
        scan_limit = torch.where(geometry["inverted"], 0.065, 0.025)[:, None]
        scanner_valid = finite & (
            torch.stack([value.amax(-1) - value.amin(-1) for value in hits], dim=-1)
            < scan_limit
        )
        terrace, terrace_valid = nominal_stair_terraces(
            foot_pos[..., :2],
            terrain.env_origins[..., :2],
            geometry["first_boundary"],
            geometry["step_width"],
            geometry["num_steps"],
            geometry["inverted"],
        )
        # Native targets lie outside the pyramid; ascent and descent both progress outward.
        progress_sign = torch.where(geometry["inverted"], 1, -1)[:, None]
        terrace = terrace * progress_sign
        ground_height = ground_height * progress_sign
        command = env.command_manager.get_term(command_name)
        toward_goal = command.pos_command_w[..., :2] - robot.data.root_pos_w[..., :2]
        distance = torch.linalg.vector_norm(toward_goal, dim=-1)
        direction = toward_goal / distance.clamp_min(1e-6)[:, None]
        moving = torch.linalg.vector_norm(command.command[..., :2], dim=-1) > 0.1
        active = (
            moving & (distance > 0.45) & (robot.data.projected_gravity_b[..., 2] < -0.7)
        )
        events = self.tracker.update(
            foot_pos=foot_pos,
            ground_height=ground_height,
            terrace=terrace,
            terrain_valid=terrace_valid & scanner_valid,
            support=support,
            airborne=airborne,
            ordinary=geometry["ordinary"],
            direction=direction,
            active=active,
            dt=env.step_dt,
            top_terrace=torch.zeros_like(geometry["num_steps"]),
        )
        return events / env.step_dt
