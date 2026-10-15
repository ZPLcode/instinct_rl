"""Behavioral CPU checks for reciprocal (not catch-up) stair landing rewards."""

import unittest
from types import SimpleNamespace

import torch

from experiments.isaac_stair_contact.stair_landing import (
    LandingTracker,
    ReciprocalStairLanding,
    _column_geometry,
    nominal_stair_terraces,
)


class Trace:
    def __init__(self, count=1, catchup_cost=0.0, top_terrace=6):
        self.tracker = LandingTracker(count, "cpu", catchup_cost=catchup_cost)
        self.top_terrace = torch.full((count,), top_terrace, dtype=torch.long)
        self.pos = torch.tensor([[[0.0, -0.1, 0.0], [0.3, 0.1, 0.1]]]).repeat(
            count, 1, 1
        )
        self.height = self.pos[..., 2].clone()
        self.terrace = torch.tensor([[0, 1]]).repeat(count, 1)
        self.support = torch.ones(count, 2, dtype=torch.bool)
        self.airborne = torch.zeros_like(self.support)
        self.valid = torch.ones_like(self.support)
        self.ordinary = torch.ones(count, dtype=torch.bool)
        self.active = torch.ones_like(self.ordinary)
        self.direction = torch.tensor([[1.0, 0.0]]).repeat(count, 1)

    def step(self, times=1):
        total = torch.zeros(len(self.pos))
        for _ in range(times):
            total += self.tracker.update(
                foot_pos=self.pos,
                ground_height=self.height,
                terrace=self.terrace,
                terrain_valid=self.valid,
                support=self.support,
                airborne=self.airborne,
                ordinary=self.ordinary,
                direction=self.direction,
                active=self.active,
                dt=0.02,
                top_terrace=self.top_terrace,
            )
        return total

    def swing(self, foot=0, times=4, env_ids=slice(None)):
        self.support[env_ids, foot] = False
        self.airborne[env_ids, foot] = True
        return self.step(times)

    def land(self, foot=0, x=0.6, z=0.2, terrace=2, times=2, env_ids=slice(None)):
        self.pos[env_ids, foot, 0] = x
        self.pos[env_ids, foot, 2] = z
        self.height[env_ids, foot] = z
        self.terrace[env_ids, foot] = terrace
        self.support[env_ids, foot] = True
        self.airborne[env_ids, foot] = False
        return self.step(times)


class LandingBehaviorTests(unittest.TestCase):
    def test_rear_foot_passes_stance_then_other_foot_does_same(self):
        trace = Trace()
        self.assertEqual(trace.step(5).item(), 0.0)
        self.assertEqual(trace.swing().item(), 0.0)
        self.assertEqual(
            trace.land(times=1).item(), 0.0, "One support frame is not stable"
        )
        self.assertEqual(trace.step().item(), 1.0)
        self.assertEqual(
            trace.step(20).item(),
            0.0,
            "Holding a rewarded foothold is not another event",
        )
        trace.swing(foot=1)
        self.assertEqual(trace.land(foot=1, x=0.9, z=0.3, terrace=3).item(), 1.0)
        self.assertEqual(trace.tracker.diagnostics()["rewarded_landings"], 2)

    def test_catch_up_step_and_equal_height_first_step_do_not_earn_bonus(self):
        trace = Trace()
        trace.step(2)
        trace.swing()
        self.assertEqual(trace.land(x=0.32, z=0.1, terrace=1).item(), 0.0)
        trace.swing(foot=1)
        self.assertEqual(trace.land(foot=1, x=0.65, z=0.2, terrace=2).item(), 0.0)

    def test_optional_interior_catchup_cost_is_one_event_and_keeps_positive_bonus(self):
        trace = Trace(catchup_cost=0.1)
        self.assertEqual(trace.step(2).item(), 0.0)
        trace.swing()
        self.assertEqual(trace.land(x=0.32, z=0.1, terrace=1, times=1).item(), 0.0)
        self.assertAlmostEqual(trace.step().item(), -0.1)
        self.assertEqual(trace.step(20).item(), 0.0)
        self.assertEqual(trace.tracker.diagnostics()["penalized_catchup_landings"], 1)
        self.assertEqual(trace.tracker.diagnostics()["rewarded_landings"], 0)
        trace = Trace(catchup_cost=0.1)
        trace.step(2)
        trace.swing()
        self.assertEqual(trace.land().item(), 1.0)
        self.assertEqual(trace.tracker.diagnostics()["penalized_catchup_landings"], 0)

    def test_catchup_default_platform_nonordinary_reset_and_recovery_exclusions(self):
        for case in (
            "default",
            "top",
            "inverted_top",
            "nonordinary",
            "reset",
            "recovery",
            "no_progress",
            "inactive",
            "descending",
            "opening",
        ):
            with self.subTest(case=case):
                trace = Trace(catchup_cost=0.0 if case == "default" else 0.1)
                if case == "top":
                    trace.top_terrace[:] = 1
                if case == "inverted_top":
                    trace.terrace -= 1
                    trace.top_terrace[:] = 0
                if case == "nonordinary":
                    trace.ordinary[:] = False
                if case == "descending":
                    trace.terrace[:, 0] = 2
                    trace.pos[:, 0, 2] = trace.height[:, 0] = 0.2
                if case == "opening":
                    trace.terrace[:] = 0
                    trace.pos[:, 1, 2] = trace.height[:, 1] = 0.0
                trace.step(2)
                trace.swing()
                if case == "inactive":
                    trace.active[:] = False
                if case == "reset":
                    trace.tracker.reset()
                if case == "recovery":
                    trace.tracker.highest_terrace[:] = 3
                landing_terrace = 0 if case == "inverted_top" else 1
                x = (
                    0.05
                    if case == "no_progress"
                    else 0.6
                    if case == "opening"
                    else 0.32
                )
                self.assertEqual(
                    trace.land(x=x, z=0.1, terrace=landing_terrace).item(), 0.0
                )
                self.assertEqual(
                    trace.tracker.diagnostics()["penalized_catchup_landings"], 0
                )
                self.assertEqual(
                    trace.tracker.diagnostics()[
                        "nonordinary_penalized_catchup_landings"
                    ],
                    0,
                )

    def test_enabled_catchup_cost_partial_reset_preserves_other_environment(self):
        trace = Trace(count=2, catchup_cost=0.1)
        trace.step(2)
        trace.swing()
        trace.tracker.reset(torch.tensor([0]))
        torch.testing.assert_close(
            trace.land(x=0.32, z=0.1, terrace=1), torch.tensor([0.0, -0.1])
        )
        self.assertEqual(
            trace.tracker.diagnostics()["per_environment"][
                "penalized_catchup_landings"
            ],
            [0, 1],
        )
        self.assertEqual(trace.step(10).sum().item(), 0.0)
        trace.tracker.reset()
        self.assertEqual(trace.step(10).sum().item(), 0.0)
        self.assertEqual(trace.tracker.diagnostics()["penalized_catchup_landings"], 1)

    def test_catchup_uses_later_verified_foothold_even_after_other_foot_lifts(self):
        for case in ("early_takeoff", "two_levels", "other_airborne"):
            with self.subTest(case=case):
                trace = Trace(catchup_cost=0.1)
                if case == "early_takeoff":
                    trace.terrace[:] = 0
                    trace.pos[..., 2] = trace.height[:] = 0.0
                if case == "two_levels":
                    trace.terrace[:, 1] = 2
                    trace.pos[:, 1, 2] = trace.height[:, 1] = 0.2
                trace.step(2)
                if case == "early_takeoff":
                    trace.swing(foot=1)
                    trace.land(foot=1, x=0.3, z=0.1, terrace=1, times=1)
                trace.swing()
                if case == "other_airborne":
                    trace.support[:, 1] = False
                    trace.airborne[:, 1] = True
                level = 2 if case == "two_levels" else 1
                self.assertAlmostEqual(
                    trace.land(x=0.32, z=0.1 * level, terrace=level).item(), -0.1
                )
                self.assertEqual(trace.step(10).item(), 0.0)
                self.assertEqual(trace.tracker.diagnostics()["rewarded_landings"], 0)
                self.assertEqual(
                    trace.tracker.diagnostics()["penalized_catchup_landings"], 1
                )

    def test_catchup_does_not_penalize_opening_or_leading_from_same_tread(self):
        for level in (0, 1):
            with self.subTest(initial_tread=level):
                trace = Trace(catchup_cost=0.1)
                trace.terrace[:] = level
                trace.pos[..., 2] = trace.height[:] = 0.1 * level
                trace.step(2)
                trace.swing(foot=1)
                self.assertEqual(
                    trace.land(
                        foot=1, x=0.65, z=0.1 * (level + 1), terrace=level + 1
                    ).item(),
                    0.0,
                )
                self.assertEqual(
                    trace.tracker.diagnostics()["penalized_catchup_landings"], 0
                )

    def test_catchup_penalizes_staggered_but_not_simultaneous_hop_landings(self):
        # A 20-ms lag makes the second foot an upward catch-up; this is intentional.
        for delay_steps in (0, 1):
            with self.subTest(delay_steps=delay_steps):
                trace = Trace(catchup_cost=0.1)
                trace.terrace[:] = 0
                trace.pos[..., 2] = trace.height[:] = 0.0
                trace.step(2)
                trace.support[:] = False
                trace.airborne[:] = True
                trace.step(4)
                trace.pos[..., 0] = torch.tensor([0.6, 0.65])
                trace.pos[..., 2] = trace.height[:] = 0.1
                trace.terrace[:] = 1
                trace.support[:, 1] = True
                trace.airborne[:, 1] = False
                if delay_steps:
                    self.assertEqual(trace.step(delay_steps).item(), 0.0)
                trace.support[:, 0] = True
                trace.airborne[:, 0] = False
                self.assertAlmostEqual(
                    trace.step(2).item(), -0.1 if delay_steps else 0.0
                )
                self.assertEqual(trace.tracker.diagnostics()["rewarded_landings"], 0)
                self.assertEqual(
                    trace.tracker.diagnostics()["penalized_catchup_landings"],
                    delay_steps,
                )

    def test_noisy_same_terrace_cannot_invent_a_step(self):
        trace = Trace()
        trace.step(2)
        trace.swing()
        # Positive scanner height and forward distance, but still the same rung.
        self.assertEqual(trace.land(x=0.6, z=0.145, terrace=1).item(), 0.0)

    def test_short_swing_side_contact_and_ambiguous_riser_do_not_pay(self):
        for rejection in ("short", "side", "edge"):
            with self.subTest(rejection=rejection):
                trace = Trace()
                trace.step(2)
                trace.swing(times=2 if rejection == "short" else 4)
                if rejection == "edge":
                    trace.valid[:, 0] = False
                if rejection == "side":
                    trace.pos[:, 0] = torch.tensor([0.6, -0.1, 0.2])
                    trace.height[:, 0] = 0.2
                    trace.terrace[:, 0] = 2
                    trace.airborne[:, 0] = False
                    self.assertEqual(trace.step(6).item(), 0.0)
                else:
                    self.assertEqual(trace.land().item(), 0.0)

    def test_nonstairs_disabled_command_excessive_rise_and_wrong_direction(self):
        for rejection in ("terrain", "command", "rise", "direction"):
            with self.subTest(rejection=rejection):
                trace = Trace()
                if rejection == "terrain":
                    trace.ordinary[:] = False
                if rejection == "command":
                    trace.active[:] = False
                if rejection == "direction":
                    trace.direction *= -1
                trace.step(2)
                trace.swing()
                self.assertEqual(
                    trace.land(z=0.5 if rejection == "rise" else 0.2).item(), 0.0
                )
                self.assertEqual(
                    trace.tracker.diagnostics()["nonordinary_rewarded_landings"], 0
                )

    def test_initial_support_and_partial_reset_never_produce_an_event(self):
        trace = Trace(count=2)
        trace.step(2)
        trace.swing()
        trace.tracker.reset(torch.tensor([0]))
        torch.testing.assert_close(trace.land(), torch.tensor([0.0, 1.0]))
        counts = trace.tracker.diagnostics()["rewarded_landings"]
        trace.tracker.reset()
        self.assertEqual(trace.step(20).sum().item(), 0.0)
        self.assertEqual(trace.tracker.diagnostics()["rewarded_landings"], counts)

    def test_same_frontier_cannot_be_farmed_by_descent_then_reascent(self):
        trace = Trace()
        trace.step(2)
        trace.swing()
        self.assertEqual(trace.land().item(), 1.0)
        # Step back to the original lower tread, then repeat the identical climb.
        trace.swing()
        self.assertEqual(trace.land(x=0.0, z=0.0, terrace=0).item(), 0.0)
        trace.swing()
        self.assertEqual(trace.land().item(), 0.0)

    def test_simultaneous_hop_is_not_a_reciprocal_step(self):
        trace = Trace()
        trace.step(2)
        trace.support[:] = False
        trace.airborne[:] = True
        trace.step(4)
        trace.land(times=1)
        trace.support[:] = True
        trace.airborne[:] = False
        self.assertEqual(trace.step(2).item(), 0.0)

    def test_invalid_new_foothold_cannot_reuse_an_old_valid_stance(self):
        trace = Trace()
        trace.step(2)
        trace.valid[:, 1] = False
        trace.step(2)
        self.assertFalse(trace.tracker.seen_stance[0, 1])
        trace.swing()
        self.assertEqual(trace.land().item(), 0.0)


class TerrainGeometryTests(unittest.TestCase):
    def config(self):
        def stairs(width=0.3, inverted=False, proportion=0.25):
            return SimpleNamespace(
                step_width=width,
                inverted=inverted,
                proportion=proportion,
                border_width=1.0,
                platform_width=2.5,
            )

        return SimpleNamespace(
            curriculum=True,
            num_cols=4,
            size=(8.0, 8.0),
            horizontal_scale=0.05,
            sub_terrains={
                "pyramid_stairs": stairs(),
                "pyramid_stairs_high": stairs(1.5),
                "pyramid_stairs_inv": stairs(inverted=True),
                "boxes": stairs(),
            },
        )

    def test_columns_and_actual_heightfield_border_discretization(self):
        geometry = _column_geometry(self.config(), "cpu")
        torch.testing.assert_close(
            geometry["ordinary"], torch.tensor([True, False, True, False])
        )
        self.assertAlmostEqual(geometry["first_boundary"][0].item(), 2.65, places=5)
        self.assertAlmostEqual(geometry["first_boundary"][2].item(), 2.70, places=5)
        self.assertEqual(geometry["num_steps"][0].item(), 6)
        self.assertEqual(geometry["num_steps"][2].item(), 6)

    def test_inverted_ascending_levels_and_edge_ambiguity(self):
        xy = torch.tensor([[[1.35, 0.0], [1.65, 0.0]], [[1.20, 0.0], [0.0, 0.0]]])
        level, valid = nominal_stair_terraces(
            xy,
            torch.zeros(2, 2),
            torch.full((2,), 2.7),
            torch.full((2,), 0.3),
            torch.full((2,), 6),
            torch.ones(2, dtype=torch.bool),
        )
        torch.testing.assert_close(level[0], torch.tensor([-5, -4]))
        self.assertTrue(valid[0].all())
        self.assertFalse(valid[1, 0])
        self.assertTrue(valid[1, 1])
        self.assertEqual(level[1, 1].item(), -6)

    def test_random_terrain_assignment_and_misnamed_stair_width_fail_loudly(self):
        cfg = self.config()
        cfg.curriculum = False
        with self.assertRaises(ValueError):
            _column_geometry(cfg, "cpu")
        cfg.curriculum = True
        cfg.sub_terrains["pyramid_stairs"].step_width = 1.5
        with self.assertRaises(ValueError):
            _column_geometry(cfg, "cpu")


class AdapterTests(unittest.TestCase):
    def test_sensor_asset_order_and_per_event_time_normalization(self):
        class Scene(dict):
            @property
            def sensors(self):
                return self

        gen = TerrainGeometryTests().config()
        positions = torch.zeros(1, 4, 3)
        positions[0, 1] = torch.tensor([1.35, -0.1, 0.135])
        positions[0, 3] = torch.tensor([1.65, 0.1, 0.235])
        force = torch.zeros(1, 3, 3)
        force[:, [2, 0], 2] = 100.0
        force[:, 1, 0] = 500.0  # unrelated sensor body must not veto support
        history = force[:, None].repeat(1, 4, 1, 1)
        left_hits, right_hits = torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)
        left_hits[..., 2], right_hits[..., 2] = 0.1, 0.2
        scene = Scene(
            robot=SimpleNamespace(
                data=SimpleNamespace(
                    body_pos_w=positions,
                    root_pos_w=torch.tensor([[1.5, 0.0, 1.0]]),
                    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
                )
            ),
            sensor=SimpleNamespace(
                data=SimpleNamespace(net_forces_w=force, net_forces_w_history=history)
            ),
            left=SimpleNamespace(data=SimpleNamespace(ray_hits_w=left_hits)),
            right=SimpleNamespace(data=SimpleNamespace(ray_hits_w=right_hits)),
            terrain=SimpleNamespace(
                cfg=SimpleNamespace(terrain_generator=gen),
                terrain_types=torch.tensor([2]),
                env_origins=torch.zeros(1, 3),
            ),
        )
        command = SimpleNamespace(
            pos_command_w=torch.tensor([[3.7, 0.0, 0.5]]),
            command=torch.tensor([[0.6, 0.0, 0.0]]),
        )
        env = SimpleNamespace(
            scene=scene,
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            command_manager=SimpleNamespace(get_term=lambda name: command),
        )
        term = ReciprocalStairLanding(SimpleNamespace(params={}), env)
        params = dict(
            sensor_cfg=SimpleNamespace(name="sensor", body_ids=[2, 0]),
            asset_cfg=SimpleNamespace(name="robot", body_ids=[1, 3]),
            left_height_scanner_cfg=SimpleNamespace(name="left"),
            right_height_scanner_cfg=SimpleNamespace(name="right"),
        )
        for _ in range(3):
            self.assertEqual(term(env, **params).item(), 0.0)
        force[:, 2] = 0
        history[:, :, 2] = 0
        for _ in range(4):
            self.assertEqual(term(env, **params).item(), 0.0)
        positions[0, 1] = torch.tensor([1.95, -0.1, 0.335])
        left_hits[..., 2] = 0.3
        force[:, 2, 2] = 100.0
        history[:, :, 2, 2] = 100.0
        self.assertEqual(term(env, **params).item(), 0.0)
        raw_value = term(env, **params).item()
        self.assertEqual(raw_value, 50.0)
        self.assertAlmostEqual(raw_value * 0.1 * env.step_dt, 0.1)
        self.assertEqual(term.diagnostics()["rewarded_landings"], 1)
        # Config normalizes the actual 0.1 cost by the manager term weight.
        for weight in (0.1, 0.5, 1.0, 5.0):
            term = ReciprocalStairLanding(
                SimpleNamespace(params={"catchup_cost": 0.1 / weight}), env
            )
            positions[0, 1] = torch.tensor([1.35, -0.1, 0.135])
            left_hits[..., 2] = 0.1
            for _ in range(3):
                term(env, **params)
            force[:, 2] = 0
            history[:, :, 2] = 0
            for _ in range(4):
                term(env, **params)
            positions[0, 1] = torch.tensor([1.68, -0.1, 0.235])
            left_hits[..., 2] = 0.2
            force[:, 2, 2] = 100.0
            history[:, :, 2, 2] = 100.0
            self.assertEqual(term(env, **params).item(), 0.0)
            self.assertAlmostEqual(
                term(env, **params).item() * weight * env.step_dt, -0.1
            )
            self.assertEqual(term.diagnostics()["penalized_catchup_landings"], 1)


if __name__ == "__main__":
    unittest.main()
