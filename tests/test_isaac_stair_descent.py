"""Physical-height descent tests through the real reward adapter."""

import unittest
from types import SimpleNamespace

import torch

from experiments.isaac_stair_contact.stair_landing import ReciprocalStairLanding


class Scene(dict):
    @property
    def sensors(self):
        return self


class AdapterTrace:
    def __init__(self, descending=True, catchup_cost=0.1):
        self.descending = descending
        self.positions = torch.tensor([[[1.35, -0.1, 0.535], [1.65, 0.1, 0.435]]])
        self.force = torch.zeros(1, 2, 3)
        self.force[..., 2] = 100
        self.history = self.force[:, None].repeat(1, 4, 1, 1)
        self.hits = [torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)]
        heights = [0.5, 0.4] if descending else [0.1, 0.2]
        for foot, height in enumerate(heights):
            self.hits[foot][..., 2] = height
            self.positions[:, foot, 2] = height + 0.035
        terrain_cfg = SimpleNamespace(
            step_width=0.3,
            inverted=not descending,
            proportion=1.0,
            border_width=1.0,
            platform_width=2.5,
        )
        generator = SimpleNamespace(
            curriculum=True,
            num_cols=1,
            size=(8.0, 8.0),
            horizontal_scale=0.05,
            sub_terrains={
                ("pyramid_stairs" if descending else "pyramid_stairs_inv"): terrain_cfg
            },
        )
        scene = Scene(
            robot=SimpleNamespace(
                data=SimpleNamespace(
                    body_pos_w=self.positions,
                    root_pos_w=torch.tensor([[1.5, 0.0, 1.0]]),
                    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
                )
            ),
            sensor=SimpleNamespace(
                data=SimpleNamespace(
                    net_forces_w=self.force, net_forces_w_history=self.history
                )
            ),
            left=SimpleNamespace(data=SimpleNamespace(ray_hits_w=self.hits[0])),
            right=SimpleNamespace(data=SimpleNamespace(ray_hits_w=self.hits[1])),
            terrain=SimpleNamespace(
                cfg=SimpleNamespace(terrain_generator=generator),
                terrain_types=torch.tensor([0]),
                env_origins=torch.zeros(1, 3),
            ),
        )
        command = SimpleNamespace(
            pos_command_w=torch.tensor([[3.7, 0.0, 0.0]]),
            command=torch.tensor([[0.6, 0.0, 0.0]]),
        )
        self.env = SimpleNamespace(
            scene=scene,
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        self.cfg = SimpleNamespace(params={"catchup_cost": catchup_cost})
        self.params = dict(
            sensor_cfg=SimpleNamespace(name="sensor", body_ids=[0, 1]),
            asset_cfg=SimpleNamespace(name="robot", body_ids=[0, 1]),
            left_height_scanner_cfg=SimpleNamespace(name="left"),
            right_height_scanner_cfg=SimpleNamespace(name="right"),
        )
        self.term = ReciprocalStairLanding(self.cfg, self.env)

    def step(self, count=1):
        return sum(
            self.term(self.env, **self.params).item() * 0.02 for _ in range(count)
        )

    def swing(self, foot=0, count=4):
        self.force[:, foot] = 0
        self.history[:, :, foot] = 0
        return self.step(count)

    def land(self, foot=0, x=1.95, height=0.3, spread=0.0, count=2):
        self.positions[:, foot, 0] = x
        self.positions[:, foot, 2] = height + 0.035
        self.hits[foot][0, :, 2] = torch.tensor(
            [height - spread / 2, height + spread / 2]
        )
        self.force[:, foot, 2] = 100
        self.history[:, :, foot, 2] = 100
        return self.step(count)


class DescentAdapterTests(unittest.TestCase):
    def test_two_alternating_lower_landings_pay_once_each(self):
        t = AdapterTrace()
        self.assertEqual(t.step(3), 0)
        self.assertEqual(t.swing(), 0)
        self.assertEqual(t.land(count=1), 0)
        self.assertEqual(t.step(), 1)
        self.assertEqual(t.step(20), 0)
        t.swing(foot=1)
        self.assertEqual(t.land(foot=1, x=2.25, height=0.2), 1)

    def test_small_and_large_cross_riser_scans_reject_initial_landing(self):
        for spread in [0.05, 0.10]:
            with self.subTest(spread=spread):
                t = AdapterTrace()
                t.step(3)
                t.swing()
                self.assertEqual(t.land(spread=spread), 0)
                self.assertEqual(t.term.diagnostics()["rewarded_landings"], 0)
                self.assertFalse(t.term.tracker.seen_stance[0, 0])

    def test_small_same_tread_roughness_is_tolerated(self):
        t = AdapterTrace()
        t.step(3)
        t.swing()
        self.assertEqual(t.land(spread=0.02), 1)

    def test_nominal_edge_does_not_pay_even_with_flat_scans(self):
        t = AdapterTrace()
        t.step(3)
        t.swing()
        self.assertEqual(t.land(x=1.75), 0)

    def test_descending_catchup_penalty_and_terminal_floor_exemption(self):
        t = AdapterTrace()
        t.step(3)
        t.swing()
        self.assertAlmostEqual(t.land(x=1.68, height=0.4), -0.1)
        self.assertEqual(t.step(10), 0)
        t = AdapterTrace()
        for foot, (x, height) in enumerate([(2.25, 0.2), (2.55, 0.1)]):
            t.positions[:, foot] = torch.tensor(
                [x, -0.1 if foot == 0 else 0.1, height + 0.035]
            )
            t.hits[foot][..., 2] = height
        t.step(3)
        t.swing()
        self.assertEqual(t.land(x=2.85, height=0), 1)
        t.swing(foot=1)
        self.assertEqual(t.land(foot=1, x=2.9, height=0), 0)

    def test_wrong_vertical_change_skipped_step_and_short_swing_do_not_pay(self):
        for case in ["reverse_height", "excessive_drop", "skip", "short_swing"]:
            with self.subTest(case=case):
                t = AdapterTrace()
                t.step(3)
                t.swing(count=2 if case == "short_swing" else 4)
                height = (
                    0.5
                    if case == "reverse_height"
                    else 0
                    if case == "excessive_drop"
                    else 0.3
                )
                self.assertEqual(
                    t.land(x=2.25 if case == "skip" else 1.95, height=height), 0
                )

    def test_descent_reset_does_not_invent_a_landing(self):
        t = AdapterTrace()
        t.step(3)
        t.swing()
        t.term.reset()
        self.assertEqual(t.land(), 0)

    def test_ascent_preserves_legacy_scan_limit_and_rewards(self):
        t = AdapterTrace(descending=False)
        t.step(3)
        t.swing()
        self.assertEqual(t.land(height=0.3, spread=0.05), 1)

    def test_mixed_ascent_descent_batch_normalizes_each_environment_independently(self):
        ascent, descent = AdapterTrace(descending=False), AdapterTrace()
        down_cfg = descent.env.scene["terrain"].cfg.terrain_generator.sub_terrains[
            "pyramid_stairs"
        ]
        up_cfg = ascent.env.scene["terrain"].cfg.terrain_generator.sub_terrains[
            "pyramid_stairs_inv"
        ]
        down_cfg.proportion = up_cfg.proportion = 0.5
        generator = SimpleNamespace(
            curriculum=True,
            num_cols=2,
            size=(8.0, 8.0),
            horizontal_scale=0.05,
            sub_terrains={"pyramid_stairs": down_cfg, "pyramid_stairs_inv": up_cfg},
        )
        scene = Scene()
        for name in ["robot", "sensor", "left", "right"]:
            data = {}
            for key in vars(ascent.env.scene[name].data):
                data[key] = torch.cat(
                    [
                        getattr(ascent.env.scene[name].data, key),
                        getattr(descent.env.scene[name].data, key),
                    ]
                )
            scene[name] = SimpleNamespace(data=SimpleNamespace(**data))
        scene["terrain"] = SimpleNamespace(
            cfg=SimpleNamespace(terrain_generator=generator),
            terrain_types=torch.tensor([1, 0]),
            env_origins=torch.zeros(2, 3),
        )
        command = SimpleNamespace(
            pos_command_w=torch.tensor([[3.7, 0.0, 0.0]]).repeat(2, 1),
            command=torch.tensor([[0.6, 0.0, 0.0]]).repeat(2, 1),
        )
        env = SimpleNamespace(
            scene=scene,
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        term = ReciprocalStairLanding(ascent.cfg, env)
        for _ in range(3):
            torch.testing.assert_close(term(env, **ascent.params), torch.zeros(2))
        sensor = scene["sensor"].data
        sensor.net_forces_w[:, 0] = 0
        sensor.net_forces_w_history[:, :, 0] = 0
        for _ in range(4):
            torch.testing.assert_close(term(env, **ascent.params), torch.zeros(2))
        scene["robot"].data.body_pos_w[:, 0] = torch.tensor([1.95, -0.1, 0.335])
        scene["left"].data.ray_hits_w[..., 2] = 0.3
        sensor.net_forces_w[:, 0, 2] = 100
        sensor.net_forces_w_history[:, :, 0, 2] = 100
        torch.testing.assert_close(term(env, **ascent.params), torch.zeros(2))
        torch.testing.assert_close(term(env, **ascent.params) * 0.02, torch.ones(2))


if __name__ == "__main__":
    unittest.main()
