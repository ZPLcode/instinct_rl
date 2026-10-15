"""CPU regressions for the native Isaac stair-contact reward experiment."""

import math
import unittest
from types import SimpleNamespace

import torch

from experiments.isaac_stair_contact.rewards import (
    feet_at_plane,
    feet_orientation_contact,
    feet_side_collision,
    foot_contact_masks,
)


class ContactClassificationTests(unittest.TestCase):
    def test_air_support_slope_side_negative_vertical_and_threshold_boundaries(self):
        forces = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # air
                [20.0, 0.0, 300.0],  # normal support
                [100.0, 0.0, 100.0],  # 45 degree support
                [50.0, 0.0, 0.0],  # pure riser impact
                [50.0, 0.0, -100.0],  # downward force must not hide a side impact
                [
                    0.0,
                    0.0,
                    -100.0,
                ],  # downward force alone is neither support nor side impact
                [20.0, 0.0, 0.0],  # horizontal threshold is strict
                [0.0, 0.0, 20.0],  # vertical support threshold is strict
                [30.0, 0.0, 20.0],  # exact ratio boundary and too little support
                [45.0, 0.0, 30.0],  # ratio boundary belongs to support
                [45.01, 0.0, 30.0],  # just beyond ratio is an impact
                [15.0, 15.0, 0.0],  # XY vector magnitude, not per-axis maximum
            ]
        )[:, None, None, :]
        side, support = foot_contact_masks(forces)
        torch.testing.assert_close(
            side[:, 0],
            torch.tensor(
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ]
            ),
        )
        torch.testing.assert_close(
            support[:, 0],
            torch.tensor(
                [
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ]
            ),
        )

    def test_each_frame_is_classified_before_history_reduction_and_impact_vetoes_stale_support(
        self,
    ):
        forces = torch.zeros(1, 4, 2, 3)
        forces[0, 0, 0] = torch.tensor([80.0, 0.0, 5.0])  # left hits wall now
        forces[0, 3, 0] = torch.tensor([0.0, 0.0, 400.0])  # left used to support
        forces[0, 2, 1] = torch.tensor([100.0, 0.0, 150.0])  # right still supports
        side, support = foot_contact_masks(forces)
        # Taking separate maxima H=80 and Fz=400 would incorrectly miss the impact.
        torch.testing.assert_close(side, torch.tensor([[True, False]]))
        torch.testing.assert_close(support, torch.tensor([[False, True]]))
        # After the impact leaves history, valid support is enabled again.
        forces[0, 0, 0] = 0
        _, support = foot_contact_masks(forces)
        torch.testing.assert_close(support, torch.tensor([[True, True]]))

    def test_opposite_directions_do_not_cancel_and_each_foot_counts_once(self):
        forces = torch.zeros(1, 4, 2, 3)
        forces[0, 0, 0] = torch.tensor([40.0, 0.0, 0.0])
        forces[0, 1, 0] = torch.tensor([-40.0, 0.0, 0.0])
        forces[0, 3, 1] = torch.tensor([0.0, -50.0, 0.0])
        side, support = foot_contact_masks(forces)
        torch.testing.assert_close(side.sum(dim=-1), torch.tensor([2]))
        self.assertFalse(support.any())


class Scene(dict):
    @property
    def sensors(self):
        return self


def make_env():
    # Sensor body order differs from the asset's body order; wrappers must
    # respect the resolved SceneEntityCfg ids instead of slicing first feet.
    forces = torch.zeros(3, 4, 3, 3)
    forces[:, 0, [2, 0], 2] = 100.0
    forces[:, :, 1, :] = torch.tensor([500.0, 0.0, 0.0])  # unselected body
    forces[1, 0, 2] = torch.tensor([80.0, 0.0, 0.0])  # env1 left impact
    forces[2, 0, [2, 0]] = torch.tensor([80.0, 0.0, -10.0])  # env2 both impacts
    quat = torch.zeros(3, 4, 4)
    quat[..., 0] = 1
    quat[:, 1] = torch.tensor(
        [math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0, 0.0]
    )  # left roll30
    quat[:, 3] = torch.tensor(
        [math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0]
    )  # right pitch45
    positions = torch.zeros(3, 4, 3)
    positions[:, 1, 2] = 0.158  # left has .1 height excess over flat ground
    positions[:, 3, 2] = 0.558
    left_hits = torch.zeros(3, 3, 3)
    left_hits[:, :, 2] = torch.tensor([0.0, 0.15, torch.inf])
    right_hits = torch.zeros(3, 3, 3)
    right_hits[:, :, 2] = torch.tensor([0.0, 0.2, 0.6])
    scene = Scene(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=positions,
                body_quat_w=quat,
                GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]]).expand(3, -1),
            )
        ),
        contacts=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces)),
        left_scan=SimpleNamespace(data=SimpleNamespace(ray_hits_w=left_hits)),
        right_scan=SimpleNamespace(data=SimpleNamespace(ray_hits_w=right_hits)),
    )
    return (
        SimpleNamespace(scene=scene),
        SimpleNamespace(name="contacts", body_ids=[2, 0]),
        SimpleNamespace(name="robot", body_ids=[1, 3]),
    )


class IsaacRewardWrapperTests(unittest.TestCase):
    def test_side_cost_integrates_selected_feet_without_modifying_sensor_history(self):
        env, sensor_cfg, _ = make_env()
        before = env.scene["contacts"].data.net_forces_w_history.clone()
        torch.testing.assert_close(
            feet_side_collision(env, sensor_cfg), torch.tensor([0.0, 0.2, 0.4])
        )
        torch.testing.assert_close(
            env.scene["contacts"].data.net_forces_w_history, before
        )

    def test_orientation_retains_sine_of_tilt_cost_only_for_supported_feet(self):
        env, sensor_cfg, asset_cfg = make_env()
        cost = feet_orientation_contact(env, sensor_cfg, asset_cfg)
        # Independent analytic rotation result: horizontal gravity magnitude
        # equals sin(tilt), summed for the two supported feet.
        torch.testing.assert_close(
            cost, torch.tensor([0.5 + math.sqrt(0.5), math.sqrt(0.5), 0.0])
        )

    def test_height_retains_offsets_miss_handling_clamps_and_ray_sum(self):
        env, sensor_cfg, asset_cfg = make_env()
        cost = feet_at_plane(
            env,
            sensor_cfg,
            SimpleNamespace(name="left_scan"),
            SimpleNamespace(name="right_scan"),
            asset_cfg,
            height_offset=0.058,
        )
        # Left rays cost [.1, 0, .1] (inf falls back to ground z=0).
        # Right rays cost [.3, .3, 0], including both clamp endpoints.
        torch.testing.assert_close(cost, torch.tensor([0.8, 0.6, 0.0]))

    def test_stale_stance_does_not_reactivate_height_or_orientation_during_impact(self):
        env, sensor_cfg, asset_cfg = make_env()
        env.scene["contacts"].data.net_forces_w_history[1:, 3, [2, 0], 2] = 500.0
        torch.testing.assert_close(
            feet_orientation_contact(env, sensor_cfg, asset_cfg),
            torch.tensor([0.5 + math.sqrt(0.5), math.sqrt(0.5), 0.0]),
        )
        torch.testing.assert_close(
            feet_at_plane(
                env,
                sensor_cfg,
                SimpleNamespace(name="left_scan"),
                SimpleNamespace(name="right_scan"),
                asset_cfg,
                height_offset=0.058,
            ),
            torch.tensor([0.8, 0.6, 0.0]),
        )
        torch.testing.assert_close(
            feet_side_collision(env, sensor_cfg), torch.tensor([0.0, 0.2, 0.4])
        )


class SideImpulseTests(unittest.TestCase):
    @staticmethod
    def reward(forces, **kwargs):
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
        env = SimpleNamespace(scene=Scene(contacts=sensor))
        return feet_side_collision(
            env, SimpleNamespace(name="contacts", body_ids=slice(None)), **kwargs
        )

    def test_side_frame_mask_excludes_large_legal_horizontal_and_vertical_support(self):
        forces = torch.zeros(1, 4, 2, 3)
        forces[0, 0, 0] = torch.tensor([50.0, 0.0, 0.0])
        forces[0, 1, 0] = torch.tensor([900.0, 0.0, 1000.0])
        forces[0, :, 1, 2] = 1000.0
        # Only 50 N * .005 s contributes: cost = -5 * .25 N s / 100 N.
        torch.testing.assert_close(
            self.reward(forces) * -5 * 0.02, torch.tensor([-0.0125])
        )

    def test_short_heavy_collision_outweighs_long_light_scrape(self):
        short = torch.zeros(1, 4, 1, 3)
        short[0, 0, 0, 0] = 900.0
        long = torch.zeros(1, 16, 1, 3)
        long[..., 0] = 25.0
        torch.testing.assert_close(
            self.reward(short) * -5 * 0.02, torch.tensor([-0.225])
        )
        torch.testing.assert_close(self.reward(long) * -5 * 0.08, torch.tensor([-0.1]))

    def test_equal_impulse_is_equal_cost_and_control_partition_does_not_matter(self):
        short = torch.zeros(1, 4, 1, 3)
        short[0, 0, 0, 0] = 900.0
        distributed = torch.zeros_like(short)
        distributed[..., 0] = 225.0
        torch.testing.assert_close(self.reward(short), self.reward(distributed))
        torch.testing.assert_close(
            self.reward(short) * -0.1,
            (self.reward(short[:, :2]) + self.reward(short[:, 2:])) * -0.05,
        )

    def test_xy_magnitudes_do_not_cancel_and_feet_and_envs_are_independent(self):
        forces = torch.zeros(3, 4, 2, 3)
        forces[0, 0, 0] = torch.tensor([60.0, 80.0, 0.0])
        forces[0, 1, 0] = torch.tensor([-60.0, -80.0, 0.0])
        forces[0, 0, 1] = torch.tensor([0.0, -200.0, 0.0])
        forces[1, 0, 1] = torch.tensor([0.0, 50.0, -100.0])
        torch.testing.assert_close(self.reward(forces), torch.tensor([1.0, 0.125, 0.0]))
        torch.testing.assert_close(
            self.reward(forces[:, :, [1, 0]]), self.reward(forces)
        )

    def test_strict_threshold_and_ratio_boundaries_are_preserved(self):
        forces = torch.tensor(
            [[20.0, 0.0, 0.0], [150.0, 0.0, 100.0], [150.01, 0.0, 100.0]]
        )[:, None, None, :]
        torch.testing.assert_close(
            self.reward(forces), torch.tensor([0.0, 0.0, 1.5001])
        )
        torch.testing.assert_close(
            self.reward(torch.zeros_like(forces), min_force=0.0), torch.zeros(3)
        )

    def test_zero_force_history_is_finite_and_empty_history_is_rejected(self):
        torch.testing.assert_close(self.reward(torch.zeros(2, 4, 2, 3)), torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "non-empty force history"):
            self.reward(torch.zeros(2, 0, 2, 3))

    def test_validation_still_rejects_bad_shapes_and_nonfinite_or_invalid_parameters(
        self,
    ):
        for shape in [(1, 4, 3), (1, 4, 2, 4)]:
            with self.subTest(shape=shape), self.assertRaises((ValueError, IndexError)):
                self.reward(torch.zeros(shape))
        forces = torch.zeros(1, 4, 2, 3)
        for key, values in [
            ("min_force", [-1.0, math.nan, math.inf]),
            ("horizontal_ratio", [0.0, -1.0, math.nan, math.inf]),
        ]:
            for value in values:
                with (
                    self.subTest(parameter=key, value=value),
                    self.assertRaises(ValueError),
                ):
                    self.reward(forces, **{key: value})


if __name__ == "__main__":
    unittest.main()
