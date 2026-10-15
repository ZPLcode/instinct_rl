"""CPU contracts for the observations consumed by the trained parkour actor.

These tests use analytic scenes and synthetic state histories; they do not need
a checkpoint, the motion-reference dataset, a display, or a GPU.
"""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from parkour_mujoco.observations import (
    History,
    NativeDepthCamera,
    ObservationBuilder,
    gaussian_blur,
)


def camera_config(*, width=7, height=5):
    return {
        "name": "camera",
        "frame": {"type": "body", "name": "robot/torso_link"},
        "pattern": {"width": width, "height": height, "fovy": 58.29},
        "ray_alignment": "yaw",
        "offset": {
            "pos": [0.0, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "convention": "world",
        },
        "focal_length": 1.0,
        "horizontal_aperture": 2.0,
        "vertical_aperture": 1.6,
        "include_geom_groups": [0, 1, 2],
        "exclude_parent_body": True,
        "min_distance": 0.1,
        "max_distance": 2.5,
        "mesh_filter_epsilon": 1e-4,
        "mesh_filter_max_hops": 6,
        "noise_pipeline": {},
        "data_histories": {"distance_to_image_plane_noised": 37},
    }


def make_model(world_geoms="", *, root_geoms="", joints=0):
    children = "".join(
        f'<body name="joint_body_{i}"><joint name="joint_{i}" type="hinge" axis="0 1 0"/>'
        '<inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/></body>'
        for i in range(joints)
    )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        + world_geoms
        + '<body name="robot/torso_link"><freejoint/>'
        + '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        + root_geoms
        + children
        + "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def box(front, *, name="wall", group=0, thickness=0.1):
    return (
        f'<geom name="{name}" type="box" pos="{front + thickness / 2} 0 0" '
        f'size="{thickness / 2} 10 10" group="{group}" contype="0" conaffinity="0"/>'
    )


def rotation_xyz(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class HistoryTests(unittest.TestCase):
    def test_first_frame_backfill_chronology_and_reset(self):
        history = History(4)
        np.testing.assert_array_equal(history.append(np.array([1, -1])), [[1, -1]] * 4)
        for frame in range(2, 7):
            history.append(np.array([frame, -frame]))
        np.testing.assert_array_equal(
            history.values, [[3, -3], [4, -4], [5, -5], [6, -6]]
        )
        history.clear()
        np.testing.assert_array_equal(
            history.append(np.array([42, 43])), [[42, 43]] * 4
        )

    def test_rejects_shape_changes(self):
        history = History(8)
        history.append(np.zeros(3))
        with self.assertRaises(ValueError):
            history.append(np.zeros(29))


class DepthCameraTests(unittest.TestCase):
    def test_gaussian_matches_training_torchvision_including_borders(self):
        import torch
        from torchvision.transforms import GaussianBlur

        image = (
            np.random.default_rng(42)
            .uniform(0.0, 2.5, size=(18, 32))
            .astype(np.float32)
        )
        image[0, 0] = 20.0  # Make an incorrect edge-padding rule visible.
        reference = GaussianBlur(kernel_size=3, sigma=1.0)(
            torch.from_numpy(image)[None]
        ).numpy()[0]
        actual = gaussian_blur(image)
        np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)

    def test_optical_axis_depth_and_euclidean_cutoff_on_analytic_wall(self):
        model, data = make_model(box(2.0))
        config = camera_config()
        camera = NativeDepthCamera(model, config)
        depth = camera.capture(data)
        # The wall's optical depth is constant; corners beyond the spherical
        # ray cutoff must become max-depth, even though their optical depth < max.
        u, v = np.meshgrid(np.arange(7) + 0.5, np.arange(5) + 0.5)
        ray_length = 2 * np.sqrt(1 + ((u - 3.5) / 3.5) ** 2 + ((v - 2.5) / 3.125) ** 2)
        expected = np.where(ray_length <= 2.5, 2.0, 2.5)
        np.testing.assert_allclose(depth, expected, atol=1e-6)
        self.assertEqual(float(depth[2, 3]), 2.0)
        self.assertEqual(float(depth[0, 0]), 2.5)

    def test_visual_group_two_is_visible_group_three_and_mount_body_are_not(self):
        model, data = make_model(
            box(2.0)
            + box(0.5, name="collision_group_three", group=3)
            + box(1.0, name="visual_group_two", group=2),
            root_geoms=box(0.2, name="mount_geometry", group=2),
        )
        camera = NativeDepthCamera(model, camera_config(width=1, height=1))
        self.assertAlmostEqual(float(camera.capture(data)[0, 0]), 1.0, places=6)
        visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_group_two")
        self.assertEqual(model.geom_contype[visual], 0)
        self.assertEqual(model.geom_conaffinity[visual], 0)
        model.geom_group[visual] = 4
        self.assertAlmostEqual(float(camera.capture(data)[0, 0]), 2.0, places=6)

    def test_near_surface_continuation_skips_both_faces(self):
        model, data = make_model(
            box(1.0) + box(0.04, name="near", group=2, thickness=0.02)
        )
        camera = NativeDepthCamera(model, camera_config(width=1, height=1))
        self.assertAlmostEqual(float(camera.capture(data)[0, 0]), 1.0, places=6)
        limited_config = camera_config(width=1, height=1)
        limited_config["mesh_filter_max_hops"] = 1
        limited = NativeDepthCamera(model, limited_config)
        self.assertEqual(float(limited.capture(data)[0, 0]), 2.5)

    def test_pose_preserves_torso_roll_and_pitch_despite_yaw_setting(self):
        model, data = make_model()
        rotation = rotation_xyz(0.3, -0.4, 0.7)
        mujoco.mju_mat2Quat(data.qpos[3:7], rotation.ravel())
        data.qpos[:3] = [1.0, 2.0, 3.0]
        mujoco.mj_forward(model, data)
        config = camera_config()
        config["offset"]["pos"] = [0.05, 0.01, 0.44]
        config["offset"]["rot"] = [np.cos(0.2), 0, np.sin(0.2), 0]
        camera = NativeDepthCamera(model, config)
        position, orientation = camera.camera_pose(data)
        np.testing.assert_allclose(
            position,
            data.qpos[:3] + rotation @ np.array([0.05, 0.01, 0.44]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            orientation, rotation @ rotation_xyz(0, 0.4, 0), atol=1e-12
        )

    def test_saved_crop_then_blur_then_normalization(self):
        model, data = make_model(box(2.0))
        config = camera_config(width=64, height=36)
        config["noise_pipeline"] = {
            "crop_and_resize": {"crop_region": [18, 0, 16, 16], "resize_shape": None},
            "gaussian_blur": {"kernel_size": 3, "sigma": 1.0},
            "depth_normalization": {
                "depth_range": [0.0, 2.5],
                "normalize": True,
                "output_range": [0.0, 1.0],
            },
        }
        camera = NativeDepthCamera(model, config)
        result = camera.capture(data)
        self.assertEqual(result.shape, (18, 32))
        self.assertEqual(camera.raw_depth.shape, (36, 64))
        np.testing.assert_allclose(
            result, gaussian_blur(camera.raw_depth[18:, 16:48]) / 2.5, atol=1e-7
        )


class ObservationBuilderTests(unittest.TestCase):
    def setUp(self):
        model, self.data = make_model(joints=29)
        # Reverse the named mapping to catch accidental qpos[7:] assumptions.
        ids = np.arange(29, 0, -1)
        self.scene = SimpleNamespace(
            model=model,
            root_body_id=mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"
            ),
            joint_names=tuple(f"joint_{i}" for i in reversed(range(29))),
            qpos_ids=model.jnt_qposadr[ids],
            qvel_ids=model.jnt_dofadr[ids],
            default_joint_pos=np.linspace(-0.1, 0.1, 29),
            initial_qvel=np.full(model.nv, 0.125),
        )
        terms = {
            name: {"history_length": 8, "clip": None, "scale": scale}
            for name, scale in (
                ("base_ang_vel", 0.25),
                ("projected_gravity", None),
                ("velocity_commands", None),
                ("joint_pos", None),
                ("joint_vel", 0.05),
                ("actions", None),
            )
        }
        terms["depth_image"] = {
            "params": {
                "sensor_cfg": {"name": "camera"},
                "num_output_frames": 8,
                "history_skip_frames": 5,
            }
        }
        self.config = {
            "observations": {"policy": {"terms": terms}},
            "scene": {"sensors": [camera_config()]},
        }
        self.builder = ObservationBuilder(self.scene, self.config)
        self.rotation = rotation_xyz(0.3, -0.4, 0.7)
        mujoco.mju_mat2Quat(self.data.qpos[3:7], self.rotation.ravel())
        self.capture = patch.object(
            self.builder.camera, "capture", return_value=np.zeros((18, 32), np.float32)
        )
        self.camera_mock = self.capture.start()
        self.addCleanup(self.capture.stop)

    def set_state(self, frame):
        self.data.qvel[3:6] = frame * np.array([0.1, 0.2, 0.3])
        self.data.qpos[self.scene.qpos_ids] = (
            self.scene.default_joint_pos + frame + np.arange(29) / 100
        )
        self.data.qvel[self.scene.qvel_ids] = frame * 2 + np.arange(29) / 10
        mujoco.mj_forward(self.scene.model, self.data)
        self.camera_mock.return_value = np.full((18, 32), frame / 50, np.float32)

    def test_full_policy_packing_named_joint_mapping_and_delayed_depth_history(self):
        self.set_state(0)
        self.builder.reset(self.data, np.array([10, 20, 30]))
        for frame in range(1, 41):
            self.set_state(frame)
            observation = self.builder.compute(
                self.data,
                np.array([frame + 10, frame + 20, frame + 30]),
                frame + np.arange(29) / 100,
            )
        self.assertEqual(observation.shape, (5376,))
        self.assertEqual(observation.dtype, np.float32)
        frames = np.arange(33, 41)[:, None]
        np.testing.assert_allclose(
            observation[:24].reshape(8, 3), frames * [0.1, 0.2, 0.3] * 0.25, atol=1e-6
        )
        np.testing.assert_allclose(
            observation[24:48].reshape(8, 3),
            np.tile(self.rotation.T @ [0, 0, -1], (8, 1)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            observation[48:72].reshape(8, 3), frames + [10, 20, 30], atol=1e-6
        )
        np.testing.assert_allclose(
            observation[72:304].reshape(8, 29), frames + np.arange(29) / 100, atol=2e-6
        )
        np.testing.assert_allclose(
            observation[304:536].reshape(8, 29),
            (2 * frames + np.arange(29) / 10 - 0.125) * 0.05,
            atol=5e-7,
        )
        np.testing.assert_allclose(
            observation[536:768].reshape(8, 29), frames + np.arange(29) / 100, atol=2e-6
        )
        np.testing.assert_allclose(
            observation[768:].reshape(8, 18, 32)[:, 0, 0],
            np.arange(4, 40, 5) / 50,
            atol=1e-7,
        )

    def test_reset_backfills_fresh_state_and_zeros_action_history(self):
        self.set_state(1)
        self.builder.compute(self.data, np.ones(3), np.ones(29))
        self.set_state(10)
        observation = self.builder.reset(self.data, np.array([0.5, 0, -0.2]))
        np.testing.assert_array_equal(observation[536:768], 0)
        np.testing.assert_allclose(
            observation[48:72].reshape(8, 3), [[0.5, 0, -0.2]] * 8
        )
        np.testing.assert_allclose(observation[768:], 0.2)

    def test_noise_precedes_clip_and_scale(self):
        config = copy.deepcopy(self.config)
        config["observations"]["policy"]["terms"]["base_ang_vel"].update(
            noise={"n_min": 0.2, "n_max": 0.2, "operation": "add"},
            clip=[-0.1, 0.1],
            scale=0.25,
        )
        builder = ObservationBuilder(self.scene, config, noise=True)
        with patch.object(
            builder.camera, "capture", return_value=np.zeros((18, 32), np.float32)
        ):
            observation = builder.reset(self.data, np.zeros(3))
        np.testing.assert_allclose(observation[:24], 0.025, atol=1e-8)

    def test_rejects_depth_delay_outside_saved_buffer(self):
        with self.assertRaises(ValueError):
            ObservationBuilder(self.scene, self.config, depth_delay=2)


if __name__ == "__main__":
    unittest.main()
