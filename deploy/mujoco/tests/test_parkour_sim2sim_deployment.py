"""Deployment contracts: saved Isaac data, real terrain surfaces, camera axes."""

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import yaml
from parkour_mujoco.camera_visualization import camera_frustum
from parkour_mujoco.isaac import G1_ISAAC_JOINT_NAMES, read_saved_yaml
from parkour_mujoco.observations import NativeDepthCamera
from parkour_mujoco.scene import _course_spec, build_scene
from test_parkour_sim2sim_observations import box, camera_config, make_model


def vertical_height(model, data, x, y=0.0):
    return 10.0 - mujoco.mj_ray(
        model,
        data,
        np.array([x, y, 10.0]),
        np.array([0.0, 0.0, -1.0]),
        None,
        True,
        -1,
        np.empty(1, dtype=np.int32),
    )


class DeploymentTests(unittest.TestCase):
    def test_saved_yaml_accepts_only_data_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text(
                "shape: !!python/tuple [8,18,32]\nids: !!python/object/apply:builtins.slice [null,null,null]\n"
            )
            self.assertEqual(read_saved_yaml(p)["shape"], [8, 18, 32])
            p.write_text("value: !!python/object/apply:os.system [false]\n")
            with self.assertRaises(yaml.constructor.ConstructorError):
                read_saved_yaml(p)

    def test_gap_in_combined_course_is_a_real_hole(self):
        spec, target = _course_spec("parkour", 0.1, stair_steps=3)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(vertical_height(model, data, 12.1), -0.8, places=6)
        self.assertAlmostEqual(vertical_height(model, data, 12.5), 0.0, places=6)
        self.assertGreater(target[0], 19.0)
        names = [g.name for g in spec.geoms]
        for name in [
            "terrain_rough",
            "terrain_step_0",
            "terrain_box_0",
            "terrain_ramp_up",
        ]:
            self.assertIn(name, names)

    def test_ramp_rays_and_physical_support_match_inclination(self):
        angle = np.deg2rad(10.0)
        spec, _ = _course_spec("slope", 0.1)
        body = spec.worldbody.add_body(name="probe", pos=[4.0, 0.0, 1.5])
        body.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], mass=1)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for x in [2.1, 3.0, 5.8]:
            self.assertAlmostEqual(
                vertical_height(model, data, x, 0.5),
                (x - 2.0) * np.tan(angle),
                places=5,
            )
        for x in [7.2, 8.5, 10.8]:
            self.assertAlmostEqual(
                vertical_height(model, data, x, 0.5),
                (11.0 - x) * np.tan(angle),
                places=5,
            )
        for _ in range(1500):
            mujoco.mj_step(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertAlmostEqual(
            data.xpos[body.id, 2], 2 * np.tan(angle) + 0.05 / np.cos(angle), delta=0.002
        )

    def test_both_ramp_meshes_have_outward_faces(self):
        for angle in [5.0, 10.0, 25.0]:
            spec, _ = _course_spec("slope", 0.1, slope_angle=angle)
            for name in ["terrain_ramp_up", "terrain_ramp_down"]:
                mesh = spec.mesh(name)
                vertices = np.array(mesh.uservert).reshape(-1, 3)
                triangles = vertices[np.array(mesh.userface).reshape(-1, 3)]
                normals = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                )
                # A convex wedge's face normals must point away from its interior.
                outward = np.sum(
                    normals * (triangles.mean(1) - vertices.mean(0)), axis=1
                )
                self.assertTrue(np.all(outward > 0), (angle, name, outward))

    def test_roughness_is_seeded_and_zero_amplitude_is_flat(self):
        fields = []
        for seed, amplitude in [(42, 0.05), (42, 0.05), (43, 0.05), (42, 0.0)]:
            spec, _ = _course_spec("rough", 0.1, seed=seed, roughness=amplitude)
            model = spec.compile()
            fields.append(model.hfield_data.copy())
        np.testing.assert_array_equal(fields[0], fields[1])
        self.assertFalse(np.array_equal(fields[0], fields[2]))
        np.testing.assert_array_equal(fields[3], np.zeros_like(fields[3]))

    def test_isaac_pixel_coordinates_use_integer_centers(self):
        model, data = make_model(box(2.0))
        config = camera_config(width=3, height=3)
        config["pixel_center_offset"] = 0.0
        camera = NativeDepthCamera(model, config)
        # Isaac pixels are 0,1,2 with principal point1.5, not half-integer pixels.
        expected = np.array([1.0, 1.0, 0.8])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(camera.local_rays[0], expected)
        self.assertTrue(np.isfinite(camera.capture(data)).all())

    def test_frustum_is_visual_only(self):
        model, data = make_model()
        camera = NativeDepthCamera(model, camera_config())
        scene = mujoco.MjvScene(model, maxgeom=20)
        before = data.qpos.copy()
        ngeom = model.ngeom
        camera_frustum(scene, camera, data)
        self.assertEqual(scene.ngeom, 8)
        self.assertEqual(model.ngeom, ngeom)
        np.testing.assert_array_equal(data.qpos, before)

    def test_native_mapping_uses_saved_isaac_joint_order(self):
        scene = build_scene()
        self.assertEqual(set(scene.joint_names), set(G1_ISAAC_JOINT_NAMES))
        self.assertEqual(scene.joint_names, G1_ISAAC_JOINT_NAMES)

    def test_heightfield_conversion_retains_ramps_and_the_gap(self):
        scene = build_scene("parkour", terrain_geometry="heightfield", stair_steps=3)
        data = mujoco.MjData(scene.model)
        mujoco.mj_forward(scene.model, data)
        self.assertAlmostEqual(
            vertical_height(scene.model, data, 12.15, 0.5), -0.8, delta=0.005
        )
        self.assertAlmostEqual(
            vertical_height(scene.model, data, 15.0, 0.5),
            np.tan(np.deg2rad(10.0)),
            delta=0.006,
        )

    def test_longer_stair_flight_has_real_treads_and_separated_obstacles(self):
        spec, target = _course_spec("parkour", 0.1, stair_steps=6)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for i in range(6):
            self.assertAlmostEqual(
                vertical_height(model, data, 2.15 + i * 0.3), (i + 1) * 0.1, places=5
            )
            self.assertAlmostEqual(
                vertical_height(model, data, 4.65 + i * 0.3), (5 - i) * 0.1, places=5
            )
        self.assertAlmostEqual(vertical_height(model, data, 4.0), 0.6, places=5)
        self.assertAlmostEqual(vertical_height(model, data, 13.15), -0.8, places=5)
        self.assertEqual(target[0], 22.0)


if __name__ == "__main__":
    unittest.main()
