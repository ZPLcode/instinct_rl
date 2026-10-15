"""Regression for the artificial vertical seam in the flat stair summit."""

import unittest

import mujoco
import numpy as np
from parkour_mujoco.scene import _course_spec


class StairGeometryTests(unittest.TestCase):
    def setUp(self):
        spec, _ = _course_spec("stairs", 0.1, step_width=0.5, stair_steps=3)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def ray(self, origin, direction):
        hit = np.empty(1, dtype=np.int32)
        return mujoco.mj_ray(
            self.model,
            self.data,
            np.asarray(origin, dtype=np.float64),
            np.asarray(direction, dtype=np.float64),
            None,
            True,
            -1,
            hit,
        )

    def test_exposed_step_profile_is_preserved(self):
        xs = [1.9, 2.1, 2.49, 2.51, 3.1, 3.49, 3.51, 3.9, 4.01, 4.51, 5.1]
        expected = [0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1, 0]
        actual = [2.0 - self.ray([x, 0, 2], [0, 0, -1]) for x in xs]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(self.ray([1.9, 0, 0.05], [1, 0, 0]), 0.1)

    def test_no_internal_vertical_face_in_flat_summit(self):
        # Six adjacent slabs wrongly returned .25 here: an internal face at
        # x=3.5. A solid one-metre summit exits only at its outer edge x=4.0.
        self.assertAlmostEqual(self.ray([3.25, 0, 0.29], [1, 0, 0]), 0.75)
        self.assertAlmostEqual(self.ray([3.75, 0, 0.29], [-1, 0, 0]), 0.75)


if __name__ == "__main__":
    unittest.main()
