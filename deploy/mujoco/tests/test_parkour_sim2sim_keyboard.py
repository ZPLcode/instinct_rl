"""Robot keys must not reach MuJoCo's visualization shortcut handler."""

import unittest
from unittest.mock import patch

import glfw
import mujoco
import numpy as np
from parkour_mujoco.control_viewer import ControlViewer
from parkour_mujoco.run import parser, robot_control_key


class KeyboardTests(unittest.TestCase):
    def test_native_viewer_is_default_and_preserves_display_shortcuts(self):
        self.assertEqual(parser().parse_args([]).viewer, "mujoco")
        for key in [ord(c) for c in "WASDXTRCF"] + [glfw.KEY_F6, glfw.KEY_F7]:
            self.assertIsNone(robot_control_key(key, "mujoco"))

    def test_native_movement_keys_and_control_window_compatibility(self):
        self.assertEqual(robot_control_key(glfw.KEY_UP, "mujoco"), ord("W"))
        self.assertEqual(robot_control_key(glfw.KEY_DOWN, "mujoco"), ord("S"))
        self.assertEqual(robot_control_key(glfw.KEY_LEFT, "mujoco"), ord("A"))
        self.assertEqual(robot_control_key(glfw.KEY_RIGHT, "mujoco"), ord("D"))
        self.assertEqual(robot_control_key(glfw.KEY_SPACE, "mujoco"), 32)
        self.assertEqual(robot_control_key(ord("W"), "control"), ord("W"))

    def viewer(self):
        viewer = ControlViewer.__new__(ControlViewer)
        viewer.opt = mujoco.MjvOption()
        viewer.follow = True
        viewer.window = object()
        viewer.events = []
        viewer.key_callback = viewer.events.append
        return viewer

    def test_robot_keys_dispatch_without_changing_visualization(self):
        viewer = self.viewer()
        before = viewer.opt.flags.copy()
        keys = [
            glfw.KEY_W,
            glfw.KEY_S,
            glfw.KEY_A,
            glfw.KEY_D,
            glfw.KEY_X,
            glfw.KEY_R,
            glfw.KEY_T,
            glfw.KEY_C,
            glfw.KEY_SPACE,
        ]
        for key in keys:
            viewer._on_key(viewer.window, key, 0, glfw.PRESS, 0)
        self.assertEqual(viewer.events, keys)
        np.testing.assert_array_equal(viewer.opt.flags, before)

    def test_release_ignored_and_only_movement_repeats(self):
        viewer = self.viewer()
        viewer._on_key(viewer.window, glfw.KEY_W, 0, glfw.PRESS, 0)
        viewer._on_key(viewer.window, glfw.KEY_W, 0, glfw.REPEAT, 0)
        viewer._on_key(viewer.window, glfw.KEY_W, 0, glfw.RELEASE, 0)
        viewer._on_key(viewer.window, glfw.KEY_R, 0, glfw.REPEAT, 0)
        viewer._on_key(viewer.window, glfw.KEY_C, 0, glfw.REPEAT, 0)
        self.assertEqual(viewer.events, [glfw.KEY_W, glfw.KEY_W])

    def test_follow_toggle_and_escape_are_viewer_controls(self):
        viewer = self.viewer()
        viewer._on_key(viewer.window, glfw.KEY_F, 0, glfw.PRESS, 0)
        self.assertFalse(viewer.follow)
        viewer._on_key(viewer.window, glfw.KEY_F, 0, glfw.RELEASE, 0)
        self.assertFalse(viewer.follow)
        with patch("glfw.set_window_should_close") as close:
            viewer._on_key(viewer.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
            close.assert_called_once_with(viewer.window, True)
        self.assertEqual(viewer.events, [])


if __name__ == "__main__":
    unittest.main()
