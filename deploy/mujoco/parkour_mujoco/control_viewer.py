"""MuJoCo renderer with owned keyboard events, without simulate UI shortcuts."""

from __future__ import annotations

import contextlib

import glfw
import mujoco
import numpy as np


class ControlViewer:
    """The small viewer interface used by the deployment loop, on one thread."""

    def __init__(self, model, data, key_callback):
        if not glfw.init():
            raise ValueError("Cannot connect to the desktop display; use --headless")
        self.window = glfw.create_window(
            1280, 800, "Parkour MuJoCo - robot control", None, None
        )
        if not self.window:
            raise ValueError("Cannot create the MuJoCo control window")
        self.model, self.data, self.key_callback = model, data, key_callback
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.user_scn = mujoco.MjvScene(model, maxgeom=256)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.follow = True
        self.status_provider = lambda: ""
        self.images = []
        self._closed = False
        self._last_mouse = glfw.get_cursor_pos(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        try:
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        except BaseException:
            glfw.destroy_window(self.window)
            raise
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    @property
    def viewport(self):
        width, height = glfw.get_framebuffer_size(self.window)
        return mujoco.MjrRect(0, 0, width, height)

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self):
        return not self._closed and not glfw.window_should_close(self.window)

    def _on_key(self, window, key, scancode, action, mods):
        moving = key in (glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D)
        if action != glfw.PRESS and not (moving and action == glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_F:
            self.follow = not self.follow
        elif key in (
            glfw.KEY_W,
            glfw.KEY_S,
            glfw.KEY_A,
            glfw.KEY_D,
            glfw.KEY_X,
            glfw.KEY_T,
            glfw.KEY_R,
            glfw.KEY_SPACE,
            glfw.KEY_C,
        ):
            self.key_callback(key)

    def _on_button(self, window, button, action, mods):
        self._last_mouse = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, x, y):
        dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
        self._last_mouse = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle):
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
            self.follow = False
        elif left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        height = max(1, glfw.get_window_size(window)[1])
        mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.cam)

    def _on_scroll(self, window, x, y):
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * y, self.cam
        )

    def set_images(self, viewport_image):
        viewport, image = viewport_image
        self.images = [(viewport, np.ascontiguousarray(np.flip(image, axis=0)))]

    def sync(self):
        # CameraViews may have selected its own context since the preceding frame.
        glfw.make_context_current(self.window)
        glfw.poll_events()
        viewport = self.viewport
        if not self.is_running() or viewport.width == 0 or viewport.height == 0:
            return
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        for i in range(self.user_scn.ngeom):
            if self.scene.ngeom >= self.scene.maxgeom:
                break
            source = self.user_scn.geoms[i]
            mujoco.mjv_initGeom(
                self.scene.geoms[self.scene.ngeom],
                source.type,
                source.size.astype(float),
                source.pos.astype(float),
                source.mat.astype(float).ravel(),
                source.rgba,
            )
            self.scene.ngeom += 1
        mujoco.mjr_render(viewport, self.scene, self.context)
        text = (
            "W/S speed | A/D turn | X stop | T auto\n"
            "R reset | Space pause | C camera | Esc quit\n"
            "Mouse: left orbit, right pan, wheel zoom | F follow\n"
            + self.status_provider()
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            text,
            "",
            self.context,
        )
        # The overlay selects 2D GL state, so scene depth cannot occlude panels.
        for rect, image in self.images:
            mujoco.mjr_drawPixels(image.ravel(), None, rect, self.context)
        glfw.swap_buffers(self.window)

    def close(self):
        if not self._closed:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
