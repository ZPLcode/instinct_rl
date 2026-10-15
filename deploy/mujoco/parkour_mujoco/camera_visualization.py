"""Display live camera RGB, raw optical depth, and the actual actor depth input."""

from __future__ import annotations

import mujoco
import numpy as np


def camera_frustum(scene, camera, data, length=0.6):
    """User-scene lines never enter the physics model or policy ray casts."""
    origin, rotation = camera.camera_pose(data)
    h, w = camera.raw_shape
    directions = camera.local_rays[[0, w - 1, h * w - 1, (h - 1) * w]]
    corners = origin + (directions * (length / directions[:, :1])) @ rotation.T
    for start, end in [(origin, p) for p in corners] + [
        (corners[i], corners[(i + 1) % 4]) for i in range(4)
    ]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.1, 0.85, 1.0, 1.0]),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
        scene.ngeom += 1


class CameraViews:
    def __init__(self, model, camera):
        self.camera = camera
        self.renderer = mujoco.Renderer(model, height=144, width=256)

    def close(self):
        self.renderer.close()

    def panel(self, data, policy_panel):
        import cv2

        self.renderer.update_scene(data, camera="robot/policy_camera")
        rgb = self.renderer.render().copy()
        normalization = self.camera.pipeline.get("depth_normalization", {})
        far = normalization.get("depth_range", [0, 2.5])[1]
        raw = np.rint(np.clip(self.camera.raw_depth / far, 0, 1) * 255).astype(np.uint8)
        raw = cv2.resize(raw, (256, 144), interpolation=cv2.INTER_NEAREST)
        panels = []
        for title, frame in [
            ("Camera RGB (live)", rgb),
            (f"Raw depth 0-{far:g}m (live)", raw[..., None].repeat(3, -1)),
        ]:
            block = np.full((170, 256, 3), 24, dtype=np.uint8)
            cv2.putText(
                block,
                title,
                (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            block[26:] = frame
            panels.append(block)
        return np.concatenate([*panels, policy_panel], axis=0)
