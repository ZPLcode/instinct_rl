"""Native MuJoCo observations matching the parkour training policy.

Depth is ray-cast on the CPU, without a graphics context. Both histories are
oldest first; proprioception is flattened term by term, as in the training
ObservationManager and InstinctRlVecEnvWrapper.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class History:
    """A small history buffer with the training buffer's first-frame fill."""

    def __init__(self, length: int):
        if length < 1:
            raise ValueError("History length must be positive")
        self.length = length
        self.values: np.ndarray | None = None

    def clear(self) -> None:
        self.values = None

    def append(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if self.values is None:
            self.values = np.repeat(value[None], self.length, axis=0)
        else:
            if self.values.shape[1:] != value.shape:
                raise ValueError("History sample shape changed")
            self.values[:-1] = self.values[1:]
            self.values[-1] = value
        return self.values


def gaussian_blur(
    image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Match torchvision GaussianBlur, including reflection at the crop edge."""
    if kernel_size < 1 or kernel_size % 2 != 1 or sigma <= 0:
        raise ValueError("Gaussian kernel must be positive and odd, with sigma > 0")
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    result = np.zeros_like(image, dtype=np.float32)
    for row in range(kernel_size):
        for col in range(kernel_size):
            result += (
                kernel[row]
                * kernel[col]
                * padded[row : row + image.shape[0], col : col + image.shape[1]]
            )
    return result


class NativeDepthCamera:
    """Reproduce GroupedRayCasterCamera's pinhole rays and depth processing.

    The training camera uses the full torso rotation despite its inherited
    ``ray_alignment='yaw'`` setting. Keep that behavior here. MuJoCo geom groups
    and the excluded mounting body are also taken from the saved sensor config.
    """

    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]):
        self.model = model
        self.config = config
        frame = config["frame"]
        if frame["type"] != "body":
            raise ValueError("The native parkour camera requires a body frame")
        body_name = (
            f"{frame['entity']}/{frame['name']}"
            if frame.get("entity")
            else frame["name"]
        )
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Camera body not found: {body_name}")
        self.excluded_body = (
            self.body_id if config.get("exclude_parent_body", True) else -1
        )
        offset = config["offset"]
        if offset.get("convention", "world") != "world":
            raise ValueError(
                "The native parkour camera expects a world-convention offset"
            )
        self.offset_pos = np.array(offset["pos"], dtype=np.float64)
        self.offset_rotation = np.empty(9, dtype=np.float64)
        offset_quat = np.asarray(offset["rot"], dtype=np.float64)
        # MuJoCo expects a unit rotation. The saved offset norm differs by ~1e-5.
        offset_quat /= np.linalg.norm(offset_quat)
        mujoco.mju_quat2Mat(self.offset_rotation, offset_quat)
        self.offset_rotation = self.offset_rotation.reshape(3, 3)
        height, width = (
            int(config["pattern"]["height"]),
            int(config["pattern"]["width"]),
        )
        self.raw_shape = (height, width)
        focal, horizontal = (
            config.get("focal_length"),
            config.get("horizontal_aperture"),
        )
        if focal is not None and horizontal is not None:
            vertical = config.get("vertical_aperture") or horizontal * height / width
            fx, fy = width * focal / horizontal, height * focal / vertical
            cx = width / 2 + config.get("horizontal_aperture_offset", 0.0) * fx
            cy = height / 2 + config.get("vertical_aperture_offset", 0.0) * fy
        else:
            fx = fy = height / (2 * np.tan(np.deg2rad(config["pattern"]["fovy"]) / 2))
            cx, cy = width / 2, height / 2
        center = float(config.get("pixel_center_offset", 0.5))
        u, v = np.meshgrid(np.arange(width) + center, np.arange(height) + center)
        rays = np.stack([np.ones_like(u), -(u - cx) / fx, -(v - cy) / fy], axis=-1)
        self.local_rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(
            -1, 3
        )
        self.geom_groups = np.zeros(6, dtype=np.uint8)
        self.geom_groups[list(config.get("include_geom_groups", (0, 1, 2)))] = 1
        if config.get("mesh_prim_paths"):
            raise ValueError(
                "Mesh-name filtering is not supported by this native depth camera"
            )
        self.max_distance = float(config["max_distance"])
        self.min_distance = float(config.get("min_distance", 0.0))
        self.pipeline = config.get("noise_pipeline", {})
        supported = {"crop_and_resize", "gaussian_blur", "depth_normalization"}
        if set(self.pipeline) - supported:
            raise ValueError(
                f"Unsupported depth processing: {set(self.pipeline) - supported}"
            )
        if self.pipeline.get("crop_and_resize", {}).get("resize_shape") is not None:
            raise ValueError(
                "Resized depth images are not supported; use the trained parkour crop"
            )
        self.raw_depth = np.full(self.raw_shape, self.max_distance, dtype=np.float32)

    def camera_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        rotation = data.xmat[self.body_id].reshape(3, 3)
        position = data.xpos[self.body_id] + rotation @ self.offset_pos
        return position, rotation @ self.offset_rotation

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        position, rotation = self.camera_pose(data)
        directions = np.ascontiguousarray(
            self.local_rays @ rotation.T, dtype=np.float64
        )
        geom_ids = np.empty(len(directions), dtype=np.int32)
        distances = np.empty(len(directions), dtype=np.float64)
        mujoco.mj_multiRay(
            self.model,
            data,
            position,
            directions.ravel(),
            self.geom_groups,
            True,
            self.excluded_body,
            geom_ids,
            distances,
            None,
            len(directions),
            self.max_distance,
        )
        # The training sensor continues rays through hits too close to the camera.
        epsilon = float(self.config.get("mesh_filter_epsilon", 1e-4))
        max_hops = int(self.config.get("mesh_filter_max_hops", 6))
        for index in np.flatnonzero(
            (distances >= 0) & (distances <= self.min_distance)
        ):
            total = distances[index]
            accepted = False
            for _ in range(max_hops):
                traveled = total + epsilon
                hit = np.empty(1, dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    data,
                    position + directions[index] * traveled,
                    directions[index],
                    self.geom_groups,
                    True,
                    self.excluded_body,
                    hit,
                )
                if distance < 0:
                    break
                total = traveled + distance
                if total > self.max_distance:
                    break
                if total > self.min_distance:
                    distances[index] = total
                    accepted = True
                    break
            if not accepted:
                distances[index] = -1.0
        valid = (distances >= 0) & (distances <= self.max_distance)
        # Policy input is optical-axis depth, not Euclidean ray length.
        depth = np.where(valid, distances * self.local_rays[:, 0], self.max_distance)
        depth = (
            np.minimum(depth, self.max_distance)
            .reshape(self.raw_shape)
            .astype(np.float32)
        )
        self.raw_depth = depth.copy()
        for name, config in self.pipeline.items():
            if name == "crop_and_resize":
                up, down, left, right = config["crop_region"]
                depth = depth[up : depth.shape[0] - down, left : depth.shape[1] - right]
            elif name == "gaussian_blur":
                depth = gaussian_blur(
                    depth, int(config["kernel_size"]), float(config["sigma"])
                )
            elif name == "depth_normalization":
                low, high = config["depth_range"]
                depth = np.clip(depth, low, high)
                if config["normalize"]:
                    a, b = config["output_range"]
                    depth = (depth - low) / (high - low) * (b - a) + a
        return np.asarray(depth, dtype=np.float32)


class ObservationBuilder:
    """Build the policy observation without a training environment or dataset."""

    def __init__(
        self,
        scene,
        env_config: dict,
        *,
        noise: bool = False,
        seed: int = 42,
        depth_delay: int = 1,
        group_name: str = "policy",
    ):
        self.scene = scene
        group = env_config["observations"][group_name]
        self.terms = group["terms"]
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.histories = {
            name: History(
                max(
                    1,
                    int(
                        group["history_length"]
                        if group.get("history_length") is not None
                        else term.get("history_length", 0)
                    ),
                )
            )
            for name, term in self.terms.items()
            if term is not None and name != "depth_image"
        }
        depth_term = self.terms["depth_image"]["params"]
        sensor_name = depth_term["sensor_cfg"]["name"]
        camera_config = next(
            sensor
            for sensor in env_config["scene"]["sensors"]
            if sensor["name"] == sensor_name
        )
        self.camera = NativeDepthCamera(scene.model, camera_config)
        self.depth_history = History(
            int(camera_config["data_histories"]["distance_to_image_plane_noised"])
        )
        frames, skip = (
            int(depth_term["num_output_frames"]),
            int(depth_term["history_skip_frames"]),
        )
        self.depth_offsets = np.arange(frames - 1, -1, -1) * max(skip, 1) + depth_delay
        if (
            depth_delay < 0
            or int(self.depth_offsets.max()) >= self.depth_history.length
        ):
            raise ValueError("Depth delay exceeds the saved image history")
        self.depth_image = None

    def reset(self, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        for history in self.histories.values():
            history.clear()
        self.depth_history.clear()
        return self.compute(
            data, command, np.zeros(len(self.scene.joint_names), dtype=np.float32)
        )

    def compute(
        self, data: mujoco.MjData, command: np.ndarray, last_action: np.ndarray
    ) -> np.ndarray:
        scene = self.scene
        rotation = data.xmat[scene.root_body_id].reshape(3, 3)
        values = {
            "base_ang_vel": rotation.T @ data.cvel[scene.root_body_id, :3],
            "projected_gravity": rotation.T @ np.array([0.0, 0.0, -1.0]),
            "velocity_commands": np.asarray(command),
            "joint_pos": data.qpos[scene.qpos_ids] - scene.default_joint_pos,
            "joint_vel": data.qvel[scene.qvel_ids] - scene.initial_qvel[scene.qvel_ids],
            "actions": np.asarray(last_action),
        }
        self.depth_image = self.camera.capture(data)
        depth_history = self.depth_history.append(self.depth_image)
        components = []
        for name, term in self.terms.items():
            if term is None:
                continue
            if name == "depth_image":
                components.append(depth_history[-1 - self.depth_offsets].ravel())
                continue
            if name not in values:
                raise ValueError(f"Unsupported native policy observation: {name}")
            value = np.asarray(values[name], dtype=np.float32).copy()
            if self.noise and term.get("noise"):
                config = term["noise"]
                if config.get("operation", "add") != "add" or "n_min" not in config:
                    raise ValueError(f"Unsupported observation noise for {name}")
                value += self.rng.uniform(
                    config["n_min"], config["n_max"], value.shape
                ).astype(np.float32)
            if term.get("clip") is not None:
                value = np.clip(value, *term["clip"])
            if term.get("scale") is not None:
                value *= np.asarray(term["scale"], dtype=np.float32)
            components.append(self.histories[name].append(value).ravel())
        result = np.concatenate(components).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise RuntimeError("Native MuJoCo observation contains NaN or infinity")
        return result
