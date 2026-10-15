"""Build a G1 model and controlled terrain courses for native MuJoCo.

Saved training configuration determines joint control and physics parameters.
Terrain occupies geometry group 0; robot visuals and collisions retain groups
2 and 3 for consistent ray-cast depth observations.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import yaml

_SHOE_XML = (
    Path(__file__).resolve().parent / "assets/g1_29dof_torsoBase_popsicle_with_shoe.xml"
)
_SHOE_FACTORY = "parkour_mujoco:g1_with_shoes"


@dataclass
class NativeScene:
    """CPU model and named mappings in policy joint order (not actuator order)."""

    model: mujoco.MjModel
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    position_actuator_ids: np.ndarray
    velocity_actuator_ids: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    root_body_id: int
    physics_dt: float
    control_dt: float
    decimation: int
    target_xy: np.ndarray
    metadata: dict[str, Any]

    @property
    def joint_qpos_ids(self) -> np.ndarray:
        return self.qpos_ids

    @property
    def joint_dof_ids(self) -> np.ndarray:
        return self.qvel_ids

    def reset(self, data: mujoco.MjData) -> None:
        """Reset state and apply the nominal posture with zero velocity targets."""
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.initial_qpos
        data.qvel[:] = self.initial_qvel
        data.ctrl[:] = 0.0
        data.ctrl[self.position_actuator_ids] = self.default_joint_pos
        mujoco.mj_forward(self.model, data)


def _resolve_values(
    values: float | Mapping[str, float] | None,
    names: tuple[str, ...],
    default: float,
) -> np.ndarray:
    if values is None:
        return np.full(len(names), default, dtype=np.float64)
    if not isinstance(values, Mapping):
        return np.full(len(names), float(values), dtype=np.float64)
    result = np.full(len(names), default, dtype=np.float64)
    for index, name in enumerate(names):
        matches = [
            value for pattern, value in values.items() if re.fullmatch(pattern, name)
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple configuration patterns match joint {name!r}")
        if matches:
            result[index] = float(matches[0])
    return result


def _current_config() -> dict[str, Any]:
    """Load nominal robot settings for geometry inspection without a checkpoint."""
    path = Path(__file__).resolve().parent / "assets/robot_defaults.yaml"
    with path.open() as stream:
        return yaml.safe_load(stream)


def _robot_spec(robot_cfg: Mapping[str, Any]) -> tuple[mujoco.MjSpec, tuple[str, ...]]:
    factory = robot_cfg.get("spec_fn")
    if factory != _SHOE_FACTORY:
        raise ValueError(
            "Native parkour sim2sim supports the trained G1 shoe model only; "
            f"saved robot spec_fn is {factory!r}."
        )
    # Reject edits instead of silently changing a checkpoint's physical model.
    for field in (
        "lights",
        "cameras",
        "textures",
        "materials",
        "meshes",
        "geoms",
        "collisions",
    ):
        if robot_cfg.get(field):
            raise ValueError(
                f"Saved robot {field} edits are not supported by native sim2sim"
            )
    if robot_cfg.get("sort_actuators", False):
        raise ValueError("Saved sort_actuators=True is not supported by native sim2sim")
    spec = mujoco.MjSpec.from_file(str(_SHOE_XML))
    for body in spec.bodies:
        for light in tuple(body.lights):
            spec.delete(light)
    joints = tuple(
        joint for joint in spec.joints if joint.type != mujoco.mjtJoint.mjJNT_FREE
    )
    names = tuple(joint.name for joint in joints)
    if len(names) != 29 or any(
        joint.type != mujoco.mjtJoint.mjJNT_HINGE for joint in joints
    ):
        raise ValueError("Expected 29 hinge joints in the G1 parkour model")
    if robot_cfg.get("policy_joint_names"):
        policy_names = tuple(robot_cfg["policy_joint_names"])
        if len(policy_names) != 29 or set(policy_names) != set(names):
            raise ValueError(
                "Saved policy joint order must name each robot joint exactly once"
            )
        names = policy_names
    used: set[str] = set()
    for group in robot_cfg["articulation"]["actuators"]:
        if group.get("transmission_type", "joint") != "joint":
            raise ValueError("Only joint PD transmissions are supported")
        selected = [
            name
            for name in names
            if any(
                re.fullmatch(pattern, name) for pattern in group["target_names_expr"]
            )
        ]
        if not selected or used.intersection(selected):
            raise ValueError("Actuator groups must each match distinct robot joints")
        used.update(selected)
        # Group position actuators before velocity actuators.
        for mode in ("pos", "vel"):
            for name in selected:
                actuator = spec.add_actuator(name=f"{name}_pd_{mode}", target=name)
                actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
                actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
                actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                actuator.inheritrange = 0.0
                actuator.ctrllimited = False
                actuator.forcelimited = False
                gain = float(group["stiffness" if mode == "pos" else "damping"])
                actuator.gainprm[0] = gain
                actuator.biasprm[1 if mode == "pos" else 2] = -gain
        for name in selected:
            joint = spec.joint(name)
            for field in ("armature", "frictionloss"):
                if group.get(field) is not None:
                    setattr(joint, field, float(group[field]))
            if group.get("viscous_damping") is not None:
                joint.damping[0] = float(group["viscous_damping"])
            if group.get("effort_limit") is not None:
                limit = float(group["effort_limit"])
                joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
                joint.actfrcrange[:] = (-limit, limit)
    if used != set(names):
        raise ValueError(f"No PD actuator configured for: {sorted(set(names) - used)}")
    return spec, names


def _add_box(
    spec: mujoco.MjSpec,
    name: str,
    x0: float,
    x1: float,
    height: float,
    *,
    bottom: float = -0.1,
    width: float = 4.0,
) -> None:
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x0 + x1) / 2, 0, (height + bottom) / 2),
        size=((x1 - x0) / 2, width / 2, (height - bottom) / 2),
        group=0,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.43, 0.50, 0.58, 1.0),
    )


def _add_rough(spec, x0, x1, amplitude, seed, width=4.0):
    """A seeded smooth heightfield with a flat border and 5 mm quantization."""
    from scipy.ndimage import gaussian_filter

    ncol, nrow = round((x1 - x0) / 0.05) + 1, round(width / 0.05) + 1
    noise = gaussian_filter(
        np.random.default_rng(seed).normal(size=(nrow, ncol)), sigma=1.2
    )
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-9)
    x, y = np.linspace(x0, x1, ncol), np.linspace(-width / 2, width / 2, nrow)
    fade = (
        np.clip(np.minimum(x - x0, x1 - x) / 0.2, 0, 1)[None]
        * np.clip((width / 2 - abs(y)) / 0.2, 0, 1)[:, None]
    )
    height = np.rint(noise * amplitude * fade / 0.005) * 0.005
    zscale = max(float(height.max()), 0.005)
    field = spec.add_hfield(
        name="rough_surface",
        nrow=nrow,
        ncol=ncol,
        size=((x1 - x0) / 2, width / 2, zscale, 0.1),
        userdata=(height / zscale).astype(np.float32).ravel().tolist(),
    )
    spec.worldbody.add_geom(
        name="terrain_rough",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=field.name,
        pos=((x0 + x1) / 2, 0, 0),
        group=0,
        friction=(1, 0.005, 0.0001),
        rgba=(0.43, 0.50, 0.58, 1),
    )


def _add_ramp(spec, name, x0, x1, z0, z1, width=4.0):
    """A closed convex wedge gives real sloping collision and ray surfaces."""
    vertices = [
        [x, y, z]
        for z in (-0.1, None)
        for x in (x0, x1)
        for y in (-width / 2, width / 2)
    ]
    for i in range(4, 8):
        vertices[i][2] = z0 if vertices[i][0] == x0 else z1
    # Outward winding: inward top faces are culled by the renderer.
    faces = [
        0,
        1,
        2,
        1,
        3,
        2,
        4,
        6,
        5,
        5,
        6,
        7,
        0,
        4,
        1,
        1,
        4,
        5,
        2,
        3,
        6,
        3,
        7,
        6,
        0,
        2,
        4,
        2,
        6,
        4,
        1,
        5,
        3,
        3,
        5,
        7,
    ]
    spec.add_mesh(
        name=name, uservert=np.array(vertices).ravel().tolist(), userface=faces
    )
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        group=0,
        friction=(1, 0.005, 0.0001),
        rgba=(0.43, 0.50, 0.58, 1),
    )


def _course_spec(
    terrain: str,
    step_height: float,
    *,
    step_width=0.3,
    stair_steps=6,
    roughness=0.05,
    slope_angle=10.0,
    seed=42,
) -> tuple[mujoco.MjSpec, np.ndarray]:
    if terrain not in ("flat", "stairs", "parkour", "boxes", "gap", "rough", "slope"):
        raise ValueError(f"Unknown terrain {terrain!r}")
    if not np.isfinite(step_height) or not 0.0 < step_height <= 0.5:
        raise ValueError("step_height must be in (0, 0.5] metres")
    if (
        not 0.1 <= step_width <= 0.8
        or not 0 <= roughness <= 0.2
        or not 0 < slope_angle <= 25
    ):
        raise ValueError(
            "Require step_width in[.1,.8], roughness in[0,.2] m, slope_angle in(0,25] degrees"
        )
    if not isinstance(stair_steps, (int, np.integer)) or not 1 <= stair_steps <= 20:
        raise ValueError("stair_steps must be an integer in[1,20]")
    stairs_end = max(5.0, 3.0 + 2 * (stair_steps - 1) * step_width)
    course_shift = stairs_end - 5.0 if terrain == "parkour" else 0.0
    spec = mujoco.MjSpec()
    spec.modelname = f"G1 parkour sim2sim ({terrain})"
    spec.worldbody.add_light(
        name="sun", pos=(0.0, -3.0, 8.0), dir=(0.0, 0.2, -1.0), diffuse=(0.8, 0.8, 0.8)
    )
    spec.add_texture(
        name="floor_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300,
        height=300,
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=(0.8, 0.8, 0.8),
    )
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=512,
        height=3072,
        rgb1=(0.3, 0.5, 0.7),
        rgb2=(0, 0, 0),
    )
    material = spec.add_material(
        name="floor_material", texuniform=True, texrepeat=(1, 1), reflectance=0.2
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_checker"
    spec.add_material(
        name="terrain_material", rgba=(0.75, 0.75, 0.75, 1), specular=0.3, shininess=0.1
    )
    if terrain in ("gap", "parkour"):
        # A real void: no infinite plane under the gap. The lower floor catches falls.
        gap_start = 12.0 + course_shift if terrain == "parkour" else 3.0
        _add_box(
            spec, "terrain_before_gap", -100, gap_start, 0.0, bottom=-0.8, width=200
        )
        _add_box(
            spec,
            "terrain_after_gap",
            gap_start + 0.35,
            100,
            0.0,
            bottom=-0.8,
            width=200,
        )
        spec.worldbody.add_geom(
            name="terrain_pit_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=(0, 0, -0.8),
            size=(100, 100, 0.1),
            group=0,
            material="floor_material",
            friction=(1, 0.005, 0.0001),
        )
    else:
        spec.worldbody.add_geom(
            name="terrain_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(100, 100, 0.1),
            group=0,
            material="floor_material",
            friction=(1, 0.005, 0.0001),
        )
    if terrain in ("stairs", "parkour"):
        # Nested solid tiers avoid internal opposing faces and summit seams.
        for index in range(stair_steps):
            _add_box(
                spec,
                f"terrain_step_{index}",
                2.0 + index * step_width,
                stairs_end - index * step_width,
                step_height * (index + 1),
            )
    if terrain in ("boxes", "parkour"):
        start = 7.0 + course_shift if terrain == "parkour" else 2.0
        for index, height in enumerate((step_height, 1.5 * step_height, step_height)):
            _add_box(
                spec,
                f"terrain_box_{index}",
                start + 2 * index,
                start + 2 * index + 0.8,
                height,
            )
    if terrain in ("rough", "parkour"):
        _add_rough(spec, 0.5, 1.8 if terrain == "parkour" else 8.0, roughness, seed)
    if terrain in ("slope", "parkour"):
        x0, length = (14.0 + course_shift, 2.0) if terrain == "parkour" else (2.0, 4.0)
        height = length * np.tan(np.deg2rad(slope_angle))
        _add_ramp(spec, "terrain_ramp_up", x0, x0 + length, 0.0, height)
        _add_box(spec, "terrain_ramp_platform", x0 + length, x0 + length + 1, height)
        _add_ramp(
            spec, "terrain_ramp_down", x0 + length + 1, x0 + 2 * length + 1, height, 0.0
        )
    ground_names = {
        "terrain_floor",
        "terrain_before_gap",
        "terrain_after_gap",
        "terrain_pit_floor",
        "terrain_rough",
    }
    for geom in spec.geoms:
        geom.material = (
            "floor_material" if geom.name in ground_names else "terrain_material"
        )
        geom.rgba[:] = (
            0.5,
            0.5,
            0.5,
            1,
        )  # Use the material's color rather than the old geom override.
    spec.visual.headlight.ambient[:] = (0.3, 0.3, 0.3)
    spec.visual.global_.azimuth = 135.0
    spec.visual.global_.elevation = -20.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.stat.extent = 8.0
    target_x = (
        21.0 + course_shift
        if terrain == "parkour"
        else 9.0 + stairs_end - 5.0
        if terrain == "stairs"
        else 12.0
        if terrain == "slope"
        else 9.0
    )
    target = np.array([target_x, 0.0])
    return spec, target


def _replace_course_with_heightfield(
    spec: mujoco.MjSpec,
    horizontal_scale: float,
    vertical_scale: float,
    *,
    x_max: float = 20.0,
) -> dict[str, Any]:
    """Sample the same primitive course using the training heightfield layout.

    A height discontinuity becomes a sloped triangle across one grid interval.
    Training uses ``int(size / horizontal_scale)`` samples including both MuJoCo
    endpoints, so the physical spacing is ``size / (sample_count - 1)``. This
    comparison intentionally preserves the course layout instead of regenerating
    training's random pyramid terrains or adding Perlin noise.
    """
    if not np.isfinite(horizontal_scale) or horizontal_scale <= 0:
        raise ValueError("Heightfield horizontal_scale must be positive and finite")
    if not np.isfinite(vertical_scale) or vertical_scale <= 0:
        raise ValueError("Heightfield vertical_scale must be positive and finite")
    x_min, y_min, y_max = -4.0, -4.0, 4.0
    x_size, y_size = x_max - x_min, y_max - y_min
    ncol, nrow = int(x_size / horizontal_scale), int(y_size / horizontal_scale)
    if min(nrow, ncol) < 2:
        raise ValueError(
            "Heightfield resolution must provide at least two samples per axis"
        )
    xs = np.linspace(x_min, x_max, ncol)
    ys = np.linspace(y_min, y_max, nrow)
    geoms = tuple(spec.geoms)
    planes = [geom for geom in geoms if geom.type == mujoco.mjtGeom.mjGEOM_PLANE]
    floor_height = max(float(geom.pos[2]) for geom in planes)
    temporary_model = spec.compile()
    temporary_data = mujoco.MjData(temporary_model)
    mujoco.mj_forward(temporary_model, temporary_data)
    heights = np.full((nrow, ncol), floor_height, dtype=np.float64)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            hit = np.empty(1, dtype=np.int32)
            distance = mujoco.mj_ray(
                temporary_model,
                temporary_data,
                np.array([x, y, 10.0]),
                np.array([0.0, 0.0, -1.0]),
                None,
                True,
                -1,
                hit,
            )
            if distance >= 0:
                heights[row, col] = 10.0 - distance
    # Same vertical discretization and minimum-height offset as HfTerrainBaseCfg.
    discrete = np.rint(heights / vertical_scale).astype(np.int32)
    minimum, maximum = int(discrete.min()), int(discrete.max())
    integer_range = max(maximum - minimum, 1)
    height_range = integer_range * vertical_scale
    normalized = ((discrete - minimum) / integer_range).astype(np.float32)
    for geom in geoms:
        spec.delete(geom)
    hfield = spec.add_hfield(
        name="course_heightfield",
        size=(x_size / 2, y_size / 2, height_range, height_range),
        nrow=nrow,
        ncol=ncol,
        userdata=normalized.ravel().tolist(),
    )
    spec.worldbody.add_geom(
        name="terrain_heightfield",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield.name,
        pos=((x_min + x_max) / 2, (y_min + y_max) / 2, minimum * vertical_scale),
        group=0,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(1.0, 0.005, 0.0001),
        material="floor_material",
    )
    # Keep a lower catch plane without duplicating contacts on the hfield surface.
    spec.worldbody.add_geom(
        name="terrain_catch_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, minimum * vertical_scale - max(height_range, 1.0)),
        size=(100, 100, 0.1),
        group=0,
        friction=(1.0, 0.005, 0.0001),
        material="floor_material",
    )
    return {
        "horizontal_scale": horizontal_scale,
        "vertical_scale": vertical_scale,
        "nrow": nrow,
        "ncol": ncol,
        "actual_spacing_xy": [x_size / (ncol - 1), y_size / (nrow - 1)],
        "bounds_xy": [[x_min, x_max], [y_min, y_max]],
        "height_range": [minimum * vertical_scale, maximum * vertical_scale],
    }


def _apply_physics(model: mujoco.MjModel, config: Mapping[str, Any]) -> None:
    enum_fields = {
        "integrator": (mujoco.mjtIntegrator, "mjINT_"),
        "cone": (mujoco.mjtCone, "mjCONE_"),
        "jacobian": (mujoco.mjtJacobian, "mjJAC_"),
        "solver": (mujoco.mjtSolver, "mjSOL_"),
    }
    scalar_fields = (
        "timestep",
        "impratio",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
    )
    for field in scalar_fields:
        if field in config:
            setattr(model.opt, field, config[field])
    for field, (enum, prefix) in enum_fields.items():
        if field in config:
            setattr(model.opt, field, getattr(enum, prefix + config[field].upper()))
    if "gravity" in config:
        model.opt.gravity[:] = config["gravity"]
    for field, enum, prefix in (
        ("disableflags", mujoco.mjtDisableBit, "mjDSBL_"),
        ("enableflags", mujoco.mjtEnableBit, "mjENBL_"),
    ):
        for flag in config.get(field, ()):
            setattr(
                model.opt,
                field,
                getattr(model.opt, field) | int(getattr(enum, prefix + flag.upper())),
            )


def build_scene(
    terrain: str = "flat",
    scene_xml: str | Path | None = None,
    env_config_path: str | Path | None = None,
    *,
    env_config: Mapping[str, Any] | None = None,
    step_height: float = 0.10,
    terrain_geometry: str = "boxes",
    step_width: float = 0.3,
    stair_steps: int = 6,
    roughness: float = 0.05,
    slope_angle: float = 10.0,
    seed: int = 42,
) -> NativeScene:
    """Build native MuJoCo from saved parameters and a deterministic test course.

    ``scene_xml`` replaces the course with a static MuJoCo terrain scene; it must
    contain no robot/joints/actuators. The trained G1 is attached automatically.
    Saved config strings are treated as data and are never dynamically imported.
    Without saved configuration, current asset defaults are used and recorded in
    metadata. Startup/reset domain randomization is intentionally not applied.
    ``terrain_geometry='heightfield'`` samples the built-in course onto the saved
    training terrain's grid for a collision-geometry comparison. Default ``boxes``
    retains exact vertical risers. External scene XML keeps its authored geometry.
    """
    if env_config is not None and env_config_path is not None:
        raise ValueError("Provide env_config or env_config_path, not both")
    if terrain_geometry not in ("boxes", "heightfield"):
        raise ValueError("terrain_geometry must be 'boxes' or 'heightfield'")
    if scene_xml is not None and terrain_geometry != "boxes":
        raise ValueError(
            "terrain_geometry applies only to generated courses, not scene_xml"
        )
    source = "current asset defaults"
    if env_config_path is not None:
        path = Path(env_config_path).expanduser().resolve()
        with path.open() as stream:
            env_config = yaml.safe_load(stream)
        source = str(path)
    elif env_config is not None:
        source = "provided saved environment configuration"
    else:
        env_config = _current_config()
    if not isinstance(env_config, Mapping):
        raise ValueError("Environment configuration must be a mapping")
    robot_cfg = env_config["scene"]["entities"]["robot"]
    robot_spec, joint_names = _robot_spec(robot_cfg)
    heightfield_metadata = None
    if scene_xml is not None:
        scene_path = Path(scene_xml).expanduser().resolve()
        spec = mujoco.MjSpec.from_file(str(scene_path))
        if spec.joints or spec.actuators or any(body.mocap for body in spec.bodies):
            raise ValueError(
                "--scene-xml must contain static terrain only, without a robot, joints or actuators"
            )
        for geom in spec.geoms:
            geom.group = 0
        # Scene keyframes, even empty ones, would precede the robot's state.
        for key in tuple(spec.keys):
            spec.delete(key)
        target_xy = np.array([9.0, 0.0])
    else:
        scene_path = None
        spec, target_xy = _course_spec(
            terrain,
            step_height,
            step_width=step_width,
            stair_steps=stair_steps,
            roughness=roughness,
            slope_angle=slope_angle,
            seed=seed,
        )
        if terrain_geometry == "heightfield":
            sub_terrains = (
                env_config["scene"]
                .get("terrain", {})
                .get("terrain_generator", {})
                .get("sub_terrains", {})
            )
            source_name = {
                "flat": "perlin_rough",
                "stairs": "pyramid_stairs",
                "parkour": "pyramid_stairs",
                "boxes": "boxes",
                "gap": "square_gaps",
                "rough": "perlin_rough",
                "slope": "hf_pyramid_slope_inv",
            }[terrain]
            terrain_cfg = sub_terrains.get(source_name, {})
            heightfield_metadata = _replace_course_with_heightfield(
                spec,
                float(terrain_cfg.get("horizontal_scale", 0.1)),
                float(terrain_cfg.get("vertical_scale", 0.005)),
                x_max=max(20.0, float(target_xy[0]) + 1.0),
            )
    camera_cfg = next(
        (
            s
            for s in env_config["scene"].get("sensors", ())
            if s.get("name") == "camera"
        ),
        None,
    )
    if camera_cfg:
        rotation = np.zeros(9)
        quat = np.asarray(camera_cfg["offset"]["rot"], dtype=float)
        mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
        # Sensor axes are +X forward/+Y left/+Z up; MuJoCo cameras look down -Z.
        rotation = rotation.reshape(3, 3) @ np.array(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        )
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        vertical = camera_cfg.get("vertical_aperture")
        if vertical is None and camera_cfg.get("horizontal_aperture") is not None:
            vertical = (
                camera_cfg["horizontal_aperture"]
                * camera_cfg["pattern"]["height"]
                / camera_cfg["pattern"]["width"]
            )
        fovy = (
            np.rad2deg(2 * np.arctan(vertical / (2 * camera_cfg["focal_length"])))
            if vertical is not None
            else camera_cfg["pattern"]["fovy"]
        )
        robot_spec.body(camera_cfg["frame"]["name"].split("/")[-1]).add_camera(
            name="policy_camera",
            pos=camera_cfg["offset"]["pos"],
            quat=quat,
            fovy=float(fovy),
        )
    spec.attach(robot_spec, prefix="robot/", frame=spec.worldbody.add_frame())
    model = spec.compile()
    _apply_physics(model, env_config["sim"]["mujoco"])
    joint_ids = np.array(
        [model.joint(f"robot/{name}").id for name in joint_names], dtype=np.int32
    )
    qpos_ids = model.jnt_qposadr[joint_ids].copy()
    qvel_ids = model.jnt_dofadr[joint_ids].copy()
    position_ids = np.array(
        [model.actuator(f"robot/{name}_pd_pos").id for name in joint_names],
        dtype=np.int32,
    )
    velocity_ids = np.array(
        [model.actuator(f"robot/{name}_pd_vel").id for name in joint_names],
        dtype=np.int32,
    )
    init = robot_cfg["init_state"]
    defaults = _resolve_values(init["joint_pos"], joint_names, 0.0)
    initial_qpos = model.qpos0.copy()
    initial_qpos[:3] = init["pos"]
    initial_qpos[3:7] = init["rot"]
    initial_qpos[qpos_ids] = defaults
    initial_qvel = np.zeros(model.nv)
    initial_qvel[:3] = init.get("lin_vel", (0.0, 0.0, 0.0))
    initial_qvel[3:6] = init.get("ang_vel", (0.0, 0.0, 0.0))
    initial_qvel[qvel_ids] = _resolve_values(init.get("joint_vel"), joint_names, 0.0)
    action_cfg = env_config["actions"]["joint_pos"]
    if action_cfg.get("preserve_order", False) or action_cfg.get(
        "actuator_names", [".*"]
    ) != [".*"]:
        raise ValueError("Native parkour requires all joints in natural policy order")
    if not action_cfg.get("use_default_offset", True):
        raise ValueError("Native parkour requires use_default_offset=True")
    if action_cfg.get("clip") is not None:
        raise ValueError("Saved per-joint action clipping is not supported")
    action_scale = _resolve_values(action_cfg["scale"], joint_names, 1.0)
    decimation = int(env_config["decimation"])
    if decimation <= 0 or model.opt.timestep <= 0:
        raise ValueError("Decimation and physics timestep must be positive")
    delay_groups = []
    for group in robot_cfg["articulation"]["actuators"]:
        delay_groups.append(
            {
                "joint_indices": [
                    index
                    for index, name in enumerate(joint_names)
                    if any(
                        re.fullmatch(pattern, name)
                        for pattern in group["target_names_expr"]
                    )
                ],
                **{
                    key: group.get(key, 0)
                    for key in ("delay_min_lag", "delay_max_lag", "delay_update_period")
                },
            }
        )
    metadata = {
        "config_source": source,
        "robot_xml": str(_SHOE_XML),
        "terrain": terrain if scene_path is None else "external",
        "terrain_geometry": terrain_geometry if scene_path is None else "external",
        "stair_layout": (
            "nested_solid_tiers_v2"
            if scene_path is None and terrain in ("stairs", "parkour")
            else None
        ),
        "heightfield": heightfield_metadata,
        "scene_xml": str(scene_path) if scene_path else None,
        "step_height": step_height,
        "step_width": step_width,
        "roughness": roughness,
        "slope_angle_degrees": slope_angle,
        "stair_steps_up": stair_steps,
        "stair_steps_down": stair_steps,
        "visual_style": "MuJoCo blue-gray checker floor and light-gray obstacles",
        "terrain_seed": seed,
        "course_sections": ["rough", "stairs", "boxes", "gap", "slope"]
        if terrain == "parkour"
        else [terrain],
        "terrain_geom_group": 0,
        "robot_geom_groups": [2, 3],
        "actuator_delay_groups": delay_groups,
        "root_joint_name": "robot/floating_base_joint",
        "domain_randomization": False,
    }
    return NativeScene(
        model=model,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        joint_names=joint_names,
        joint_ids=joint_ids,
        qpos_ids=qpos_ids,
        qvel_ids=qvel_ids,
        position_actuator_ids=position_ids,
        velocity_actuator_ids=velocity_ids,
        default_joint_pos=defaults,
        action_scale=action_scale,
        root_body_id=model.body("robot/torso_link").id,
        physics_dt=float(model.opt.timestep),
        control_dt=float(model.opt.timestep) * decimation,
        decimation=decimation,
        target_xy=target_xy,
        metadata=metadata,
    )
