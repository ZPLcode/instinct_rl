"""Load parkour policies without constructing a training environment.

Checkpoint loading uses the saved YAML configuration and the original
instinct_rl network classes. The exported ONNX graph includes both the depth
encoder and the actor; its input is the same flat observation as training.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .isaac import normalize_env_config, read_saved_yaml


def resolve_checkpoint(
    load_run: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path:
    """Resolve a checkpoint path, iteration number, or latest saved iteration."""
    root = Path.cwd() / "logs" / "instinct_rl" / "g1_parkour"
    if checkpoint is not None:
        direct = Path(checkpoint).expanduser()
        if direct.is_file():
            return direct.resolve()
        if direct.is_absolute():
            raise FileNotFoundError(f"Checkpoint does not exist: {direct}")
    if load_run is None:
        runs = (
            sorted(
                (
                    p
                    for p in root.iterdir()
                    if p.is_dir()
                    and not p.name.startswith(("_", "."))
                    and any(p.glob("model_*.pt"))
                ),
                key=lambda p: p.name,
            )
            if root.is_dir()
            else []
        )
        if not runs:
            raise FileNotFoundError(
                f"No training runs with checkpoints found in {root}"
            )
        run = runs[-1]
    else:
        run = Path(load_run).expanduser()
        if not run.is_dir():
            run = root / str(load_run)
        if not run.is_dir():
            raise FileNotFoundError(f"Training run does not exist: {load_run}")
    if checkpoint is not None:
        name = str(checkpoint)
        if name.isdigit():
            name = f"model_{name}.pt"
        result = run / name
        if not result.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {result}")
        return result.resolve()
    numbered = [
        (int(match.group(1)), path)
        for path in run.glob("model_*.pt")
        if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))
    ]
    if not numbered:
        raise FileNotFoundError(f"No numbered checkpoints found in {run}")
    return max(numbered, key=lambda item: item[0])[1].resolve()


def _observation_format(
    env_cfg: dict, num_actions: int, group: str
) -> dict[str, tuple[int, ...]]:
    """Recover the supported parkour observation shapes from saved configuration."""
    groups = env_cfg["observations"]
    config = groups[group]
    sizes = {
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos": num_actions,
        "joint_vel": num_actions,
        "actions": num_actions,
    }
    result: dict[str, tuple[int, ...]] = {}
    for name, term in config["terms"].items():
        if term is None:
            continue
        if name == "depth_image":
            params = term["params"]
            sensor_name = params["sensor_cfg"]["name"]
            sensor = next(
                s for s in env_cfg["scene"]["sensors"] if s["name"] == sensor_name
            )
            height = int(sensor["pattern"]["height"])
            width = int(sensor["pattern"]["width"])
            crop = sensor.get("noise_pipeline", {}).get("crop_and_resize")
            if crop:
                up, down, left, right = crop["crop_region"]
                height -= up + down
                width -= left + right
                if crop.get("resize_shape"):
                    height, width = map(int, crop["resize_shape"])
            result[name] = (int(params["num_output_frames"]), height, width)
        elif name in sizes:
            history = config.get("history_length")
            if history is None:
                history = term.get("history_length", 0)
            length = max(1, int(history))
            flatten = (
                config.get("flatten_history_dim", True)
                if config.get("history_length")
                else term.get("flatten_history_dim", True)
            )
            result[name] = (
                (length * sizes[name],)
                if flatten or not history
                else (length, sizes[name])
            )
        else:
            raise ValueError(
                f"Unsupported parkour observation term {group}.{name}; update the sim2sim mapping."
            )
    return result


class _InferenceModule(torch.nn.Module):
    """Export the inference path, including any saved policy normalizer."""

    def __init__(
        self, actor_critic: torch.nn.Module, normalizer: torch.nn.Module | None
    ):
        super().__init__()
        self.actor_critic = actor_critic
        self.normalizer = normalizer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            observations = self.normalizer(observations)
        return self.actor_critic.act_inference(observations)


class ParkourPolicy:
    """NumPy inference interface shared by PyTorch and ONNX policies.

    Pass either one flat observation or a batch. The returned raw actions keep
    the same batch convention; action scaling and joint PD belong to the runner.
    """

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        module: torch.nn.Module | None = None,
        session: Any = None,
        device: str = "cpu",
        env_config: dict | None = None,
    ):
        self.metadata = metadata
        self.observation_segments = {
            name: tuple(shape)
            for name, shape in metadata["observation_segments"].items()
        }
        self.observation_size = sum(
            math.prod(shape) for shape in self.observation_segments.values()
        )
        self.num_actions = int(metadata["num_actions"])
        self.device = torch.device(device)
        self.module = module.to(self.device).eval() if module is not None else None
        self.session = session
        self.env_config = (
            env_config if env_config is not None else metadata.get("env_config")
        )

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        single = observations.ndim == 1
        if single:
            observations = observations[None, :]
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError(
                f"Expected observations [B, {self.observation_size}], got {observations.shape}"
            )
        if not np.isfinite(observations).all():
            raise ValueError("Policy observations contain NaN or infinity")
        if self.module is not None:
            with torch.inference_mode():
                result = (
                    self.module(torch.as_tensor(observations, device=self.device))
                    .cpu()
                    .numpy()
                )
        else:
            result = self.session.run(
                None, {self.session.get_inputs()[0].name: observations}
            )[0]
        if (
            result.shape != (observations.shape[0], self.num_actions)
            or not np.isfinite(result).all()
        ):
            raise RuntimeError(
                f"Policy returned invalid actions with shape {result.shape}"
            )
        return result[0] if single else result

    def export_onnx(self, path: str | Path, metadata_extra: dict | None = None) -> Path:
        """Export one graph, metadata JSON, and verify CPU ONNX numerical parity."""
        if self.module is None:
            raise ValueError("ONNX export requires a checkpoint-backed PyTorch policy")
        import onnx
        import onnxruntime as ort

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix != ".onnx":
            raise ValueError("The export path must end in .onnx")
        module = copy.deepcopy(self.module).cpu().eval()
        dummy = torch.zeros(1, self.observation_size, dtype=torch.float32)
        with torch.inference_mode():
            torch.onnx.export(
                module,
                dummy,
                str(output),
                input_names=["observations"],
                output_names=["actions"],
                dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
                opset_version=17,
                dynamo=False,
                external_data=False,
            )
        onnx.checker.check_model(str(output))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(output), sess_options=options, providers=["CPUExecutionProvider"]
        )
        test_obs = (
            np.random.default_rng(42)
            .normal(0, 0.25, (3, self.observation_size))
            .astype(np.float32)
        )
        with torch.inference_mode():
            expected = module(torch.from_numpy(test_obs)).numpy()
        actual = session.run(None, {"observations": test_obs})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        metadata = {
            **self.metadata,
            "env_config": self.env_config,
            **(metadata_extra or {}),
        }
        metadata["onnx_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        return output


def load_policy(
    load_run: str | Path | None = None,
    checkpoint: str | Path | None = None,
    *,
    onnx: str | Path | None = None,
    device: str = "cpu",
) -> ParkourPolicy:
    """Load a saved parkour checkpoint or a self-contained exported ONNX policy.

    This function never constructs a scene, loads AMP motion data, or initializes
    a PPO runner. Checkpoints are loaded using PyTorch's weights-only unpickler.
    """
    if onnx is not None:
        if checkpoint is not None:
            raise ValueError("Choose either a checkpoint or an ONNX policy")
        if device != "cpu":
            raise ValueError(
                "The native sim2sim ONNX path currently supports --device cpu"
            )
        import onnxruntime as ort

        path = Path(onnx).expanduser().resolve()
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"ONNX policy metadata is required: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        env = metadata.get("env_config")
        if not isinstance(env, dict) or not env:
            raise ValueError(
                "ONNX metadata must include the saved env_config for native deployment"
            )
        policy_group = metadata.get("policy_observation_group", "policy")
        expected_segments = _observation_format(
            env, int(metadata["num_actions"]), policy_group
        )
        segments = metadata.get("observation_segments")
        if not isinstance(segments, dict):
            raise ValueError("ONNX metadata must include observation_segments")
        actual_ordered = [(name, tuple(shape)) for name, shape in segments.items()]
        expected_ordered = list(expected_segments.items())
        if actual_ordered != expected_ordered:
            raise ValueError(
                "ONNX observation_segments must match the saved environment's term order and shapes: "
                f"expected {expected_ordered}, got {actual_ordered}"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        return ParkourPolicy(metadata=metadata, session=session)

    path = resolve_checkpoint(load_run, checkpoint)
    params = path.parent / "params"
    agent_path, env_path = params / "agent.yaml", params / "env.yaml"
    if not agent_path.is_file() or not env_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint requires saved params/agent.yaml and params/env.yaml in {path.parent}"
        )
    agent = read_saved_yaml(agent_path)
    env = normalize_env_config(read_saved_yaml(env_path))
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["model_state_dict"]
    num_actions = int(state["std"].numel())
    policy_group = agent.get("policy_observation_group", "policy")
    critic_group = agent.get("critic_observation_group", "critic")
    obs_format = {"policy": _observation_format(env, num_actions, policy_group)}
    obs_format["critic"] = _observation_format(env, num_actions, critic_group)

    from instinct_rl import modules

    config = copy.deepcopy(agent["policy"])
    class_name = config.pop("class_name")
    # The module constructor prints complete network structures; keep CLI output concise.
    with contextlib.redirect_stdout(io.StringIO()):
        model = modules.build_actor_critic(
            class_name, config, obs_format, num_actions, num_rewards=1
        )
    model.load_state_dict(state, strict=True)
    normalizer = None
    normalizer_cfg = copy.deepcopy(agent.get("normalizers", {}).get("policy"))
    if normalizer_cfg:
        normalizer = modules.build_normalizer(
            input_shape=sum(
                math.prod(shape) for shape in obs_format["policy"].values()
            ),
            normalizer_class_name=normalizer_cfg.pop("class_name"),
            normalizer_kwargs=normalizer_cfg,
        )
        normalizer.load_state_dict(saved["policy_normalizer_state_dict"], strict=True)
    metadata = {
        "format_version": 1,
        "task": "Instinct-Parkour-Target-Amp-G1-v0",
        "checkpoint": str(path),
        "env_config_path": str(env_path),
        "training_backend": env.get("source_training_backend", "MuJoCo"),
        "iteration": int(saved.get("iter", -1)),
        "policy_class": class_name,
        "policy_observation_group": policy_group,
        "critic_observation_group": critic_group,
        "observation_segments": {
            name: list(shape) for name, shape in obs_format["policy"].items()
        },
        "num_actions": num_actions,
        "action_semantics": "raw joint-position action; apply saved action scale and default joint offset",
    }
    return ParkourPolicy(
        metadata=metadata,
        module=_InferenceModule(model, normalizer),
        device=device,
        env_config=env,
    )
