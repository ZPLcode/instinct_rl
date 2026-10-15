# G1 parkour in MuJoCo

This package runs Isaac-trained G1 parkour policies with native MuJoCo physics
and CPU inference. It loads the saved robot parameters, joint order, observation
layout and network configuration without starting Isaac Sim.

## Installation

From the repository root, in a Python 3.11 environment:

```bash
python -m pip install -e deploy/mujoco
```

For ONNX export and inference:

```bash
python -m pip install -e 'deploy/mujoco[export]'
```

The robot MJCF and mesh files are included in the package. PyTorch checkpoint
loading requires Instinct RL; the package declares the tested revision as a dependency.

## Checkpoints

Keep the run directory in this form:

```text
run/
  model_58700.pt
  params/
    env.yaml
    agent.yaml
```

Use a checkpoint path, or select the largest checkpoint number in a run directory:

```bash
python -m parkour_mujoco \
  --checkpoint /path/to/run/model_58700.pt --terrain parkour --show-camera

python -m parkour_mujoco \
  --load-run /path/to/run --terrain stairs --command 0 0 0 --duration 600
```

`parkour-sim2sim` is equivalent to `python -m parkour_mujoco`.
With neither argument, the runner searches `logs/instinct_rl/g1_parkour` under
the current directory. Explicit paths are preferable for repeatable evaluations.

## Terrain and movement

| Option | Description |
| --- | --- |
| `--terrain flat` | Flat ground |
| `--terrain rough` | Seeded rough heightfield |
| `--terrain stairs` | Ascending and descending stairs |
| `--terrain boxes` | Three separated low boxes |
| `--terrain gap` | A gap with a lower catch surface |
| `--terrain slope` | An uphill ramp, platform and downhill ramp |
| `--terrain parkour` | Combined rough, stair, box, gap and ramp course |
| `--step-height 0.10` | Height of each stair in metres |
| `--step-width 0.3` | Tread width in metres |
| `--stair-steps 6` | Number of stairs on each side of the platform, from 1 to 20 |
| `--roughness 0.05` | Peak roughness in metres |
| `--slope-angle 10` | Ramp angle in degrees |
| `--seed 42` | Terrain seed |
| `--terrain-geometry heightfield` | Sample the course onto a heightfield |
| `--scene-xml scene.xml` | Use a static external terrain scene |

The default stair flight reaches 0.60 m. Increasing the stair count shifts later
obstacles and the goal to keep the sections separated.
External scenes must contain static terrain only; the runner adds the robot.

By default, a heading controller follows the course goal at up to 0.5 m/s.
`--speed` changes that limit, and `--target X Y` changes the world-space goal.
Use `--command VX VY WZ` for a fixed body-frame velocity command:

```bash
python -m parkour_mujoco --load-run /path/to/run \
  --terrain flat --command 0.5 0 0 --duration 20
```

## Native viewer controls

Click the window to give it keyboard focus. The native MuJoCo sidebars and
visualization shortcuts remain available. Robot controls use separate keys:

| Key | Action |
| --- | --- |
| Up / Down | Increase / decrease forward speed by 0.1 m/s |
| Left / Right | Increase left / right yaw rate by 0.2 rad/s |
| F8 | Clear all three movement-command components |
| F9 | Reset robot state and observation history |
| F10 | Resume automatic target following |
| F11 | Toggle live camera panels and the camera frustum |
| F12 | Toggle view following |
| Space | Pause / continue the control loop |

Commands persist after a key is released. Down changes forward speed only;
F8 also clears any remaining turn command. To start with zero movement commands,
pass `--command 0 0 0`.

Use the mouse to orbit, pan and zoom. Disable following with F12 before panning
freely. Close the window to exit; Esc retains MuJoCo's free-camera shortcut.
F6/F7 retain the native frame and label visualization shortcuts.

The optional `--viewer control` window uses W/S for speed, A/D for yaw, X to stop,
R to reset, T for automatic movement, C for the camera, F for following, and Esc
to close the window.

## Camera visualization

`--show-camera` displays mounted-camera RGB, raw optical depth and the latest depth
frame used by the policy. RGB and raw depth show the current state; the policy
panel includes the configured image processing and observation delay.
A cyan frustum marks the camera pose in the world view. F11 hides the live panels
and frustum while retaining the policy-depth panel.

These visual aids do not enter the collision model or alter policy observations.

## Headless evaluation and video

```bash
python -m parkour_mujoco --load-run /path/to/run \
  --terrain parkour --headless --duration 50 \
  --report outputs/mujoco/course.json --trace outputs/mujoco/course.npz
```

Reports include the checkpoint, terrain parameters, fall count, displacement,
tracking error and termination reason. Traces record commands, actions and state
at each control step. `--auto-reset` resets after a fall; otherwise a fall ends the run.
`--duration` counts simulated time. Headless runs are unpaced unless `--realtime` is set.

To record video without a desktop display:

```bash
MUJOCO_GL=egl python -m parkour_mujoco --load-run /path/to/run \
  --terrain parkour --headless --duration 50 --show-camera \
  --video outputs/mujoco/course.mp4 --report outputs/mujoco/course.json
```

Camera-panel recordings use 960 by 720 pixels at 25 FPS. Other recordings use
640 by 480. Rendering needs a working OpenGL or EGL context; headless inference
without video uses CPU ray casts and does not need a rendering context.

## ONNX

The export includes both the depth encoder and actor, plus a JSON configuration
sidecar. Its input is the complete flat policy observation.

```bash
python -m parkour_mujoco --checkpoint /path/to/run/model_58700.pt \
  --export-onnx outputs/mujoco/policy.onnx --export-only

python -m parkour_mujoco --onnx outputs/mujoco/policy.onnx \
  --terrain flat --headless --duration 20
```

Keep the ONNX and JSON files together. An actor-only export from another playback
script does not have this input contract.

## Simulation conventions

Saved Isaac configuration determines joint order, nominal pose, action scale,
PD gains, armature and effort limits. The runner defaults to zero action delay
for Isaac implicit PD. Physics and policy periods follow the saved configuration.
Depth observations preserve optical-depth conventions, camera pose, processing,
term order and history sampling. Policy inference uses deterministic actor output.

Training randomization and pushes are not applied during deployment. The simplified
courses are intended for controlled comparisons. Differences between PhysX and
MuJoCo remain, and successful traversal is specific to the tested checkpoint and
terrain settings. Evaluate stopping and standing separately from forward walking.

## Source and tests

- `scene.py`: robot model, actuator setup and terrain geometry.
- `observations.py`: proprioception, ray-cast depth and observation history.
- `policy.py`: checkpoint loading, inference and ONNX export.
- `run.py`: CLI, keyboard controls, rollout loop, reports and video.
- `diagnose_vision.py`: visual-input ablations and foot-contact analysis.

```bash
python -m unittest discover -s deploy/mujoco/tests -p 'test_*.py'
```
