# InstinctLab Parkour

Isaac Lab training and native MuJoCo deployment for Unitree G1 parkour.
The repository includes the [InstinctLab](https://github.com/project-instinct/InstinctLab)
environments, stair-contact and reciprocal-landing experiments, and a standalone
MuJoCo runner for saved parkour policies.

## Repository layout

| Directory | Contents |
| --- | --- |
| `source/instinctlab` | Isaac Lab environments, robot assets, terrain generators and task configurations |
| `scripts/instinct_rl` | Training and playback entry points |
| `experiments/isaac_stair_contact` | Stair reward configuration, fine-tuning and evaluation |
| `deploy/mujoco` | CPU MuJoCo simulation, policy loading, camera display and ONNX export |
| `tests` | CPU tests for stair-contact and landing rewards |

## Isaac training setup

The training code uses Python 3.11, Isaac Sim 5.1.0 and Isaac Lab 2.3.2
(Isaac Lab commit `f73c331738`). Install Isaac Lab with its
[installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Run the following commands with the Python environment used by Isaac Lab:

```bash
git clone https://github.com/ZPLcode/instinct_rl.git instinctlab-parkour
cd instinctlab-parkour
python -m pip install "git+https://github.com/project-instinct/instinct_rl.git@687d0ce"
python -m pip install -e source/instinctlab
```

Set the location of the retargeted motion dataset before starting a parkour task:

```bash
export PARKOUR_MOTION_DIR=/path/to/parkour_motion_reference
```

The directory must contain `parkour_motion_without_run.yaml` and the motion files
listed in its `selected_files` field. Motion datasets and trained checkpoints are
not bundled with the repository.

Train the baseline parkour task:

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Target-Amp-G1-v0 \
  --num_envs 2048 --max_iterations 20000 --headless
```

Training writes checkpoints and `params/env.yaml` / `params/agent.yaml` under
`logs/instinct_rl/g1_parkour/`. Keep these configuration files with each checkpoint;
the MuJoCo runner uses them to recover the network, joint order and controller settings.

For stair-contact and landing fine-tuning, see the
[experiment guide](experiments/isaac_stair_contact/README.md).
The general environment documentation is in [DOCS.md](DOCS.md).

## MuJoCo deployment

The deployment package can run in a separate Python environment. Install it from
this repository, then load a checkpoint together with its saved configuration:

```bash
python -m pip install -e deploy/mujoco
python -m parkour_mujoco \
  --checkpoint /path/to/run/model_58700.pt \
  --terrain parkour --show-camera --command 0 0 0 --duration 600
```

The combined course contains rough ground, stairs, boxes, a gap and ramps.
The default window is MuJoCo's native viewer. Arrow keys adjust movement,
F8 clears the movement command, F9 resets, F10 enables target following,
F11 toggles camera visualization, F12 toggles camera following, and Space pauses.

See the [deployment guide](deploy/mujoco/README.md) for terrain parameters,
keyboard controls, headless evaluation, video recording and ONNX export.

## Validation

CPU tests do not start Isaac Sim or training:

```bash
python -m unittest discover -s tests -p 'test_isaac_stair*.py'
python -m unittest discover -s deploy/mujoco/tests -p 'test_*.py'
```

When evaluating a policy, check terrain completion, foot contacts, falls, and
walking-to-standing transitions. Zero velocity commands can still produce stepping
or drift with some checkpoints. Simplified MuJoCo courses do not reproduce the
full Isaac terrain generator or guarantee sim-to-real performance.

## Upstream and license

Based on [InstinctLab](https://github.com/project-instinct/InstinctLab).
The MuJoCo deployment was adapted from
[InstinctMJ](https://github.com/project-instinct/InstinctMJ), with Isaac joint-order
reference from [Instinct onboard](https://github.com/project-instinct/instinct_onboard).
The PPO / AMP implementation is provided by
[Instinct RL](https://github.com/project-instinct/instinct_rl).

The repository retains the upstream [CC BY-NC 4.0 license](LICENSE), contributor
notices and source-file licenses. See [UPSTREAM.md](UPSTREAM.md) for source revisions.
