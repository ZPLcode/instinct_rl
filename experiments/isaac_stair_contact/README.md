# Stair-contact and reciprocal-landing training

These scripts resume an Isaac parkour checkpoint with additional foot-contact
and landing rewards. They use the existing task, observation layout and terrain
mixture. Run them with an Isaac Lab Python environment from the repository root.

## Reward terms

`config.py` adds a separate two-foot contact sensor that records every physics
sample within a policy step. `rewards.py` distinguishes riser impacts from floor
support and uses those masks for the foot orientation and foot-to-plane terms.

`stair_landing.py` rewards an alternating landing only after a foot leaves support,
swings past the other foot's established foothold, and obtains stable support on
the next stair. It covers ordinary ascending and descending stairs. The optional
catch-up cost applies when a foot joins the other foot on an interior stair.
Terrain labels and foot scanners are reward inputs; they do not enter actor observations.

## Resume training

```bash
python experiments/isaac_stair_contact/train.py \
  --checkpoint /path/to/run/model.pt \
  --output outputs/parkour/stair_finetune \
  --num-envs 2048 --iterations 20000 --save-interval 100 \
  --learning-rate 3e-5 --side-weight -50 \
  --landing-weight 1 --catchup-cost 0.1 \
  --capped-airtime-weight 1 \
  --upstairs-proportion 0.4 --upstairs-height-range 0.09 0.12 \
  --episode-length-s 10 --device cuda:0
```

Use a new output directory for each run. `--iterations` is the number of additional
iterations; checkpoint numbering continues from the source model. The script saves
its effective configuration, source snapshot and checkpoint provenance. Checkpoint
writes use a temporary file followed by an atomic rename.

The parameter set above was used for stair fine-tuning. Treat it as an experiment
configuration and compare it against the source checkpoint before changing deployment.
`--landing-weight 0` disables the landing term. Omitting `--capped-airtime-weight`
retains the original airtime term; setting it selects the capped biped term, which
is inactive for zero linear-velocity commands.

## Evaluate

Evaluate a checkpoint on the same terrain seeds and starting states:

```bash
python experiments/isaac_stair_contact/evaluate.py \
  --checkpoint /path/to/run/model.pt \
  --output outputs/parkour/evaluation --duration 20 --headless

python experiments/isaac_stair_contact/analyze_stair_gait.py \
  --help
```

`evaluate_gaps.py` evaluates gap terrain. `analyze_gaps.py` reads the corresponding
traces and uses `assets/shoe_geometry.json` for shoe bounds. `compare.py` and
`compare_gaps.py` compare recorded evaluations without running training.

Compare completion, falls, side-contact impulses, foot placement and alternating
landings. Also check flat-ground walking, zero-command standing, and transitions
between them; improved stair rewards alone do not establish those behaviors.

## CPU tests

```bash
python -m unittest discover -s tests -p 'test_isaac_stair*.py'
```
