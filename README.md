# SO-101 ACT Training

Train an [ACT](https://arxiv.org/abs/2304.13705) policy for a SO-101 robot arm, in
simulation or on real hardware.

Two ways in:

| I want to… | Go to |
|---|---|
| Train on a dataset that already exists | [Quick start](#quick-start) below |
| Collect data in simulation and iterate | [source/witsense/README.md](source/witsense/README.md) |
| Record on the real robot | [Real robot](#real-robot) below |

---

## Requirements

- Linux, NVIDIA GPU (6 GB is enough at batch 4)
- [`uv`](https://docs.astral.sh/uv/) and `ffmpeg`
- A Hugging Face account with access to the `witsense-ai` org

Simulation additionally needs Isaac Sim — see
[source/witsense/README.md](source/witsense/README.md).

## Quick start

```bash
git clone https://github.com/WitsenseAI/so101-custom-training
cd so101-custom-training
cp .env.example .env
```

Edit `.env` and set your token and dataset:

```bash
export HF_TOKEN=hf_...
export DATASET_REPO=witsense-ai/synthetic_so101_ring_insert
```

Then train:

```bash
bash scripts/train_act.sh smoke     # 60 steps, confirms it runs
bash scripts/train_act.sh           # the real run
```

The first run creates `.venv/` and installs lerobot into it. The dataset downloads
automatically. Checkpoints land in `outputs/train/<run>/checkpoints/`.

Run the real one under `tmux` — it takes hours.

## Where everything lives

Everything is inside the repo. Nothing is written outside it.

```
.venv/            training environment (uv, Python 3.12, lerobot)   ~7 GB
.cache/lerobot/   downloaded datasets
.cache/hf/        Hugging Face download cache
outputs/train/    checkpoints and logs
Datasets/         datasets you record yourself
Assets/           USD files for simulation
```

All four paths come from `.env` and are derived from the repo root, so moving or
renaming the repo needs no edits. All are gitignored. To put them on another disk,
override the variable in `.env`.

## Configuration

Everything is set in `.env`. The values you are likely to change:

| Variable | What it does |
|---|---|
| `HF_TOKEN` | Hugging Face write token |
| `DATASET_REPO` | dataset to train on |
| `POLICY_REPO` | where a trained policy is pushed |
| `STEPS` | training steps |
| `BATCH_SIZE` | drives VRAM |
| `RUN_NAME` | output directory name |

Override per run without editing the file:

```bash
DATASET_REPO=witsense-ai/other_dataset STEPS=20000 bash scripts/train_act.sh
```

### Batch size and VRAM

Measured on a 6 GB RTX 4050 with two cameras (640×480 + 1280×720):

| batch | VRAM | 50k steps |
|---|---|---|
| 2 | 2.6 GB | ~2.6 h |
| 4 | 4.4 GB | ~5.2 h |

Memory use is constant per step, so if `smoke` fits, the whole run fits.

### How many steps

Aim for 10–25 epochs:

```
STEPS = frames × 15 / BATCH_SIZE
```

Too few steps looks exactly like a bad dataset, so check this whenever the dataset grows.

## Push a policy

```bash
source .env && source "$LEROBOT_VENV/bin/activate"
python scripts/push_checkpoint_to_hf.py witsense-ai/so101_ring_act \
    outputs/train/<run>/checkpoints/030000
```

Pick the checkpoint by success rate on the robot, not by training loss — they often
disagree. Set `PUSH_TO_HUB=true` in `.env` to push automatically at the end instead.

## Real robot

Requires both arms connected.

```bash
lerobot-find-port                # find serial ports    -> .env
lerobot-find-cameras opencv      # find cameras         -> .env
bash scripts/calibrate.sh        # calibrate both arms
bash scripts/record.sh 50        # record 50 episodes
bash scripts/eval_act.sh         # run a trained policy
```

While recording: **→** save and continue, **←** redo episode, **Esc** stop.

Camera resolution at eval must match training, and calibration must be unchanged.

## Scripts

| Script | Purpose |
|---|---|
| `train_act.sh` | train on any Hub dataset |
| `merge_datasets.py` | combine datasets |
| `push_dataset_to_hf.py` | push a dataset with its version tag |
| `push_checkpoint_to_hf.py` | push a checkpoint |
| `check_dataset.py` | verify a recording before training on it |
| `convert_dataset_v30.py` | convert v2.1 → v3.0 |
| `filter_rollouts.py` | keep the good episodes from a rollout batch |
| `run_eval.py` | evaluate in simulation |
| `record.sh`, `calibrate.sh`, `eval_act.sh` | real robot |

## The two environments

This project has two Python environments, and both contain a package called `lerobot` at
different versions.

| | `.venv/` | `env_isaaclab` |
|---|---|---|
| type | uv venv, Python 3.12 | conda env, Python 3.11 |
| holds | lerobot 0.6.2, torch 2.11, numpy 2.2 | Isaac Sim 5.1, lerobot 0.3.3, numpy 1.26 |
| used for | training, dataset tools, Hub pushes | simulation, recording, evaluation |
| created by | `scripts/train_act.sh` | you — see the simulation README |

They are separate because lerobot ≥ 0.4 needs numpy ≥ 2, while Isaac Sim pins
numpy == 1.26. Installing current lerobot next to Isaac Sim breaks Isaac Lab.

The practical consequence: simulation records datasets in LeRobot **v2.1**, training reads
**v3.0**, so every sim recording needs `scripts/convert_dataset_v30.py`.

## Further reading

- [source/witsense/README.md](source/witsense/README.md) — simulation, and the full
  data-collection loop
- [ACT_PIPELINE.md](ACT_PIPELINE.md) — hardware, dataset format, what each flag does
