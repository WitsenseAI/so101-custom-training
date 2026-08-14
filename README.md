# SO-101 ACT Training

Train an [ACT](https://arxiv.org/abs/2304.13705) policy on **any LeRobot-format dataset
hosted on the Hugging Face Hub**, and push the result to the org.

Deep background — hardware, teleoperation, dataset format, what every training flag does —
is in [ACT_PIPELINE.md](ACT_PIPELINE.md).

---

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env`. The two values that matter:

```bash
export HF_TOKEN=hf_...                                        # write access to push
export DATASET_REPO=witsense-ai/so101_pick_and_place_ring_33   # any Hub dataset
```

Everything else has a working default. `POLICY_REPO` is derived as
`$HF_ORG/<dataset-name>_act` unless you set it. Caches (`HF_LEROBOT_HOME`, `HF_HOME`) and
the lerobot venv (`LEROBOT_VENV`) default under `$HOME` — point them at a roomy disk if
your datasets are large. Weights & Biases is off by default; set `WANDB_ENABLE=true`.

Requires [`uv`](https://docs.astral.sh/uv/) and `ffmpeg`. Everything else installs itself.

## The two environments

Training and simulation cannot share one environment, so this project has two. Knowing
which is which saves a lot of confusion, because both contain a package called `lerobot`
and they are different versions.

| | `$LEROBOT_VENV` — training | `env_isaaclab` — simulation |
|---|---|---|
| what | a `uv` venv, Python 3.12 | a conda env, Python 3.11 |
| holds | lerobot 0.6.2, torch 2.11, numpy 2.2 | Isaac Sim 5.1, Isaac Lab, lerobot 0.3.3, numpy 1.26 |
| runs | `train_act.sh`, dataset tools, Hub pushes | `dataset_sim.py`, `run_eval.py`, anything importing `witsense` |
| created by | `scripts/train_act.sh` on first run | you, separately — see `source/witsense/README.md` |

They are split because **lerobot ≥ 0.4 needs numpy ≥ 2 while `isaacsim-kernel` pins
numpy == 1.26.0**. Installing current lerobot next to Isaac Sim downgrades it to 0.3.3 and
takes `gymnasium` and `packaging` with it, which breaks Isaac Lab. The consequence you
will actually hit: the simulator records datasets in LeRobot **v2.1** format, and the
trainer only reads **v3.0**, so every sim recording needs a conversion step
(`scripts/convert_dataset_v30.py`).

### Creating the training venv

`scripts/train_act.sh` builds it on first run if `$LEROBOT_VENV/bin/lerobot-train` is
missing — nothing to do. To create it by hand, or to rebuild it:

```bash
source .env
uv venv "$LEROBOT_VENV" --python 3.12
uv pip install --python "$LEROBOT_VENV/bin/python" \
    "lerobot[dataset,training] @ git+https://github.com/huggingface/lerobot.git@22bd7a2f489b367d8df42de803b1e8c4ca63a3f9"
```

The commit is pinned in `train_act.sh` as `LEROBOT_REF`. It is pinned because an unpinned
install is how a renamed CLI flag silently breaks a run — `--eval_freq` disappeared between
versions and took a training job with it. Bump it deliberately and re-run `smoke` after.

Anything in this README that says `source .env && source "$LEROBOT_VENV/bin/activate"` is
selecting this environment. `$LEROBOT_VENV` comes from `.env`; it defaults to
`$HOME/.venvs/lerobot` and is set to `/media/zarus101/ssd2/WITSENSE/lerobot-venv` here.

## 2. Train

```bash
bash scripts/train_act.sh smoke     # 60 steps — confirms it fits in VRAM first
bash scripts/train_act.sh           # the real run
```

The first run creates the venv and pins lerobot to a known-good commit. The dataset is
downloaded automatically — no manual snapshot or symlink step.

Batch size drives VRAM. Measured on a 6 GB RTX 4050 with two cameras (640×480 + 1280×720):

| batch | VRAM | s/step | 50k steps |
|---|---|---|---|
| 2 | 2.6 GB | 0.19 | ~2.6 h |
| 4 | 4.4 GB | 0.38 | ~5.2 h |

Always run `smoke` after changing the dataset or batch size. ACT's memory is constant per
step, so if 60 steps fit, the whole run fits — there is no late OOM.

Run it under `tmux` so a closed terminal doesn't kill a multi-hour job.

## 3. Push to the Hub

Default is `PUSH_TO_HUB=false`, then push a checkpoint you actually chose:

```bash
source .env && source "$LEROBOT_VENV/bin/activate"
python scripts/push_checkpoint_to_hf.py witsense-ai/so101_ring_act \
    outputs/train/<run>/checkpoints/030000
```

This is deliberate. The best checkpoint is the one with the highest success rate **on the
robot**, which is often not the last step — and training loss cannot tell you which one it is.

To push automatically at the end of training instead, set `PUSH_TO_HUB=true` in `.env`.

> Setting `POLICY_REPO` to a repo that already holds someone else's model **overwrites it**.
> Check before you push.

---

## Recording your own dataset

Only needed if you're not training on an existing Hub dataset. Requires both arms connected.

```bash
lerobot-find-port                # identify the serial ports -> .env
lerobot-find-cameras opencv      # identify the cameras -> .env
bash scripts/calibrate.sh        # both arms; back up the output
bash scripts/record.sh 50        # 50 teleoperated episodes
```

During recording: **→** save and continue, **←** redo this episode, **Esc** stop. Demonstrate
consistently — same approach, same grasp, same speed — and vary only where the object starts.
See [ACT_PIPELINE.md](ACT_PIPELINE.md) §4 for what separates a good demo from a bad one.

## Evaluating on the robot

```bash
bash scripts/eval_act.sh                                          # policy from .env
bash scripts/eval_act.sh outputs/train/<run>/checkpoints/030000/pretrained_model
```

Camera resolutions at eval **must match training**, and calibration must be unchanged.
Keep a hand on the e-stop for the first rollout. Score by counting successes.

## Layout

```
.env.example                     config: HF, caches, training, W&B, robot ports
scripts/train_act.sh             train on any Hub dataset            ← main entry point
scripts/push_checkpoint_to_hf.py push a chosen checkpoint to the org
scripts/record.sh                teleoperated data collection
scripts/calibrate.sh             one-time arm calibration
scripts/eval_act.sh              on-robot rollouts
robot_config.yaml                hardware reference (ports, cameras)
ACT_PIPELINE.md                  full pipeline explanation
```
