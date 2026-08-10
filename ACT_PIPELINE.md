# SO-101 → ACT: the whole pipeline, end to end

Everything from plugging in the arms to a policy that moves the follower, for
`witsense-ai/so101_pick_and_place_ring_33` and `train_act.sh`.

Facts below marked **(live)** were read from the actual dataset on the Hub and from
lerobot `main` source, not from memory.

---

## 0. What you are actually building

ACT (Action Chunking Transformer) is **behaviour cloning**, nothing more. It learns
`observation → next 100 actions`. There is no reward, no exploration, no simulator.
Consequences that drive every decision below:

- The policy can only be as good as your teleoperated demonstrations. Sloppy demos → sloppy policy.
- It learns *your* camera views. Move a camera after recording and the policy breaks.
- It learns joint positions in *calibrated* units. Re-calibrate differently → the policy breaks.
- "Loss went down" means nothing. The only real metric is success rate on the robot.

The action space here is 6 joint **positions** (not velocities, not torques) — **(live)** from
`meta/info.json`:

```
shoulder_pan.pos  shoulder_lift.pos  elbow_flex.pos
wrist_flex.pos    wrist_roll.pos     gripper.pos
```

`observation.state` is the same 6 values read back from the follower. So the model maps
(2 camera frames + 6 joint readings) → (100 × 6 future joint targets).

---

## 1. Hardware setup

Two SO-101 arms and two USB cameras.

| Piece | Role | Repo default |
|---|---|---|
| Leader arm | You hold this. It is a position sensor with a handle. Motors are limp. | `/dev/ttyACM2`, id `left_leader` |
| Follower arm | Mirrors the leader. Later runs the policy. | `/dev/ttyACM0`, id `left_follower` |
| `top` camera | Scene / workspace overview | `/dev/video2` |
| `wrist` camera | Mounted on the follower gripper | `/dev/video4` |

(from `robot_config.yaml` — note `scripts/*.sh` disagree with each other on ports and
video nodes; the ports are whatever `lerobot-find-port` tells you *today*.)

### 1.1 Find the serial ports

```bash
lerobot-find-port          # unplug the arm when prompted; it tells you which node vanished
```

Do it once per arm. `/dev/ttyACM*` numbering is assignment-order and **will** change across
reboots or replug order. If you get tired of that, pin them with a udev rule by serial:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -m1 ATTRS{serial}
# /etc/udev/rules.d/99-so101.rules
# SUBSYSTEM=="tty", ATTRS{serial}=="XXXX", SYMLINK+="so101_leader"
```

### 1.2 Find the cameras

```bash
lerobot-find-cameras opencv
```

Prints every `/dev/videoN` that actually yields frames and saves sample images. `/dev/videoN`
also renumbers on replug. Check both the index **and** the picture — swapping `top` and
`wrist` is invisible during training and fatal at eval.

---

## 2. Calibration — do not skip, do not redo casually

```bash
lerobot-calibrate --teleop.type=so101_leader   --teleop.port=$ROBOT_LEADER_PORT   --teleop.id=left_leader
lerobot-calibrate --robot.type=so101_follower  --robot.port=$ROBOT_FOLLOWER_PORT  --robot.id=left_follower
```

What it does: drives each joint to its physical stops, records the min/max encoder counts and
a homing offset, and writes a JSON per arm id under `~/.cache/huggingface/lerobot/calibration/`
(this repo also keeps copies in `calibration/`).

Why it matters more than it looks: calibration defines the **units** of everything downstream.
`shoulder_pan.pos = 0.31` only means a physical angle relative to that calibration file. Your
dataset, and therefore your policy, is expressed in those units.

> **Rule:** back up the calibration files next to the dataset. If you re-calibrate the follower
> after training, the policy's outputs land in a different frame and it will miss the object.
> `scripts/calibrate.sh` already copies them to `~/lerobot-calibration-backup/` — keep that.

The leader and follower are calibrated separately because they have different mounting offsets;
teleoperation maps leader normalised position → follower normalised position.

---

## 3. Teleoperation sanity check

Before recording anything:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$ROBOT_FOLLOWER_PORT --robot.id=left_follower \
  --teleop.type=so101_leader  --teleop.port=$ROBOT_LEADER_PORT  --teleop.id=left_leader \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30},
                    wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

Checklist:
- Follower tracks the leader 1:1, no lag, no jitter, no joint racing to a limit (→ bad calibration).
- Gripper opens and closes over its full range.
- Both camera windows show the right view, right way up, and the object is visible in both.
- The wrist camera doesn't go black when the gripper closes (exposure) and its cable doesn't
  snag at the extremes of the motion.

Fix everything here. Anything wrong now is baked into every episode you record.

---

## 4. Recording the dataset (the hand-teleoperated part)

This is `scripts/record_pick_green.sh`. The command:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=$ROBOT_FOLLOWER_PORT --robot.id=left_follower \
  --robot.cameras="{top: {...}, wrist: {...}}" \
  --teleop.type=so101_leader --teleop.port=$ROBOT_LEADER_PORT --teleop.id=left_leader \
  --dataset.repo_id=witsense-ai/so101_pick_and_place_ring_33 \
  --dataset.single_task="pickup the ring and place it on the toy" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.camera_encoder.vcodec=h264 --dataset.streaming_encoding=true \
  --display_data=true --play_sounds=true
```

### What happens per control tick (30 Hz)

1. Read follower joint positions → `observation.state`
2. Grab a frame from each camera → `observation.images.top`, `observation.images.wrist`
3. Read leader joint positions → `action` (this is the label: where you told the arm to go)
4. Append the frame; sleep to hold exactly 1/30 s

So **the action at time t is the leader position at time t**, and the state is where the
follower actually is. The model learns the correction from "where I am" to "where I should go" —
which is why teleop lag shows up as a learned offset.

### Loop structure

`episode_time_s=25` recording, then `reset_time_s=15` where recording stops and you put the
ring back. Repeat `num_episodes` times.

### Keyboard controls during recording **(live, from `lerobot/utils/keyboard_input.py`)**

| Key | Effect |
|---|---|
| **→** right arrow | End this episode early, save it, move to reset |
| **←** left arrow | End and **re-record** this episode (discards it) |
| **Esc** | Stop recording entirely |

Use ← liberally. A failed or ugly episode in the dataset is worse than no episode.

### How to actually demonstrate (this is the part that decides your success rate)

- **Be consistent.** Same approach direction, same grasp point, same lift height, same place
  motion, every time. ACT has no way to know that two different strategies are both valid; it
  averages them, and the average is often a strategy that fails.
- **Vary only the object pose.** Move the ring/target to a different spot each episode, covering
  the workspace you want to work at test time. Variation in *the world*, consistency in *your motion*.
- **Move smoothly and at a moderate speed.** Jerky leader motion becomes jerky action labels.
- **Pause briefly at the key moments** (just before grasp, just after lift). Extra frames at the
  hard part is free upweighting of the hard part.
- **Do not correct mid-episode with a big jerk.** If you mess up, hit ← and redo.
- **End episodes in a consistent pose.** Don't let the arm drift for the last 3 seconds; that
  teaches "do nothing" as a valid final action.
- **Only record successes.** Failures are not useful to plain behaviour cloning.

### How many episodes

30–50 for a single fixed task with a fixed camera — which is what you have. If the object can
start anywhere on the table, or lighting changes, you want 100+. Your 33 is at the thin end of
workable; the first thing to try if the policy is mediocre is *more and better episodes*, not
more training steps.

---

## 5. What the recorded dataset actually is

**(live)** `meta/info.json` of `witsense-ai/so101_pick_and_place_ring_33`:

| Field | Value |
|---|---|
| `codebase_version` | `v3.0` |
| `total_episodes` | 33 |
| `total_frames` | 18,861 |
| `fps` | 30 |
| `total_tasks` | 1 |
| `robot_type` | `so_follower` |
| `observation.images.top` | video, **480 × 640**, h264, crf 23 |
| `observation.images.wrist` | video, **720 × 1280**, h264, crf 23 |
| `action` / `observation.state` | float32, shape `[6]` |

That's 18,861 / 30 ≈ **629 s ≈ 10.5 minutes** of robot data, ~571 frames (≈19 s) per episode.

Layout on disk / in the repo:

```
meta/info.json                       # schema, fps, feature shapes  ← the file training reads first
meta/stats.json                      # per-feature mean/std/min/max ← used for normalisation
meta/tasks.parquet                   # task_index → language string
meta/episodes/chunk-000/*.parquet    # per-episode index ranges
data/chunk-000/file-000.parquet      # the numeric columns, all episodes concatenated
videos/observation.images.top/...    # one mp4 per chunk; frames decoded on demand
videos/observation.images.wrist/...
```

Images are **not** stored as frames — they are h264 video, decoded lazily by the dataloader.
That is why `ffmpeg` is installed in the training script and why `num_workers` matters.

> ⚠️ **Two things worth checking on this specific dataset.**
> 1. The two cameras have **different resolutions** (top 640×480, wrist 1280×720). ACT handles
>    this — it runs the backbone per camera and concatenates tokens — but the 720p wrist view
>    produces roughly 4× as many visual tokens as the top view, so it dominates memory and step
>    time. Recording both at 640×480 would train meaningfully faster for probably no loss.
> 2. `scripts/record_pick_green.sh` configures `top` at 1280×720 and `wrist` at 640×480 — the
>    **opposite** of what landed in the dataset. Either the names are swapped in the dataset or a
>    different config produced it. Visualise before you trust it, because whatever is labelled
>    `wrist` at training time must be labelled `wrist` at eval time or the policy sees its inputs
>    swapped.

Inspect it:

```bash
lerobot-dataset-viz --repo-id=witsense-ai/so101_pick_and_place_ring_33 --episode-index=0
```

Watch a few episodes. Look for: object visible in both views the whole time, no black frames,
no episode where the demo failed, consistent motion across episodes.

---

## 6. Publishing to the Hub

Either record straight to the Hub (`--dataset.push_to_hub=true`) or push afterwards:

```bash
HF_TOKEN=... python scripts/push_dataset_to_hf.py \
  --path ~/.cache/huggingface/lerobot/witsense-ai/so101_pick_and_place_ring_33 \
  --repo-id witsense-ai/so101_pick_and_place_ring_33
```

Training then pulls it back down on the GPU box. That indirection exists purely so the recording
machine (with the arms) and the training machine (with the GPU) don't have to be the same box.

---

## 7. `train_act.sh`, line by line

### 7.1 Arguments and paths

```bash
HF_TOKEN="${1:-${HF_TOKEN:-}}"     # positional arg, else env var
STEPS="${2:-50000}"
BATCH_SIZE="${3:-8}"
VENV="/workspace/vla_jepa_env"      # reused from the VLA-JEPA experiments
export HF_HOME="/workspace/.hf_home"
```

`/workspace` is the vast.ai / RunPod convention — that's the persistent volume. `HF_HOME` is
redirected there so a 2 GB dataset download survives a container restart and doesn't fill the
small container root.

```bash
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1
```

Redirects the script's stdout through `tee` (log to file **and** to terminal), then points stderr
at the same place. Everything from here on is captured in `/workspace/act_training.log`. Useful
when the SSH session dies mid-run.

`set -euo pipefail` at the top: exit on error, exit on undefined variable, fail a pipeline if any
stage fails.

### 7.2 Block [1/3] — environment, only once

```bash
if [ ! -f "$VENV/bin/lerobot-train" ]; then ... fi
```

The guard makes the script idempotent: re-running it on a warm box skips the 10-minute install.

```bash
uv venv "$VENV" --python 3.12
uv pip install "lerobot[dataset,training] @ git+https://github.com/huggingface/lerobot.git@main"
```

- `uv` instead of pip because the dependency resolve is minutes faster on a rented GPU you pay for.
- `[dataset]` pulls the video decoding stack, `[training]` pulls `accelerate` + `wandb`.
- **`@main` means you get whatever lerobot HEAD is that day.** This is the single biggest
  reproducibility risk in the script — see §9. Pin a commit for anything you want to repeat.
- `ffmpeg` is an apt dependency for the h264 decoding.

The install log goes to a file and is only dumped on failure, keeping the console clean.

### 7.3 Block [2/3] — Hub auth, tag, download

```python
login(token=...)
api.create_tag(repo, tag="v3.0", repo_type="dataset")   # if absent
snapshot_download(repo, repo_type="dataset")
```

The `v3.0` tag exists because lerobot resolves a dataset's codebase version by looking for a
matching git tag on the repo. Your dataset is `codebase_version: v3.0` **(live)**; tagging the
repo `v3.0` makes the loader's version check pass instead of erroring or warning about a format
mismatch. It's a compatibility handshake, not a data change.

`snapshot_download` pulls the whole repo into `$HF_HOME/hub/datasets--witsense-ai--…/snapshots/<sha>/`
up front, so training doesn't stall on network I/O at step 1.

### 7.4 The symlink

```bash
ln -sf "$DATASET_CACHE/$SNAPSHOT" "$HF_HOME/lerobot/witsense-ai/so101_pick_and_place_ring_33"
```

lerobot looks for local datasets under `$HF_LEROBOT_HOME/<repo_id>` (defaults to
`$HF_HOME/lerobot/<repo_id>`). The HF hub cache stores things under a `datasets--org--name/snapshots/<sha>/`
path instead. The symlink bridges the two so `lerobot-train` finds the already-downloaded copy
and doesn't re-fetch. Pure plumbing.

### 7.5 Block [3/3] — the training command

```bash
lerobot-train \
  --dataset.repo_id=witsense-ai/so101_pick_and_place_ring_33 \
  --policy.type=act \
  --policy.repo_id=witsense-ai/so101_act_fewshot \
  --policy.device=cuda \
  --output_dir=/workspace/act_training \
  --job_name=act_so101_fewshot \
  --wandb.enable=false \
  --steps=50000 --batch_size=8 --num_workers=4 \
  --save_freq=2000 --log_freq=200 --eval_freq=5000
```

| Flag | Meaning | Note |
|---|---|---|
| `--dataset.repo_id` | Which data | Resolved via the symlink above |
| `--policy.type=act` | Build ACT from scratch with dataset-derived input/output shapes | Not fine-tuning; ACT has no pretrained checkpoint (only its ResNet18 backbone is ImageNet-pretrained) |
| `--policy.repo_id` | Where the **final** model gets pushed | `push_to_hub` defaults to **true**, so the run uploads at the end **(live)** |
| `--policy.device=cuda` | Falls back to CPU automatically if CUDA is absent | |
| `--output_dir` | Checkpoints + `train_config.json` land here | Errors if it already exists and `--resume` is not set |
| `--wandb.enable=false` | Logs to stdout/file only | |
| `--steps=50000` | **Optimizer steps**, not epochs | See the arithmetic below |
| `--batch_size=8` | Frames per step | 8 is also the lerobot default |
| `--num_workers=4` | Dataloader processes decoding video | Raise if `data_s` in the logs is large |
| `--save_freq=2000` | Checkpoint every 2000 steps | 25 checkpoints over the run |
| `--log_freq=200` | Metrics line every 200 steps | |
| `--eval_freq=5000` | ⚠️ **not a valid flag on current lerobot main** — see §9 | |

**Steps vs epochs.** 18,861 frames ÷ batch 8 ≈ **2,358 steps per pass over the data**. So
50,000 steps ≈ **21 epochs**. That is a reasonable amount for ACT on 33 episodes; the useful
checkpoints are usually somewhere between 10k and 40k, and you pick between them by testing on
the robot, not by loss.

---

## 8. What `lerobot-train` does internally

Order of operations **(live, from `lerobot/scripts/lerobot_train.py`)**:

1. **Parse & validate config.** All fail-fasts fire here, before anything expensive.
2. **Build the dataset.** Reads `meta/info.json`, builds the frame index, sets up lazy video decoding.
3. **Build the policy.** `--policy.type=act` with `input_features`/`output_features` inferred from
   the dataset: two image inputs at their respective shapes, a 6-dim state input, a 6-dim action output.
4. **Build the pre/post processors.** This is where `meta/stats.json` becomes normalisation —
   ACT uses **mean/std** for visual, state, and action **(live)**. Actions come out of the model
   normalised and get un-normalised by the postprocessor. This is why the stats file matters and
   why you can't mix a policy with a different dataset's stats.
5. **Build optimizer.** ACT's training preset: **AdamW, lr 1e-5, backbone lr 1e-5, weight decay 1e-4,
   no LR schedule** **(live)**.
6. **Build the dataloader** with `EpisodeAwareSampler` — samples frames while respecting episode
   boundaries, seeded so the order is reproducible and resumable.
7. **Loop `steps` times:**
   - fetch a batch → preprocess (normalise, to device)
   - forward: ResNet18 on each camera view → tokens; + state token + latent token → transformer
     encoder (4 layers) → decoder (1 layer) queries 100 action slots → 100 × 6 actions
   - loss = **L1(predicted chunk, ground-truth chunk) + 10 × KL** (the VAE term) **(live: `kl_weight=10.0`)**
   - backward, grad clip, AdamW step
   - every `log_freq`: print `loss`, `grad_norm`, `lr`, `data_s`, `updt_s`, `smp/s`, `mem_gb`
   - every `save_freq`: write `checkpoints/<step>/` (model + optimizer + RNG) and update `checkpoints/last`
8. **Push the final model** to `--policy.repo_id`.

### ACT's architecture defaults **(live, `configuration_act.py`)**

| Parameter | Default | What it does |
|---|---|---|
| `chunk_size` | 100 | Predicts 100 future actions (≈3.3 s at 30 fps) |
| `n_action_steps` | 100 | Executes all 100 before re-querying |
| `n_obs_steps` | 1 | Single frame in, no history |
| `vision_backbone` | `resnet18`, ImageNet weights | Per camera |
| `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 | |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 | 1 decoder layer matches the original ACT's effective behaviour |
| `use_vae` / `latent_dim` / `kl_weight` | true / 32 / 10.0 | The CVAE that lets ACT model multi-modal demos |
| `dropout` | 0.1 | |
| `temporal_ensemble_coeff` | `None` (off) | See below |

**Action chunking is the whole idea.** Predicting 100 steps at once and executing them open-loop
avoids the compounding-error and stuttering problems of per-step behaviour cloning. The cost is
reactivity: for 3.3 seconds the robot ignores what it sees. If the object moves mid-episode, it
will not adapt until the next chunk.

**Temporal ensembling** (`--policy.temporal_ensemble_coeff=0.01` with `--policy.n_action_steps=1`)
re-queries every step and blends overlapping chunks. Smoother and more reactive, but ~100× more
inference compute. Worth trying at eval time if the motion looks steppy.

### Reading the training log

```
step:2K smpl:16K ep:28 epch:0.87 loss:0.412 grdn:1.23 lr:1.0e-05 updt_s:0.081 data_s:0.002
```

- `loss` — L1 + KL. **Falling is necessary, not sufficient.** A very low loss on 33 episodes
  usually means memorisation.
- `grdn` — gradient norm. Spikes to 10× baseline mean bad frames or a bad batch.
- `data_s` — time waiting for the dataloader. If it's a meaningful fraction of `updt_s`, raise
  `--num_workers`.
- `epch` — fractional epochs completed.

---

## 9. Bugs and sharp edges in the current script

**1. `--eval_freq=5000` will kill the run.** On current lerobot `main` the field no longer exists;
it was replaced by `env_eval_freq` (simulator rollouts) and `eval_steps` (held-out loss).
lerobot's CLI parser calls draccus with strict parsing, which errors on unrecognised arguments
**(verified in `draccus/argparsing.py`: unparsed args → `parser.error`)**. Since the script pins
`@main`, the training command exits ~immediately with `unrecognized arguments: --eval_freq=5000`
after you've already paid for the install and the download.

Fix — either drop the flag, or ask for what you actually want (offline eval loss on held-out episodes):

```bash
  --dataset.eval_split=0.1 \
  --eval_steps=5000
```

Note that holding out 10% of 33 episodes is ~3 episodes; on a dataset this small the eval loss is
noisy and robot testing is still the real metric.

**2. `--eval_freq` never meant what it looks like anyway.** Even in older lerobot, periodic eval
required a simulation `--env`. With a real-robot dataset and no env, it did nothing.

**3. `@main` is unpinned.** The flag above broke exactly because of this. For anything you want to
reproduce:

```bash
"lerobot[dataset,training] @ git+https://github.com/huggingface/lerobot.git@<commit-sha>"
```

**4. `--output_dir=/workspace/act_training` must not exist.** lerobot raises `FileExistsError` if
it does and `--resume` isn't set **(live)**. Second run on the same box → immediate crash. Either
`rm -rf` it or timestamp the directory.

**5. The token is interpolated into a heredoc.** `login(token="$HF_TOKEN")` puts your token into
the Python source *and* into `/workspace/act_training.log` if anything echoes it. Prefer
`export HF_TOKEN` and let `huggingface_hub` read it from the environment.

**6. `apt-get install` without `apt-get update`** works on most GPU images and fails on stale ones.

**7. The final model is pushed automatically.** `push_to_hub` defaults to true and `--policy.repo_id`
is set, so the run publishes to `witsense-ai/so101_act_fewshot` at the end **(live)**. Add
`--policy.private=true` if that repo shouldn't be public. To also push intermediate checkpoints
(useful for a run you might lose), add `--save_checkpoint_to_hub=true`.

---

## 10. Resuming an interrupted run

```bash
lerobot-train --config_path=/workspace/act_training/checkpoints/last/pretrained_model/train_config.json --resume=true
```

The checkpoint's config wins over CLI flags, except flags you pass explicitly. The dataloader
resumes at the right epoch and sample offset, so the data order is preserved.

---

## 11. Evaluating on the robot

This is `scripts/eval_act_robot.sh`:

```bash
lerobot-rollout \
  --policy.path=witsense-ai/so101_act_fewshot \
  --robot.type=so101_follower --robot.port=$ROBOT_FOLLOWER_PORT --robot.id=left_follower \
  --robot.cameras="{top: {...}, wrist: {...}}" \
  --fps=30 \
  --dataset.repo_id=witsense-ai/rollout_eval_act \
  --dataset.single_task="pickup the ring and place it on the toy" \
  --dataset.num_episodes=10 --dataset.episode_time_s=25
```

Non-negotiable preconditions — the eval environment must match the recording environment:

- **Same calibration files** on the follower.
- **Same camera positions**, same names, and **the same resolutions the policy was trained on**
  (see the §5 warning: your dataset trained `wrist` at 1280×720, but `eval_act_robot.sh` opens it
  at 640×480 — reconcile these).
- Similar lighting and table layout.
- **Hand on the e-stop / power switch for the first rollout.** A freshly trained ACT policy can
  drive straight into the table.

Judge it by counting successes out of 10, per starting position. If it reaches correctly but
fails to grasp, that's usually a wrist-camera / grasp-consistency problem in the demos. If it
does nothing much, that's usually too few episodes or too-varied demos.

### Testing an intermediate checkpoint

```bash
bash scripts/eval_act_robot.sh /workspace/act_training/checkpoints/030000/pretrained_model
```

Test 20k / 30k / 40k / last. More steps is genuinely not always better on 33 episodes.

---

## 11b. Reproducing the baseline locally (measured, RTX 4050 6 GB)

`witsense-ai/so101_act_fewshot` is the existing model trained from this dataset. Its
`train_config.json` on the Hub is the ground truth for what to match:

| | Baseline | Local repro |
|---|---|---|
| steps / batch | 50,000 / 4 | same |
| lr / weight decay | 1e-5 / 1e-4 (ACT preset) | same |
| seed | 1000 | same |
| image_transforms | disabled | same |
| precision | fp32 | **bf16** (to fit 6 GB) |
| lerobot | older (its config still has `eval_freq`) | pinned `22bd7a2f` |
| GPU | RTX 4070 Laptop 8 GB | RTX 4050 Laptop 6 GB |
| final train loss | ~0.11 | — |

Run it with `scripts/train_act_local.sh`. Measured on this machine:

| batch | VRAM | s/step | 50k steps |
|---|---|---|---|
| 2 | 2.64 GB | 0.19 | ~2.6 h (but half the samples) |
| **4** | **4.37 GB** | **0.376** | **~5.2 h** |

`data_s` is ~0.002 s, so the dataloader is not the bottleneck and `num_workers=4` is ample.
ACT's memory is constant per step (fixed chunk size, fixed image dims), so a passing 60-step
smoke test predicts the whole run — no late OOM.

> The script sets `--policy.push_to_hub=false` on purpose. The original `train_act.sh` pushes
> to `witsense-ai/so101_act_fewshot`, which would **overwrite the baseline you are comparing
> against**. Push your own run to a new repo id, explicitly, once you're happy with it.

## 12. The whole thing as a checklist

```
[ ] lerobot-find-port          → note both ports
[ ] lerobot-find-cameras       → note both video nodes, confirm the views
[ ] lerobot-calibrate x2       → back up the calibration files
[ ] lerobot-teleoperate        → follower tracks cleanly, cameras look right
[ ] lerobot-record             → 30-50 consistent successful episodes, ← to redo bad ones
[ ] lerobot-dataset-viz        → watch several episodes before trusting them
[ ] push dataset to the Hub
[ ] train_act.sh               → fix --eval_freq first; ~21 epochs at 50k steps / batch 8
[ ] lerobot-rollout            → several checkpoints, hand on the e-stop, count successes
```

If the policy underperforms, the order of things that actually help:
**more/cleaner episodes** → **an earlier or later checkpoint** → **temporal ensembling at eval** →
hyperparameters. Hyperparameters are last for a reason.
