# witsense — simulation

Isaac Lab tasks for the SO-101 arm. Record demonstrations by teleoperation, train ACT on
them, evaluate, and use the policy's own rollouts as extra training data.

Start at [Setup](#setup). Once it runs, follow [the pipeline](#the-pipeline) top to bottom.

---

## Setup

### 1. Requirements

| | |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| Isaac Lab | installed editable |
| conda env | `env_isaaclab` |
| warp-lang | 1.8.1 |

Install Isaac Sim and Isaac Lab first, following
[NVIDIA's guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Then pin warp:

```bash
conda activate env_isaaclab
pip install --no-deps "warp-lang==1.8.1"
```

### 2. Install this package

```bash
cd so101-custom-training
pip install -e source/witsense
```

### 3. Install lerobot alongside Isaac Sim

Recording needs `lerobot`. Install exactly this, in this order:

```bash
pip install "lerobot==0.3.3"
pip install --no-deps "gymnasium==1.2.1" "packaging==23.0"
```

pip prints a conflict warning. Ignore it.

### 4. Download the assets

The scene, arm, ring and ghost are USD files kept out of git.

```bash
hf download witsense-ai/witsense_sim_assets --repo-type dataset --local-dir Assets
```

989 MB. Needs read access to `witsense-ai` — run `hf auth login` if it 401s.

### 5. Check it works

```bash
python -c "import witsense.utils.constant as c; print(c.ASSETS_ROOT)"
python scripts/sim_bedroom.py --task LeHome-SO101-Direct-RingInsert-v0 \
    --enable_cameras --device cpu
```

The first prints the `Assets` path. The second opens the scene with the arm, ring and
ghost on the table.

**Run everything from the repo root.** Asset paths are resolved relative to the working
directory.
---

## The task

`ring_insert` — a single SO-101 picks up the ring and places it around the ghost toy.

```
tasks/ring_insert/ring_insert_cfg.py    RingInsertEnvCfg — geometry, cameras, home pose
tasks/ring_insert/ring_insert.py        RingInsertEnv — reset, success, progress
```

Two measures per episode:

- **success** — ring centred on the ghost in xy and low enough. Binary.
- **progress** — 0 to 1. Roughly 0.5 for reaching the ghost, 0.9 for seated. This is what
  separates a near miss from an attempt that never touched the ring.

Registered task ids:

| id | |
|---|---|
| `LeHome-SO101-Direct-RingInsert-v0` | ring insert, one arm |
| `LeHome-BiSO101-Direct-Garment-v2` | bedroom, two arms, particle garment |

---

## The pipeline

```
1  record demos       dataset_sim record        -> Datasets/record/ring_insert/00N
2  check              check_dataset.py          <- must print "looks good"
3  convert to v3.0    convert_dataset_v30.py
4  merge              merge_datasets.py         -> merged_NN
5  push               push_dataset_to_hf.py
6  train              train_act.sh              -> outputs/train/<run>
7  evaluate           run_eval.py               -> success rate
8  collect rollouts   run_eval.py --record      -> outputs/rollouts_runN
9  filter             filter_rollouts.py
10 merge + retrain    back to step 4
```

Steps 1, 7 and 8 run in `env_isaaclab`. Steps 2–6 and 9 run in the training venv, written
`$V` below:

```bash
source .env
V="$LEROBOT_VENV"
```

### 1. Record demonstrations

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-RingInsert-v0 \
    --teleop_device gamepad --sensitivity 0.5 \
    --enable_record --disable_depth --num_episode 20 \
    --dataset_root Datasets/record/ring_insert \
    --task_description "place the ring around the ghost toy" \
    --enable_cameras --device cpu
```

Aim for 40–50 episodes. Record in batches — each run writes a new numbered directory,
and step 4 merges them.

Controls are in [Teleoperation](#teleoperation) below. Do the first episode without
`--enable_record` to check reach and framing.

Use `--teleop_device so101leader --port /dev/ttyACM0` to drive from a physical leader arm,
or `keyboard`.

> Do not add `--headless`. Cameras stop rendering and you get correct joint data with
> frozen video. The flag `--num_episode` is singular; the plural form is ignored.

### 2. Check the recording

```bash
$V/bin/python scripts/check_dataset.py Datasets/record/ring_insert/002
```

Must print `looks good`. Do not skip this — it catches datasets with frozen state or dead
video, which train without error and produce a policy that does nothing.

Add `--min-gripper-travel 0` when checking rollouts rather than demonstrations.

### 3. Convert to v3.0

```bash
$V/bin/python scripts/convert_dataset_v30.py Datasets/record/ring_insert/002
```

Converts in place; the original moves to `002_old`. Do not run it twice on one directory.

### 4. Merge

```bash
$V/bin/python scripts/merge_datasets.py \
    --out Datasets/record/ring_insert/merged_40 \
    --repo-id witsense-ai/synthetic_so101_ring_insert \
    Datasets/record/ring_insert/002 Datasets/record/ring_insert/003
```

### 5. Push

```bash
$V/bin/python scripts/push_dataset_to_hf.py \
    witsense-ai/synthetic_so101_ring_insert \
    Datasets/record/ring_insert/merged_40
```

Private by default; `--public` to change that.

### 6. Train

```bash
DATASET_REPO=witsense-ai/synthetic_so101_ring_insert bash scripts/train_act.sh smoke
DATASET_REPO=witsense-ai/synthetic_so101_ring_insert STEPS=50000 bash scripts/train_act.sh
```

Set `STEPS ≈ frames × 15 / batch_size` to land near 15 epochs. **Recheck this every time
the dataset grows** — a schedule that fitted 20k frames is far too short for 50k, and the
result looks like a bad dataset rather than a short run.

### 7. Evaluate

```bash
python -m scripts.run_eval \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --enable_cameras --device cpu --num_episodes 100 --max_steps 400
```

Use 100 episodes for any number you intend to compare. At 10 episodes the error bars are
wider than the differences you are looking for.

**Keep `--num_episodes` and `--max_steps` fixed across runs.** Raising the step cap turns
slow successes into successes on its own and invalidates the comparison.

Add `--debug` to write per-episode traces and camera frames to `outputs/sim_frames/`.

### 8. Collect rollouts

```bash
python -m scripts.run_eval \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --enable_cameras --device cpu --num_episodes 150 --max_steps 400 --record \
  --out outputs/rollouts_run2

$V/bin/python scripts/check_dataset.py outputs/rollouts_run2/dataset --min-gripper-travel 0
```

About 2.5 min per episode. Run under `tmux`, and do a 3-episode test first.

### 9. Filter

```bash
$V/bin/python scripts/convert_dataset_v30.py outputs/rollouts_run2/dataset
$V/bin/python scripts/filter_rollouts.py outputs/rollouts_run2 --min-progress 0.5
```

Keeps successes plus near-misses above the progress threshold.

### 10. Merge and retrain

```bash
$V/bin/python scripts/merge_datasets.py \
    --out Datasets/record/ring_insert/merged_v3 \
    --repo-id witsense-ai/synthetic_so101_ring_insert_v3 \
    Datasets/record/ring_insert/merged_v2 outputs/rollouts_run2/dataset_filtered
```

Merge into the **previous merged set**, not back into the original demonstrations —
otherwise earlier rollouts are dropped.

Then push (step 5), train (step 6), evaluate (step 7), and compare. Repeat from step 8.

---

## Results

Each round adds the previous policy's filtered rollouts to the training set.

| round | training data | frames | success | mean progress |
|---|---|---|---|---|
| 1 | 40 demos | 8,691 | 32 % | 0.43 |
| 2 | + 43 rollouts | 21,924 | 50 % | 0.52 |
| 3 | + 104 rollouts | 56,125 | — | — |

Measured over 100 episodes at `--max_steps 400`. Round 1 → 2 is a real gain (p = 0.01).

Round 2's failures, by how far they got:

| progress | count | |
|---|---|---|
| < 0.35 | 32 | never engaged the ring |
| 0.35–0.64 | 13 | grasped, then lost it |
| ≥ 0.64 | 5 | reached the ghost, did not seat |

The bottleneck is the grasp, not the insertion.

Expect diminishing returns — the dataset is increasingly the policy's own behaviour. When
a round stops helping, add fresh teleoperated demonstrations rather than more rollouts.

---

## Teleoperation

**Press start first** — `A` on the gamepad, `B` on the keyboard. Nothing moves until you do.

| | keyboard | gamepad |
|---|---|---|
| shoulder_pan | `T`/`G` | left stick ←→ |
| shoulder_lift | `Y`/`H` | left stick ↑↓ |
| elbow_flex | `U`/`J` | right stick ↑↓ |
| wrist_flex | `I`/`K` | right stick ←→ |
| wrist_roll | `O`/`L` | D-pad ←→ |
| gripper | `Q`/`A` | RT open / LT close |
| start control | `B` | `A` |
| start recording | `S` | `X` |
| discard episode | `D` | `B` |
| save as success | `N` | `Y` |

Movement is hold-to-move. Start with `--sensitivity 0.2` and raise it.

The joint keys only work while the **Isaac Sim window is focused**; the record keys work
from anywhere. So: click the Isaac Sim window, press `B`, then `S`, then drive.

---

## Troubleshooting

**`AttributeError: module 'warp.types' has no attribute 'array'`**
warp-lang is too new. `pip install --no-deps "warp-lang==1.8.1"`.

**Recorded video is frozen but joint data looks fine**
The run was headless. Cameras only render when a GUI or RTX sensor is active. Re-record
without `--headless`.

**`BackwardCompatibilityError` when training on a pushed dataset**
lerobot reads the format version from a git tag, not `info.json`.
`scripts/push_dataset_to_hf.py` sets it; a dataset pushed by hand may not have it.

**`merge_datasets` says features differ when they look identical**
It compares the whole feature dict, including `names`. `run_eval.py` and
`utils/dataset_record.py` must declare identical schemas — edit one, edit both.

**Trained policy does nothing**
Check the dataset with `check_dataset.py`, then check the step count was right for the
dataset size. Both failures look the same at eval.

**Dataset conversion refuses: metadata over 100 MB**
Depth was recorded. Use `--disable_depth`; ACT never reads it.

**A private dataset 401s as "Repository Not Found"**
`HF_HOME` points inside the repo, where there is no login token. Set `HF_TOKEN` in `.env`.

---

## Notes

`decimation = 3` puts the environment at 30 Hz, so one step is one dataset frame at the
declared `fps=30`.

Recording writes 6-dim `observation.state` and `action` with SO-101 joint names, plus
`observation.images.top_rgb` (480×640) and `observation.images.wrist_rgb` (720×1280).

The task id deliberately contains no `Bi` — `dataset_record.py` keys off that substring to
choose the single-arm schema.

`self.robot` is the attribute name `devices/action_process.py` reads for teleop. Renaming
it breaks teleop silently.

`TABLE_Z = 0.521` is measured from the table's world bounding box.

`LEHOME_*` environment variable names are unchanged from upstream, because the installed
package reads those names.
