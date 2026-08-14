# witsense — Isaac Lab extension

Bimanual SO-101 simulation tasks. Copied from
[`lehome-challenge/source/lehome`](https://github.com/IliaLarchenko/lehome-challenge/tree/5ea947ed83abf414180f4c503dbb31b9d6aa39f8/source/lehome)
and renamed `lehome` → `witsense`.

## Requirements

| | |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| Isaac Lab | `/home/zarus101/IsaacLab`, installed editable |
| conda env | `env_isaaclab` |
| warp-lang | **1.8.1** — see below |

`isaaclab` declares `warp-lang` with no upper bound, so a plain install pulls 1.13, which
moved `array` out of `warp.types`. The pip package shadows the copy Isaac Sim bundles
(`omni.warp.core-1.8.2`), and app startup dies with
`AttributeError: module 'warp.types' has no attribute 'array'` while loading
`isaaclab_assets`. Fix:

```bash
pip install --no-deps "warp-lang==1.8.1"
```

1.8.2, the bundled version, is an NVIDIA build and is not on PyPI; 1.8.1 has the attribute.

## Rename fixes applied to the copy

If you re-copy from upstream, these are the four things that break:

1. **Internal imports.** Every module still said `from lehome...`. Fixed with
   `sed -i 's/\blehome\b/witsense/g'` over `*.py` and `*.toml`.
2. **`config/extension.toml`.** `[[python.module]] name` was `lehome`.
3. **Typo'd package inits.** `witsense/assets/__inti__.py` and
   `witsense/devices/keyboard/__inti__.py` — renamed to `__init__.py`. The keyboard one
   also only exported `Se3Keyboard`, while `devices/__init__.py` imports `BiKeyboard` too.
4. **`witsense/assets/object/`** (`Garment.py`, `utils.py`) was missing from the copy;
   `tasks/bedroom/garment_bi_v2.py` needs `GarmentObject`.

One more, specific to this repo: `witsense/tasks/ring_insert/` is a copy of
`tasks/bedroom/` and registered the **same gym ids**. `witsense.tasks` imports both, so
the later one silently replaced the bedroom env. The ring-insert id is now
`LeHome-SO101-Direct-RingInsert-v0` — deliberately without `Bi`, see below.

## Install

```bash
conda activate env_isaaclab
pip install -e source/witsense
hf download witsense-ai/witsense_sim_assets --repo-type dataset --local-dir Assets
```

`setup.py` pulls in the runtime deps that Isaac Lab does not provide: `pyserial`,
`deepdiff`, `plotly`, `omegaconf`, `GitPython`, `scipy`, `tqdm`, `pynput`, `psutil`.

Check it, from the repo root:

```bash
python -c "import witsense.utils.constant as c; print(c.ASSETS_ROOT)"
```

Expect `/media/zarus101/ssd2/WITSENSE/so101-custom-training/Assets`. A `pxr` /
`ModuleNotFoundError` here is normal for anything under `witsense.tasks` — those modules
only import once Isaac Sim is running.

## Assets — download these first

Nothing runs without them: the scene, the arm, the ring and the ghost are all USD files,
and they are kept out of git (the root `.gitignore` excludes `**/*.usd*`; `Assets/.gitignore`
is `*`). They live in a private dataset repo in the org.

```bash
cd so101-custom-training
hf download witsense-ai/witsense_sim_assets --repo-type dataset --local-dir Assets
```

989 MB, 450 files. Needs read access to `witsense-ai` — `hf auth login` first if the
download 401s. Verify:

```bash
ls Assets/scenes/marble/Scene_00_Apartment.usd Assets/objects/ring/roundtape.usda
python -c "import witsense.utils.constant as c; print(c.ASSETS_ROOT)"
```

`utils/constant.py` sets `ASSETS_ROOT = <git root>/Assets`, so **run everything from the
repo root** — `garment_cfg_base_path` and `particle_cfg_path` in `GarmentEnvCfg` are
relative to the working directory too.

```
Assets/scenes/marble/Scene_00_Apartment.usd        the bedroom scene, both tasks
Assets/scenes/marble/Table038/                     the table; retextured at runtime
Assets/robots/lerobot/so101_follower_good.usd      the arm
Assets/objects/ring/roundtape.usda                 ring_insert — generated, not upstream
Assets/objects/ghost/ghost.usd                     ring_insert target
Assets/objects/Challenge_Garment/Release/          bedroom garment task only (266 MB)
Assets/textures/surface/real_mat.png               orange mat matching the real setup
```

If you re-upload that repo, two traps cost an hour here:

- **Do not upload `Assets/.gitignore`.** It is `*` plus `!.gitignore`, and the Hub applies
  a repo's `.gitignore` **server-side** — commits then succeed while adding no files, with
  no error. The symptom is `list_repo_commits` showing your commit and `list_repo_files`
  showing nothing.
- **Upload from outside the git tree.** `huggingface_hub` honours the enclosing repo's
  `.gitignore` too, and the root one excludes every `*.usd`. Stage with
  `cp -al Assets /somewhere/outside` (hardlinks, so no 946 MB copy) and upload that.

## Run the bedroom scene

```bash
python scripts/sim_bedroom.py --enable_cameras --device cpu
python scripts/sim_bedroom.py --enable_cameras --device cpu --headless --steps 20
```

Loads the scene, spawns both arms and the garment, then holds joint position for
`--steps`. Prints `[sim_bedroom] <n> steps OK` on success.

- `--device cpu` — upstream advises CPU for particle-cloth stability; `cuda` is worth
  testing but validate it before trusting a dataset recorded on it.
- `--enable_cameras` — the env has three `TiledCamera`s and constructs them on reset.
- `--garment_name Top_Long_Seen_0` — required by the env; there is no default garment,
  a missing one raises `FileNotFoundError` from `ChallengeGarmentLoader`.
- The script sets `LEHOME_DISABLE_KEYBOARD=1`, since `witsense.devices` grabs a `pynput`
  listener at import time and that needs a display server.

### Registered gym ids

| id | status |
|---|---|
| `LeHome-BiSO101-Direct-Garment-v2` | bedroom, two arms, particle garment |
| `LeHome-SO101-Direct-RingInsert-v0` | ring insert, one arm, rigid ring + ghost |
| `LeHome-BiSO101-Direct-Garment-v0`, `-fling-v0`, `LeHome-SO101-Direct-Garment-v0` | registered by `tasks/bedroom` but the modules were never copied; `gym.make` fails |

## ring_insert — the custom task

Single SO-101 picks up the ring and places it around the ghost toy. Sim counterpart of
the real `pick_and_place_ring` recordings.

```bash
python scripts/sim_bedroom.py --task LeHome-SO101-Direct-RingInsert-v0 \
    --enable_cameras --device cpu
```

```
tasks/ring_insert/ring_insert_cfg.py    RingInsertEnvCfg
tasks/ring_insert/ring_insert.py        RingInsertEnv
```

The garment copies (`garment_bi_v2.py`, `garment_bi_cfg_v2.py`,
`challenge_garment_loader.py`, `config_file/`) were deleted from this package — they were
unmodified copies of `tasks/bedroom/`, and a rigid-body task needs no cloth solver,
garment loader or particle config.

What differs from bedroom:

| | bedroom | ring_insert |
|---|---|---|
| arms | `left_arm` + `right_arm` | `robot` (one) |
| action / state | 12 | 6 |
| images | `top_rgb`, `left_rgb`, `right_rgb` | `top_rgb`, `wrist_rgb` |
| object | `GarmentObject` particle cloth | two `RigidObject`s |
| success | sleeve/pant fold checks | ring centred on ghost in xy, and low |

The arm keeps the pose the bimanual task gave its **right** arm, so the authored top-camera
offset still points where it did. `self.robot` is the attribute name
`devices/action_process.py` reads for single-arm keyboard teleop — renaming it breaks teleop
silently. The id deliberately has no `Bi` in it: `dataset_record.py` keys off that substring
to choose the 6-dim single-arm dataset schema over the 12-dim one.

## Pipeline: demonstrations → policy → rollouts

The whole loop, in order. Every step has a check, because several of these failed
silently and cost hours — the notes under each are the failures that actually happened,
not hypotheticals.

```
1 record teleop demos      dataset_sim record            -> Datasets/record/ring_insert/00N
2 check the recording      check_dataset.py              <- MUST print "looks good"
3 convert to v3.0          convert_dataset_v30.py        (in place; original -> 00N_old)
4 merge batches            merge_datasets (lerobot)      -> merged_NN
5 push                     HfApi.upload_folder           -> witsense-ai/<name>
6 train                    train_act.sh                  -> outputs/train/<run>
7 evaluate                 run_eval.py                   -> success rate + progress
8 collect rollouts         run_eval.py --record          -> outputs/rollouts_runN
9 filter                   filter_rollouts.py            -> keep successes / near-misses
10 merge + retrain         merge_datasets -> steps 5,6,7 (filtered behaviour cloning)
                           then back to step 8 for the next round
```

`$V` below is the ACT venv, `/media/zarus101/ssd2/WITSENSE/lerobot-venv`. Steps 1, 7 and 8
run in `env_isaaclab`; steps 2–5 and 9 run in `$V`, because the two envs hold different
lerobot versions (see "Installing lerobot next to Isaac Sim").

### 1. Record teleoperated demonstrations

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-RingInsert-v0 \
    --teleop_device gamepad --sensitivity 0.5 \
    --enable_record --disable_depth --num_episode 20 \
    --dataset_root Datasets/record/ring_insert \
    --task_description "place the ring around the ghost toy" \
    --enable_cameras --device cpu
```

- **`--num_episode`, singular.** `--num_episodes` (plural) exists on other subcommands and
  is silently ignored here, so you get the default 20 no matter what you pass.
- **`--disable_depth`.** Depth is stored as a raw 480×640 array, so lerobot computes
  per-element statistics for it and 20 episodes produce ~187 MB of episode metadata —
  past the 100 MB limit the v2.1→v3.0 converter refuses. ACT never reads depth.
- **Do not run headless.** The cameras only re-render inside `step()` when
  `has_gui()` or `has_rtx_sensors()` is true; headless they stay frozen at the reset
  frame and the recording gets correct joint data alongside dead video.
- Aim for 40–50 episodes total; the real dataset for this task has 33. Record in batches
  and merge (step 4) — the recorder writes a fresh numbered directory each run.

`--teleop_device so101leader --port /dev/ttyACM0` drives it from the physical leader arm
instead, and `keyboard` still works. The `bi-` variants are rejected by
`validate_task_and_device` because the task id has no `Bi` in it.

### 2. Check the recording — before anything else

```bash
$V/bin/python scripts/check_dataset.py Datasets/record/ring_insert/002
```

It must print `looks good`. This exists because a 20-episode dataset was recorded,
converted, pushed and trained on (2 h of GPU) before anyone noticed `observation.state`
never changed. It checks:

| check | what it caught |
|---|---|
| state varies per episode | `_get_observations` returned numpy **views** onto live sim buffers, so every frame aliased one array and held the final step's values |
| state correlates with action | 0.22 on the broken data, 1.00 on good data |
| gripper travel | a pick task where the gripper never moved |
| video motion | run_eval recorded perfect states beside footage frozen at the reset frame |

Pass `--min-gripper-travel 0` when checking *rollouts* — a policy that fails to grasp is
a legitimate rollout, not a broken recording.

### 3. Convert v2.1 → v3.0

```bash
$V/bin/python scripts/convert_dataset_v30.py Datasets/record/ring_insert/002
```

Isaac Sim pins lerobot 0.3.3 (newer needs numpy≥2, `isaacsim-kernel` pins numpy==1.26.0),
so the recorder writes v2.1 while training needs v3.0. lerobot's own converter fails three
ways here — a `datasets`/pandas dtype bug, an arrow write error on 2-D depth, and it reads
the version from the Hub rather than `--root`. This wraps it with those patched.

Converts **in place**: the v3.0 result takes the original path, the v2.1 original moves to
`002_old`. Do not run it twice on the same directory.

### 4. Merge batches

```bash
$V/bin/python scripts/merge_datasets.py \
    --out Datasets/record/ring_insert/merged_40 \
    --repo-id witsense-ai/synthetic_so101_ring_insert \
    Datasets/record/ring_insert/002 Datasets/record/ring_insert/003
```

Refuses to overwrite an existing `--out`, and checks the episode count adds up. Inputs
must already be v3.0 (step 3) and must share identical feature metadata — see the note at
the end of step 10.

### 5. Push to the Hub

```bash
$V/bin/python scripts/push_dataset_to_hf.py \
    witsense-ai/synthetic_so101_ring_insert \
    Datasets/record/ring_insert/merged_40
```

Private by default; `--public` to change that. Two things bite here, and the script
handles both:

- **Deleting the old layout.** v3.0 chunk filenames do not overlap v2.1 ones, so a plain
  upload leaves both in the repo and the loader trips over the old one. It uploads with
  `delete_patterns=["data/**", "videos/**", "meta/**"]`. `--keep-existing` skips that,
  which is only safe when the remote is already the same format version.
- **The version tag.** lerobot reads the dataset format from a **git tag**, not from
  `meta/info.json`. A repo whose files are v3.0 but whose tag says `v2.1` still fails with
  `BackwardCompatibilityError`, and no amount of re-uploading fixes it. The script tags
  whatever `info.json` says and moves a stale tag, leaving a correct one untouched.

Credentials come from `HF_TOKEN`, else `~/.cache/huggingface/token` — read explicitly
because `.env` points `HF_HOME` at the dataset cache, a directory with no token in it, so
a perfectly good `hf auth login` goes unseen and a private repo then fails with a
misleading `401 … Repository Not Found`.

### 6. Train

```bash
bash scripts/train_act.sh smoke      # 60 steps, confirms it reads the dataset
STEPS=50000 bash scripts/train_act.sh
```

~5 h for 50k steps on a 6 GB RTX 4050 at batch 4 (4.37 GB, 2.7 step/s). Keep
steps×batch ÷ frames near 10–25 epochs; 50k over 8.7k frames is 23.

### 7. Evaluate

```bash
python -m scripts.run_eval \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --enable_cameras --device cpu --num_episodes 20 --max_steps 400 --debug
```

`--debug` writes `trace_ep*.csv` (commanded vs measured joints, end effector, object
positions, progress) and camera frames to `outputs/sim_frames/`, beside the real dataset's
frames in `outputs/real_frames/` for comparison.

Reported per episode: `success` (binary, `_get_success`) and `progress` (continuous, see
`insertion_progress` — 0.5 for reaching the ghost, ~0.9 seated). Progress is what
separates a near miss from a rollout that never touched the ring.

Units and camera keys are read from the checkpoint, so a policy trained on the real robot
(degrees, `observation.images.top`) and one trained in sim (radians, `..._top_rgb`) both
run without flags.

### 8. Collect rollouts

```bash
python -m scripts.run_eval \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --enable_cameras --device cpu --num_episodes 100 --max_steps 400 --record \
  --out outputs/rollouts_run1

$V/bin/python scripts/check_dataset.py outputs/rollouts_run1/dataset --min-gripper-travel 0
```

~2.5 min/episode, so 100 ≈ 4 h. Chain the two with `&&` after a 3-episode verify run so a
dead-camera bug stops the batch instead of wasting the night. Run under `tmux`.

### 9. Filter

```bash
$V/bin/python scripts/convert_dataset_v30.py outputs/rollouts_run1/dataset
$V/bin/python scripts/filter_rollouts.py outputs/rollouts_run1 --min-progress 0.5
```

Keeps successes, plus near-misses above the progress threshold, using lerobot's
`delete_episodes`. 100 rollouts at 32% success kept 43 episodes / 13k frames.

### 10. Merge demos + rollouts, retrain, compare

```bash
$V/bin/python scripts/merge_datasets.py \
    --out Datasets/record/ring_insert/merged_v2 \
    --repo-id witsense-ai/synthetic_so101_ring_insert_v2 \
    Datasets/record/ring_insert/merged_40 outputs/rollouts_run1/dataset_filtered

$V/bin/python scripts/push_dataset_to_hf.py \
    witsense-ai/synthetic_so101_ring_insert_v2 \
    Datasets/record/ring_insert/merged_v2
```

Then retrain and re-evaluate:

```bash
DATASET_REPO=witsense-ai/synthetic_so101_ring_insert_v2 bash scripts/train_act.sh smoke
DATASET_REPO=witsense-ai/synthetic_so101_ring_insert_v2 STEPS=50000 bash scripts/train_act.sh

python -m scripts.run_eval \
  --checkpoint outputs/train/synthetic_so101_ring_insert_v2_act/checkpoints/last/pretrained_model \
  --enable_cameras --device cpu --num_episodes 100 --max_steps 400
```

**The comparison must be like-for-like** — same `--num_episodes`, same `--max_steps`, same
jitter, and not headless. A 10-episode run cannot distinguish 60% from the 32% baseline
(p ≈ 0.07); 100 episodes can. Changing the step cap between runs invalidates the number:
some successes land at 380–400 steps, so a cap raised to 450 converts failures into
successes on its own.

Push a checkpoint once it beats the previous one:

```bash
source .env && source "$LEROBOT_VENV/bin/activate"
python scripts/push_checkpoint_to_hf.py witsense-ai/synthetic_so101_ring_insert_v2_act \
    outputs/train/synthetic_so101_ring_insert_v2_act/checkpoints/last
```

Then round 2 is the same loop from step 8, recording with the new policy into
`outputs/rollouts_run2`.

#### Measured

| round | training data | success (100 ep, 400 steps) | mean max progress |
|---|---|---|---|
| 1 | 40 teleop demos | 32 % | 0.43 |
| 2 | 40 demos + 43 filtered rollouts | **50 %** | 0.52 |

z = 2.59, p = 0.0097 — a real gain, not sampling noise.

Round 2's 50 failures, by how far they got:

| progress | count | reading |
|---|---|---|
| < 0.35 | 32 | never engaged the ring — approach/grasp missed |
| 0.35–0.64 | 13 | grasped, then dropped or misaligned |
| ≥ 0.64 | 5 | reached the success band, did not seat |

The bottleneck is the **grasp**, not the insertion. `--debug` traces show
`closest approach to ring` at 0.067–0.083 m in every episode including the successes: the
approach is off-centre every time, and half the time it is off-centre enough to miss.
Timeouts are not the constraint — only 2 successes needed more than 350 of the 400 steps.

Expect diminishing returns: after round 2 the dataset is 40 human demos against ~120
policy episodes, so self-distillation is amplifying the policy's own habits. If a round
gains much less than the first did, add 10–15 fresh teleop demos of centred grasps rather
than more rollouts — that attacks the actual bottleneck and is cheaper than another round.

> Rollouts and demonstrations must carry **identical feature metadata** or
> `merge_datasets` refuses them — `validate_all_metadata` compares the whole dict, so
> `names=None` vs joint names, or `"channel"` vs `"channels"`, is enough to block a merge
> of otherwise identical data. `run_eval.py` and `utils/dataset_record.py` are kept in
> sync; if you edit the feature schema in one, edit both.

### Teleoperation controls

**Press start before anything else** — `B` on the keyboard, `A` on the gamepad.
`Device.advance()` returns `None` until it sees it, and the record loop feeds a
hold-position action instead, so the arm sits still whatever else you press.

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
| success + save | `N` | `Y` |

The gamepad's vertical stick axes are **inverted on purpose**: a positive shoulder_lift or
elbow_flex command lowers the gripper on this asset (measured — `+0.05` moved the jaw from
z 0.617 to 0.608), and since the action is `current_position + delta`, binding stick-up to
positive made the arm sink for as long as the stick was held.

The two key sets go through different input systems, which is why some keys work from the
terminal and some do not:

| keys | system | window focus |
|---|---|---|
| `B` start control, `S` record, `D` discard, `N` save, `ESC` abort | pynput | global |
| `T`/`G` `Y`/`H` `U`/`J` `I`/`K` `O`/`L` pan…roll, `Q`/`A` gripper | carb | **Isaac Sim window must be focused** |

So: click the Isaac Sim window, `B`, then `S`, then drive.

Movement is hold-to-move: `_delta_pos` accumulates on key-press and subtracts on
key-release, and each env step adds it to the current joint position. Default sensitivity
is 0.25 rad per step, which is very fast at 90 Hz — start with `--sensitivity 0.2` and
raise it. If a joint runs away because a key-release was missed (window focus changed
while a key was held), the stuck delta clears on the next `teleop_interface.reset()`.

> `lerobot.add_frame` takes the task as an argument and rejects frame keys that are not
> declared features. The upstream harness passed it inside the frame, which fails with
> `add_frame() missing 1 required positional argument: 'task'` the moment you press `S`.
> Patched in `dataset_record.py`, `dataset_replay.py` and `evaluation.py`.

`decimation = 3` over the 1/90 s physics step puts the env at **30 Hz**, so one env step is
one dataset frame at the declared `fps=30`. The garment task uses `decimation = 1`, i.e.
90 Hz, and relies on episodes being "treated as 30 Hz for dataset purposes regardless of
physics_hz" (see `apply_camera_overrides`). That fudge makes a recording play back at a
third speed and makes a 30 Hz action chunk execute three times too fast at eval.

Recording writes the single-arm schema: `observation.state` and `action` are 6-dim with
the SO-101 joint names, images are `observation.images.top_rgb` and
`observation.images.wrist_rgb` at 480×640. Add `--disable_depth` to drop
`observation.top_depth` — the top camera then renders RGB only, which is cheaper and is
what the real-robot ACT recordings look like.

Do the first episode without `--enable_record` to check reach and framing. `ring_pos`,
`ghost_pos` and `ring_xy_jitter` are guesses; an unreachable ring or an off-frame ghost
is much cheaper to find now than after fifty episodes.

`TABLE_Z = 0.521` is measured from Table038's world bounding box, not inherited from the
bimanual task — see "Geometry" above.

## Record / replay in sim

`scripts/dataset_sim.py` + `scripts/utils/` are the upstream teleop harness. Same rename
fixes were applied there:

- `scripts/utils/parser.py` — `--particle_cfg_path` defaulted to
  `source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml`
- `scripts/utils/common.py`, `process_parquet_to_pc.py` — `from lehome.utils...` imports
- `scripts/utils/evaluation.py` — a partial rename had left `witsense_CAMERA_*` and
  `witsense_WORKER_LABEL` in place of the `LEHOME_*` names the code actually reads

`LEHOME_*` environment variable names are left as-is throughout, because the installed
`witsense` package reads those same names (`LEHOME_DISABLE_KEYBOARD`,
`LEHOME_CHECK_INTERVAL`).

### Installing lerobot next to Isaac Sim

Recording needs `lerobot` for `LeRobotDataset`. **Install exactly this, in this order:**

```bash
pip install "lerobot==0.3.3"
pip install --no-deps "gymnasium==1.2.1" "packaging==23.0"
```

`lerobot>=0.4` cannot coexist with Isaac Sim 5.1: its dependency chain needs `numpy>=2`,
and `isaacsim-kernel` pins `numpy==1.26.0`. Asking pip for a newer lerobot under a
`numpy<2` constraint does not fail — it silently walks lerobot back to 0.3.3 anyway and,
on the way, downgrades `gymnasium` to 0.29.1 (isaaclab pins `==1.2.1`) and bumps
`packaging` to 26.x (isaacsim-core pins `==23.0`), leaving Isaac Lab broken. The second
command puts those two back; `--no-deps` stops pip re-resolving and undoing it.

lerobot 0.3.3 declares `gymnasium<1.0` and `packaging>=24.2`, so pip prints a conflict
warning about both. Ignore it — Isaac Lab's pins win, and the recording path only uses
`LeRobotDataset`, which does not touch either.

The one thing 0.3.3 lacks is `lerobot.datasets.dataset_tools`, needed by
`scripts/utils/dataset_processing.py` (dataset merge/augment). `scripts/utils/__init__.py`
no longer imports that module eagerly, so `record` and `replay` work without it — the same
lazy-import treatment the file already gives `evaluation`, `dataset_record` and
`dataset_replay`. To merge datasets, do it in the separate lerobot venv used for ACT
training, not here.

```bash
python -m scripts.dataset_sim record --help
python -m scripts.dataset_sim replay --dataset_root Datasets/record/001 \
    --num_replays 1 --disable_depth --enable_cameras --device cpu
```

Recording defaults to `Datasets/record`, teleop device `bi-so101leader` or `bi-keyboard`
(the `Bi` in the task id must match — see `validate_task_and_device`). Keyboard teleop
needs a display, so drop the `LEHOME_DISABLE_KEYBOARD=1` that `sim_bedroom.py` sets.

> Steps above are written from the install as configured; run
> `scripts/sim_bedroom.py` to confirm the scene loads on this machine.
