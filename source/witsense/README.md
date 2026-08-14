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

### Geometry

Measured from the assets, not guessed:

| | |
|---|---|
| Table038 top | **z = 0.521** (x −0.522…0.468, y −0.400…0.400) |
| ring | 100 mm outside, 90 mm hole (5 mm wall), 24 mm tall, origin at **centre** → rests at 0.533 |
| ghost | 62 × 70 mm, 48 mm tall, origin at its **base** → rests at 0.521 |
| robot | lowest geometry is 30 mm above its origin → origin goes at 0.491 |

| field | value | why |
|---|---|---|
| `ring_pos` | `(0.16, -0.05, 0.533)` | 0.21 m from the base, on the table |
| `ghost_pos` | `(0.30, -0.03, 0.521)` | 0.23 m from the base, 0.14 m clear of the ring |
| `ring_xy_jitter` | `0.04` | worst-case spawn still 0.24 m from base, 0.10 m off the ghost |
| `ghost_kinematic` | `True` | pinned; a free ghost is knocked over every early attempt |
| `success_xy_tol` | `0.015` | slack around the 5 mm the geometry actually allows |
| `success_z_max` | `0.551` | ring centre is 0.533 down on the table, 0.581 perched on the ghost |

The insertion clearance is 10 mm per side: a 90 mm hole over a 70 mm ghost. Tight but
demonstrable. `roundtape.usda` is a generated ring — 64 angular segments, 4 points each,
ordered outer+z / outer−z / inner−z / inner+z. To resize it, regenerate `points`,
`extent`, `physics:mass` and `physics:diagonalInertia` together; mass follows the wall
volume at 948.5 kg/m³ and the inertia is a hollow cylinder,
`Ixx=Iyy=(1/12)m(3(ro²+ri²)+h²)`, `Izz=½m(ro²+ri²)`. Leaving the inertia stale makes the
ring tumble wrongly without any error.

The arm base is at `(0.23, -0.25)` rotated 180° about z, so it reaches out along **+y**
over both objects. The top camera keeps the bimanual task's rotation and moves only its
position, by the offset that recentres the same camera→workspace vector on the new
single-arm workspace — so the viewing angle is the one that setup was authored with.
| `table_texture` | `real_mat.png` | orange mat matching the real setup; `None` keeps the scene's white table |

The bedroom table ships white, which leaves a white ghost and a pale ring invisible on it.
`table_texture_id` indexes `Assets/textures/surface/<id>.png`, applied to the same shader
the bimanual task's randomiser targets. 76 is the darkest of the 100 — a near-flat neutral
grey, mean rgb 63/62/59 against the white table's ~255. 10 and 51 are warm wood tones.
Fixed, not randomised. If the ring still reads too pale, its own colour is one line:
`inputs:diffuseColor = (0.88, 0.74, 0.44)` in `Assets/objects/ring/roundtape.usda`.

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
10 merge + retrain         back to step 4                (filtered behaviour cloning)
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
$V/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
parts = [LeRobotDataset(repo_id=f"local/{n}", root=f"Datasets/record/ring_insert/{n}")
         for n in ("002", "003")]
merged = merge_datasets(parts, output_repo_id="witsense-ai/synthetic_so101_ring_insert",
                        output_dir="Datasets/record/ring_insert/merged_40")
print(merged.num_episodes, "episodes,", merged.num_frames, "frames")
PY
```

### 5. Push to the Hub

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
$V/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "witsense-ai/synthetic_so101_ring_insert"
api.upload_folder(folder_path="Datasets/record/ring_insert/merged_40",
                  repo_id=REPO, repo_type="dataset",
                  delete_patterns=["data/**", "videos/**", "meta/**"])
refs = api.list_repo_refs(REPO, repo_type="dataset")
main = refs.branches[0].target_commit
for t in refs.tags:
    if t.name == "v3.0" and t.target_commit != main:
        api.delete_tag(REPO, tag="v3.0", repo_type="dataset")
        api.create_tag(REPO, tag="v3.0", revision="main", repo_type="dataset")
PY
```

Two things bite here:

- **`delete_patterns`** — v3.0 chunk filenames do not overlap v2.1 ones, so a plain upload
  leaves both layouts in the repo and the loader trips over the old one.
- **The version tag.** lerobot reads the dataset format from a **git tag**, not from
  `meta/info.json`. A repo whose files are v3.0 but whose tag says `v2.1` still fails with
  `BackwardCompatibilityError`, and no amount of re-uploading fixes it.

`HF_TOKEN` is set explicitly because `.env` points `HF_HOME` at the dataset cache, a
directory with no token in it, so a perfectly good `hf auth login` goes unseen. A private
dataset then fails with a misleading `401 … Repository Not Found`.

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

### 9. Filter, then back to step 4

```bash
$V/bin/python scripts/convert_dataset_v30.py outputs/rollouts_run1/dataset
$V/bin/python scripts/filter_rollouts.py outputs/rollouts_run1 --min-progress 0.5
```

Keeps successes, plus near-misses above the progress threshold, using lerobot's
`delete_episodes`. Merge the result with the demonstrations (step 4) and retrain (step 6)
— that is filtered behaviour cloning, the simplest form of the flywheel. Re-evaluate and
compare success against the previous number; if it does not move, more machinery
(a success head, best-of-N, advantages) will not help either.

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
