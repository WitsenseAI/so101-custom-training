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
the later one silently replaced the bedroom env. Ring-insert ids are now
`LeHome-BiSO101-Direct-RingInsert-*`.

## Install

```bash
conda activate env_isaaclab
pip install -e source/witsense
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

## Assets

`utils/constant.py` sets `ASSETS_ROOT = <git root>/Assets`, so **run everything from the
repo root** — `garment_cfg_base_path` and `particle_cfg_path` in `GarmentEnvCfg` are
relative to the working directory too.

```
Assets/scenes/marble/Scene_00_Apartment.usd        the bedroom
Assets/objects/Challenge_Garment/Release/          Top_Long_*, Top_Short_*, Pant_*
Assets/robots/lerobot/so101_follower.usd
```

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
| ring | 100 mm outside, 80 mm hole, 24 mm tall, origin at **centre** → rests at 0.533 |
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

> **The insertion clearance is 5 mm per side.** The ring's hole is 80 mm and the ghost is
> 70 mm across. That is a demanding teleoperated insertion — if it proves too fiddly to
> demonstrate consistently, scale the ghost down or swap in a wider ring before recording
> fifty episodes of near-misses.

The arm base is at `(0.23, -0.25)` rotated 180° about z, so it reaches out along **+y**
over both objects. The top camera keeps the bimanual task's rotation and moves only its
position, by the offset that recentres the same camera→workspace vector on the new
single-arm workspace — so the viewing angle is the one that setup was authored with.
| `table_texture_id` | `76` | dark table top; `None` keeps the scene's white material |

The bedroom table ships white, which leaves a white ghost and a pale ring invisible on it.
`table_texture_id` indexes `Assets/textures/surface/<id>.png`, applied to the same shader
the bimanual task's randomiser targets. 76 is the darkest of the 100 — a near-flat neutral
grey, mean rgb 63/62/59 against the white table's ~255. 10 and 51 are warm wood tones.
Fixed, not randomised. If the ring still reads too pale, its own colour is one line:
`inputs:diffuseColor = (0.88, 0.74, 0.44)` in `Assets/objects/ring/roundtape.usda`.

### Teleoperate and record

Install lerobot first — see "Installing lerobot next to Isaac Sim" above; the version and
order matter.

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-RingInsert-v0 \
    --teleop_device keyboard \
    --enable_record \
    --dataset_root Datasets/record/ring_insert \
    --task_description "place the ring around the ghost toy" \
    --enable_cameras --device cpu
```

`--teleop_device so101leader --port /dev/ttyACM0` drives it from the physical leader arm
instead. Both single-arm choices are correct here; the `bi-` variants are rejected by
`validate_task_and_device` because the task id has no `Bi` in it.

**Press `B` before anything else.** `Device.advance()` returns `None` until it sees `B`
(`if not action["started"]: return None`), and the record loop then feeds a hold-position
action instead — the arm sits still no matter which movement key you press.

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

Recording writes the single-arm schema: `observation.state` and `action` are 6-dim with
the SO-101 joint names, images are `observation.images.top_rgb` and
`observation.images.wrist_rgb` at 480×640. Add `--disable_depth` to drop
`observation.top_depth` — the top camera then renders RGB only, which is cheaper and is
what the real-robot ACT recordings look like.

Do the first episode without `--enable_record` to check reach and framing. `ring_pos`,
`ghost_pos` and `ring_xy_jitter` are guesses; an unreachable ring or an off-frame ghost
is much cheaper to find now than after fifty episodes.

`TABLE_Z = 0.5` comes from the bimanual task: both arm bases sit at that height and the
garment settles onto the same surface.

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
