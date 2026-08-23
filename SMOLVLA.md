# SmolVLA on ring_insert

How to train SmolVLA on the same dataset as ACT and evaluate it in the same simulator, so
the two numbers are comparable.

Everything below was checked against the two environments actually installed here:
`.venv` (lerobot 0.6.2, training) and `env_isaaclab` (lerobot 0.3.3, simulation).

---

## What differs from ACT

| | ACT | SmolVLA |
|---|---|---|
| params | 52 M | ~450 M (SmolVLM2-500M + action expert) |
| needs a language instruction | no | **yes** — `batch["task"]` |
| chunk_size / n_action_steps | 100 / 100 | 50 / 50 |
| image preprocessing | none | resized to 512×512 with padding, scaled to [-1, 1] |
| normalizer file in checkpoint | `..._step_3_...` | `..._step_5_...` |
| extra config keys 0.3.3 rejects | 3 | 6 |

The language input is the change that matters. SmolVLA will not run without it, and
`run_eval.py` never builds one.

---

## 1. Training

The training venv already has SmolVLA — nothing to install.

```bash
source .env
DATASET=witsense-ai/synthetic_so101_ring_insert_v2   # the 187-episode set

.venv/bin/lerobot-train \
  --dataset.repo_id=$DATASET \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.top_rgb": "observation.images.camera1", "observation.images.wrist_rgb": "observation.images.camera2"}' \
  --policy.device=cuda \
  --output_dir=outputs/train/ring_insert_smolvla \
  --job_name=ring_insert_smolvla \
  --batch_size=2 \
  --steps=30000 \
  --save_freq=2000 \
  --log_freq=200 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

### `--rename_map` is required

`smolvla_base` was pretrained with cameras called `observation.images.camera1`, `camera2`,
`camera3`. This dataset calls them `top_rgb` and `wrist_rgb`. Without the rename, training
stops before the first step:

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.camera1', 'observation.images.camera2', ...]
- Extra features: ['observation.images.top_rgb', 'observation.images.wrist_rgb']
```

Mapping two cameras onto three is fine. `validate_visual_features_consistency` accepts
either set being a subset of the other, and `{camera1, camera2}` is a subset of the
policy's three. `prepare_images` then skips the absent `camera3` — it only substitutes
blank images up to `config.empty_cameras`, which defaults to 0.

Which camera becomes `camera1` is arbitrary but must stay fixed between training and
evaluation.

### Use `--policy.path`, not `--policy.type`

`--policy.type=smolvla` builds the action expert from random weights. `--policy.path=lerobot/smolvla_base`
starts from the pretrained VLA checkpoint. With 187 episodes you want the pretrained one —
that is the entire point of using a VLA here. The existing `scripts/train_smolvla.sh` uses
`--policy.type`, so change that line if you reuse the script.

### VRAM

This is the part most likely to stop you. On a 6 GB card, ACT ran at batch 4 using 4.4 GB.
SmolVLA is ~9× the parameters and processes two 512×512 images per sample.

Defaults already help: `freeze_vision_encoder=True` and `train_expert_only=True`, so the
VLM backbone is frozen and only the action expert trains.

Run a short one first and watch `nvidia-smi`:

```bash
.venv/bin/lerobot-train ... --steps=60 --save_freq=0 --output_dir=outputs/train/smolvla_smoke
```

If it OOMs, in order of what costs least:

1. `--batch_size=1`
2. `--policy.num_expert_layers=…` fewer expert layers
3. `--policy.resize_imgs_with_padding="(256,256)"` — quarter the vision tokens
4. Train on one camera (see `empty_cameras`)

If none of it fits, train on a rented GPU. Do not compare a crippled SmolVLA against a
full-size ACT and conclude anything.

### Steps

Fine-tuning from `smolvla_base` needs far fewer steps than ACT from scratch. 20k–30k at
batch 2 is a reasonable first run. Do not carry over ACT's `STEPS=140000`.

---

## 2. Evaluation — changes to `scripts/run_eval.py`

Six changes. The first four are required; the last two are correctness details that fail
silently if skipped.

### 2.1 Import the policy class

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
```

Confirmed present in lerobot 0.3.3 in `env_isaaclab`, and it imports cleanly.

### 2.2 Pick the class from the checkpoint

`load_act_policy()` hardcodes `ACTPolicy`. Read `config.json`'s `type` field instead:

```python
POLICY_CLASSES = {"act": ACTPolicy, "smolvla": SmolVLAPolicy}

cfg = json.loads((model_dir / "config.json").read_text())
policy_cls = POLICY_CLASSES[cfg["type"]]
...
policy = policy_cls.from_pretrained(str(model_dir))
```

### 2.3 Drop three more config keys

`load_act_policy()` drops `use_peft`, `pretrained_path`, `pretrained_revision`. SmolVLA
checkpoints written by 0.6.2 carry three more that 0.3.3's `SmolVLAConfig` rejects:

```python
DROP_KEYS = ("use_peft", "pretrained_path", "pretrained_revision",
             "compile_mode", "compile_model", "rtc_config")
```

That list is the exact set difference between the two versions' config dataclasses.

### 2.4 Find the normalizer file by glob, not by name

This line only ever matches ACT:

```python
stats_file = ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"
```

The step number is the normalizer's index in the preprocessing pipeline, and SmolVLA has
two extra steps before it (a newline processor and a tokenizer), so its file is
`policy_preprocessor_step_5_normalizer_processor.safetensors`. Glob instead:

```python
matches = sorted(ckpt.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
if len(matches) != 1:
    raise FileNotFoundError(f"expected exactly one normalizer beside {ckpt}, found {matches}")
stats_file = matches[0]
```

Keep the graft logic and the `isfinite` check exactly as they are — SmolVLA has the same
`normalize_inputs` / `unnormalize_outputs` modules, with the same uninitialised-parameter
trap. Without the graft the policy emits noise and does not raise.

### 2.5 Add the task string to the observation

This is the one that has no ACT equivalent. `SmolVLAPolicy.prepare_language()` reads
`batch["task"]` and raises `KeyError` without it.

In `to_policy_obs()`:

```python
out["task"] = task_description   # e.g. "place the ring around the ghost toy"
```

**Use the exact string the dataset was recorded with.** It is stored in the dataset, so
read it rather than retyping it:

```bash
.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset(repo_id='local/x', root='Datasets/record/ring_insert/merged_v2')
print(d.meta.tasks)"
```

A different wording at eval is a silent distribution shift — it will run, and score worse,
for a reason nothing reports.

`"task"` is a plain string in a dict of tensors. That is fine: `populate_queues` skips keys
it has no queue for, and `normalize_inputs` only touches declared features.

### 2.6 The camera rename follows you into evaluation

Because training used `--rename_map`, the trained checkpoint declares its inputs as
`observation.images.camera1` and `camera2`. The simulator emits `observation.images.top_rgb`
and `wrist_rgb`. `build_image_key_map()` only tries `key`, `key + "_rgb"` and
`key.replace("_rgb", "")`, none of which turn `camera1` into `top_rgb`, so it will exit
with `policy wants 'observation.images.camera1'; env emits [...]`.

Give it the same mapping you trained with:

```python
RENAME = {
    "observation.images.camera1": "observation.images.top_rgb",
    "observation.images.camera2": "observation.images.wrist_rgb",
}
```

and try `RENAME.get(key)` first inside the candidate loop. **The direction must match
training** — if `top_rgb` was `camera1` during training it must be `camera1` here. Swapping
them runs without error and destroys the policy, because the wrist view arrives where the
top view is expected.

### 2.7 Check the declared image shapes before the first eval

`to_policy_obs()` raises if the env renders a different resolution than the policy
declares. Confirm what the trained checkpoint actually recorded:

```bash
python -c "
import json; c = json.load(open('outputs/train/ring_insert_smolvla/checkpoints/last/pretrained_model/config.json'))
print({k: v['shape'] for k, v in c['input_features'].items()})"
```

`smolvla_base`'s own config declares 256×256, and the dataset's are 480×640 and 720×1280.
If the saved config kept 256×256 rather than adopting the dataset's, that guard will trip.
SmolVLA resizes internally to 512×512 regardless, so the declared shape is bookkeeping —
but decide deliberately whether to relax the check for SmolVLA or leave it strict. Do not
pre-resize the images yourself; that would resize twice.

---

## 3. Running it

```bash
conda activate env_isaaclab
python -m scripts.run_eval \
  --checkpoint outputs/train/ring_insert_smolvla/checkpoints/last/pretrained_model \
  --enable_cameras --device cpu --num_episodes 100 --max_steps 400
```

First run downloads the SmolVLM2 tokenizer from the Hub, so it needs network access once.

**Do not change `--num_episodes` or `--max_steps` from the ACT runs.** The comparison is
only meaningful at 100 episodes / 400 steps / no `--headless` / jitter 0.

Expect it to be slower per step than ACT — a 500M VLM on CPU is not fast. If it is
unusably slow, run with `--device cuda`, but then re-run the ACT baseline on `cuda` too so
both sides match.

---

## 4. Comparison

Fill this in as you go:

| policy | training data | frames | success (100 ep, 400 steps) | mean progress |
|---|---|---|---|---|
| ACT round 1 | 40 demos | 8,691 | 32 % | 0.43 |
| ACT round 2 | + 43 rollouts | 21,924 | 50 % | 0.52 |
| ACT round 3 | + 104 rollouts | 56,125 | | |
| SmolVLA | | | | |

Train SmolVLA on **the same dataset as one of the ACT rows**, not a different one.
Otherwise you are comparing two things at once and cannot attribute the difference.

A two-proportion z-test tells you whether a gap is real:

```python
import math
x1, x2, n = 50, 0, 100          # successes for each policy, out of n
p = (x1 + x2) / (2 * n)
z = (x2 - x1) / n / math.sqrt(p * (1 - p) * 2 / n)
print(z, math.erfc(abs(z) / math.sqrt(2)))
```

At 100 episodes per side, gaps below about 14 points are not distinguishable from noise.

---

## 5. Things that will waste your time

**SmolVLA scores near zero and the log looks fine.** Almost certainly the normalization
graft (2.4) or a missing/wrong task string (2.5). Both fail silently. Check the grafted
count printed at load, and print the task string once per run.

**`KeyError: 'task'`** — 2.5 is missing.

**`TypeError: __init__() got an unexpected keyword argument 'rtc_config'`** — 2.3 is missing.

**`FileNotFoundError: no normalization stats beside the checkpoint`** — 2.4 is missing.

**`ValueError: Feature mismatch between dataset/environment and policy config`** —
`--rename_map` is missing. See the training section.

**`policy wants 'observation.images.camera1'; env emits [...]`** — 2.6 is missing.

**OOM during training** — see the VRAM section. It is not a bug.

**Policy works in training-time eval but not in sim.** Check the image key mapping printed
by `build_image_key_map`, and confirm the env renders the resolutions the policy declares.

**`scripts/train_smolvla.sh` as it stands does not work here.** It sources
`~/ws/lerobot/.venv`, points at a dataset path that no longer exists, and uses
`--policy.type`. Treat it as a starting point, not a working script.
