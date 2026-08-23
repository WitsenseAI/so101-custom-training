#!/usr/bin/env python3
"""Run a trained SmolVLA policy in the ring_insert sim task and score the rollouts.

    python -m scripts.run_eval_smolvla \
        --checkpoint outputs/train/ring_insert_smolvla/checkpoints/last/pretrained_model \
        --enable_cameras --device cpu --num_episodes 100 --max_steps 400

Same protocol and summary.json as scripts/run_eval.py, so the numbers compare directly.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("LEHOME_DISABLE_KEYBOARD", "1")

from isaaclab.app import AppLauncher

DEFAULT_TASK = "LeHome-SO101-Direct-RingInsert-v0"
DEFAULT_CKPT = "outputs/train/ring_insert_smolvla/checkpoints/last/pretrained_model"

# Dominant instruction in the merged dataset (84% of v3 frames). Double space is verbatim.
TASK_STRING = "place the ring and put around the  ghost toy"

# Must match the --rename_map used for training.
CAMERA_MAP = {
    "observation.images.camera1": "observation.images.top_rgb",
    "observation.images.camera2": "observation.images.wrist_rgb",
}

# Config keys the newer trainer writes that lerobot 0.3.3's SmolVLAConfig rejects.
DROP_KEYS = ("use_peft", "pretrained_path", "pretrained_revision",
             "compile_mode", "compile_model", "rtc_config")

NORM_MODULES = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--task_description", default=TASK_STRING)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--out", default="outputs/rollouts_smolvla")
    p.add_argument("--record", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm_jitter_deg", type=float, default=None)
    return p


parser = build_parser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from safetensors.torch import load_file

import witsense.tasks  # noqa: F401  (registers the gym ids)
from scripts.utils.common import save_episode as save_episode_safe
from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- loading -----------------------------------------------------------------
def patched_config(ckpt: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="smolvla_ckpt_")) / "model"
    shutil.copytree(ckpt, tmp)
    cfg_path = tmp / "config.json"
    cfg = json.loads(cfg_path.read_text())
    for key in DROP_KEYS:
        cfg.pop(key, None)
    cfg_path.write_text(json.dumps(cfg))
    return tmp


def normalizer_file(ckpt: Path) -> Path:
    """SmolVLA's normalizer sits at a different pipeline step than ACT's, so glob it."""
    hits = sorted(ckpt.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one normalizer beside {ckpt}, found {hits}")
    return hits[0]


def feature_map(stats: dict) -> dict:
    names = {key.rsplit(".", 1)[0] for key in stats}
    return {f"buffer_{name.replace('.', '_')}": name for name in names}


def graft_module(mod, stats: dict, fmap: dict) -> tuple[int, list]:
    done, missing = 0, []
    for name, tensor in list(mod.named_parameters()) + list(mod.named_buffers()):
        head, stat = name.rsplit(".", 1)
        key = f"{fmap.get(head, '')}.{stat}"
        if key not in stats:
            missing.append(name)
            continue
        with torch.no_grad():
            tensor.copy_(stats[key].to(tensor.dtype).reshape(tensor.shape))
        done += 1
    return done, missing


def graft_stats(policy, stats_file: Path) -> int:
    """0.3.3 does not read the preprocessor safetensors; without this the policy is noise."""
    stats = load_file(str(stats_file))
    fmap = feature_map(stats)
    total, missing = 0, []
    for name in NORM_MODULES:
        mod = getattr(policy, name, None)
        if mod is None:
            continue
        done, miss = graft_module(mod, stats, fmap)
        total += done
        missing += [f"{name}.{m}" for m in miss]
    check_grafted(total, missing)
    return total


def check_grafted(total: int, missing: list) -> None:
    if missing:
        raise SystemExit(f"normalization stats not found for: {missing}")
    if total == 0:
        raise SystemExit("grafted no normalization stats — the policy would emit noise")


def assert_finite(policy) -> None:
    stale = [
        f"{name}.{n}"
        for name in NORM_MODULES
        if (mod := getattr(policy, name, None)) is not None
        for n, t in list(mod.named_parameters()) + list(mod.named_buffers())
        if t.numel() and not torch.isfinite(t).all()
    ]
    if stale:
        raise SystemExit(f"normalization stats still uninitialised: {stale}")


def load_policy(ckpt: Path):
    policy = SmolVLAPolicy.from_pretrained(str(patched_config(ckpt)))
    grafted = graft_stats(policy, normalizer_file(ckpt))
    assert_finite(policy)
    policy.to(DEVICE).eval()
    print(f"[smolvla] loaded {ckpt} (grafted {grafted} stats)", flush=True)
    return policy


def detect_units(stats_file: Path) -> str:
    """Real-robot checkpoints speak degrees, sim-trained ones radians."""
    stats = load_file(str(stats_file))
    keys = [f"observation.state.{k}" for k in ("min", "max", "mean")]
    span = max(float(stats[k].abs().max()) for k in keys if k in stats)
    return "deg" if span > 6.5 else "rad"


# --- observation / action ----------------------------------------------------
def image_key_map(policy, env_obs: dict) -> dict:
    """Unmatched cameras are dropped: smolvla_base declares three, this task has two."""
    mapping = {}
    for key, feat in policy.config.input_features.items():
        if feat.type != FeatureType.VISUAL:
            continue
        options = (CAMERA_MAP.get(key), key, f"{key}_rgb", key.replace("_rgb", ""))
        hit = next((c for c in options if c and c in env_obs), None)
        if hit is not None:
            mapping[key] = hit
    if not mapping:
        raise SystemExit(f"no policy camera matches {[k for k in env_obs if 'images' in k]}")
    return mapping


def image_tensor(img) -> torch.Tensor:
    """No resize: SmolVLA pads to 512x512 itself, so resizing here would do it twice."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    img = img[..., :3].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(DEVICE)


def to_policy_obs(env_obs: dict, keys: dict, units: str, task: str) -> dict:
    out = {pkey: image_tensor(env_obs[ekey]) for pkey, ekey in keys.items()}
    state = np.asarray(env_obs["observation.state"], dtype=np.float32).reshape(-1)
    if units == "deg":
        state = np.rad2deg(state)
    out["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
    out["task"] = task  # SmolVLA is language-conditioned; it raises without this
    return out


def to_env_action(action: torch.Tensor, units: str) -> np.ndarray:
    a = action.detach().cpu().numpy().reshape(-1)
    return (np.deg2rad(a) if units == "deg" else a).astype(np.float32)


# --- environment and recording -----------------------------------------------
def make_env(args):
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if args.arm_jitter_deg is not None and hasattr(cfg, "arm_start_jitter_deg"):
        cfg.arm_start_jitter_deg = args.arm_jitter_deg
    return gym.make(args.task, cfg=cfg).unwrapped, cfg


def video_feature(cam) -> dict:
    return {"dtype": "video", "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"]}


def make_dataset(out_dir: Path, env_cfg):
    """Schema must match utils/dataset_record.py exactly or merge_datasets refuses."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id="local/ring_insert_smolvla_rollouts", fps=30, root=str(out_dir / "dataset"),
        features={
            "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
            "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
            "observation.images.top_rgb": video_feature(env_cfg.top_camera),
            "observation.images.wrist_rgb": video_feature(env_cfg.wrist_camera),
        },
        use_videos=True)


def add_frame(dataset, obs: dict, action: np.ndarray, task: str) -> None:
    dataset.add_frame({
        "observation.state": np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1),
        "action": action,
        "observation.images.top_rgb": obs["observation.images.top_rgb"],
        "observation.images.wrist_rgb": obs["observation.images.wrist_rgb"],
    }, task)


# --- rollout -----------------------------------------------------------------
def step_once(env, policy, env_obs: dict, ctx: dict):
    obs = to_policy_obs(env_obs, ctx["keys"], ctx["units"], ctx["task"])
    with torch.inference_mode():
        action = policy.select_action(obs)
    env_action = to_env_action(action, ctx["units"])
    cmd = torch.from_numpy(env_action).unsqueeze(0).to(env.device)
    _, _, _, truncated, _ = env.step(cmd)
    env.render()  # headless never re-renders, and the policy then sees one stale frame
    if ctx["dataset"] is not None:
        add_frame(ctx["dataset"], env_obs, env_action, ctx["task"])
    return env._get_observations(), truncated


def episode_over(env, truncated) -> tuple[bool, bool]:
    success = bool(env._get_success().any())
    stop = bool(truncated.any() if torch.is_tensor(truncated) else truncated)
    return success, success or stop


def run_episode(env, policy, args, ctx: dict, seed: int) -> dict:
    env.reset(seed=seed)
    policy.reset()
    env_obs, success, steps, best = env._get_observations(), False, 0, 0.0
    for step in range(args.max_steps):
        env_obs, truncated = step_once(env, policy, env_obs, ctx)
        steps, best = step + 1, max(best, float(env.insertion_progress()[0]))
        success, stop = episode_over(env, truncated)
        if stop:
            break
    return {"success": success, "steps": steps, "max_progress": round(best, 4),
            "final_progress": round(float(env.insertion_progress()[0]), 4)}


def run_all(env, policy, args, ctx: dict, dataset) -> list:
    results = []
    for ep in range(args.num_episodes):
        row = run_episode(env, policy, args, ctx, args.seed + ep) | {"episode": ep}
        if dataset is not None:
            save_episode_safe(dataset)
        results.append(row)
        print(f"[ep {ep:03d}] success={row['success']} steps={row['steps']} "
              f"progress max={row['max_progress']:.2f} final={row['final_progress']:.2f}",
              flush=True)
    return results


def summarize(results: list, ckpt: Path, args, out_dir: Path) -> dict:
    n, ok = len(results), sum(r["success"] for r in results)
    prog = [r["max_progress"] for r in results]
    summary = {"checkpoint": str(ckpt), "policy": "smolvla", "task": args.task,
               "task_description": args.task_description,
               "mean_max_progress": round(float(np.mean(prog)), 4) if prog else 0.0,
               "episodes": n, "successes": ok, "success_rate": ok / n if n else 0.0,
               "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_context(env, policy, args, units: str, dataset) -> dict:
    env.reset(seed=args.seed)
    keys = image_key_map(policy, env._get_observations())
    unused = [k for k, f in policy.config.input_features.items()
              if f.type == FeatureType.VISUAL and k not in keys]
    print(f"[smolvla] cameras {keys}", flush=True)
    print(f"[smolvla] unused policy cameras {unused}", flush=True)
    print(f"[smolvla] units {units}  task {args.task_description!r}", flush=True)
    return {"keys": keys, "units": units, "dataset": dataset, "task": args.task_description}


def main() -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"not a policy directory (no config.json): {ckpt}")
    policy = load_policy(ckpt)
    env, env_cfg = make_env(args)
    dataset = make_dataset(out_dir, env_cfg) if args.record else None
    ctx = build_context(env, policy, args, detect_units(normalizer_file(ckpt)), dataset)
    summary = summarize(run_all(env, policy, args, ctx, dataset), ckpt, args, out_dir)
    report(summary, out_dir)
    env.close()


def report(summary: dict, out_dir: Path) -> None:
    print(f"\nsuccess rate: {summary['successes']}/{summary['episodes']} = "
          f"{summary['success_rate']:.1%}   "
          f"mean max progress: {summary['mean_max_progress']:.2f}")
    print(f"summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    sys.exit(exit_code)
