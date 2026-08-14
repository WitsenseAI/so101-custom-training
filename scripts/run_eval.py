#!/usr/bin/env python3
"""Run a trained ACT policy in the ring_insert sim task and score the rollouts.

    python -m scripts.run_eval --enable_cameras --device cpu --num_episodes 10

Defaults point at the local 50k checkpoint and the registered single-arm task, so
usually there is nothing to pass but --num_episodes.

NOTE ON WHAT THIS MEASURES. The checkpoint under outputs/train/act_33_local was trained
on REAL SO-101 camera frames. Running it here is a sim-to-real transfer test across a
large visual gap — different table, lighting, renderer, and a wrist camera at a different
resolution. A low success rate is the expected result and says nothing about the
checkpoint's quality on the real robot. It is a plumbing check and a baseline number.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

# Nothing here teleoperates, and witsense.devices grabs a pynput listener on import.
os.environ.setdefault("LEHOME_DISABLE_KEYBOARD", "1")

from isaaclab.app import AppLauncher

DEFAULT_CHECKPOINT = "outputs/train/act_33_local/checkpoints/last/pretrained_model"
DEFAULT_TASK = "LeHome-SO101-Direct-RingInsert-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=600, help="Timeout per episode, in env steps")
parser.add_argument("--out", type=str, default="outputs/rollouts")
parser.add_argument("--record", action="store_true", help="Also write a LeRobot dataset")
parser.add_argument("--debug", action="store_true",
                    help="Write per-step commanded/measured joints and object positions to trace_ep*.csv")
parser.add_argument("--frames_dir", type=str, default="outputs/sim_frames",
                    help="Where --debug writes sim_top.png / sim_wrist.png, to sit beside outputs/real_frames")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--arm_jitter_deg", type=float, default=None,
                    help="Override RingInsertEnvCfg.arm_start_jitter_deg. Pass 0 to start every "
                         "episode from the exact home pose, which is how the demonstrations were "
                         "recorded — the fair comparison for a policy trained on them.")
parser.add_argument("--start_pose_deg", type=float, nargs=6, default=None,
                    help="Joint pose (deg) to begin each episode from. The env's home pose is "
                         "not necessarily where the training episodes start — recording begins "
                         "wherever the operator left the arm — and starting a BC policy outside "
                         "its training distribution is not a fair test of it.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True  # the policy is vision-based

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from safetensors.torch import load_file

import witsense.tasks  # noqa: F401  (registers the gym ids)
# Datasets here live on a fuseblk mount, where lerobot's save_episode() races its own
# async image writer and FUSE's lazy unlinks and dies with [Errno 39] Directory not
# empty. Same wrapper the teleop recorder uses: drain the writer, retry the rmdir.
from scripts.utils.common import save_episode as save_episode_safe
from lerobot.configs.types import FeatureType
from lerobot.policies.act.modeling_act import ACTPolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Policy feature key -> key in this env's observation dict. The env suffixes its camera
# keys with _rgb; the policy was trained without the suffix.
def build_image_key_map(policy, env_obs: dict) -> dict:
    """Map each of the policy's visual inputs onto a key this env actually emits.

    The real dataset names its cameras observation.images.top / .wrist; the sim recorder
    appends _rgb. A policy trained on either has to run here, so match by name and fall
    back to the _rgb variant rather than assuming one convention.
    """
    mapping = {}
    for key, feat in policy.config.input_features.items():
        if feat.type != FeatureType.VISUAL:
            continue
        for candidate in (key, f"{key}_rgb", key.replace("_rgb", "")):
            if candidate in env_obs:
                mapping[key] = candidate
                break
        else:
            raise SystemExit(
                f"policy wants {key!r}; env emits "
                f"{[k for k in env_obs if 'images' in k]}"
            )
    return mapping


def detect_state_units(ckpt: Path) -> str:
    """Whether the policy speaks degrees or radians, read off its own statistics.

    The real SO-101 reports degrees (state spans -45..+66); Isaac Lab joints are radians
    (-1.99..+1.73). Every joint on this arm is limited well inside +/-2*pi rad, so a span
    beyond that can only be degrees. Guessing wrong feeds the policy values ~57x off with
    no error anywhere.
    """
    stats = load_file(str(ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    span = max(
        float(stats["observation.state.min"].abs().max()),
        float(stats["observation.state.max"].abs().max()),
    )
    return "deg" if span > 6.5 else "rad"

# Verbatim from the training dataset, double space included.
TASK_STRING = "place the ring and put around the  ghost toy"


# ---------------------------------------------------------------------------
# Loading a newer-lerobot checkpoint under the 0.3.3 that fits next to Isaac Sim
# ---------------------------------------------------------------------------
def load_act_policy(ckpt: Path) -> ACTPolicy:
    """Load the checkpoint and restore its normalization stats.

    Two incompatibilities with the lerobot version pinned here (0.3.3 — newer releases
    need numpy>=2, which isaacsim-kernel forbids):

    1. ACTConfig rejects `use_peft` / `pretrained_path` / `pretrained_revision`, which the
       newer trainer writes. Dropped from a scratch copy of config.json.
    2. Normalization stats live in policy_preprocessor_*.safetensors, which 0.3.3 does not
       read — it expects them as buffers inside the model. Without them every
       normalize/unnormalize buffer stays uninitialised and the policy emits noise, with
       no error. They are copied across below.
    """
    tmp = Path(tempfile.mkdtemp(prefix="act_ckpt_"))
    shutil.copytree(ckpt, tmp / "model")
    model_dir = tmp / "model"

    cfg = json.loads((model_dir / "config.json").read_text())
    dropped = [k for k in ("use_peft", "pretrained_path", "pretrained_revision") if k in cfg]
    for k in dropped:
        cfg.pop(k)
    (model_dir / "config.json").write_text(json.dumps(cfg))

    policy = ACTPolicy.from_pretrained(str(model_dir))

    stats_file = ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if not stats_file.is_file():
        raise FileNotFoundError(f"no normalization stats beside the checkpoint: {stats_file}")
    stats = load_file(str(stats_file))

    # lerobot names each buffer by replacing '.' with '_' in the feature name, which is
    # not reversible: observation.images.top_rgb and observation.images.top.rgb both
    # become buffer_observation_images_top_rgb. Build the map forwards, from the feature
    # names the stats file actually contains, instead of guessing backwards.
    features = {key.rsplit(".", 1)[0] for key in stats}
    buffer_to_feature = {f"buffer_{name.replace('.', '_')}": name for name in features}

    grafted, missing = 0, []
    for mod_name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        mod = getattr(policy, mod_name, None)
        if mod is None:
            continue
        # lerobot holds these as nn.Parameter initialised to inf, not as buffers, so
        # named_buffers() alone silently iterates nothing and leaves the policy noise-only.
        entries = list(mod.named_parameters()) + list(mod.named_buffers())
        for name, tensor in entries:
            head, stat = name.rsplit(".", 1)
            key = f"{buffer_to_feature.get(head, '')}.{stat}"
            if key in stats:
                with torch.no_grad():
                    tensor.copy_(stats[key].to(tensor.dtype).reshape(tensor.shape))
                grafted += 1
            else:
                missing.append(f"{mod_name}.{name}")

    if missing:
        raise RuntimeError(f"normalization stats not found for: {missing}")
    if grafted == 0:
        raise RuntimeError("grafted no normalization stats — the policy would emit noise")

    # Anything still infinite means a stat did not land where the policy reads it.
    stale = [
        f"{m}.{n}"
        for m in ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
        if (mod := getattr(policy, m, None)) is not None
        for n, t in list(mod.named_parameters()) + list(mod.named_buffers())
        if t.numel() and not torch.isfinite(t).all()
    ]
    if stale:
        raise RuntimeError(f"normalization stats still uninitialised: {stale}")

    print(f"[run_eval] loaded {ckpt} (dropped {dropped}, grafted {grafted} stats)")

    policy.to(DEVICE).eval()
    return policy


# ---------------------------------------------------------------------------
# Observation / action adapters
# ---------------------------------------------------------------------------
def to_policy_obs(env_obs: dict, policy_shapes: dict, image_keys: dict, units: str) -> dict:
    """One env observation -> the batched tensors ACT expects."""
    out = {}
    for pkey, ekey in image_keys.items():
        img = env_obs[ekey]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:  # (num_envs, H, W, C)
            img = img[0]
        if img.shape[-1] == 4:  # drop alpha
            img = img[..., :3]

        # RingInsertEnvCfg renders at the policy's own resolutions, so this should never
        # fire. Resizing here would silently change the framing the policy was trained on
        # (and 640x480 -> 1280x720 stretches 4:3 into 16:9), so a mismatch is an error,
        # not something to paper over — fix the camera in the task config instead.
        _, want_h, want_w = policy_shapes[pkey]
        if img.shape[:2] != (want_h, want_w):
            raise SystemExit(
                f"{pkey}: env renders {img.shape[1]}x{img.shape[0]}, policy wants "
                f"{want_w}x{want_h}. Set the matching camera in RingInsertEnvCfg."
            )

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        out[pkey] = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(DEVICE)

    state = np.asarray(env_obs["observation.state"], dtype=np.float32).reshape(-1)
    # Isaac Lab joints are radians. Only convert for a policy trained on the real robot's
    # degrees; a sim-trained policy already speaks radians and must be left alone.
    if units == "deg":
        state = np.rad2deg(state)
    out["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
    return out


def to_env_action(action: torch.Tensor, units: str) -> np.ndarray:
    """Policy action -> env joint position targets, always radians."""
    a = action.detach().cpu().numpy().reshape(-1)
    return (np.deg2rad(a) if units == "deg" else a).astype(np.float32)


def save_frames(env_obs: dict, frames_dir: Path) -> None:
    """Dump the sim camera frames next to the real ones, for side-by-side comparison.

    Writes outputs/sim_frames/sim_{top,wrist}.png to mirror outputs/real_frames/. Later
    episodes overwrite earlier ones — this is for checking what the scene looks like, not
    for archiving every episode. The real counterparts came from the cached dataset:

        ffmpeg -i <dataset>/videos/observation.images.top/chunk-000/file-000.mp4 \
               -vf "select=eq(n\\,0)" -vframes 1 outputs/real_frames/real_top.png
    """
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    for ekey, tag in (("observation.images.top_rgb", "top"),
                      ("observation.images.wrist_rgb", "wrist")):
        img = np.asarray(env_obs[ekey])
        if img.ndim == 4:
            img = img[0]
        img = img[..., :3].astype(np.uint8)
        path = frames_dir / f"sim_{tag}.png"
        cv2.imwrite(str(path), img[:, :, ::-1])  # RGB -> BGR
        r, g, b = img.reshape(-1, 3).mean(0)
        print(f"[run_eval] {path}: {img.shape[1]}x{img.shape[0]} "
              f"mean RGB=({r:.0f},{g:.0f},{b:.0f})", flush=True)


def gripper_body_index(robot) -> int:
    """Index of the body to treat as the end effector, for the debug trace."""
    names = list(robot.body_names)
    for want in ("jaw", "gripper", "wrist"):
        for i, n in enumerate(names):
            if want in n.lower():
                return i
    return len(names) - 1


def trace_row(env, gi: int, step: int, action_deg, state_deg) -> dict:
    """One row of the debug trace: what was commanded, what moved, where things are."""
    ee = env.robot.data.body_pos_w[0, gi].tolist()
    ring = env.ring.data.root_pos_w[0].tolist()
    ghost = env.ghost.data.root_pos_w[0].tolist()
    row = {"step": step}
    row |= {f"cmd_{i}": round(float(v), 2) for i, v in enumerate(action_deg)}
    row |= {f"meas_{i}": round(float(v), 2) for i, v in enumerate(state_deg)}
    row |= {"progress": round(float(env.insertion_progress()[0]), 4)}
    row |= {"ee_x": round(ee[0], 4), "ee_y": round(ee[1], 4), "ee_z": round(ee[2], 4)}
    row |= {"ring_x": round(ring[0], 4), "ring_y": round(ring[1], 4), "ring_z": round(ring[2], 4)}
    row |= {"ghost_x": round(ghost[0], 4), "ghost_y": round(ghost[1], 4), "ghost_z": round(ghost[2], 4)}
    return row


def main() -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"not a policy directory (no config.json): {ckpt}")
    policy = load_act_policy(ckpt)

    policy_shapes = {k: tuple(v.shape) for k, v in policy.config.input_features.items()}
    units = detect_state_units(ckpt)
    print(f"[run_eval] policy state/action units: {units}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if args.arm_jitter_deg is not None and hasattr(env_cfg, "arm_start_jitter_deg"):
        env_cfg.arm_start_jitter_deg = args.arm_jitter_deg
        print(f"[run_eval] arm start jitter: +/-{args.arm_jitter_deg} deg", flush=True)
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    dataset = None
    if args.record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Same schema the teleop harness writes, so rollouts and demonstrations merge.
        # Image shapes come from the cameras rather than being repeated as literals —
        # they differ per camera (top 480x640, wrist 720x1280) and a mismatch here is
        # only caught deep inside lerobot's frame validation.
        # These must match scripts/utils/dataset_record.py EXACTLY, or lerobot's
        # merge_datasets refuses to combine rollouts with demonstrations:
        # validate_all_metadata compares the whole feature dict, names included.
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll", "gripper"]

        def _video_feature(cam):
            return {
                "dtype": "video",
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
            }

        dataset = LeRobotDataset.create(
            repo_id="local/ring_insert_rollouts",
            fps=30,
            root=str(out_dir / "dataset"),
            features={
                "observation.state": {"dtype": "float32", "shape": (6,), "names": joint_names},
                "action": {"dtype": "float32", "shape": (6,), "names": joint_names},
                "observation.images.top_rgb": _video_feature(env_cfg.top_camera),
                "observation.images.wrist_rgb": _video_feature(env_cfg.wrist_camera),
            },
            use_videos=True,
        )

    gi = gripper_body_index(env.robot)
    if args.debug:
        print(f"[run_eval] tracing end effector body {env.robot.body_names[gi]!r}", flush=True)

    results = []
    for ep in range(args.num_episodes):
        env_obs, _ = env.reset(seed=args.seed + ep)
        if args.start_pose_deg is not None:
            q = torch.deg2rad(
                torch.tensor(args.start_pose_deg, dtype=torch.float32, device=env.device)
            ).unsqueeze(0)
            env.robot.write_joint_position_to_sim(q)
            env.robot.write_joint_velocity_to_sim(torch.zeros_like(q))
            for _ in range(5):  # settle, and let the cameras re-render the new pose
                env_obs, *_ = env.step(q)
        if args.debug:
            start = np.rad2deg(np.asarray(env_obs["observation.state"], dtype=np.float32))
            print(f"[run_eval] ep {ep:03d} starts at {start.round(1).tolist()} deg", flush=True)
        if ep == 0:
            image_keys = build_image_key_map(policy, env_obs)
            print(f"[run_eval] camera mapping: {image_keys}", flush=True)
        policy.reset()  # clears ACT's action queue; skipping it leaks the last episode

        if args.debug:
            save_frames(env_obs, Path(args.frames_dir))

        trace = []
        success, steps = False, 0
        max_progress = 0.0
        for step in range(args.max_steps):
            obs = to_policy_obs(env_obs, policy_shapes, image_keys, units)
            with torch.inference_mode():
                action = policy.select_action(obs)

            env_action = to_env_action(action, units)
            prev_obs = env_obs
            env_obs, _reward, _terminated, truncated, _info = env.step(
                torch.from_numpy(env_action).unsqueeze(0).to(env.device)
            )
            # Isaac Lab only re-renders inside step() when has_gui() or has_rtx_sensors()
            # is true, and headless it is not — so the TiledCameras keep handing back the
            # frame captured at reset and the policy runs the whole episode on one stale
            # image. The teleop recorder calls render() every step for exactly this
            # reason; without it the cameras are decorative.
            env.render()
            env_obs = env._get_observations()

            if dataset is not None:
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(
                            prev_obs["observation.state"], dtype=np.float32
                        ).reshape(-1),
                        "action": env_action,
                        "observation.images.top_rgb": prev_obs["observation.images.top_rgb"],
                        "observation.images.wrist_rgb": prev_obs["observation.images.wrist_rgb"],
                    },
                    TASK_STRING,
                )

            if args.debug:
                # meas is the state the policy SAW (pre-step), not the state after acting.
                # Logging the post-step state instead makes the arm look as though it never
                # started from the reset pose, because the first action moves it ~30 deg
                # within one control step.
                trace.append(
                    trace_row(
                        env, gi, step,
                        np.rad2deg(env_action),
                        np.rad2deg(np.asarray(prev_obs["observation.state"], dtype=np.float32)),
                    )
                )

            steps = step + 1
            max_progress = max(max_progress, float(env.insertion_progress()[0]))
            if bool(env._get_success().any()):
                success = True
                break
            if bool(truncated.any() if torch.is_tensor(truncated) else truncated):
                break

        if dataset is not None:
            save_episode_safe(dataset)

        if args.debug and trace:
            import csv

            trace_path = out_dir / f"trace_ep{ep:03d}.csv"
            with trace_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
            # Did the arm follow the commands, and did it ever reach the ring?
            ee = np.array([[r["ee_x"], r["ee_y"], r["ee_z"]] for r in trace])
            ring = np.array([[r["ring_x"], r["ring_y"], r["ring_z"]] for r in trace])
            # Median, not max: the first few steps are a large transient while the arm
            # swings from its reset pose to the policy's first target, and reporting that
            # peak makes healthy tracking look broken.
            cmd = np.array([[r[f"cmd_{i}"] for i in range(6)] for r in trace])
            meas = np.array([[r[f"meas_{i}"] for i in range(6)] for r in trace])
            follow = np.median(np.abs(cmd - meas))
            print(
                f"          ee travelled {np.linalg.norm(ee[-1] - ee[0]):.3f} m, "
                f"closest approach to ring {np.linalg.norm(ee - ring, axis=1).min():.3f} m, "
                f"lowest ee z {ee[:, 2].min():.3f} (table 0.521), "
                f"ring moved {np.linalg.norm(ring[-1] - ring[0]):.3f} m, "
                f"median tracking error {follow:.1f} deg -> {trace_path.name}",
                flush=True,
            )

        final_progress = float(env.insertion_progress()[0])
        results.append({"episode": ep, "success": success, "steps": steps,
                        "max_progress": round(max_progress, 4),
                        "final_progress": round(final_progress, 4)})
        print(f"[ep {ep:03d}] success={success} steps={steps} "
              f"progress max={max_progress:.2f} final={final_progress:.2f}", flush=True)

    n = len(results)
    n_ok = sum(r["success"] for r in results)
    prog = [r["max_progress"] for r in results]
    summary = {
        "checkpoint": str(ckpt),
        "mean_max_progress": round(float(np.mean(prog)), 4) if prog else 0.0,
        "task": args.task,
        "episodes": n,
        "successes": n_ok,
        "success_rate": n_ok / n if n else 0.0,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsuccess rate: {n_ok}/{n} = {summary['success_rate']:.1%}"
          f"   mean max progress: {summary['mean_max_progress']:.2f}")
    print(f"summary written to {out_dir / 'summary.json'}")

    env.close()


if __name__ == "__main__":
    # simulation_app.close() tears the process down hard, so anything still sitting in a
    # stdio buffer — including a traceback — is lost. Print and flush before closing, or
    # a crash in here looks identical to a clean exit that produced no episodes.
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
