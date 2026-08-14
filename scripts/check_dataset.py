#!/usr/bin/env python3
"""Sanity-check a recorded LeRobot dataset before spending hours training on it.

    python scripts/check_dataset.py Datasets/record/ring_insert/002

Exists because a whole 20-episode dataset was recorded, converted, pushed and trained on
(2 h of GPU) before anyone noticed that observation.state never changed: the env returned
numpy views onto Isaac Lab's live buffers, so every frame in an episode aliased one array
and was written holding the final step's values. Nothing errored — the loss curve looked
healthy and the policy half-worked off the images alone.

Checks, in order of how quietly they fail:

  1. observation.state actually varies within each episode (the alias bug)
  2. state tracks action, rather than being unrelated to it
  3. the gripper opens and closes (a pick task with a frozen gripper is unlearnable)
  4. episode lengths are plausible and consistent

Exits non-zero if anything looks wrong, so it can gate a training run.
"""

import argparse
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def load(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"no data parquet under {root}")
    return pd.concat(pq.read_table(f).to_pandas(ignore_metadata=True) for f in files)


def video_motion(root: Path, episodes: int = 3, samples: int = 5) -> dict[str, float]:
    """Mean frame-to-frame change in each camera's video, to catch dead footage.

    A dataset can pass every state/action check and still be worthless: run_eval recorded
    correct joint trajectories alongside videos frozen at the reset frame, because Isaac
    Lab only re-renders inside step() when a GUI or RTX sensor is active and headless it
    is not. Values here were ~1.0 for that dead footage and 22-49 for real teleoperation,
    so anything below a couple of units means the cameras were not updating.
    """
    out = {}
    for cam_dir in sorted((root / "videos").rglob("observation.images.*")):
        if not cam_dir.is_dir():
            continue
        diffs = []
        for video in sorted(cam_dir.glob("*.mp4"))[:episodes]:
            with tempfile.TemporaryDirectory() as tmp:
                cmd = ["ffmpeg", "-loglevel", "error", "-i", str(video),
                       "-vf", f"select='not(mod(n,{max(1, 400 // samples)}))'",
                       "-vsync", "0", f"{tmp}/f%03d.png"]
                if subprocess.run(cmd, capture_output=True).returncode != 0:
                    continue
                frames = sorted(Path(tmp).glob("*.png"))
                if len(frames) < 2:
                    continue
                imgs = [np.asarray(Image.open(f), dtype=np.float32) for f in frames]
                diffs.append(float(np.mean([np.abs(imgs[i] - imgs[0]).mean()
                                            for i in range(1, len(imgs))])))
        if diffs:
            out[cam_dir.name] = float(np.mean(diffs))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--min-video-motion", type=float, default=3.0,
                   help="Mean frame-to-frame change a camera must show. Dead footage scores ~1, "
                        "real motion 20+")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--min-gripper-travel", type=float, default=10.0,
                   help="Degrees the gripper must move for the episode to contain a grasp")
    args = p.parse_args()

    df = load(args.root)
    episodes = sorted(df.episode_index.unique())
    print(f"{len(episodes)} episodes, {len(df)} frames\n")

    frozen, no_grip, lengths = [], [], []
    print(f"{'ep':>3s} {'len':>5s} {'state span (deg, per joint)':>34s} {'grip travel':>12s}")
    for ep in episodes:
        d = df[df.episode_index == ep].sort_values("frame_index")
        state = np.rad2deg(np.stack(d["observation.state"].map(np.asarray)))
        span = state.max(0) - state.min(0)
        lengths.append(len(d))
        if span.max() < 1e-3:
            frozen.append(int(ep))
        if span[5] < args.min_gripper_travel:
            no_grip.append(int(ep))
        print(f"{int(ep):3d} {len(d):5d} " + " ".join(f"{v:5.1f}" for v in span) + f" {span[5]:12.1f}")

    # state should follow action; if they are unrelated the wrong thing was recorded
    state = np.rad2deg(np.stack(df["observation.state"].map(np.asarray)))
    action = np.rad2deg(np.stack(df["action"].map(np.asarray)))
    print("\nper-joint correlation between action and state:")
    corrs = []
    for i, name in enumerate(JOINTS):
        if state[:, i].std() < 1e-6 or action[:, i].std() < 1e-6:
            corrs.append(0.0)
            print(f"   {name:6s}   n/a (no variation)")
            continue
        c = float(np.corrcoef(action[:, i], state[:, i])[0, 1])
        corrs.append(c)
        print(f"   {name:6s} {c:6.3f}")

    motion = {} if args.skip_video else video_motion(args.root)
    if motion:
        print("\nvideo motion (mean frame-to-frame change; dead footage is ~1):")
        for cam, v in motion.items():
            print(f"   {cam:34s} {v:7.2f}{'   DEAD' if v < args.min_video_motion else ''}")

    print()
    problems = []
    dead = [c for c, v in motion.items() if v < args.min_video_motion]
    if dead:
        problems.append(f"video is frozen for {dead} — the cameras were not re-rendering. "
                        f"run_eval must call env.render() each step; states can look perfect "
                        f"while the footage is a single repeated frame")
    if frozen:
        problems.append(f"observation.state NEVER CHANGES in episodes {frozen} — the env is "
                        f"returning views onto live sim buffers; add .copy() in _get_observations")
    if no_grip:
        problems.append(f"gripper moves < {args.min_gripper_travel} deg in episodes {no_grip} — "
                        f"no grasp was demonstrated there")
    if np.mean(corrs) < 0.5:
        problems.append(f"action and state are barely correlated (mean {np.mean(corrs):.2f}); "
                        f"the arm is not tracking, or the wrong signal was recorded")
    if len(lengths) > 1 and min(lengths) < 30:
        problems.append(f"some episodes are very short (min {min(lengths)} frames)")

    if problems:
        print("PROBLEMS FOUND:")
        for t in problems:
            print(f"  - {t}")
        return 1
    print(f"looks good: state varies, tracks action (mean corr {np.mean(corrs):.2f}), "
          f"gripper active, lengths {min(lengths)}-{max(lengths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
