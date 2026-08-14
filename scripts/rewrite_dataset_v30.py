#!/usr/bin/env python3
"""Rewrite a LeRobot v2.1 dataset as v3.0, by replaying it through the new writer.

    /path/to/lerobot-venv/bin/python scripts/rewrite_dataset_v30.py \
        Datasets/record/ring_insert/001_no_depth \
        --repo-id witsense-ai/synthetic_so101_ring_insert --push

Why not lerobot's own convert_dataset_v21_to_v30? On this machine it fails three ways in
a row: a datasets/pandas dtype bug in its pd.concat, an arrow write error on the 2-D depth
column, and finally it resolves the dataset version from the Hub rather than from --root,
so it re-reads v2.1 no matter what you point it at.

This instead reads the v2.1 parquet and videos directly and feeds them to the current
LeRobotDataset writer, which produces v3.0 natively. Slower — the videos are decoded and
re-encoded — but it only depends on APIs both sides actually support.

The sim recorder writes v2.1 because Isaac Sim pins lerobot 0.3.3 (newer lerobot needs
numpy>=2, isaacsim-kernel pins numpy==1.26.0), so every sim recording needs this step
before it can be trained on in the ACT venv.
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def video_frames(path: Path):
    """Yield RGB frames from an episode video, in order."""
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="v2.1 dataset root")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out", type=Path, default=None, help="where to write (default: <src>_v30)")
    p.add_argument("--push", action="store_true")
    p.add_argument("--private", action="store_true", default=True)
    args = p.parse_args()

    src = args.src
    out = args.out or src.parent / f"{src.name}_v30"
    info = json.loads((src / "meta" / "info.json").read_text())

    # Carry the features across unchanged, minus the bookkeeping columns the writer adds
    # back itself. Video features keep their declared shapes, so top stays 480x640 and
    # wrist stays 720x1280.
    managed = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {k: v for k, v in info["features"].items() if k not in managed}
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    table_keys = [k for k in features if k not in video_keys]
    print(f"features: {table_keys} + video {video_keys}")

    task = json.loads((src / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"]
    print(f"task: {task!r}")

    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=info["fps"], features=features, root=out,
        use_videos=True, image_writer_threads=8,
    )

    episodes = sorted(glob.glob(str(src / "data" / "**" / "*.parquet"), recursive=True))
    for ep_path in episodes:
        ep_path = Path(ep_path)
        ep_idx = int(ep_path.stem.split("_")[-1])
        table = pq.read_table(ep_path).to_pandas(ignore_metadata=True)
        # One decoder per camera, advanced in lockstep with the table rows.
        readers = {
            k: video_frames(next(iter(src.glob(f"videos/**/{k}/episode_{ep_idx:06d}.mp4"))))
            for k in video_keys
        }
        n = 0
        for _, row in table.iterrows():
            frame = {k: np.asarray(row[k], dtype=np.float32) for k in table_keys}
            for k, reader in readers.items():
                frame[k] = next(reader)
            frame["task"] = task
            ds.add_frame(frame)
            n += 1
        ds.save_episode()
        print(f"  episode {ep_idx:03d}: {n} frames", flush=True)

    print(f"\nwrote {ds.num_episodes} episodes / {ds.num_frames} frames -> {out}")
    if args.push:
        ds.push_to_hub(private=args.private,
                       tags=["robotics", "lerobot", "so101", "isaac-sim", "synthetic", "ring-insert"])
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
