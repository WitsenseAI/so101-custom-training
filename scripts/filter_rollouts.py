#!/usr/bin/env python3
"""Keep only the rollout episodes worth retraining on.

    /path/to/lerobot-venv/bin/python scripts/filter_rollouts.py outputs/rollouts_run1

Reads summary.json written by run_eval alongside the recorded dataset, and drops every
episode that neither succeeded nor got far enough. What survives is self-imitation data:
the policy's own good attempts, ready to merge with the human demonstrations and retrain
on (filtered behaviour cloning).

Two selection knobs:

    --min-progress   keep failures that got at least this far (0 disables, i.e. keep
                     successes only). max_progress is insertion_progress at its best
                     point in the episode: ~0.5 means the ring reached the ghost,
                     ~0.9 means it was nearly seated. Including strong near-misses gives
                     the next policy more to learn from than successes alone, at the cost
                     of teaching it some trajectories that did not finish the task.
    --max-episodes   cap the result, keeping the highest-progress episodes.

The rollout dataset must be v3.0 — run scripts/convert_dataset_v30.py on it first, since
run_eval writes v2.1 under Isaac Sim's pinned lerobot.
"""

import argparse
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="run_eval --out directory (holds summary.json and dataset/)")
    p.add_argument("--dataset", type=Path, default=None, help="Defaults to <run_dir>/dataset")
    p.add_argument("--out", type=Path, default=None, help="Defaults to <run_dir>/dataset_filtered")
    p.add_argument("--repo-id", default="local/rollouts_filtered")
    p.add_argument("--min-progress", type=float, default=0.0,
                   help="Also keep failures whose max_progress reached this (0 = successes only)")
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    summary_path = args.run_dir / "summary.json"
    if not summary_path.is_file():
        sys.exit(f"no summary.json at {summary_path}")
    results = json.loads(summary_path.read_text())["results"]

    root = args.dataset or args.run_dir / "dataset"
    ds = LeRobotDataset(repo_id="local/rollouts", root=root)
    if len(results) != ds.num_episodes:
        print(f"WARNING: summary has {len(results)} episodes, dataset has {ds.num_episodes}. "
              f"Matching by episode index.", file=sys.stderr)

    keep = []
    for r in results:
        if r["episode"] >= ds.num_episodes:
            continue
        if r["success"] or (args.min_progress > 0 and r.get("max_progress", 0) >= args.min_progress):
            keep.append(r)

    keep.sort(key=lambda r: (not r["success"], -r.get("max_progress", 0)))
    if args.max_episodes:
        keep = keep[: args.max_episodes]
    keep_idx = sorted(r["episode"] for r in keep)

    n_ok = sum(r["success"] for r in keep)
    print(f"{ds.num_episodes} episodes -> keeping {len(keep_idx)} "
          f"({n_ok} successes, {len(keep_idx) - n_ok} near-misses above {args.min_progress})")
    if not keep_idx:
        sys.exit("nothing passed the filter — lower --min-progress or collect more rollouts")

    drop = [i for i in range(ds.num_episodes) if i not in set(keep_idx)]
    out = args.out or args.run_dir / "dataset_filtered"
    filtered = delete_episodes(ds, episode_indices=drop, output_dir=out, repo_id=args.repo_id)
    print(f"wrote {filtered.num_episodes} episodes / {filtered.num_frames} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
