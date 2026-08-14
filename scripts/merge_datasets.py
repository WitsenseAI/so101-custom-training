#!/usr/bin/env python3
"""Merge LeRobot v3.0 datasets into one.

    $V/bin/python scripts/merge_datasets.py \
        --out Datasets/record/ring_insert/merged_40 \
        --repo-id witsense-ai/synthetic_so101_ring_insert \
        Datasets/record/ring_insert/002 Datasets/record/ring_insert/003

Inputs must already be v3.0 (scripts/convert_dataset_v30.py) and must share identical
feature metadata, `names` included. lerobot compares the whole feature dict, so
`names=None` against joint names, or "channel" against "channels", blocks a merge of
otherwise identical data.
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parts", type=Path, nargs="+", help="Dataset roots to merge, in order")
    p.add_argument("--out", type=Path, required=True, help="Output dataset root (must not exist)")
    p.add_argument("--repo-id", default="local/merged",
                   help="repo_id recorded in the merged metadata; the eventual Hub name")
    args = p.parse_args()

    if len(args.parts) < 2:
        return _fail("need at least two datasets to merge")
    if args.out.exists():
        return _fail(f"{args.out} already exists — remove it or pick another --out")
    for path in args.parts:
        if not (path / "meta" / "info.json").is_file():
            return _fail(f"{path} is not a LeRobot dataset (no meta/info.json)")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    parts = []
    for path in args.parts:
        ds = LeRobotDataset(repo_id=f"local/{path.name}", root=str(path))
        print(f"  {path}: {ds.num_episodes} episodes, {ds.num_frames} frames")
        parts.append(ds)

    merged = merge_datasets(parts, output_repo_id=args.repo_id, output_dir=str(args.out))
    print(f"\n{merged.num_episodes} episodes, {merged.num_frames} frames -> {args.out}")

    expected = sum(d.num_episodes for d in parts)
    if merged.num_episodes != expected:
        return _fail(f"expected {expected} episodes, got {merged.num_episodes}")
    return 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
