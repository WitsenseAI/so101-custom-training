#!/usr/bin/env python3
"""Drop a feature column from a LeRobot v2.1 dataset.

    python scripts/strip_column_v21.py Datasets/record/ring_insert/001 \
        Datasets/record/ring_insert/001_no_depth

Why this exists: the sim recorder stores observation.top_depth as a raw 480x640 uint16
array feature rather than a video. lerobot computes per-element statistics for array
features, so twenty episodes of depth produce ~187 MB of episode metadata and the
v2.1 -> v3.0 converter refuses it:

    NotImplementedError: Episodes dataset is too large (187 MB) to write to a single
    file. The current limit is 100 MB.

ACT never reads depth, so dropping it makes the dataset convertible and roughly ten
times smaller. Record with --disable_depth to avoid producing it in the first place.

Only the data parquets and the two metadata files that name the column are touched;
videos are copied through untouched.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq


def strip(src: Path, dst: Path, column: str) -> int:
    if not (src / "meta" / "info.json").is_file():
        sys.exit(f"not a LeRobot dataset (no meta/info.json): {src}")
    if dst.exists():
        sys.exit(f"refusing to overwrite existing {dst}")

    shutil.copytree(src, dst)

    # 1. data parquets
    rewritten = 0
    for path in sorted(dst.rglob("data/**/*.parquet")):
        table = pq.read_table(path)
        if column not in table.column_names:
            continue
        pq.write_table(table.drop_columns([column]), path)
        rewritten += 1

    # 2. meta/info.json — the feature declaration
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.get("features", {}).pop(column, None)
    info_path.write_text(json.dumps(info, indent=4))

    # 3. meta/episodes_stats.jsonl — the per-element statistics, the actual bulk
    stats_path = dst / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        lines = []
        for line in stats_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            entry.get("stats", {}).pop(column, None)
            lines.append(json.dumps(entry))
        stats_path.write_text("\n".join(lines) + "\n")

    return rewritten


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--column", default="observation.top_depth")
    args = p.parse_args()

    n = strip(args.src, args.dst, args.column)
    before = sum(f.stat().st_size for f in args.src.rglob("*") if f.is_file())
    after = sum(f.stat().st_size for f in args.dst.rglob("*") if f.is_file())
    print(f"dropped {args.column!r} from {n} parquet file(s)")
    print(f"{before / 1e6:.0f} MB -> {after / 1e6:.0f} MB   {args.dst}")


if __name__ == "__main__":
    main()
