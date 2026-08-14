#!/usr/bin/env python3
"""Convert a sim-recorded LeRobot v2.1 dataset to v3.0, working around three upstream bugs.

    /path/to/lerobot-venv/bin/python scripts/convert_dataset_v30.py \
        Datasets/record/ring_insert/002

Isaac Sim pins lerobot 0.3.3 (newer lerobot needs numpy>=2, isaacsim-kernel pins
numpy==1.26.0), so the recorder writes v2.1 while training needs v3.0. lerobot ships
convert_dataset_v21_to_v30 for exactly this, but on this machine it fails three ways:

1. It does pd.concat([pd.read_parquet(f) ...]). Because lerobot imports `datasets`,
   pyarrow hands pandas datasets' PandasArrayExtensionDtype columns, whose concat is
   broken against pandas 2.3.3 / datasets 4.8.5 — datasets sets _metadata to the string
   "value_type" where pandas expects a tuple of attribute names, so pandas iterates the
   characters and raises "no attribute 'v'".
2. Once patched, the same concat fails on shape. Reading the parquet without the HF
   metadata and casting to plain object dtype fixes it; the data written is identical.
3. A 2-D column (observation.top_depth) cannot be written back as list<list<uint16>>
   from a bare ndarray, so those are handed over as lists of rows. Record with
   --disable_depth and this never arises.

It also resolves the dataset version from the Hub rather than from --root, so pass a
repo id that does not exist remotely and let it work purely locally.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

import datasets  # noqa: F401  - imported first, exactly as lerobot does
import pandas as pd
import pyarrow.parquet as pq
from datasets.features.features import PandasArrayExtensionDtype

if isinstance(PandasArrayExtensionDtype._metadata, str):
    PandasArrayExtensionDtype._metadata = (PandasArrayExtensionDtype._metadata,)

_original_read_parquet = pd.read_parquet


def _read_parquet_as_object(path, *args, **kwargs):
    try:
        df = pq.read_table(path).to_pandas(ignore_metadata=True)
    except Exception:
        return _original_read_parquet(path, *args, **kwargs)
    for col in df.columns:
        if isinstance(df[col].dtype, PandasArrayExtensionDtype):
            df[col] = pd.Series(list(df[col].to_numpy()), index=df.index, dtype=object)
        if df[col].dtype == object and len(df) and getattr(df[col].iloc[0], "ndim", 1) > 1:
            df[col] = df[col].map(lambda a: list(a) if getattr(a, "ndim", 1) > 1 else a)
    return df


pd.read_parquet = _read_parquet_as_object


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="v2.1 dataset directory")
    p.add_argument("--repo-id", default=None,
                   help="Only used to name the conversion; defaults to a local-only id so "
                        "the converter cannot resolve a stale version from the Hub.")
    args, rest = p.parse_known_args()

    root = args.root.resolve()
    repo_id = args.repo_id or f"local/{root.name}_{root.parent.name}"
    print(f"[convert] converting {root} in place (repo-id {repo_id})", flush=True)

    sys.argv = ["convert_dataset_v21_to_v30", f"--repo-id={repo_id}",
                f"--root={root}", "--push-to-hub=False", *rest]
    try:
        runpy.run_module("lerobot.scripts.convert_dataset_v21_to_v30", run_name="__main__")
    except SystemExit as err:  # the converter exits non-zero after writing, on the hub check
        if err.code not in (None, 0):
            print(f"[convert] converter exited {err.code}; checking the output anyway", flush=True)

    # The converter writes <root>_v30, then swaps: the original becomes <root>_old and
    # the v3.0 result takes the original path. So the answer is at `root`, not `root_v30`.
    info = json.loads((root / "meta" / "info.json").read_text())
    version = info.get("codebase_version")
    if version != "v3.0":
        sys.exit(f"[convert] {root} is still {version}")
    print(f"[convert] {root} is now v3.0 — {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames (original kept at {root.parent / (root.name + '_old')})")


if __name__ == "__main__":
    main()
