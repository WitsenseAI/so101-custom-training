#!/usr/bin/env python3
"""Download a trained MuSHR policy from the Hub into a run folder play_policy.py can use.

    python scripts/download_hf_checkpoint.py
    python scripts/download_hf_checkpoint.py --repo-id witsense-ai/mushr_wheeledlab_elevation

Writes logs/<repo-name>/ with the same layout training produces, so playback is then:

    python scripts/play_policy.py -p logs/mushr_wheeledlab_elevation --steps 1000 \
        env.scene.num_envs=2

play_policy.py needs both run_config.pkl and models/model_*.pt, so the repo carries both.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_REPO = "witsense-ai/mushr_wheeledlab_elevation"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "logs"


def resolve_token() -> str | None:
    """HF_TOKEN, else the cached `hf auth login`. None lets huggingface_hub try its own."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        return token
    cached = Path.home() / ".cache" / "huggingface" / "token"
    return cached.read_text().strip() if cached.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help="Parent directory; the run folder is created inside it")
    p.add_argument("--name", default=None, help="Run folder name (default: the repo name)")
    p.add_argument("--force", action="store_true", help="Overwrite an existing run folder")
    return p


def check_layout(run_dir: Path) -> None:
    models = sorted((run_dir / "models").glob("model_*.pt"))
    if not models:
        sys.exit(f"no models/model_*.pt in {run_dir} — not a playable run folder")
    if not (run_dir / "run_config.pkl").is_file():
        print(f"WARNING: no run_config.pkl in {run_dir}; play_policy.py needs "
              f"--task and --policy-path instead of -p", file=sys.stderr)
    print(f"  checkpoints: {', '.join(m.name for m in models)}")


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.dest / (args.name or args.repo_id.split("/")[-1])
    if run_dir.exists() and not args.force:
        sys.exit(f"{run_dir} already exists — pass --force to overwrite")

    token = resolve_token()
    try:
        HfApi(token=token).model_info(args.repo_id)
    except Exception as exc:
        sys.exit(f"cannot reach {args.repo_id}: {type(exc).__name__}\n"
                 f"  private repo? set HF_TOKEN, or run: hf auth login")

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo_id, local_dir=str(run_dir), token=token)
    print(f"downloaded {args.repo_id} -> {run_dir}")
    check_layout(run_dir)
    print(f"\nplay it:\n  python scripts/play_policy.py -p {run_dir} "
          f"--steps 1000 env.scene.num_envs=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
