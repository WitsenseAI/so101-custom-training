#!/usr/bin/env python3
"""Push a trained LeRobot policy checkpoint to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/push_checkpoint_to_hf.py witsense-ai/so101_ring_act outputs/train/<run>/checkpoints/last

Point it at a checkpoint *step* directory (or straight at a pretrained_model/ dir).
Use this rather than training with PUSH_TO_HUB=true when you want to choose the
checkpoint by robot success rate instead of automatically shipping the last step.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", help="Target model repo, e.g. witsense-ai/so101_ring_act")
    p.add_argument("checkpoint", type=Path, help="Checkpoint step dir, or a pretrained_model dir")
    p.add_argument("--public", action="store_true", help="Create the repo public (default private)")
    p.add_argument("--message", default=None, help="Commit message")
    args = p.parse_args()

    # None makes huggingface_hub fall back to a cached `hf auth login`, so this works
    # whether the token comes from .env or from a previous login on the machine.
    # It looks under $HF_HOME, which .env points at the dataset cache — a directory with
    # no token in it — so read the real login explicitly before giving up.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        token = cached.read_text().strip() if cached.is_file() else None
    try:
        who = HfApi(token=token).whoami()["name"]
    except Exception:
        print(
            "ERROR: no usable Hugging Face credentials.\n"
            "  Set HF_TOKEN in .env, or run: hf auth login",
            file=sys.stderr,
        )
        return 2

    # A checkpoint dir holds pretrained_model/ (weights + pre/postprocessor); accept either.
    model_dir = args.checkpoint / "pretrained_model"
    if not model_dir.is_dir():
        model_dir = args.checkpoint
    if not (model_dir / "config.json").is_file():
        print(f"ERROR: no config.json under {model_dir} — not a policy checkpoint.", file=sys.stderr)
        return 3

    print(f"Pushing as {who} -> {args.repo_id} ({'public' if args.public else 'private'})")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message or f"Upload policy from {args.checkpoint.name}",
    )
    print(f"Pushed {model_dir} -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
