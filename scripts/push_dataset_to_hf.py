#!/usr/bin/env python3
"""Push a LeRobot v3.0 dataset to the Hugging Face Hub, with the right version tag.

    $V/bin/python scripts/push_dataset_to_hf.py witsense-ai/synthetic_so101_ring_insert \
        Datasets/record/ring_insert/merged_40

Two things make this more than an upload_folder call, and both cost an hour the first
time they bite:

  * v3.0 chunk filenames do not overlap v2.1 ones, so a plain upload over an existing
    v2.1 repo leaves both layouts in place and the loader trips over the old one. This
    deletes data/, videos/ and meta/ in the same commit.

  * lerobot resolves the dataset format from a **git tag**, not from meta/info.json.
    A repo whose files are v3.0 but whose tag says v2.1 fails with
    BackwardCompatibilityError, and no amount of re-uploading fixes it. This retags to
    match the version actually in info.json, moving a stale tag if one exists.

Runs in the training venv ($LEROBOT_VENV) alongside the rest of the dataset tooling.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))  # so the sibling import works under -m too
from push_checkpoint_to_hf import resolve_token  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", help="Target dataset repo, e.g. witsense-ai/synthetic_so101_ring_insert")
    p.add_argument("root", type=Path, help="Local dataset root (holds meta/, data/, videos/)")
    p.add_argument("--public", action="store_true", help="Create the repo public (default private)")
    p.add_argument("--message", default=None, help="Commit message")
    p.add_argument("--keep-existing", action="store_true",
                   help="Do not delete the repo's existing data/videos/meta first. Only safe "
                        "when the remote is already the same format version.")
    args = p.parse_args()

    info_path = args.root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"ERROR: {args.root} is not a LeRobot dataset (no meta/info.json)", file=sys.stderr)
        return 1
    info = json.loads(info_path.read_text())
    version = info.get("codebase_version", "v3.0")

    token, who = resolve_token()
    if who is None:
        print("ERROR: no usable Hugging Face credentials.\n"
              "  Set HF_TOKEN in .env, or run: hf auth login", file=sys.stderr)
        return 2

    print(f"Pushing as {who} -> {args.repo_id} ({'public' if args.public else 'private'})")
    print(f"  {args.root}: {info.get('total_episodes', '?')} episodes, "
          f"{info.get('total_frames', '?')} frames, {version}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.root),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.message or f"Upload {args.root.name} ({version})",
        delete_patterns=None if args.keep_existing else ["data/**", "videos/**", "meta/**"],
        ignore_patterns=[".cache/**", "**/.gitignore"],
    )

    # Retag only when it is wrong: create_tag on an existing tag raises, and deleting
    # then recreating a correct tag churns the repo for nothing.
    refs = api.list_repo_refs(args.repo_id, repo_type="dataset")
    main_commit = next((b.target_commit for b in refs.branches if b.name == "main"), None)
    existing = next((t for t in refs.tags if t.name == version), None)
    if existing is None or existing.target_commit != main_commit:
        if existing is not None:
            api.delete_tag(args.repo_id, tag=version, repo_type="dataset")
        api.create_tag(args.repo_id, tag=version, revision="main", repo_type="dataset")
        print(f"  tagged {version}")

    print(f"Pushed -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
