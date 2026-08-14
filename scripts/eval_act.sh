#!/usr/bin/env bash
# Run a trained ACT policy on the robot and record the rollouts.
#
#   bash scripts/eval_act.sh                                     # $POLICY_REPO from .env
#   bash scripts/eval_act.sh outputs/train/<run>/checkpoints/030000/pretrained_model
#
# Keep a hand on the e-stop for the first rollout: a fresh policy can drive into the table.
#
# Camera resolutions are read from the policy's own config.json rather than from .env:
# a policy trained with mismatched camera sizes (e.g. top 640x480 + wrist 1280x720) will
# silently receive the wrong input shape if you guess a single size for both.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[ -f .env ] || { echo "ERROR: no .env. Run: cp .env.example .env" >&2; exit 1; }
set -a; source .env; set +a
source "${LEROBOT_VENV:-$HOME/.venvs/lerobot}/bin/activate"

DATASET_NAME="${DATASET_REPO##*/}"
POLICY_PATH="${1:-${POLICY_REPO:-$HF_ORG/${DATASET_NAME}_act}}"
EVAL_REPO="${EVAL_REPO:-$HF_ORG/eval_${DATASET_NAME}}"
: "${EVAL_EPISODES:=10}"; : "${EPISODE_TIME_S:=25}"

# Ask the policy what image shapes it expects, and open each camera at exactly that size.
CAMERAS=$(TOP_CAMERA="$TOP_CAMERA" WRIST_CAMERA="$WRIST_CAMERA" python - "$POLICY_PATH" <<'PY'
import json, os, sys
from pathlib import Path

src = Path(sys.argv[1])
local = src / "config.json"
if not local.is_file():
    local = src / "pretrained_model" / "config.json"
if local.is_file():
    cfg = json.loads(local.read_text())
else:  # a Hub repo id
    from huggingface_hub import hf_hub_download
    cfg = json.loads(Path(hf_hub_download(str(src), "config.json")).read_text())

devices = {"top": os.environ["TOP_CAMERA"], "wrist": os.environ["WRIST_CAMERA"]}
entries = []
for key, feat in cfg["input_features"].items():
    if "image" not in key:
        continue
    name = key.rsplit(".", 1)[-1]           # observation.images.top -> top
    if name not in devices:
        sys.exit(f"policy expects camera '{name}', which is not configured in .env")
    _, h, w = feat["shape"]
    entries.append(f"{name}: {{type: opencv, index_or_path: {devices[name]}, "
                   f"width: {w}, height: {h}, fps: 30}}")
if not entries:
    sys.exit("policy declares no image inputs")
print("{" + ", ".join(entries) + "}")
PY
)
echo "  cameras   $CAMERAS"

echo "  policy    $POLICY_PATH"
echo "  rollouts  $EVAL_EPISODES -> $EVAL_REPO"
echo ""

lerobot-rollout \
  --policy.path="$POLICY_PATH" \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --fps=30 \
  --dataset.repo_id="$EVAL_REPO" \
  --dataset.single_task="$TASK_DESC" \
  --dataset.num_episodes="$EVAL_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --play_sounds=true

echo ""
echo "Score it by counting successes out of $EVAL_EPISODES. Loss cannot tell you this."