#!/usr/bin/env bash
# Run a trained ACT policy on the robot and record the rollouts.
#
#   bash scripts/eval_act.sh                                     # $POLICY_REPO from .env
#   bash scripts/eval_act.sh outputs/train/<run>/checkpoints/030000/pretrained_model
#
# Keep a hand on the e-stop for the first rollout: a fresh policy can drive into the table.
#
# The camera resolutions here MUST match what the policy was trained on. Check with:
#   python -c "import json,urllib.request as u; \
#     print(json.load(u.urlopen('https://huggingface.co/<policy>/resolve/main/config.json'))['input_features'])"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[ -f .env ] || { echo "ERROR: no .env. Run: cp .env.example .env" >&2; exit 1; }
set -a; source .env; set +a
source "${LEROBOT_VENV:-$HOME/.venvs/lerobot}/bin/activate"

DATASET_NAME="${DATASET_REPO##*/}"
POLICY_PATH="${1:-${POLICY_REPO:-$HF_ORG/${DATASET_NAME}_act}}"
EVAL_REPO="${EVAL_REPO:-$HF_ORG/eval_${DATASET_NAME}}"
: "${CAMERA_WIDTH:=640}"; : "${CAMERA_HEIGHT:=480}"
: "${EVAL_EPISODES:=10}"; : "${EPISODE_TIME_S:=25}"

CAMERAS="{
  top:   {type: opencv, index_or_path: $TOP_CAMERA,   width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: 30},
  wrist: {type: opencv, index_or_path: $WRIST_CAMERA, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: 30}
}"

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
