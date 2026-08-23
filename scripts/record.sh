#!/usr/bin/env bash
# Record teleoperated demonstrations into a LeRobot dataset.
#
#   bash scripts/record.sh 50      # 50 episodes
#
# Keys during recording:  →  save episode & continue   ←  redo this episode   Esc  stop
# Use ← freely: a bad demo in the dataset is worse than one fewer demo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[ -f .env ] || { echo "ERROR: no .env. Run: cp .env.example .env" >&2; exit 1; }
set -a; source .env; set +a
source "${LEROBOT_VENV:-$ROOT_DIR/.venv}/bin/activate"

EPISODES="${1:-50}"
: "${DATASET_REPO:?set DATASET_REPO in .env}"
: "${EPISODE_TIME_S:=25}"; : "${RESET_TIME_S:=15}"
: "${CAMERA_WIDTH:=640}"; : "${CAMERA_HEIGHT:=480}"

# Both cameras at the same resolution on purpose: mismatched sizes still train, but the
# larger view produces disproportionately more visual tokens and dominates memory.
CAMERAS="{
  top:   {type: opencv, index_or_path: $TOP_CAMERA,   width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: 30},
  wrist: {type: opencv, index_or_path: $WRIST_CAMERA, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: 30}
}"

echo "  dataset   $DATASET_REPO"
echo "  task      $TASK_DESC"
echo "  episodes  $EPISODES  (${EPISODE_TIME_S}s record / ${RESET_TIME_S}s reset)"
echo ""

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$ROBOT_LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.single_task="$TASK_DESC" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.push_to_hub=true \
  --dataset.streaming_encoding=true \
  --display_data=true \
  --play_sounds=true \
  --resume=true

echo ""
echo "Review before training:  lerobot-dataset-viz --repo-id=$DATASET_REPO --episode-index=0"
