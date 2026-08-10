#!/usr/bin/env bash
# Calibrate both SO-101 arms. Run once per physical setup, before recording.
#
# Calibration defines the UNITS of every joint value in your dataset and policy.
# Re-calibrating differently after training moves the policy's outputs into a
# different frame and it will miss the object — so back the files up and keep them.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[ -f .env ] || { echo "ERROR: no .env. Run: cp .env.example .env" >&2; exit 1; }
set -a; source .env; set +a
source "${LEROBOT_VENV:-$HOME/.venvs/lerobot}/bin/activate"

echo "Leader on $ROBOT_LEADER_PORT ..."
lerobot-calibrate --teleop.type=so101_leader --teleop.port="$ROBOT_LEADER_PORT" --teleop.id="$LEADER_ID"

echo "Follower on $ROBOT_FOLLOWER_PORT ..."
lerobot-calibrate --robot.type=so101_follower --robot.port="$ROBOT_FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"

mkdir -p calibration
cp -r ~/.cache/huggingface/lerobot/calibration/* calibration/ 2>/dev/null || true
echo ""
echo "Backed up to $ROOT_DIR/calibration/ — keep these alongside the dataset."
echo "Next, sanity-check teleoperation (see ACT_PIPELINE.md section 3) before recording."
