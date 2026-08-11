#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Calibrating leader (right_leader) on $ROBOT_LEADER_PORT..."
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$ROBOT_LEADER_PORT --teleop.id=right_leader

echo "Calibrating follower (left_follower) on $ROBOT_FOLLOWER_PORT..."
lerobot-calibrate --robot.type=so101_follower --robot.port=$ROBOT_FOLLOWER_PORT --robot.id=left_follower

cp -r ~/.cache/huggingface/lerobot/ ~/lerobot-calibration-backup/
echo "Calibration backed up to ~/lerobot-calibration-backup/"
