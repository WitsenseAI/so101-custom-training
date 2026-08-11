#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Teleoperation. Ctrl+C to stop."
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=$ROBOT_FOLLOWER_PORT \
  --robot.id=left_follower \
  --teleop.type=so101_leader \
  --teleop.port=$ROBOT_LEADER_PORT \
  --teleop.id=right_leader \
  --robot.cameras="$ROBOT_CAMERAS" \
  --display_data=false
