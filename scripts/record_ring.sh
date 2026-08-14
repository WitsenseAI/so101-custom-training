#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TASK_DESC="pickup the ring and place it on the toy"
DATASET_REPO=$HF_USER/so101_pick_and_place_ring
EPISODES="${1:-50}"

echo "Task: $TASK_DESC | Dataset: $DATASET_REPO | Episodes: $EPISODES"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=$ROBOT_FOLLOWER_PORT \
  --robot.id=left_follower \
  --robot.cameras="{
    top:   {type: opencv, index_or_path: /dev/video4, width: 640,  height: 480, fps: 30},
    wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}
  }" \
  --teleop.type=so101_leader \
  --teleop.port=$ROBOT_LEADER_PORT \
  --teleop.id=right_leader \
  --dataset.repo_id=$DATASET_REPO \
  --dataset.single_task="$TASK_DESC" \
  --dataset.num_episodes=$EPISODES \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.camera_encoder.vcodec=h264 \
  --dataset.camera_encoder.crf=23 \
  --dataset.camera_encoder.preset=fast \
  --dataset.streaming_encoding=true \
  --display_data=true \
  --play_sounds=true

echo "Done. Saved to $HF_LEROBOT_HOME/$DATASET_REPO"
