#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

POLICY_PATH="${1:-witsense-ai/so101_act_fewshot}"
EPISODES="${2:-10}"
EVAL_DATASET=witsense-ai/rollout_eval_act

echo "Evaluating ACT on SO-101"
echo "  Policy:   $POLICY_PATH"
echo "  Episodes: $EPISODES -> $EVAL_DATASET"

lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="$POLICY_PATH" \
  --policy.n_action_steps=40 \
  --inference.type=sync \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_FOLLOWER_PORT" \
  --robot.id=left_follower \
  --robot.max_relative_target=20 \
  --robot.cameras="$ROBOT_CAMERAS" \
  --fps=30 \
  --dataset.repo_id="$EVAL_DATASET" \
  --dataset.single_task="pickup the ring and place it on the toy" \
  --dataset.num_episodes=$EPISODES \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --dataset.camera_encoder.vcodec=h264 \
  --dataset.camera_encoder.preset=ultrafast \
  --dataset.streaming_encoding=false \
  --play_sounds=false

echo "✓ Eval complete: $EVAL_DATASET"
