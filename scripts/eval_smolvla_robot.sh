#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CHECKPOINT_PATH="${1:-$ROOT_DIR/outputs/train/smolvla/checkpoints/075000/pretrained_model}"
[ -d "$CHECKPOINT_PATH/pretrained_model" ] && CHECKPOINT_PATH="$CHECKPOINT_PATH/pretrained_model"
[ -d "$CHECKPOINT_PATH" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }

EVAL_DATASET=witsense-ai/rollout_eval_smolvla

lerobot-rollout \
  --strategy.type=sentry \
  --policy.path="$CHECKPOINT_PATH" \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_FOLLOWER_PORT" \
  --robot.id=left_follower \
  --robot.cameras="$ROBOT_CAMERAS" \
  --dataset.repo_id="$EVAL_DATASET" \
  --dataset.single_task="pickup the ring and place it on the toy" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=true \
  --dataset.camera_encoder.vcodec=h264 \
  --dataset.camera_encoder.preset=fast \
  --dataset.streaming_encoding=true \
  --play_sounds=false

echo "✓ Done: $EVAL_DATASET"
