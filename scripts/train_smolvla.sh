#!/bin/bash
source ~/ws/lerobot/.venv/bin/activate
set -e

export HF_USER=witsense-ai

OUTPUT_DIR="outputs/train/smolvla"
STEPS=100000
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

DATASET_ROOT=~/ws/lerobot/so101-custom-training/witsense-ai/so101_pick_and_place_ring_33

if [ ! -d "$DATASET_ROOT" ]; then
  echo "Error: Dataset not found in witsense-ai folder"
  exit 1
fi

echo "Training SmolVLA on dataset: $DATASET_ROOT"
echo "Steps: $STEPS"
echo "Output: $OUTPUT_DIR"

# Check if resuming from existing checkpoint
RESUME=false
CONFIG_PATH=""

if [ -d "$CHECKPOINT_DIR" ]; then
  LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_DIR"/*/ 2>/dev/null | head -1)
  if [ -n "$LATEST_CHECKPOINT" ]; then
    if [ -f "$LATEST_CHECKPOINT/training_state/training_step.json" ]; then
      STEP=$(grep -o '"step": [0-9]*' "$LATEST_CHECKPOINT/training_state/training_step.json" | grep -o '[0-9]*')
      echo ""
      echo "Found checkpoint at step $STEP"
      echo "Resuming from: $LATEST_CHECKPOINT"
      RESUME=true
      CONFIG_PATH="$LATEST_CHECKPOINT/pretrained_model/train_config.json"
      echo ""
    fi
  fi
fi

# Build training command
if [ "$RESUME" = true ]; then
  echo "🔄 RESUMING training from checkpoint..."
  lerobot-train \
    --config_path="$CONFIG_PATH" \
    --resume=true \
    --steps=$STEPS
else
  echo "🚀 STARTING new training..."
  lerobot-train \
    --dataset.repo_id=local_dataset \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=smolvla \
    --output_dir="$OUTPUT_DIR" \
    --job_name=smolvla_so101 \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=$STEPS \
    --save_freq=5000 \
    --policy.push_to_hub=false
fi

echo "✓ Done: $OUTPUT_DIR"
