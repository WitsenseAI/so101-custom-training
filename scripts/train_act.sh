#!/usr/bin/env bash
# Train an ACT policy on any LeRobot-format dataset hosted on the Hugging Face Hub.
#
#   cp .env.example .env      # set HF_TOKEN + DATASET_REPO
#   bash scripts/train_act.sh smoke     # 60 steps: proves it fits in VRAM
#   bash scripts/train_act.sh           # the real run
#
# Everything is configured through .env. Override per-run from the CLI:
#   DATASET_REPO=someorg/some_dataset STEPS=20000 bash scripts/train_act.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ─── config ──────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "ERROR: no .env found. Run: cp .env.example .env" >&2
    exit 1
fi

# Fall back to a cached `hf auth login`. huggingface_hub looks for its token under
# $HF_HOME, and .env points HF_HOME at the dataset cache, which has no token in it — so a
# perfectly good login under ~/.cache/huggingface goes unseen and any PRIVATE dataset
# fails with a misleading "401 ... Repository Not Found". Public datasets need no token,
# which is why this only bites the first time you train from a private one.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    export HF_TOKEN
fi

: "${DATASET_REPO:?set DATASET_REPO in .env}"
: "${HF_ORG:=witsense-ai}"
: "${LEROBOT_VENV:=$ROOT_DIR/.venv}"
: "${STEPS:=50000}"; : "${BATCH_SIZE:=4}"; : "${NUM_WORKERS:=4}"; : "${SEED:=1000}"
: "${SAVE_FREQ:=2000}"; : "${LOG_FREQ:=200}"
: "${PUSH_TO_HUB:=false}"; : "${PUSH_PRIVATE:=true}"
: "${WANDB_ENABLE:=false}"

# Pinned: an unpinned install is how a renamed CLI flag silently breaks a run.
# Bump deliberately, and re-run `smoke` after you do.
LEROBOT_REF="${LEROBOT_REF:-22bd7a2f489b367d8df42de803b1e8c4ca63a3f9}"

DATASET_NAME="${DATASET_REPO##*/}"
POLICY_REPO="${POLICY_REPO:-$HF_ORG/${DATASET_NAME}_act}"
RUN_NAME="${RUN_NAME:-${DATASET_NAME}_act}"

if [ "${1:-full}" = "smoke" ]; then
    STEPS=60; SAVE_FREQ=0; LOG_FREQ=10; PUSH_TO_HUB=false
    RUN_NAME="${RUN_NAME}_smoke"
    rm -rf "outputs/train/$RUN_NAME"        # smoke runs are disposable
fi
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN_NAME"

if [ -e "$OUTPUT_DIR" ]; then
    cat >&2 <<EOF
ERROR: $OUTPUT_DIR already exists (lerobot will not overwrite it). Either:
  RUN_NAME=another_name bash $0
  rm -rf $OUTPUT_DIR
  lerobot-train --config_path=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json --resume=true
EOF
    exit 1
fi

# ─── environment (installed once) ────────────────────────────────────────────
if [ ! -x "$LEROBOT_VENV/bin/lerobot-train" ]; then
    echo "[setup] installing lerobot @ ${LEROBOT_REF:0:8} -> $LEROBOT_VENV"
    command -v uv >/dev/null || { echo "ERROR: uv not found. https://docs.astral.sh/uv/" >&2; exit 1; }
    command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not on PATH; video decoding may fail"
    uv venv "$LEROBOT_VENV" --python 3.12
    uv pip install --python "$LEROBOT_VENV/bin/python" -q \
        "lerobot[dataset,training] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_REF}"
fi
source "$LEROBOT_VENV/bin/activate"

# bf16 roughly halves activation memory. Ampere and newer support it; older cards don't,
# so ask the GPU rather than assuming.
MIXED_PRECISION=$(python -c "
import torch
print('bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'no')")
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")

cat <<EOF

  dataset    $DATASET_REPO
  policy     $POLICY_REPO $([ "$PUSH_TO_HUB" = true ] && echo "(pushed on completion)" || echo "(local only)")
  run        $RUN_NAME -> $OUTPUT_DIR
  training   $STEPS steps, batch $BATCH_SIZE, seed $SEED
  device     $DEVICE, mixed precision: $MIXED_PRECISION

EOF

# ─── train ───────────────────────────────────────────────────────────────────
ARGS=(
  --dataset.repo_id="$DATASET_REPO"
  --policy.type=act
  --policy.device="$DEVICE"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --output_dir="$OUTPUT_DIR"
  --job_name="$RUN_NAME"
  --seed="$SEED"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --wandb.enable="$WANDB_ENABLE"
)
# Only passed when actually enabled: "no" is already the default, and draccus parses
# CLI values as YAML, where the bare word `no` becomes boolean False and fails validation.
[ "$MIXED_PRECISION" = bf16 ] && ARGS+=(--accelerator.mixed_precision=bf16)
# repo_id is only valid alongside push_to_hub; passing it otherwise is harmless but noisy.
[ "$PUSH_TO_HUB" = true ] && ARGS+=(--policy.repo_id="$POLICY_REPO" --policy.private="$PUSH_PRIVATE")
[ "$WANDB_ENABLE" = true ] && ARGS+=(--wandb.project="${WANDB_PROJECT:-so101-act}")
[ "$WANDB_ENABLE" = true ] && [ -n "${WANDB_ENTITY:-}" ] && ARGS+=(--wandb.entity="$WANDB_ENTITY")

mkdir -p "$ROOT_DIR/outputs"
lerobot-train "${ARGS[@]}" 2>&1 | tee -a "$ROOT_DIR/outputs/${RUN_NAME}.log"

cat <<EOF

Done. Checkpoints: $OUTPUT_DIR/checkpoints/
Push a chosen checkpoint (pick it by robot testing, not by loss):
  python scripts/push_checkpoint_to_hf.py $POLICY_REPO $OUTPUT_DIR/checkpoints/last
EOF
