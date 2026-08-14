#!/bin/bash
set -euo pipefail

HF_TOKEN="${1:-${HF_TOKEN:-}}"
STEPS="${2:-50000}"
BATCH_SIZE="${3:-8}"
VENV="/workspace/vla_jepa_env"
export HF_HOME="/workspace/.hf_home"
LOG_FILE="/workspace/act_training.log"

exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "=== ACT Training ==="
echo "Started at: $(date)"
echo "Steps: $STEPS | Batch: $BATCH_SIZE"
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Usage: bash train_act.sh <HF_TOKEN> [steps] [batch_size]"
    exit 1
fi

UV=$(command -v uv || echo "$HOME/.cargo/bin/uv")

if [ ! -f "$VENV/bin/lerobot-train" ]; then
    echo "[1/3] Setting up environment..."
    apt-get install -y -qq git ffmpeg python3.12-venv 2>&1 | tail -1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    rm -rf "$VENV"
    "$UV" venv "$VENV" --python 3.12
    source "$VENV/bin/activate"
    "$UV" pip install -q "lerobot[dataset,training] @ git+https://github.com/huggingface/lerobot.git@main" \
        > /workspace/lerobot_install.log 2>&1 || { cat /workspace/lerobot_install.log; exit 1; }
    python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')"
else
    echo "✓ Environment already installed"
fi

source "$VENV/bin/activate"

echo "[2/3] HuggingFace setup..."
python3 << EOF
from huggingface_hub import login, HfApi

login(token="$HF_TOKEN")

api = HfApi(token="$HF_TOKEN")
refs = api.list_repo_refs("witsense-ai/so101_pick_and_place_ring_33", repo_type="dataset")
existing_tags = [t.name for t in refs.tags]
if "v3.0" not in existing_tags:
    api.create_tag("witsense-ai/so101_pick_and_place_ring_33", tag="v3.0", repo_type="dataset")
    print("  ✓ Tag v3.0 created")
else:
    print(f"  ✓ Tags: {existing_tags}")

from huggingface_hub import snapshot_download
snapshot_download("witsense-ai/so101_pick_and_place_ring_33", repo_type="dataset", token="$HF_TOKEN")
print("  ✓ Dataset ready")
EOF

DATASET_CACHE="$HF_HOME/hub/datasets--witsense-ai--so101_pick_and_place_ring_33/snapshots"
SNAPSHOT=$(ls "$DATASET_CACHE" | head -1)
mkdir -p "$HF_HOME/lerobot/witsense-ai"
rm -f "$HF_HOME/lerobot/witsense-ai/so101_pick_and_place_ring_33"
ln -sf "$DATASET_CACHE/$SNAPSHOT" "$HF_HOME/lerobot/witsense-ai/so101_pick_and_place_ring_33"
echo "✓ Dataset symlinked"

echo ""
echo "[3/3] Starting ACT training: $STEPS steps, batch_size=$BATCH_SIZE"
echo ""

lerobot-train \
  --dataset.repo_id=witsense-ai/so101_pick_and_place_ring_33 \
  --policy.type=act \
  --policy.repo_id=witsense-ai/so101_act_fewshot \
  --policy.device=cuda \
  --output_dir=/workspace/act_training \
  --job_name=act_so101_fewshot \
  --wandb.enable=false \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --num_workers=4 \
  --save_freq=2000 \
  --log_freq=200 \
  --eval_freq=5000

echo ""
echo "=== ACT Training Complete ==="
echo "Finished at: $(date)"
echo "Model: https://huggingface.co/witsense-ai/so101_act_fewshot"
