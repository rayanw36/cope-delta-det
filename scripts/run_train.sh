#!/bin/bash
# Training pipeline: Stage 1 (pre-training) + Stage 2 (fine-tuning)
# Usage: bash scripts/run_train.sh

set -e

echo "=== CoPE-Δ-Det Training Pipeline ==="

# Stage 1: Δ-Encoder Pre-training
echo ""
echo "--- Stage 1: Δ-Encoder Alignment Pre-training ---"
python training/pretrain.py \
    --config configs/train_pretrain.yaml

# Stage 2: End-to-End Fine-tuning
echo ""
echo "--- Stage 2: End-to-End Fine-tuning ---"
python training/finetune.py \
    --config configs/train_finetune.yaml \
    --pretrain_ckpt checkpoints/pretrain/best.pt

echo ""
echo "=== Training complete ==="
echo "Best model: checkpoints/finetune/best.pt"
