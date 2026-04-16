#!/bin/bash
# Full evaluation pipeline: main eval + baselines + ablations + codec sweep + Pareto
# Usage: bash scripts/run_eval.sh

set -e

CHECKPOINT="checkpoints/finetune/best.pt"
CONFIG="configs/eval.yaml"

echo "=== CoPE-Δ-Det Evaluation Pipeline ==="

# 1. Main evaluation
echo ""
echo "--- Main Evaluation ---"
python evaluation/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output results/main_results.json

# 2. Codec robustness sweep (QP x GOP)
echo ""
echo "--- Codec Robustness Sweep ---"
python evaluation/codec_robustness.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir results/codec_robustness \
    --qp 22 27 32 37 \
    --gop 8 16 32

# 3. Pareto curves
echo ""
echo "--- Pareto Analysis ---"
python evaluation/pareto.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir results/pareto \
    --gop 8 16 32

# 4. Ablation studies
echo ""
echo "--- Ablation Studies ---"
python evaluation/ablations.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir results/ablations \
    --ablations branch fusion refresh

# 5. Baselines
echo ""
echo "--- Baseline: Full-Decode YOLO ---"
python baselines/full_decode_yolo.py \
    --images_dir ./data/bdd100k/images/val \
    --annotations ./data/bdd100k/annotations_val_coco.json

echo ""
echo "--- Baseline: Mean-MV Propagation ---"
python baselines/mean_mv.py --config "$CONFIG"

echo ""
echo "=== All evaluations complete ==="
echo "Results saved to results/"
