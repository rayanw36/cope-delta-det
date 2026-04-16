#!/bin/bash
# Batch HEVC encoding at multiple QP/GOP settings
# Usage: bash scripts/run_encode.sh

set -e

INPUT_DIR="./data/bdd100k/bdd100k/videos/train"
OUTPUT_DIR="./data/bdd100k/hevc"

echo "=== HEVC Encoding Pipeline ==="
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# Encode at all QP/GOP combinations
python data/encode_hevc.py \
    --input_dir "$INPUT_DIR" \
    --output_base "$OUTPUT_DIR" \
    --workers 4

echo "=== Encoding complete ==="
