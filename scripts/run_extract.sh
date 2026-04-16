#!/bin/bash
# Batch feature extraction from HEVC encoded videos
# Usage: bash scripts/run_extract.sh

set -e

HEVC_ROOT="./data/bdd100k_hevc"
FEAT_ROOT="./data/bdd100k_features"

echo "=== Feature Extraction Pipeline ==="

# Extract for each QP/GOP combination
for QP in 22 27 32 37; do
    for GOP in 8 16 32; do
        INPUT="$HEVC_ROOT/qp${QP}_gop${GOP}"
        OUTPUT="$FEAT_ROOT/qp${QP}_gop${GOP}"

        if [ ! -d "$INPUT" ]; then
            echo "Skipping qp${QP}_gop${GOP}: input not found"
            continue
        fi

        echo "Extracting features: QP=$QP, GOP=$GOP"
        python data/extract_features.py \
            --input_dir "$INPUT" \
            --output_dir "$OUTPUT"

        # Validate
        echo "Validating..."
        python data/extract_features.py \
            --input_dir "$INPUT" \
            --output_dir "$OUTPUT" \
            --validate
    done
done

echo "=== Feature extraction complete ==="
