#!/bin/bash
# Run end-to-end k-fold pattern learning on sample dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/sample_pattern_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/sample_output"

# Check if sample dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Sample dataset not found at: $DATA_DIR"
    echo "Please run: python process_mining/create_sample_dataset.py"
    exit 1
fi

echo "======================================"
echo "Running K-Fold Pattern Learning E2E"
echo "======================================"
echo ""
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count files
TRAIN_COUNT=$(ls -1 "$DATA_DIR/train"/*.txt 2>/dev/null | wc -l | tr -d ' ')
VAL_COUNT=$(ls -1 "$DATA_DIR/val"/*.txt 2>/dev/null | wc -l | tr -d ' ')
TEST_COUNT=$(ls -1 "$DATA_DIR/test"/*.txt 2>/dev/null | wc -l | tr -d ' ')

echo "Dataset size:"
echo "  Train: $TRAIN_COUNT files"
echo "  Val: $VAL_COUNT files"
echo "  Test: $TEST_COUNT files"
echo ""

# Run the main script
echo "Starting k-fold pattern learning..."
echo ""

python -m process_mining.kfold_pattern_learning.main \
    --train-dir "$DATA_DIR/train" \
    --val-dir "$DATA_DIR/val" \
    --test-dir "$DATA_DIR/test" \
    --output-dir "$OUTPUT_DIR" \
    --n-folds 3 \
    --platform nim \
    --workers 2 \
    --max-iterations 3 \
    --eval-sample-size 10

echo ""
echo "======================================"
echo "E2E Run Complete!"
echo "======================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - Phase 1: $OUTPUT_DIR/phase1_kfold_results/phase1_complete_results.json"
echo "  - Phase 2: $OUTPUT_DIR/phase2_refinement_results/phase2_complete_results.json"
echo "  - Final patterns: $OUTPUT_DIR/final_patterns/"
echo ""