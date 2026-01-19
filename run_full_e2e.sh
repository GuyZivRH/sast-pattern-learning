#!/bin/bash
# Run end-to-end k-fold pattern learning on FULL dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/full_pattern_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/full_output"

echo "======================================"
echo "FULL DATASET E2E RUN"
echo "======================================"
echo ""
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count files
TRAIN_COUNT=$(ls -1 "$DATA_DIR/train"/*.txt 2>/dev/null | wc -l | tr -d ' ')
VAL_COUNT=$(ls -1 "$DATA_DIR/validation"/*.txt 2>/dev/null | wc -l | tr -d ' ')
TEST_COUNT=$(ls -1 "$DATA_DIR/test"/*.txt 2>/dev/null | wc -l | tr -d ' ')

echo "Dataset size:"
echo "  Train: $TRAIN_COUNT files"
echo "  Validation: $VAL_COUNT files"
echo "  Test: $TEST_COUNT files"
echo ""

# Run the main script with caffeinate
echo "Starting k-fold pattern learning with caffeinate..."
echo ""

caffeinate -i python -m process_mining.kfold_pattern_learning.main \
    --train-dir "$DATA_DIR/train" \
    --val-dir "$DATA_DIR/validation" \
    --test-dir "$DATA_DIR/test" \
    --output-dir "$OUTPUT_DIR" \
    --n-folds 5 \
    --platform nim \
    --workers 2 \
    --max-iterations 3 \
    --eval-sample-size 500

echo ""
echo "======================================"
echo "E2E Run Complete!"
echo "======================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""