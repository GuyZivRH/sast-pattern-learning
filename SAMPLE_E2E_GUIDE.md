# Sample E2E Testing Guide

## Overview

This guide explains how to run end-to-end testing of the k-fold pattern learning pipeline using a small sample dataset.

## Quick Start

```bash
# 1. Create sample dataset (already done)
python process_mining/create_sample_dataset.py

# 2. Validate dataset
python process_mining/validate_sample_dataset.py

# 3. Run e2e test
./process_mining/run_sample_e2e.sh
```

## Sample Dataset Details

**Location:** `process_mining/data/sample_pattern_data/`

**Size:**
- Train: 10 files, 435 entries
- Val: 10 files, 337 entries
- Test: 10 files, 390 entries
- Total: 30 files, 1,162 entries

**Issue Types:**
- RESOURCE_LEAK (most common)
- OVERRUN
- INTEGER_OVERFLOW
- UNINIT
- USE_AFTER_FREE
- Plus 13 additional types across splits

**Ground Truth Distribution:**
- FALSE_POSITIVE: ~80%
- TRUE_POSITIVE: ~20%

## E2E Pipeline Configuration

The sample e2e script (`run_sample_e2e.sh`) runs with these settings:

```bash
--n-folds 3                  # 3-fold cross-validation in Phase 1
--platform nim               # NVIDIA NIM for LLM
--workers 2                  # 2 parallel workers
--max-iterations 3           # Max 3 refinement iterations in Phase 2
--eval-sample-size 10        # Sample 10 entries max for Phase 2 eval
```

### Why These Settings?

- **Small folds (n=3)**: With only 10 train files, keeps folds meaningful
- **Low iteration count**: Faster testing, still validates convergence logic
- **Small eval sample**: Phase 2 samples max 10 entries per eval to speed up testing

## Expected Runtime

With these settings, expect:
- Phase 1 (k-fold learning): ~5-10 minutes per fold
- Phase 2 (refinement): ~2-5 minutes per iteration
- Total: ~30-45 minutes

## Output Structure

After running, you'll find:

```
process_mining/outputs/sample_output/
├── phase1_kfold_results/
│   ├── phase1_complete_results.json
│   ├── fold_1/
│   │   ├── patterns_RESOURCE_LEAK.json
│   │   ├── patterns_OVERRUN.json
│   │   └── ...
│   ├── fold_2/
│   └── fold_3/
├── phase2_refinement_results/
│   ├── phase2_complete_results.json
│   ├── RESOURCE_LEAK/
│   │   ├── iteration_1/
│   │   ├── iteration_2/
│   │   └── ...
│   └── OVERRUN/
└── final_patterns/
    ├── RESOURCE_LEAK_patterns.json
    ├── OVERRUN_patterns.json
    └── ...
```

## Validating Results

After the run completes, check:

1. **Phase 1 Results:**
   ```bash
   cat process_mining/outputs/sample_output/phase1_kfold_results/phase1_complete_results.json | jq '.summary'
   ```

2. **Phase 2 Convergence:**
   ```bash
   cat process_mining/outputs/sample_output/phase2_refinement_results/phase2_complete_results.json | jq '.issue_types.RESOURCE_LEAK.convergence_info'
   ```

3. **Final Patterns:**
   ```bash
   ls -lh process_mining/outputs/sample_output/final_patterns/
   ```

## Customizing the Run

### Run with Different Settings

Edit `run_sample_e2e.sh` or run directly:

```bash
python -m process_mining.kfold_pattern_learning.main \
    --train-dir process_mining/data/sample_pattern_data/train \
    --val-dir process_mining/data/sample_pattern_data/val \
    --test-dir process_mining/data/sample_pattern_data/test \
    --output-dir process_mining/sample_output \
    --n-folds 5 \
    --max-iterations 5 \
    --eval-sample-size 20
```

### Run Individual Phases

**Phase 1 only (k-fold learning):**
```bash
python -m process_mining.kfold_pattern_learning.main \
    --train-dir process_mining/data/sample_pattern_data/train \
    --output-dir process_mining/sample_output \
    --phase 1
```

**Phase 2 only (refinement):**
```bash
python -m process_mining.kfold_pattern_learning.main \
    --train-dir process_mining/data/sample_pattern_data/train \
    --val-dir process_mining/data/sample_pattern_data/val \
    --output-dir process_mining/sample_output \
    --phase 2
```

**Phase 3 only (final evaluation):**
```bash
python -m process_mining.kfold_pattern_learning.main \
    --test-dir process_mining/data/sample_pattern_data/test \
    --output-dir process_mining/sample_output \
    --phase 3
```

### Run Specific Issue Types

```bash
./process_mining/run_sample_e2e.sh --issue-types RESOURCE_LEAK OVERRUN
```

## Troubleshooting

### Out of Memory
If you run out of memory, reduce:
- `--eval-sample-size` (lower = less memory)
- `--workers` (1 = sequential processing)

### LLM Rate Limits
The pipeline includes automatic rate limiting. If you hit limits:
- Reduce `--workers` to 1
- Add delays between LLM calls (edit code if needed)

### Missing API Key
Ensure you have the NVIDIA NIM API key set:
```bash
export LLM_API_KEY="your-api-key"
```

Or use local LLM:
```bash
# Edit run_sample_e2e.sh, change:
--platform local
```

## Next Steps

After validating with the sample dataset, run the full pipeline:

```bash
python -m process_mining.kfold_pattern_learning.main \
    --train-dir process_mining/full_pattern_data \
    --output-dir process_mining/full_output \
    --n-folds 5 \
    --max-iterations 10 \
    --eval-sample-size 500
```

## Files Reference

| File | Purpose |
|------|---------|
| `create_sample_dataset.py` | Creates sample dataset from full data |
| `validate_sample_dataset.py` | Validates dataset structure and content |
| `run_sample_e2e.sh` | Runs full e2e pipeline on sample data |
| `sample_pattern_data/` | Sample dataset (train/val/test splits) |
| `sample_output/` | Output from e2e run |

## Test Coverage

Before running e2e, verify tests pass:

```bash
pytest tests/unit/process_mining/kfold_pattern_learning/ -v
```

Expected: 59/59 tests passing (100%)