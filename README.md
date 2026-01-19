# K-Fold Pattern Learning System

A 3-phase machine learning pipeline for learning and refining SAST false positive classification patterns using k-fold cross-validation and iterative refinement.

---

## Quick Start

```bash
# Run complete pipeline on your data
python main.py \
  --train-dir process_mining/full_pattern_data/train \
  --val-dir process_mining/full_pattern_data/val \
  --test-dir process_mining/full_pattern_data/test \
  --output-dir output/kfold_learning \
  --platform nim \
  --n-folds 5 \
  --max-iterations 10
```

**Expected runtime**: ~30-60 minutes per issue type (depends on LLM rate limits)

---

## Table of Contents

- [Overview](#overview)
- [Data Requirements](#data-requirements)
- [Pipeline Architecture](#pipeline-architecture)
  - [Phase 1: K-Fold Cross-Validation](#phase-1-k-fold-cross-validation)
  - [Phase 2: Iterative Refinement](#phase-2-iterative-refinement)
  - [Phase 3: Final Test Evaluation](#phase-3-final-test-evaluation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Output](#output)
- [How It Works (Deep Dive)](#how-it-works-deep-dive)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system learns patterns that distinguish between:
- **False Positives (FP)**: SAST flags safe code incorrectly
- **True Positives (TP)**: SAST correctly identifies vulnerabilities

### Three-Phase Approach

```
Phase 1: K-Fold CV        Phase 2: Refinement       Phase 3: Final Test
(on train/ 60%)          (train + val, 80%)        (test/ 20%)
     │                          │                          │
     ├─► Learn patterns         ├─► Refine on TRAIN       ├─► Final metrics
     │   from 5 folds           │   Validate on VAL       │   on held-out test
     └─► Merge best ────────────┴─► Converge ─────────────┴─► Report
```

### Key Features

✅ **Proper ML Pipeline**: Train/Val/Test split with no data leakage
✅ **Stratified K-Fold**: Preserves FP/TP ratio across folds
✅ **Early Stopping**: Prevents overfitting using validation set
✅ **Overfitting Detection**: Tracks train vs val metrics
✅ **Pattern Deduplication**: LLM merges similar patterns
✅ **Iterative Refinement**: Learns from misclassifications

---

## Data Requirements

### Input Format

Each `.txt` file contains validation entries in this format:

```
Entry ID: package-name/file.c:123-RESOURCE_LEAK-001
Package Name: package-name
Issue Type: RESOURCE_LEAK
CWE: CWE-401
Error Trace: [SAST error trace here]
Source Code: [Code snippet here]
Ground Truth Classification: TRUE_POSITIVE
Ground Truth Justification: [Analyst explanation]
---
```

### Directory Structure

**You must provide 3 directories** (pre-split recommended):

```
your_data/
├── train/              # 60% of data - for k-fold pattern learning
│   ├── package1.txt
│   ├── package2.txt
│   └── ...
├── val/                # 20% of data - for early stopping in Phase 2
│   ├── package10.txt
│   └── ...
└── test/               # 20% of data - held-out for final evaluation
    ├── package15.txt
    └── ...
```

**Example with existing data**:
```bash
ls process_mining/full_pattern_data/train  # 118 packages
ls process_mining/full_pattern_data/val    # 39 packages
ls process_mining/full_pattern_data/test   # 39 packages
```

### Data Split Recommendations

| Total Packages | Train (60%) | Val (20%) | Test (20%) |
|----------------|-------------|-----------|------------|
| 50 packages    | 30 files    | 10 files  | 10 files   |
| 100 packages   | 60 files    | 20 files  | 20 files   |
| 200 packages   | 120 files   | 40 files  | 40 files   |

**How to split your data**:
```bash
# Option 1: Manual split (recommended)
# Ensure balanced FP/TP ratio in each split

# Option 2: Random split (quick, but may be imbalanced)
python scripts/split_data.py \
  --input-dir all_data/ \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --output-dir process_mining/full_pattern_data/
```

---

## Pipeline Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     Input: Labeled Dataset                    │
│              (SAST findings + ground truth labels)            │
└───────────────────────────┬──────────────────────────────────┘
                            │
                       Manual Split
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    ┌────────┐         ┌────────┐        ┌────────┐
    │ Train  │         │  Val   │        │  Test  │
    │ (60%)  │         │ (20%)  │        │ (20%)  │
    └───┬────┘         └───┬────┘        └───┬────┘
        │                  │                  │
        │                  │                  │ (held-out)
        │                  │                  │
   ┌────▼──────────────────▼───┐             │
   │   PHASE 1: K-FOLD CV       │             │
   │   (Learn from train/)      │             │
   │   • 5 folds × 80/20 split  │             │
   │   • Learn patterns         │             │
   │   • Merge best patterns    │             │
   └────────────┬───────────────┘             │
                │                             │
         Merged Patterns                      │
                │                             │
   ┌────────────▼───────────────┐             │
   │   PHASE 2: REFINEMENT      │             │
   │   (Refine on train+val)    │             │
   │   • Refine on train/       │             │
   │   • Validate on val/       │             │
   │   • Early stopping         │             │
   └────────────┬───────────────┘             │
                │                             │
         Final Patterns                       │
                │                             │
   ┌────────────▼───────────────┐             │
   │   PHASE 3: TEST            │◄────────────┘
   │   (Evaluate on test/)      │
   │   • Final metrics          │
   │   • Generate reports       │
   └────────────┬───────────────┘
                ▼
         Final Report
```

---

## Phase 1: K-Fold Cross-Validation

### Goal
Learn initial patterns from training data using k-fold cross-validation.

### Input
- `train/` directory (60% of original data)
- Issue type (e.g., `RESOURCE_LEAK`)

### Process

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Stratified K-Fold Split (n=5 folds)                 │
├─────────────────────────────────────────────────────────────┤
│ Input: 118 files from train/                                │
│                                                              │
│ StratifiedKFoldSplitter:                                     │
│   1. Analyze each file → count issues, FP ratio             │
│   2. Assign strata: {size}_{fp_bucket}                      │
│      • size: small/medium/large (by issue count)            │
│      • fp_bucket: low/medium/high (by FP ratio)             │
│   3. Distribute files across 5 folds evenly                 │
│                                                              │
│ Output: 5 folds, each with:                                 │
│   • train_files (80% of train/ = ~94 files)                 │
│   • val_files (20% of train/ = ~24 files)                   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Learn Patterns for Each Fold (5 iterations)         │
├─────────────────────────────────────────────────────────────┤
│ For each fold (0-4):                                         │
│                                                              │
│   A. PATTERN LEARNING (PatternLearner)                      │
│      ─────────────────────────────────                       │
│      train_files (94 files) → PatternLearner                │
│        1. Parse ALL entries from all 94 files               │
│        2. Filter to issue_type (e.g., RESOURCE_LEAK)        │
│           → Found: 1068 entries                             │
│        3. Random sample if > 50 entries                     │
│           → Shuffle & take first 50 entries                 │
│           → Why 50? Token limit (~10k tokens for input)     │
│        4. Build LLM prompt with 50 entries                  │
│           → ~25 FP examples                                 │
│           → ~25 TP examples                                 │
│        5. LLM generates patterns                            │
│           → Response: {"fp": [...], "tp": [...]}            │
│                                                              │
│      Save: RESOURCE_LEAK_fold_0_patterns.json               │
│                                                              │
│   B. FOLD EVALUATION (FoldEvaluator)                        │
│      ─────────────────────────                               │
│      val_files (24 files) + fold_patterns → Evaluate        │
│        1. Parse ALL entries from val_files                  │
│           (NO sampling - evaluate on ALL data)              │
│        2. For each entry:                                   │
│           • LLM classifies using learned patterns           │
│           • Compare with ground truth                       │
│        3. Calculate metrics: F1, Precision, Recall          │
│                                                              │
│      Save: RESOURCE_LEAK_fold_0_evaluation.json             │
│        • metrics: {f1, precision, recall, accuracy}         │
│        • confusion_matrix                                   │
│        • per-entry results                                  │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Merge Patterns Across Folds (PatternMerger)         │
├─────────────────────────────────────────────────────────────┤
│ Input: 5 pattern files                                       │
│   • fold_0_patterns.json (4 FP, 3 TP patterns)              │
│   • fold_1_patterns.json (5 FP, 4 TP patterns)              │
│   • fold_2_patterns.json (3 FP, 3 TP patterns)              │
│   • fold_3_patterns.json (4 FP, 3 TP patterns)              │
│   • fold_4_patterns.json (4 FP, 4 TP patterns)              │
│                                                              │
│ LLM Deduplication:                                           │
│   • Identify duplicate patterns (same code idiom)           │
│   • Preserve unique patterns                                │
│   • Combine insights from duplicates                        │
│   • Renumber pattern IDs sequentially                       │
│                                                              │
│ Output: RESOURCE_LEAK_merged_patterns.json                  │
│   • ~5-7 unique FP patterns (after dedup)                   │
│   • ~4-5 unique TP patterns (after dedup)                   │
│                                                              │
│ Metrics: Average F1 across 5 folds                          │
│   avg_f1 = mean([0.72, 0.75, 0.71, 0.74, 0.73]) = 0.73     │
└─────────────────────────────────────────────────────────────┘
                           ▼
                  MERGED PATTERNS
               (Input to Phase 2)
```

### Key Details

**Q: How many entries are used for learning?**
A: **Maximum 50 entries per issue type per fold** (randomly sampled if more)

**Q: How many entries are used for evaluation?**
A: **ALL entries** in the validation fold (no sampling)

**Q: What's the train/val split per fold?**
A: **80% train, 20% val** (e.g., 94 files train, 24 files val)

**Q: Why sample to 50 entries?**
A: **LLM context window limit**. 50 entries ≈ 10k tokens, leaving room for LLM response.

### Output

```
output/phase1_kfold_results/
├── RESOURCE_LEAK_fold_0_patterns.json      # Patterns learned from fold 0
├── RESOURCE_LEAK_fold_0_evaluation.json    # Fold 0 evaluation metrics
├── RESOURCE_LEAK_fold_1_patterns.json
├── RESOURCE_LEAK_fold_1_evaluation.json
├── ...
├── RESOURCE_LEAK_merged_patterns.json      # ⭐ Deduplicated patterns
└── phase1_complete_results.json            # Complete Phase 1 results
```

---

## Phase 2: Iterative Refinement

### Goal
Refine patterns using train+val data with **proper ML pipeline** (no data leakage).

### Input
- Merged patterns from Phase 1
- `train/` directory (60% of original) - for refinement
- `val/` directory (20% of original) - for early stopping

### Process

```
┌─────────────────────────────────────────────────────────────┐
│ Initialization                                               │
├─────────────────────────────────────────────────────────────┤
│ • Load merged patterns from Phase 1                         │
│ • Keep train/ and val/ separate (NO combining!)             │
│ • Set best_val_f1 = 0.0                                     │
│ • Set no_improvement_count = 0                              │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Iteration Loop (max 10 iterations, or until convergence)    │
├─────────────────────────────────────────────────────────────┤
│ While iteration < max_iterations AND not converged:         │
│                                                              │
│   Step A: Evaluate on TRAIN (find refinement targets)       │
│   ──────────────────────────────────────────────────        │
│   current_patterns + train/ → FoldEvaluator                 │
│     • Parse ALL entries from train/ files                   │
│     • Classify each entry using current patterns            │
│     • Calculate metrics: train_f1, train_precision, etc.    │
│     • Track misclassified entries                           │
│                                                              │
│   Example Output:                                            │
│     Train F1: 0.850                                         │
│     Train Precision: 0.820                                  │
│     Train Recall: 0.880                                     │
│     Train Misclassified: 15 entries                         │
│                                                              │
│   Step B: Evaluate on VAL (check generalization)            │
│   ─────────────────────────────────────────────             │
│   current_patterns + val/ → FoldEvaluator                   │
│     • Parse ALL entries from val/ files                     │
│     • Classify each entry using current patterns            │
│     • Calculate metrics: val_f1, val_precision, etc.        │
│                                                              │
│   Example Output:                                            │
│     Val F1: 0.780                                           │
│     Val Precision: 0.750                                    │
│     Val Recall: 0.810                                       │
│                                                              │
│   Overfitting Check:                                         │
│     IF train_f1 - val_f1 > 0.1:                             │
│       ⚠️  Warning: Potential overfitting                    │
│                                                              │
│   Step C: Check Convergence (based on VAL F1)               │
│   ──────────────────────────────────────────                │
│   improvement = val_f1 - best_val_f1                        │
│                                                              │
│   IF improvement <= 0.01 (1%):                              │
│      no_improvement_count += 1                              │
│      IF no_improvement_count >= 3:                          │
│         CONVERGED → Exit loop                               │
│   ELSE:                                                      │
│      best_val_f1 = val_f1                                   │
│      no_improvement_count = 0                               │
│                                                              │
│   IF train_misclassified == 0:                              │
│      PERFECT TRAIN CLASSIFICATION → Exit loop               │
│                                                              │
│   Step D: Refine Patterns (based on TRAIN misclass only)    │
│   ─────────────────────────────────────────────────────     │
│   train_misclassified_entries → PatternRefiner              │
│     1. Categorize misclassifications:                       │
│        • False Negatives (FN): real vulns missed            │
│        • False Positives Missed: safe code flagged          │
│     2. Sample max 20 misclassified entries                  │
│     3. Build refinement prompt                              │
│     4. LLM generates refinement suggestions:                │
│        {                                                     │
│          "add": [new patterns for uncovered cases],         │
│          "modify": [improve existing patterns],             │
│          "remove": [redundant patterns]                     │
│        }                                                     │
│                                                              │
│   Step E: Apply Refinements                                 │
│   ───────────────────────                                   │
│   current_patterns = apply_refinements(current_patterns,    │
│                                         refinements)         │
│     • Add new patterns to fp/tp lists                       │
│     • Update summaries for modified patterns                │
│     • Remove patterns by pattern_id                         │
│                                                              │
│   Save:                                                      │
│     • RESOURCE_LEAK_iteration_1_patterns.json               │
│     • RESOURCE_LEAK_iteration_1_refinements.json            │
│                                                              │
│   iteration += 1                                             │
└─────────────────────────────────────────────────────────────┘
                           ▼
                  FINAL PATTERNS
               (Input to Phase 3)
```

### ML Best Practices (NEW - 2026-01-18 Fix)

✅ **No Data Leakage**: Val set NEVER used for refinement
✅ **Early Stopping**: Converges when val F1 stops improving
✅ **Overfitting Detection**: Warns if train F1 >> val F1
✅ **Proper Generalization**: Uses val set to check pattern quality

### Example Iteration Log

```
--- Iteration 1/10 ---
Evaluating on TRAIN (118 files)...
  Train F1: 0.750
  Train Precision: 0.720
  Train Recall: 0.780
Evaluating on VAL (39 files)...
  Val F1: 0.720
  Val Precision: 0.700
  Val Recall: 0.740
  Val F1 Improvement: +0.000 (first iteration)
  Train Misclassified: 25

--- Iteration 2/10 ---
Evaluating on TRAIN (118 files)...
  Train F1: 0.820
  Train Precision: 0.800
  Train Recall: 0.840
Evaluating on VAL (39 files)...
  Val F1: 0.780
  Val Precision: 0.760
  Val Recall: 0.800
  Val F1 Improvement: +0.060 (above threshold)
  Train Misclassified: 18

--- Iteration 3/10 ---
  ...
  ⚠️  Potential overfitting: Train F1 (0.920) >> Val F1 (0.770)
  Val F1 Improvement: -0.010 (below threshold)

Convergence reached: No VAL F1 improvement for 3 iterations
```

### Output

```
output/phase2_refinement_results/
├── RESOURCE_LEAK_iteration_1_patterns.json     # Patterns after iter 1
├── RESOURCE_LEAK_iteration_1_refinements.json  # Refinements applied
├── RESOURCE_LEAK_iteration_2_patterns.json
├── RESOURCE_LEAK_iteration_2_refinements.json
├── ...
└── phase2_complete_results.json                # ⭐ Complete Phase 2 results
    └── Contains: initial/final patterns, all iterations, convergence info
```

---

## Phase 3: Final Test Evaluation

### Goal
Evaluate refined patterns on **completely held-out** test set.

### Input
- Final patterns from Phase 2
- `test/` directory (20% of original, never seen before)

### Process

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Evaluate on Test Set                                │
├─────────────────────────────────────────────────────────────┤
│ final_patterns + test/ → FoldEvaluator                      │
│   • Parse ALL entries from test/ files                      │
│   • For each test entry:                                    │
│     - LLM classifies using final patterns                   │
│     - Compare with ground truth                             │
│   • Calculate final metrics: F1, Precision, Recall          │
│                                                              │
│ Output:                                                      │
│   • Final F1, Precision, Recall, Accuracy                   │
│   • Confusion matrix                                        │
│   • Misclassified examples with details                     │
│   • Per-issue-type breakdown                                │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Generate Comprehensive Reports                      │
├─────────────────────────────────────────────────────────────┤
│ A. JSON Results                                              │
│    • phase3_complete_results.json                           │
│    • RESOURCE_LEAK_test_results.json (per issue type)       │
│                                                              │
│ B. Markdown Report (phase3_report.md)                       │
│    ├─ Overall Summary Table                                 │
│    │   (F1, Precision, Recall per issue type)               │
│    ├─ Detailed Per-Issue-Type Analysis                      │
│    │   • Test set statistics (FP/TP counts)                 │
│    │   • Pattern counts                                     │
│    │   • Sample misclassified examples (up to 10)           │
│    └─ Final Recommendations                                 │
│                                                              │
│ C. Complete Pipeline Report (final_report.md)               │
│    ├─ Phase 1 Summary (k-fold F1)                          │
│    ├─ Phase 2 Summary (iterations, improvement)            │
│    ├─ Phase 3 Summary (final test metrics)                 │
│    └─ Per-issue-type progression table:                    │
│       Phase 1 F1 → Phase 2 F1 → Final Test F1              │
└─────────────────────────────────────────────────────────────┘
                           ▼
                   FINAL REPORT
```

### Output

```
output/phase3_test_results/
├── RESOURCE_LEAK_test_results.json     # Per-issue-type results
├── phase3_complete_results.json        # Complete Phase 3 results
└── phase3_report.md                    # ⭐ Human-readable report

output/
├── complete_pipeline_results.json      # ⭐ All 3 phases combined
└── final_report.md                     # ⭐ Executive summary
```

---

## Usage Examples

### Example 1: Run Complete Pipeline

```bash
python main.py \
  --train-dir process_mining/full_pattern_data/train \
  --val-dir process_mining/full_pattern_data/val \
  --test-dir process_mining/full_pattern_data/test \
  --output-dir output/kfold_learning \
  --platform nim \
  --n-folds 5 \
  --max-iterations 10
```

**Expected Output**:
```
Phase 1: K-Fold CV completed in 25.3 minutes
  Avg F1 across folds: 0.730
  Merged patterns: 6 FP, 5 TP

Phase 2: Refinement completed in 18.7 minutes
  Iterations: 6
  Final Val F1: 0.785
  Improvement: +0.055

Phase 3: Test evaluation completed in 8.2 minutes
  Final Test F1: 0.772
  Final Test Precision: 0.750
  Final Test Recall: 0.795
```

### Example 2: Run Only Phase 1 (Pattern Learning)

```bash
python main.py \
  --phase 1 \
  --train-dir process_mining/full_pattern_data/train \
  --output-dir output/phase1_only \
  --issue-types RESOURCE_LEAK UNINIT
```

### Example 3: Run Only Phase 2 (Refinement)

Requires Phase 1 results to exist first.

```bash
python main.py \
  --phase 2 \
  --train-dir process_mining/full_pattern_data/train \
  --val-dir process_mining/full_pattern_data/val \
  --output-dir output/phase2_only \
  --max-iterations 5
```

### Example 4: Run Only Phase 3 (Test Evaluation)

Requires Phase 2 results to exist first.

```bash
python main.py \
  --phase 3 \
  --test-dir process_mining/full_pattern_data/test \
  --output-dir output/phase3_only
```

---

## Configuration

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--train-dir` | Yes | - | Training data directory (60% of data) |
| `--val-dir` | Conditional | - | Validation directory (20%, required for full pipeline) |
| `--test-dir` | Conditional | - | Test directory (20%, required for full pipeline) |
| `--output-dir` | Yes | - | Output directory for all results |
| `--platform` | No | `nim` | LLM platform: `local` or `nim` |
| `--n-folds` | No | `5` | Number of folds for k-fold CV |
| `--max-iterations` | No | `10` | Max refinement iterations (Phase 2) |
| `--workers` | No | `1` | Parallel workers for evaluation |
| `--seed` | No | `42` | Random seed for reproducibility |
| `--issue-types` | No | auto-detect | Specific issue types (e.g., `RESOURCE_LEAK UNINIT`) |
| `--phase` | No | all | Run specific phase: `1`, `2`, or `3` |

### Platform Configuration

| Platform | Model | Rate Limit | Use Case |
|----------|-------|------------|----------|
| `nim` | `meta/llama-3.3-70b-instruct` | 30 req/min | Production (recommended) |
| `local` | `llama3.1:70b` | unlimited | Local testing |

---

## Output

### Complete Output Structure

```
output_dir/
├── phase1_kfold_results/
│   ├── RESOURCE_LEAK_fold_0_patterns.json
│   ├── RESOURCE_LEAK_fold_0_evaluation.json
│   ├── RESOURCE_LEAK_fold_1_patterns.json
│   ├── RESOURCE_LEAK_fold_1_evaluation.json
│   ├── ...
│   ├── RESOURCE_LEAK_merged_patterns.json      ← Phase 1 output
│   └── phase1_complete_results.json
│
├── phase2_refinement_results/
│   ├── RESOURCE_LEAK_iteration_1_patterns.json
│   ├── RESOURCE_LEAK_iteration_1_refinements.json
│   ├── RESOURCE_LEAK_iteration_2_patterns.json
│   ├── RESOURCE_LEAK_iteration_2_refinements.json
│   ├── ...
│   └── phase2_complete_results.json            ← Phase 2 output
│
├── phase3_test_results/
│   ├── RESOURCE_LEAK_test_results.json
│   ├── phase3_complete_results.json            ← Phase 3 output
│   └── phase3_report.md
│
├── complete_pipeline_results.json              ← All phases combined
└── final_report.md                             ← Executive summary
```

### Pattern Format

```json
{
  "fp": [
    {
      "pattern_id": "RESOURCE_LEAK-FP-001",
      "group": "A: Custom Allocator Patterns",
      "summary": "SAST reports freeing an invalid pointer when code uses a custom allocator that prepends metadata. The pattern: allocator adds header struct, returns offset pointer, free() subtracts offset. Key indicators: magic/size fields, pointer arithmetic in free(), paired alloc/free. Safe because calculated pointer IS the original allocation."
    }
  ],
  "tp": [
    {
      "pattern_id": "RESOURCE_LEAK-TP-001",
      "group": "A: Unfreed Memory",
      "summary": "Memory leak confirmed as 'file' opened at line 891 is not closed before going out of scope at line 920. No close() call in any code path. Severity: Critical - can lead to file descriptor exhaustion. Fix: Add fclose() before all return statements."
    }
  ]
}
```

---

## How It Works (Deep Dive)

### Stratified K-Fold Splitting

**Why stratify?** Ensures each fold has similar FP/TP ratio distribution.

**9 Strata** based on:
1. **Size**: small (≤5 issues), medium (≤20), large (>20)
2. **FP Ratio**: low (<0.25), medium (<0.75), high (≥0.75)

**Example**:
```
Stratum: large_high
  • Packages with >20 issues AND FP ratio ≥0.75
  • 12 packages total
  • Distributed: 2-3 packages per fold
```

### Sampling Strategy

| Stage | Data | Sampling | Purpose |
|-------|------|----------|---------|
| **Phase 1 Learning** | train_files (80% of fold) | Random 50 entries | Fit LLM context window |
| **Phase 1 Evaluation** | val_files (20% of fold) | ALL entries | Accurate metrics |
| **Phase 2 Refinement** | train_misclassified | Max 20 entries | LLM refinement suggestions |
| **Phase 2 Evaluation** | train/ and val/ | ALL entries | Accurate train/val metrics |
| **Phase 3 Test** | test/ | ALL entries | Final evaluation |

### Convergence Criteria (Phase 2)

Phase 2 stops when ANY of these conditions is met:

1. **Val F1 Improvement < 1%** for 3 consecutive iterations (early stopping)
2. **Perfect TRAIN Classification** (no train misclassifications)
3. **Max Iterations** reached (default: 10)

### Overfitting Detection

```
Iteration | Train F1 | Val F1 | Gap  | Status
----------|----------|--------|------|--------
   1      |  0.750   | 0.720  | 0.03 | ✅ Good
   2      |  0.820   | 0.780  | 0.04 | ✅ Good
   3      |  0.880   | 0.785  | 0.10 | ⚠️  Warning threshold
   4      |  0.920   | 0.782  | 0.14 | ❌ Overfitting detected
   5      |  0.950   | 0.778  | 0.17 | ❌ Stop (val F1 dropped)
```

**Action**: Use patterns from iteration 2-3 (best val F1).

---

## Troubleshooting

### Issue: "No patterns generated"

**Symptoms**: Empty pattern files in Phase 1.

**Causes**:
- No entries found for specified issue type
- LLM platform inaccessible
- Insufficient data (< 5 entries)

**Solutions**:
```bash
# Check if issue type exists in your data
grep "Issue Type:" train/*.txt | sort | uniq -c

# Verify LLM platform
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.3-70b-instruct","messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# Increase samples if data is sparse
python main.py --max-entries 100  # Default is 50
```

### Issue: "Phase 2 not converging"

**Symptoms**: Runs all 10 iterations without converging.

**Causes**:
- Patterns too generic (can't improve further)
- Patterns too specific (overfitting)
- Val set too small

**Solutions**:
```bash
# Check iteration logs
cat output/phase2_refinement_results/phase2_complete_results.json | jq '.issue_types.RESOURCE_LEAK.iterations[] | {iter:.iteration, train_f1, val_f1}'

# If val F1 oscillates: reduce max iterations
python main.py --max-iterations 5

# If val F1 plateaus early: patterns are as good as they'll get (success!)
```

### Issue: "Test F1 << Val F1" (Overfitting)

**Symptoms**:
```
Phase 2 Final Val F1: 0.850
Phase 3 Test F1: 0.720  ← Much lower!
```

**Causes**:
- Overfitting to train+val distribution
- Test set has different characteristics

**Solutions**:
```bash
# Check overfitting gap in Phase 2 results
cat output/phase2_refinement_results/phase2_complete_results.json | \
  jq '.issue_types.RESOURCE_LEAK.convergence_info.overfitting_gap'

# If gap > 0.15: Patterns overfit, use earlier iteration
# Review iteration files to find best val F1 with smallest train/val gap

# Use patterns from earlier iteration
cp output/phase2_refinement_results/RESOURCE_LEAK_iteration_3_patterns.json \
   final_patterns.json
```

### Issue: "Rate limit errors"

**Symptoms**: `HTTP 429 Too Many Requests`

**Cause**: NVIDIA NIM rate limit (30 req/min)

**Solution**: Automatic rate limiting is built-in. Just wait - the pipeline will retry.

---

## Performance Considerations

| Factor | Impact | Details |
|--------|--------|---------|
| **LLM Calls** | 30-60 min/issue | Phase 1: k×2 calls, Phase 2: iters×3 calls, Phase 3: 1 call |
| **Rate Limiting** | Automatic | 30 req/min for NVIDIA NIM |
| **Parallelization** | Use `--workers 4` | Speeds up evaluation (not pattern learning) |
| **Token Usage** | Max 50 entries | Learning uses ~10k tokens input, ~6k output |

---

## Module Organization

The k-fold pattern learning system uses these core modules from `process_mining/core/`:

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `core/data_models.py` | Data structures | ValidationEntry, ClassificationResult |
| `core/parsers.py` | Parse validation .txt files | ValidationEntryParser |
| `core/classifiers.py` | LLM-based classification | PatternBasedClassifier |
| `core/metrics.py` | Evaluation metrics | EvaluationMetrics |
| `core/evaluators.py` | End-to-end evaluation | PatternEvaluator |

**Import examples**:
```python
from process_mining.core.parsers import ValidationEntryParser
from process_mining.core.classifiers import PatternBasedClassifier
from process_mining.core.metrics import EvaluationMetrics

# K-fold specific modules
from process_mining.kfold_pattern_learning.pattern_learner import PatternLearner
from process_mining.kfold_pattern_learning.config import TOP_10_ISSUE_TYPES, LLM_DEFAULTS
```

---

## References

- **ML Best Practices**: [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) (Hastie et al.)
- **Early Stopping**: [Prechelt (1998)](https://link.springer.com/chapter/10.1007/3-540-49430-8_3)
- **Core Modules**: `process_mining/core/` (data models, parsers, classifiers, metrics, evaluators)
- **Configuration**: `process_mining/kfold_pattern_learning/config.py` (centralized constants)

---

## Authors

Developed for SAST AI Workflow project.

**Last Updated**: 2026-01-19 (Directory reorganization)