# K-Fold Pattern Learning System

A 3-phase machine learning pipeline for learning and refining SAST false positive classification patterns using k-fold cross-validation and iterative refinement.

---

## Quick Start

```bash
# Run complete pipeline on your data
python -m process_mining.kfold_pattern_learning.main \
  --train-dir data/train \
  --val-dir data/val \
  --test-dir data/test \
  --output-dir outputs/kfold_learning \
  --platform vertex \
  --n-folds 3 \
  --max-iterations 10 \
  --workers 4
```

**Quick test with sample data**:
```bash
./run_sample_e2e.sh  # Runs on small sample dataset (~30-45 min)
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
     │   from 3 folds           │   Validate on VAL       │   on held-out test
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
# Option 1: Use the stratified data splitter (recommended)
# Stratifies by (issue_type, classification) to maintain distribution
python -m process_mining.kfold_pattern_learning.stratified_data_splitter \
  source_dir/ \
  output_dir/ \
  --train 0.6 \
  --val 0.2 \
  --test 0.2 \
  --seed 42

# Option 2: Include all issue types (default: top 10 only)
python -m process_mining.kfold_pattern_learning.stratified_data_splitter \
  source_dir/ \
  output_dir/ \
  --all-types

# Option 3: Manual split
# Ensure balanced FP/TP ratio in each split
```

**Stratified splitting ensures**:
- Each (issue_type, classification) group is distributed across train/val/test
- Maintains TP/FP balance for each issue type in all splits
- All issue types appear in train, val, and test (when possible)
- By default, filters to top 10 issue types for pattern learning
- No data leakage between splits

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
   │   • 3 folds (stratified)   │             │
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

## Phase 1: K-Fold Cross-Validation with Iterative Refinement

### Goal
Learn and refine patterns using k-fold cross-validation with iterative refinement across fold combinations.

### Input
- `train/` directory (60% of original data, package files)
- Issue type (e.g., `RESOURCE_LEAK`)

### Process

```
┌─────────────────────────────────────────────────────────────┐
│ Initialization: Stratified K-Fold Split (n=3 folds)         │
├─────────────────────────────────────────────────────────────┤
│ Input: Package files from train/                            │
│                                                              │
│ StratifiedKFoldSplitter (stratified_kfold.py):              │
│   Strategy: PACKAGE-level splitting (no contamination)      │
│                                                              │
│   1. Load ALL package files and compute statistics          │
│      • Count TP/FP per package                              │
│      • Identify primary issue type per package              │
│   2. Group packages by primary issue type                   │
│   3. For each issue type:                                   │
│      • Distribute packages evenly across k folds            │
│      • Each package goes entirely into ONE fold             │
│   4. Return original package files (no temp files)          │
│                                                              │
│ Output: 3 folds, each containing ~33% of packages           │
│   • Fold 0: 42 packages (e.g., 67 TP, 289 FP)              │
│   • Fold 1: 42 packages (e.g., 68 TP, 290 FP)              │
│   • Fold 2: 41 packages (e.g., 65 TP, 289 FP)              │
│                                                              │
│ Guarantees:                                                  │
│   • Every fold contains EVERY issue type                    │
│   • NO package contamination (unique packages per fold)     │
│   • Fails if any issue type has < n_folds packages          │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Learn Initial Patterns                              │
├─────────────────────────────────────────────────────────────┤
│ Train: Folds 0,1 (84 packages)                              │
│ Validate: Fold 2 (41 packages)                              │
│                                                              │
│ A. PATTERN LEARNING (PatternLearner)                        │
│    train_files (folds 0+1) → PatternLearner                 │
│      1. Parse ALL entries from train files                  │
│      2. Filter to issue_type (e.g., RESOURCE_LEAK)          │
│         → Found: 712 entries (135 TP, 577 FP)               │
│      3. Batch processing strategy (processes ALL data):     │
│         • TP: Sample 3 with replacement → 3 TP              │
│         • FP: Process ALL in sliding batches of 3           │
│           - Batch 1: FP[0:3]                                │
│           - Batch 2: FP[3:6]                                │
│           - ...                                             │
│           - Batch 192: FP[576:577] + 2 sampled              │
│           - Total: 577 FP in 192 batches                    │
│      4. For each batch: 3 TP + 3 FP → LLM generates patterns│
│      5. Deduplicate patterns by summary                     │
│                                                              │
│ B. VALIDATION EVALUATION (FoldEvaluator)                    │
│    patterns_0 + val_files (fold 2) → Evaluate               │
│      • Evaluate on ALL entries in fold 2 (no sampling)      │
│      • Calculate metrics: F1, Precision, Recall             │
│                                                              │
│ Output:                                                      │
│   • RESOURCE_LEAK_step_0_patterns.json                      │
│   • RESOURCE_LEAK_step_0_evaluation.json                    │
│   • Val F1: 0.750                                           │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Refine Patterns                                     │
├─────────────────────────────────────────────────────────────┤
│ Train: Folds 0,2 (83 packages)                              │
│ Validate: Fold 1 (42 packages)                              │
│                                                              │
│ A. TRAIN EVALUATION (find misclassifications)               │
│    patterns_0 + train_files (folds 0+2) → FoldEvaluator     │
│      • Evaluate current patterns on training data           │
│      • Find misclassified entries                           │
│      • Sample max 20 misclassifications                     │
│                                                              │
│ B. PATTERN REFINEMENT (PatternRefiner)                      │
│    patterns_0 + misclassified → PatternRefiner              │
│      • Analyze misclassifications (FN vs FP-missed)         │
│      • LLM suggests refinements:                            │
│        - Add: New patterns for uncovered cases              │
│        - Modify: Improve existing patterns                  │
│        - Remove: Delete redundant patterns                  │
│      • Apply refinements → patterns_1                       │
│                                                              │
│ C. VALIDATION EVALUATION                                    │
│    patterns_1 + val_files (fold 1) → Evaluate               │
│      • Evaluate refined patterns on fold 1                  │
│      • Calculate metrics: F1, Precision, Recall             │
│                                                              │
│ Output:                                                      │
│   • RESOURCE_LEAK_step_1_patterns.json                      │
│   • RESOURCE_LEAK_step_1_refinements.json                   │
│   • RESOURCE_LEAK_step_1_evaluation.json                    │
│   • Val F1: 0.780                                           │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Further Refinement                                  │
├─────────────────────────────────────────────────────────────┤
│ Train: Folds 1,2 (83 packages)                              │
│ Validate: Fold 0 (42 packages)                              │
│                                                              │
│ A. TRAIN EVALUATION (find misclassifications)               │
│    patterns_1 + train_files (folds 1+2) → FoldEvaluator     │
│      • Evaluate on different fold combination               │
│      • Find misclassified entries                           │
│      • Sample max 20 misclassifications                     │
│                                                              │
│ B. PATTERN REFINEMENT (PatternRefiner)                      │
│    patterns_1 + misclassified → PatternRefiner              │
│      • Analyze new misclassifications                       │
│      • LLM suggests refinements                             │
│      • Apply refinements → patterns_2                       │
│                                                              │
│ C. VALIDATION EVALUATION                                    │
│    patterns_2 + val_files (fold 0) → Evaluate               │
│      • Evaluate on fold 0                                   │
│      • Calculate final metrics                              │
│                                                              │
│ Output:                                                      │
│   • RESOURCE_LEAK_step_2_patterns.json (FINAL)              │
│   • RESOURCE_LEAK_step_2_refinements.json                   │
│   • RESOURCE_LEAK_step_2_evaluation.json                    │
│   • Val F1: 0.795                                           │
└─────────────────────────────────────────────────────────────┘
                           ▼
                  FINAL PATTERNS (patterns_2)
               (Input to Phase 1.5)
```

### Key Details

**Q: How many entries are used for learning?**
A: **ALL entries** are processed in batches of 6 (3 TP + 3 FP). For 868 FP entries, this creates ~290 batches and 290 LLM calls.

**Q: How many entries are used for evaluation?**
A: **ALL entries** in the validation fold (no sampling)

**Q: What's the train/val split per fold?**
A: **~67% train, ~33% val** per fold (2 folds train, 1 fold val)

**Q: Why batch size of 6?**
A: **LLM context window + balanced TP/FP**. 6 entries ≈ 2k tokens, allowing for rich pattern generation while maintaining balance.

### Output

```
output/phase1_kfold_results/
├── RESOURCE_LEAK_step_0_patterns.json       # Initial patterns (Step 0)
├── RESOURCE_LEAK_step_0_evaluation.json     # Step 0 validation metrics
├── RESOURCE_LEAK_step_1_patterns.json       # Refined patterns (Step 1)
├── RESOURCE_LEAK_step_1_refinements.json    # Step 1 refinement details
├── RESOURCE_LEAK_step_1_evaluation.json     # Step 1 validation metrics
├── RESOURCE_LEAK_step_2_patterns.json       # ⭐ Final patterns (Step 2)
├── RESOURCE_LEAK_step_2_refinements.json    # Step 2 refinement details
├── RESOURCE_LEAK_step_2_evaluation.json     # Step 2 validation metrics
├── RESOURCE_LEAK_final_evaluation.json      # Phase 1.5: Final pattern baseline on full val set
├── phase1_5_final_evaluation.json           # Phase 1.5: All issue types combined
└── phase1_complete_results.json             # Complete Phase 1 results
```

---

## Phase 2: Further Iterative Refinement

### Goal
Further refine patterns using train+val data with **proper ML pipeline** (no data leakage).

### Input
- Final patterns from Phase 1 (patterns_2 from iterative refinement)
- `train/` directory (60% of original) - for finding misclassifications
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
│   Step A: Predict & Evaluate on TRAIN (find refinement targets) │
│   ──────────────────────────────────────────────────────────    │
│   current_patterns + train/ → FoldEvaluator                     │
│     • Sample up to 500 entries from train/ files               │
│     • LLM classifies each entry using current patterns          │
│     • Calculate metrics: train_f1, train_precision, etc.        │
│     • Track misclassified entries for refinement               │
│                                                              │
│   Example Output:                                            │
│     Train F1: 0.850                                         │
│     Train Precision: 0.820                                  │
│     Train Recall: 0.880                                     │
│     Train Misclassified: 15 entries                         │
│                                                              │
│   Step B: Predict & Evaluate on VAL (check generalization)  │
│   ──────────────────────────────────────────────────────    │
│   current_patterns + val/ → FoldEvaluator                   │
│     • Sample up to 500 entries from val/ files              │
│     • LLM classifies each entry using current patterns      │
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

### ML Best Practices

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
python -m process_mining.kfold_pattern_learning.main \
  --train-dir data/pattern_data/train \
  --val-dir data/pattern_data/val \
  --test-dir data/pattern_data/test \
  --output-dir outputs/kfold_learning \
  --platform nim \
  --n-folds 5 \
  --max-iterations 10 \
  --eval-sample-size 500
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
python -m process_mining.kfold_pattern_learning.main \
  --phase 1 \
  --train-dir data/pattern_data/train \
  --output-dir outputs/phase1_only \
  --n-folds 3 \
  --issue-types RESOURCE_LEAK UNINIT
```

### Example 3: Run Only Phase 2 (Refinement)

Requires Phase 1 results to exist first.

```bash
python -m process_mining.kfold_pattern_learning.main \
  --phase 2 \
  --train-dir data/pattern_data/train \
  --val-dir data/pattern_data/val \
  --output-dir outputs/phase2_only \
  --max-iterations 5 \
  --eval-sample-size 100
```

### Example 4: Run Only Phase 3 (Test Evaluation)

Requires Phase 2 results to exist first.

```bash
python -m process_mining.kfold_pattern_learning.main \
  --phase 3 \
  --test-dir data/pattern_data/test \
  --output-dir outputs/phase3_only
```

---

## Configuration

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--train-dir` | Yes | - | Training data directory (60% of data) |
| `--val-dir` | Conditional | - | Validation directory (20%, required for Phase 2/3) |
| `--test-dir` | Conditional | - | Test directory (20%, required for Phase 3) |
| `--output-dir` | Yes | - | Output directory for all results |
| `--platform` | No | `vertex` | LLM platform: `local`, `nim`, or `vertex` (Google Vertex AI)|
| `--n-folds` | No | `3` | Number of folds for k-fold CV (Phase 1) |
| `--max-iterations` | No | `10` | Max refinement iterations (Phase 2) |
| `--eval-sample-size` | No | `500` | Max entries to sample for Phase 2 evaluations (0=no sampling) |
| `--workers` | No | `1` | Parallel workers for LLM evaluation |
| `--seed` | No | `42` | Random seed for reproducibility |
| `--issue-types` | No | auto-detect | Specific issue types (space-separated) |
| `--phase` | No | all | Run specific phase: `1`, `2`, or `3` |

### Platform Configuration

| Platform | Model | Rate Limit | Use Case |
|----------|-------|------------|----------|
| `vertex` | `claude-sonnet-4-5@20250929` | High (API-dependent) | Production (recommended, fastest) |
| `nim` | `qwen/qwen3-coder-480b-a35b-instruct` | 30 req/min | NVIDIA NIM platform |
| `local` | `llama3.1:70b` | Unlimited | Local testing |

**Environment Variables**:
```bash
# Vertex AI (Google Cloud)
export VERTEX_PROJECT_ID="your-gcp-project"
export VERTEX_REGION="us-east5"
export VERTEX_MODEL="claude-sonnet-4-5@20250929"

# NVIDIA NIM
export NIM_API_KEY="your-nvidia-api-key"
export NIM_MODEL="qwen/qwen3-coder-480b-a35b-instruct"

# Local vLLM
export LOCAL_BASE_URL="http://localhost:8000/v1"
```

---

## Output

### Complete Output Structure

```
output_dir/
├── phase1_kfold_results/
│   ├── RESOURCE_LEAK_step_0_patterns.json       # Initial patterns (Step 0)
│   ├── RESOURCE_LEAK_step_0_evaluation.json     # Step 0 validation metrics
│   ├── RESOURCE_LEAK_step_1_patterns.json       # Refined patterns (Step 1)
│   ├── RESOURCE_LEAK_step_1_refinements.json    # Step 1 refinement details
│   ├── RESOURCE_LEAK_step_1_evaluation.json     # Step 1 validation metrics
│   ├── RESOURCE_LEAK_step_2_patterns.json       # ⭐ Final patterns (Step 2)
│   ├── RESOURCE_LEAK_step_2_refinements.json    # Step 2 refinement details
│   ├── RESOURCE_LEAK_step_2_evaluation.json     # Step 2 validation metrics
│   ├── RESOURCE_LEAK_final_evaluation.json      # Phase 1.5: Final pattern baseline
│   ├── phase1_5_final_evaluation.json           # Phase 1.5: All issue types combined
│   └── phase1_complete_results.json             # Complete Phase 1 results
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

**Why stratify at package level?** Ensures:
1. ALL issue types appear in EVERY fold
2. NO package contamination between folds (each package → one fold only)
3. Realistic evaluation (packages are independent units)

**Package-Level Stratification Strategy**:
- Works at **package level** (no contamination)
- Each package file goes entirely into ONE fold
- Groups packages by **primary issue type** (most common issue in package)
- For each issue type:
  - Distributes packages evenly across k folds
- **Guarantees**: Every fold contains every issue type with unique packages

**Example with 3 folds**:
```
Issue Type: RESOURCE_LEAK (118 packages)
  • Fold 0: 39 packages → 67 TP, 289 FP
  • Fold 1: 40 packages → 68 TP, 290 FP
  • Fold 2: 39 packages → 65 TP, 289 FP

Issue Type: UNINIT (104 packages)
  • Fold 0: 35 packages → 48 TP, 152 FP
  • Fold 1: 35 packages → 49 TP, 151 FP
  • Fold 2: 34 packages → 47 TP, 150 FP

Result: Each fold contains ALL issue types with UNIQUE packages
        No package appears in multiple folds (no contamination)
```

### Batching and Sampling Strategy

| Stage | Data | Processing | Purpose |
|-------|------|------------|---------|
| **Phase 1 Learning** | train_files (67% of entries) | ALL data in batches of 6 (3 TP + 3 FP) | Process all data for pattern learning |
| **Phase 1 Evaluation** | val_files (33% of entries) | ALL entries (no sampling) | Accurate fold metrics |
| **Phase 1.5 Baseline** | val/ (20% of original) | Sample up to 500 entries | Merged pattern baseline |
| **Phase 2 Train Eval** | train/ (60% of original) | Sample up to 500 entries | Find misclassifications for refinement |
| **Phase 2 Val Eval** | val/ (20% of original) | Sample up to 500 entries | Early stopping criterion |
| **Phase 2 Refinement** | train_misclassified | Max 20 entries | LLM refinement suggestions |
| **Phase 3 Test** | test/ (20% of original) | ALL entries (no sampling) | Final evaluation |

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