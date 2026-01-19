#!/usr/bin/env python3
"""
K-Fold Pattern Learning Configuration

Centralized configuration constants for the k-fold pattern learning system.
Replaces hardcoded values scattered across multiple files.
"""

from typing import List

# Top 10 Issue Types (93.7% of data)
# Focus pattern learning on these types for better quality
TOP_10_ISSUE_TYPES: List[str] = [
    'RESOURCE_LEAK',
    'OVERRUN',
    'UNINIT',
    'INTEGER_OVERFLOW',
    'USE_AFTER_FREE',
    'CPPCHECK_WARNING',
    'BUFFER_SIZE',
    'VARARGS',
    'COMPILER_WARNING',
    'COPY_PASTE_ERROR'
]

# K-Fold Cross-Validation Defaults
KFOLD_DEFAULTS = {
    'n_folds': 3,
    'random_seed': 42,
    'max_entries_per_fold': 50,
    'workers': 1,
    'validate_all_types_per_fold': True,
    'top_n_only': True
}

# Balanced Batch Sampling Defaults
SAMPLING_DEFAULTS = {
    'max_tp_per_batch': 3,
    'max_fp_per_batch': 3
}

# LLM Configuration Defaults
LLM_DEFAULTS = {
    'max_tokens': 16000,
    'temperature': 0.0,
    'platform': 'nim'
}

# Phase 1.5 Evaluation Defaults
PHASE_1_5_DEFAULTS = {
    'max_eval_samples': 500
}

# Data Split Ratios (for stratified_data_splitter.py)
SPLIT_RATIOS = {
    'train': 0.6,
    'validation': 0.2,
    'test': 0.2
}

# Minimum Sample Requirements
MIN_SAMPLES = {
    'min_samples_per_type': 5,
    'min_samples_per_fold': 1
}