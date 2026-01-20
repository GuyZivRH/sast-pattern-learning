#!/usr/bin/env python3
"""
K-Fold Pattern Learning Configuration

Centralized configuration for the k-fold pattern learning system.
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
    'workers': 1,
    'validate_all_types_per_fold': True,
    'top_n_only': True
}

# Balanced Batch Sampling for Pattern Learning
SAMPLING_DEFAULTS = {
    'max_tp_per_batch': 3,
    'max_fp_per_batch': 3
}

# LLM Configuration Defaults
LLM_DEFAULTS = {
    'max_tokens': 16000,
    'temperature': 0.0,
    'platform': 'vertex'  # Options: 'local', 'nim', 'vertex'
}

# Phase 2 Refinement Defaults
REFINEMENT_DEFAULTS = {
    'max_iterations': 10,
    'convergence_threshold': 0.01,
    'convergence_patience': 3,
    'max_misclassified_per_iteration': 20,
    'eval_sample_size': 500
}