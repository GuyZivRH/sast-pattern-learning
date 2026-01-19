"""
K-Fold Pattern Learning System for SAST False Positive Classification

This package implements a k-fold cross-validation approach to learning patterns
from human-annotated SAST findings, with iterative refinement.

Main Components:
- stratified_kfold: Stratified k-fold splitting preserving FP/TP ratios
- pattern_learner: LLM-based pattern extraction from training data
- fold_evaluator: Pattern evaluation on held-out folds
- pattern_merger: LLM-based pattern deduplication and merging
- kfold_orchestrator: Phase 1 orchestration (k-fold CV)
- pattern_refiner: LLM-based pattern refinement from errors
- refinement_orchestrator: Phase 2 orchestration (iterative refinement)
- test_evaluator: Final evaluation on test set
- main: CLI entry point for full pipeline
"""

__version__ = "1.0.0"