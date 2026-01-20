"""
K-Fold Pattern Learning System for SAST False Positive Classification

This package implements a k-fold cross-validation approach to learning patterns
from human-annotated SAST findings, with iterative refinement across folds.

Main Components:
- stratified_kfold: Package-level stratified k-fold splitting (no contamination)
- pattern_learner: LLM-based pattern extraction from training data
- fold_evaluator: Pattern evaluation on validation folds
- pattern_refiner: LLM-based pattern refinement from misclassifications
- kfold_orchestrator: Phase 1 orchestration (k-fold CV with iterative refinement)
- refinement_orchestrator: Phase 2 orchestration (further refinement on train+val)
- test_evaluator: Phase 3 - Final evaluation on held-out test set
- main: CLI entry point for full 3-phase pipeline
"""

__version__ = "1.0.0"