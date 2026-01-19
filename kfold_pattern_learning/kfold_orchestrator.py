#!/usr/bin/env python3
"""
K-Fold Orchestrator - Phase 1 Orchestration

Orchestrates the full k-fold cross-validation process:
1. Split data into k stratified folds
2. For each fold: learn patterns on train, evaluate on val
3. Merge patterns across all folds
4. Generate Phase 1 report
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.stratified_kfold import StratifiedKFoldSplitter
from process_mining.kfold_pattern_learning.pattern_learner import PatternLearner
from process_mining.kfold_pattern_learning.fold_evaluator import FoldEvaluator
from process_mining.kfold_pattern_learning.pattern_merger import PatternMerger
from process_mining.kfold_pattern_learning.config import (
    TOP_10_ISSUE_TYPES,
    KFOLD_DEFAULTS,
    PHASE_1_5_DEFAULTS
)
from process_mining.core.parsers import ValidationEntryParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KFoldOrchestrator:
    """
    Orchestrates Phase 1: K-Fold Cross-Validation.

    Coordinates stratified splitting, pattern learning, fold evaluation,
    and pattern merging across k folds.
    """

    def __init__(
        self,
        train_dir: Path,
        output_dir: Path,
        n_folds: int = 3,
        platform: str = "nim",
        random_seed: int = 42,
        max_entries_per_fold: int = 50,
        workers: int = 1,
        issue_types: Optional[List[str]] = None,
        validate_all_types_per_fold: bool = True,
        top_n_only: bool = True,
        top_n_list: Optional[List[str]] = None
    ):
        """
        Initialize k-fold orchestrator.

        Args:
            train_dir: Directory containing training .txt files
            output_dir: Output directory for Phase 1 results
            n_folds: Number of folds (default: 3)
            platform: LLM platform ("local" or "nim")
            random_seed: Random seed for reproducibility
            max_entries_per_fold: Max entries per issue type to send to LLM
            workers: Number of parallel workers for evaluation
            issue_types: List of issue types to process (None = auto-detect)
            validate_all_types_per_fold: Ensure all issue types in each fold
            top_n_only: Filter to top N issue types only (default: True)
            top_n_list: Custom list of top issue types (default: TOP_10_ISSUE_TYPES)
        """
        self.train_dir = Path(train_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.platform = platform
        self.random_seed = random_seed
        self.max_entries_per_fold = max_entries_per_fold
        self.workers = workers
        self.issue_types = issue_types
        self.validate_all_types_per_fold = validate_all_types_per_fold
        self.top_n_only = top_n_only
        self.top_n_list = top_n_list or TOP_10_ISSUE_TYPES

        # Create output directories
        self.phase1_dir = self.output_dir / "phase1_kfold_results"
        self.phase1_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.splitter = StratifiedKFoldSplitter(
            n_splits=n_folds,
            random_state=random_seed
        )
        self.pattern_learner = PatternLearner(platform=platform)
        self.fold_evaluator = FoldEvaluator(platform=platform, workers=workers)
        self.pattern_merger = PatternMerger(platform=platform)

        logger.info(f"Initialized KFoldOrchestrator:")
        logger.info(f"  Train dir: {self.train_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  K-folds: {self.n_folds}")
        logger.info(f"  Platform: {self.platform}")
        if validate_all_types_per_fold:
            logger.info(f"  Validation: Ensure all issue types in each fold")
        if top_n_only:
            logger.info(f"  Pattern learning enabled for top {len(self.top_n_list)} issue types only")
            logger.info(f"    {', '.join(self.top_n_list)}")

    def _detect_issue_types(self, train_files: List[Path]) -> List[str]:
        """Auto-detect issue types from training files."""
        logger.info("Auto-detecting issue types from training data...")

        parser = ValidationEntryParser()
        issue_type_counts = defaultdict(int)

        for txt_file in train_files[:20]:  # Sample first 20 files
            entries = parser.parse_file(txt_file)
            for entry in entries:
                issue_type_counts[entry.issue_type] += 1

        issue_types = sorted(issue_type_counts.keys())
        logger.info(f"  Detected {len(issue_types)} issue types: {issue_types}")

        return issue_types

    def run_phase1(self) -> Dict:
        """
        Run Phase 1: K-Fold Cross-Validation.

        Returns:
            Dictionary with complete Phase 1 results:
            {
                'metadata': {...},
                'issue_types': {
                    'RESOURCE_LEAK': {
                        'fold_results': [...],
                        'fold_patterns': [...],
                        'merged_patterns': {...},
                        'avg_metrics': {...}
                    },
                    ...
                },
                'overall_summary': {...}
            }
        """
        logger.info("="*80)
        logger.info("PHASE 1: K-FOLD CROSS-VALIDATION")
        logger.info("="*80)

        start_time = datetime.now()

        # Step 1: Create k-fold splits
        logger.info("\n[Step 1/4] Creating stratified k-fold splits...")
        folds = self.splitter.split(self.train_dir)

        logger.info(f"  Created {len(folds)} folds")

        # Step 2: Detect or use provided issue types
        if self.issue_types is None:
            all_train_files = [f for f in self.train_dir.glob("*.txt") if not f.name.startswith("_")]
            all_issue_types = self._detect_issue_types(all_train_files)

            # Filter to top N if requested
            if self.top_n_only:
                self.issue_types = [it for it in all_issue_types if it in self.top_n_list]
                skipped = len(all_issue_types) - len(self.issue_types)
                logger.info(f"  Pattern learning: {len(self.issue_types)} types (skipping {skipped} rare types)")
            else:
                self.issue_types = all_issue_types

        logger.info(f"\n[Step 2/4] Processing {len(self.issue_types)} issue types...")

        # Step 3: Process each issue type
        results_by_issue_type = {}

        for issue_type in self.issue_types:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Issue Type: {issue_type}")
            logger.info(f"{'='*80}")

            # Validate fold coverage if requested
            if self.validate_all_types_per_fold:
                coverage_ok = self._validate_fold_coverage(folds, issue_type)
                if not coverage_ok:
                    logger.error(f"Issue type {issue_type} not in all folds. Consider reducing k or using stratified split.")
                    logger.error(f"Skipping pattern learning for {issue_type}")
                    continue

            fold_results = []
            fold_patterns = []

            # Process each fold
            for fold_idx, (train_files, val_files) in enumerate(folds):
                logger.info(f"\n  --- Fold {fold_idx + 1}/{self.n_folds} ---")

                # Learn patterns on training fold
                logger.info(f"  Learning patterns from {len(train_files)} training files...")
                patterns = self.pattern_learner.learn_patterns(
                    train_files=train_files,
                    issue_type=issue_type
                )

                fold_patterns.append(patterns)

                # Save fold-specific patterns
                fold_pattern_file = self.phase1_dir / f"{issue_type}_fold_{fold_idx}_patterns.json"
                with open(fold_pattern_file, 'w') as f:
                    json.dump(patterns, f, indent=2)

                # Evaluate on validation fold
                logger.info(f"  Evaluating on {len(val_files)} validation files...")
                fold_eval = self.fold_evaluator.evaluate_fold(
                    patterns_dict=patterns,
                    val_files=val_files,
                    issue_type=issue_type
                )

                fold_results.append(fold_eval)

                # Save fold evaluation results
                fold_eval_file = self.phase1_dir / f"{issue_type}_fold_{fold_idx}_evaluation.json"
                with open(fold_eval_file, 'w') as f:
                    json.dump(fold_eval, f, indent=2)

            # Step 4: Merge patterns across folds
            logger.info(f"\n  Merging patterns from {self.n_folds} folds...")
            merged_patterns = self.pattern_merger.merge_patterns(
                fold_patterns=fold_patterns,
                issue_type=issue_type
            )

            # Save merged patterns
            merged_pattern_file = self.phase1_dir / f"{issue_type}_merged_patterns.json"
            with open(merged_pattern_file, 'w') as f:
                json.dump(merged_patterns, f, indent=2)

            # Calculate average metrics across folds
            avg_metrics = self._calculate_average_metrics(fold_results)

            # Compile results for this issue type
            results_by_issue_type[issue_type] = {
                'fold_results': fold_results,
                'fold_patterns': fold_patterns,
                'merged_patterns': merged_patterns,
                'avg_metrics': avg_metrics
            }

            logger.info(f"\n  Average metrics across {self.n_folds} folds:")
            logger.info(f"    F1: {avg_metrics['f1']:.3f}")
            logger.info(f"    Precision: {avg_metrics['precision']:.3f}")
            logger.info(f"    Recall: {avg_metrics['recall']:.3f}")

        # Generate overall summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        overall_summary = {
            'total_issue_types': len(self.issue_types),
            'total_folds': self.n_folds,
            'duration_seconds': duration,
            'avg_f1_across_issue_types': sum(
                results_by_issue_type[it]['avg_metrics']['f1']
                for it in self.issue_types
            ) / len(self.issue_types)
        }

        # Compile final results
        phase1_results = {
            'metadata': {
                'phase': 1,
                'train_dir': str(self.train_dir),
                'n_folds': self.n_folds,
                'platform': self.platform,
                'random_seed': self.random_seed,
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration
            },
            'issue_types': results_by_issue_type,
            'overall_summary': overall_summary
        }

        # Save complete Phase 1 results
        phase1_file = self.phase1_dir / "phase1_complete_results.json"
        with open(phase1_file, 'w') as f:
            json.dump(phase1_results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info("PHASE 1 COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Avg F1 across all issue types: {overall_summary['avg_f1_across_issue_types']:.3f}")
        logger.info(f"Results saved to: {self.phase1_dir}")

        return phase1_results

    def _calculate_average_metrics(self, fold_results: List[Dict]) -> Dict:
        """Calculate average metrics across folds."""
        if not fold_results:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_accuracy = 0.0

        for fold_result in fold_results:
            metrics = fold_result.get('metrics', {}).get('overall', {})
            total_f1 += metrics.get('f1', 0.0)
            total_precision += metrics.get('precision', 0.0)
            total_recall += metrics.get('recall', 0.0)
            total_accuracy += metrics.get('accuracy', 0.0)

        n = len(fold_results)
        return {
            'f1': total_f1 / n,
            'precision': total_precision / n,
            'recall': total_recall / n,
            'accuracy': total_accuracy / n
        }

    def _validate_fold_coverage(self, folds: List, issue_type: str) -> bool:
        """
        Validate that issue_type appears in all folds.

        Args:
            folds: List of (train_files, val_files) tuples
            issue_type: Issue type to check

        Returns:
            True if issue_type in all folds, False otherwise
        """
        parser = ValidationEntryParser()

        for fold_idx, (train_files, val_files) in enumerate(folds):
            # Check if issue_type in this fold's validation set
            has_issue_type = False

            for val_file in val_files:
                try:
                    entries = parser.parse_file(val_file)
                    if any(e.issue_type == issue_type for e in entries):
                        has_issue_type = True
                        break
                except Exception as e:
                    logger.warning(f"Error parsing {val_file.name}: {e}")
                    continue

            if not has_issue_type:
                logger.warning(f"  Issue type {issue_type} missing from fold {fold_idx + 1}")
                return False

        return True

    def run_phase1_5(
        self,
        merged_patterns_by_issue_type: Dict[str, Dict],
        val_dir: Path,
        max_eval_samples: int = 500
    ) -> Dict:
        """
        Phase 1.5: Evaluate merged patterns on validation set.

        Provides true baseline before Phase 2 refinement starts.

        Args:
            merged_patterns_by_issue_type: Dict mapping issue_type -> merged patterns
            val_dir: Validation directory
            max_eval_samples: Max samples to evaluate (500 for speed)

        Returns:
            Evaluation results for merged patterns
        """
        logger.info("="*80)
        logger.info("PHASE 1.5: MERGED PATTERN EVALUATION")
        logger.info("="*80)

        val_files = sorted(val_dir.glob("*.txt"))
        logger.info(f"Evaluating on {len(val_files)} validation files (max {max_eval_samples} samples per type)")

        results_by_issue_type = {}

        for issue_type, merged_patterns in merged_patterns_by_issue_type.items():
            logger.info(f"\nEvaluating merged patterns for {issue_type}...")

            # Evaluate on validation set with sampling
            eval_result = self.fold_evaluator.evaluate_fold(
                patterns_dict=merged_patterns,
                val_files=val_files,
                issue_type=issue_type,
                max_entries=max_eval_samples
            )

            metrics = eval_result.get('metrics', {}).get('overall', {})
            logger.info(f"  Merged Pattern Metrics:")
            logger.info(f"    F1: {metrics.get('f1', 0.0):.3f}")
            logger.info(f"    Precision: {metrics.get('precision', 0.0):.3f}")
            logger.info(f"    Recall: {metrics.get('recall', 0.0):.3f}")

            # Save results
            results_by_issue_type[issue_type] = eval_result

            # Save per-issue-type evaluation
            eval_file = self.phase1_dir / f"{issue_type}_merged_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_result, f, indent=2)

        # Save complete Phase 1.5 results
        phase1_5_results = {
            'metadata': {
                'phase': 1.5,
                'val_dir': str(val_dir),
                'max_eval_samples': max_eval_samples,
                'description': 'Evaluation of merged patterns on validation set'
            },
            'issue_types': results_by_issue_type
        }

        phase1_5_file = self.phase1_dir / "phase1_5_merged_evaluation.json"
        with open(phase1_5_file, 'w') as f:
            json.dump(phase1_5_results, f, indent=2)

        logger.info(f"\nPhase 1.5 complete. Results saved to: {phase1_5_file}")

        return phase1_5_results


def main():
    """Test k-fold orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1: K-Fold Cross-Validation")
    parser.add_argument("train_dir", type=Path, help="Training data directory")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--platform", "-p", choices=["local", "nim"],
                       default="nim", help="LLM platform")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-entries", type=int, default=50,
                       help="Max entries per fold per issue type")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--issue-types", nargs="+", help="Specific issue types to process")

    args = parser.parse_args()

    orchestrator = KFoldOrchestrator(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        platform=args.platform,
        random_seed=args.seed,
        max_entries_per_fold=args.max_entries,
        workers=args.workers,
        issue_types=args.issue_types
    )

    results = orchestrator.run_phase1()

    print("\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    print(f"Issue Types: {results['overall_summary']['total_issue_types']}")
    print(f"Folds: {results['overall_summary']['total_folds']}")
    print(f"Duration: {results['overall_summary']['duration_seconds']:.1f}s")
    print(f"Avg F1: {results['overall_summary']['avg_f1_across_issue_types']:.3f}")


if __name__ == "__main__":
    main()