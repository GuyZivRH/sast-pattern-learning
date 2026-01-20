#!/usr/bin/env python3
"""
K-Fold Orchestrator - Phase 1 Orchestration with Iterative Refinement

Orchestrates Phase 1 with iterative refinement across folds:
Step 0: Learn initial patterns from folds 0,1 (validate on fold 2)
Step 1: Refine patterns using folds 0,2 (validate on fold 1)
Step 2: Refine patterns using folds 1,2 (validate on fold 0)

Result: patterns_2 (refined across all fold combinations)
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
from process_mining.kfold_pattern_learning.pattern_refiner import PatternRefiner
from process_mining.kfold_pattern_learning.config import (
    TOP_10_ISSUE_TYPES,
    KFOLD_DEFAULTS,
    REFINEMENT_DEFAULTS
)
from process_mining.core.parsers import ValidationEntryParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KFoldOrchestrator:
    """
    Orchestrates Phase 1: K-Fold Cross-Validation with Iterative Refinement.

    New approach:
    - Step 0: Learn patterns from train folds → patterns_0
    - Step 1-2: Iteratively refine using different fold combinations
    - No pattern merging (refinement keeps patterns clean)
    """

    def __init__(
        self,
        train_dir: Path,
        output_dir: Path,
        n_folds: int = KFOLD_DEFAULTS['n_folds'],
        platform: str = "nim",
        random_seed: int = KFOLD_DEFAULTS['random_seed'],
        workers: int = KFOLD_DEFAULTS['workers'],
        issue_types: Optional[List[str]] = None,
        validate_all_types_per_fold: bool = KFOLD_DEFAULTS['validate_all_types_per_fold'],
        top_n_only: bool = KFOLD_DEFAULTS['top_n_only'],
        top_n_list: Optional[List[str]] = None,
        max_misclassified_per_step: int = REFINEMENT_DEFAULTS['max_misclassified_per_iteration']
    ):
        """
        Initialize k-fold orchestrator.

        Args:
            train_dir: Directory containing training .txt files
            output_dir: Output directory for Phase 1 results
            n_folds: Number of folds (default: 3)
            platform: LLM platform ("local", "nim", or "vertex")
            random_seed: Random seed for reproducibility
            workers: Number of parallel workers for evaluation
            issue_types: List of issue types to process (None = auto-detect)
            validate_all_types_per_fold: Ensure all issue types in each fold
            top_n_only: Filter to top N issue types only (default: True)
            top_n_list: Custom list of top issue types (default: TOP_10_ISSUE_TYPES)
            max_misclassified_per_step: Max misclassifications to analyze per refinement step
        """
        self.train_dir = Path(train_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.platform = platform
        self.random_seed = random_seed
        self.workers = workers
        self.issue_types = issue_types
        self.validate_all_types_per_fold = validate_all_types_per_fold
        self.top_n_only = top_n_only
        self.top_n_list = top_n_list or TOP_10_ISSUE_TYPES
        self.max_misclassified_per_step = max_misclassified_per_step

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
        self.pattern_refiner = PatternRefiner(platform=platform)

        logger.info(f"Initialized KFoldOrchestrator (Iterative Refinement):")
        logger.info(f"  Train dir: {self.train_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  K-folds: {self.n_folds}")
        logger.info(f"  Platform: {self.platform}")
        logger.info(f"  Strategy: Step 0 (learn) → Steps 1-{n_folds-1} (refine)")
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
        Run Phase 1: K-Fold Cross-Validation with Iterative Refinement.

        New approach:
        - Step 0: Learn patterns from folds 0,1 (val on fold 2) → patterns_0
        - Step 1: Refine using folds 0,2 (val on fold 1) → patterns_1
        - Step 2: Refine using folds 1,2 (val on fold 0) → patterns_2

        Returns:
            Dictionary with complete Phase 1 results including final refined patterns
        """
        logger.info("="*80)
        logger.info("PHASE 1: K-FOLD CROSS-VALIDATION WITH ITERATIVE REFINEMENT")
        logger.info("="*80)

        start_time = datetime.now()

        # Step 1: Create k-fold splits
        logger.info("\n[Step 1/4] Creating stratified k-fold splits...")
        folds = self.splitter.split(self.train_dir)

        logger.info(f"  Created {len(folds)} folds")
        logger.info(f"  Strategy: Step 0 (learn) → Steps 1-{self.n_folds-1} (iterative refinement)")

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

        # Step 3: Process each issue type with iterative refinement
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

            # Run iterative refinement across folds
            issue_results = self._run_iterative_refinement(folds, issue_type)
            results_by_issue_type[issue_type] = issue_results

        # Generate overall summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        overall_summary = {
            'total_issue_types': len(self.issue_types),
            'total_folds': self.n_folds,
            'refinement_steps': self.n_folds,
            'duration_seconds': duration
        }

        # Compile final results
        phase1_results = {
            'metadata': {
                'phase': 1,
                'train_dir': str(self.train_dir),
                'n_folds': self.n_folds,
                'platform': self.platform,
                'random_seed': self.random_seed,
                'strategy': 'iterative_refinement',
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
        logger.info(f"Results saved to: {self.phase1_dir}")

        return phase1_results

    def _run_iterative_refinement(self, folds: List, issue_type: str) -> Dict:
        """
        Run iterative refinement for one issue type.

        Step 0: Learn patterns from folds 0,1 (val on fold 2) → patterns_0
        Step 1: Refine using folds 0,2 (val on fold 1) → patterns_1
        Step 2: Refine using folds 1,2 (val on fold 0) → patterns_2

        Returns:
            Results dictionary with step_results and final_patterns
        """
        step_results = []
        current_patterns = None

        # Define refinement steps
        refinement_steps = self._get_refinement_steps(self.n_folds)

        for step_idx, (train_fold_indices, val_fold_idx) in enumerate(refinement_steps):
            logger.info(f"\n  --- Step {step_idx}/{len(refinement_steps)-1} ---")
            logger.info(f"  Train folds: {train_fold_indices}, Val fold: {val_fold_idx}")

            # Get train and val files
            train_files = []
            for fold_idx in train_fold_indices:
                train_files.extend(folds[fold_idx][0])  # Get train portion of each fold
                train_files.extend(folds[fold_idx][1])  # Get val portion of each fold (all files)

            val_files = list(folds[val_fold_idx][0]) + list(folds[val_fold_idx][1])  # All files from val fold

            if step_idx == 0:
                # Step 0: Learn patterns from scratch
                logger.info(f"  Learning initial patterns from {len(train_files)} training files...")
                current_patterns = self.pattern_learner.learn_patterns(
                    train_files=train_files,
                    issue_type=issue_type
                )

                # Save initial patterns
                pattern_file = self.phase1_dir / f"{issue_type}_step_0_patterns.json"
                with open(pattern_file, 'w') as f:
                    json.dump(current_patterns, f, indent=2)

            else:
                # Steps 1-2: Refine existing patterns
                logger.info(f"  Refining patterns using {len(train_files)} training files...")

                # Evaluate on train files to find misclassifications
                train_eval = self.fold_evaluator.evaluate_fold(
                    patterns_dict=current_patterns,
                    val_files=train_files,
                    issue_type=issue_type
                )

                # Extract misclassified entries
                misclassified = self._extract_misclassified(train_eval, max_entries=self.max_misclassified_per_step)

                if misclassified:
                    logger.info(f"  Found {len(misclassified)} misclassifications, refining patterns...")

                    # Refine patterns
                    refinements = self.pattern_refiner.refine_patterns(
                        current_patterns=current_patterns,
                        misclassified_entries=misclassified,
                        issue_type=issue_type
                    )

                    # Apply refinements
                    current_patterns = self._apply_refinements(current_patterns, refinements)

                    # Save refinement details
                    refinement_file = self.phase1_dir / f"{issue_type}_step_{step_idx}_refinements.json"
                    with open(refinement_file, 'w') as f:
                        json.dump(refinements, f, indent=2)
                else:
                    logger.info(f"  No misclassifications found, patterns unchanged")

                # Save refined patterns
                pattern_file = self.phase1_dir / f"{issue_type}_step_{step_idx}_patterns.json"
                with open(pattern_file, 'w') as f:
                    json.dump(current_patterns, f, indent=2)

            # Evaluate on validation fold
            logger.info(f"  Evaluating on {len(val_files)} validation files...")
            val_eval = self.fold_evaluator.evaluate_fold(
                patterns_dict=current_patterns,
                val_files=val_files,
                issue_type=issue_type
            )

            metrics = val_eval.get('metrics', {})
            logger.info(f"  Validation Metrics:")
            logger.info(f"    F1: {metrics.get('f1_score', 0.0):.3f}")
            logger.info(f"    Precision: {metrics.get('precision', 0.0):.3f}")
            logger.info(f"    Recall: {metrics.get('recall', 0.0):.3f}")

            # Save step evaluation
            eval_file = self.phase1_dir / f"{issue_type}_step_{step_idx}_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(val_eval, f, indent=2)

            # Record step results
            step_results.append({
                'step': step_idx,
                'train_folds': train_fold_indices,
                'val_fold': val_fold_idx,
                'metrics': metrics,
                'misclassified_count': len(misclassified) if step_idx > 0 else None
            })

        # Return results with final patterns
        return {
            'step_results': step_results,
            'final_patterns': current_patterns,
            'final_metrics': step_results[-1]['metrics'] if step_results else {}
        }

    def _get_refinement_steps(self, n_folds: int) -> List[tuple]:
        """
        Get refinement step configuration.

        For n_folds=3:
        - Step 0: train=[0,1], val=2
        - Step 1: train=[0,2], val=1
        - Step 2: train=[1,2], val=0
        """
        steps = []

        if n_folds == 3:
            steps = [
                ([0, 1], 2),  # Step 0
                ([0, 2], 1),  # Step 1
                ([1, 2], 0)   # Step 2
            ]
        else:
            # General case: rotate validation fold
            for val_fold in range(n_folds):
                train_folds = [i for i in range(n_folds) if i != val_fold]
                steps.append((train_folds, val_fold))

        return steps

    def _extract_misclassified(self, eval_result: Dict, max_entries: int = 20) -> List:
        """Extract misclassified entries from evaluation result."""
        misclassified = []

        for result in eval_result.get('detailed_results', []):
            predicted = result.get('predicted_classification', '')
            ground_truth = result.get('ground_truth_classification', '')

            if predicted != ground_truth:
                misclassified.append(result.get('entry'))

        # Limit to max_entries
        if len(misclassified) > max_entries:
            import random
            random.shuffle(misclassified)
            misclassified = misclassified[:max_entries]

        return misclassified

    def _apply_refinements(self, current_patterns: Dict, refinements: Dict) -> Dict:
        """
        Apply refinements to current patterns.

        Refinements structure:
        {
            "add": {"fp": [...], "tp": [...]},
            "modify": {"fp": [...], "tp": [...]},
            "remove": {"fp": [...], "tp": [...]}
        }
        """
        result_patterns = {
            'fp': list(current_patterns.get('fp', [])),
            'tp': list(current_patterns.get('tp', []))
        }

        # Apply additions
        for pattern_type in ['fp', 'tp']:
            new_patterns = refinements.get('add', {}).get(pattern_type, [])
            result_patterns[pattern_type].extend(new_patterns)

        # Apply modifications
        for pattern_type in ['fp', 'tp']:
            modifications = refinements.get('modify', {}).get(pattern_type, [])
            for mod_pattern in modifications:
                pattern_id = mod_pattern.get('pattern_id')
                # Find and replace
                for idx, p in enumerate(result_patterns[pattern_type]):
                    if p.get('pattern_id') == pattern_id:
                        result_patterns[pattern_type][idx] = mod_pattern
                        break

        # Apply removals
        for pattern_type in ['fp', 'tp']:
            remove_ids = [p.get('pattern_id') for p in refinements.get('remove', {}).get(pattern_type, [])]
            result_patterns[pattern_type] = [
                p for p in result_patterns[pattern_type]
                if p.get('pattern_id') not in remove_ids
            ]

        return result_patterns

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
            # Check if issue_type in this fold (train + val)
            has_issue_type = False

            for file in train_files + val_files:
                try:
                    entries = parser.parse_file(file)
                    if any(e.issue_type == issue_type for e in entries):
                        has_issue_type = True
                        break
                except Exception as e:
                    logger.warning(f"Error parsing {file.name}: {e}")
                    continue

            if not has_issue_type:
                logger.warning(f"  Issue type {issue_type} missing from fold {fold_idx}")
                return False

        return True

    def run_phase1_5(
        self,
        final_patterns_by_issue_type: Dict[str, Dict],
        val_dir: Path,
        max_eval_samples: int = REFINEMENT_DEFAULTS['eval_sample_size']
    ) -> Dict:
        """
        Phase 1.5: Evaluate final refined patterns on validation set.

        Provides baseline before Phase 2 refinement starts.

        Args:
            final_patterns_by_issue_type: Dict mapping issue_type -> final patterns (patterns_2)
            val_dir: Validation directory
            max_eval_samples: Max samples to evaluate (500 for speed)

        Returns:
            Evaluation results for final patterns
        """
        logger.info("="*80)
        logger.info("PHASE 1.5: FINAL PATTERN EVALUATION")
        logger.info("="*80)

        val_files = sorted(val_dir.glob("*.txt"))
        logger.info(f"Evaluating on {len(val_files)} validation files (max {max_eval_samples} samples per type)")

        results_by_issue_type = {}

        for issue_type, final_patterns in final_patterns_by_issue_type.items():
            logger.info(f"\nEvaluating final patterns for {issue_type}...")

            # Evaluate on validation set with sampling
            eval_result = self.fold_evaluator.evaluate_fold(
                patterns_dict=final_patterns,
                val_files=val_files,
                issue_type=issue_type,
                max_entries=max_eval_samples
            )

            metrics = eval_result.get('metrics', {}).get('overall', {})
            logger.info(f"  Final Pattern Metrics:")
            logger.info(f"    F1: {metrics.get('f1', 0.0):.3f}")
            logger.info(f"    Precision: {metrics.get('precision', 0.0):.3f}")
            logger.info(f"    Recall: {metrics.get('recall', 0.0):.3f}")

            # Save results
            results_by_issue_type[issue_type] = eval_result

            # Save per-issue-type evaluation
            eval_file = self.phase1_dir / f"{issue_type}_final_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_result, f, indent=2)

        # Save complete Phase 1.5 results
        phase1_5_results = {
            'metadata': {
                'phase': 1.5,
                'val_dir': str(val_dir),
                'max_eval_samples': max_eval_samples,
                'description': 'Evaluation of final refined patterns on validation set'
            },
            'issue_types': results_by_issue_type
        }

        phase1_5_file = self.phase1_dir / "phase1_5_final_evaluation.json"
        with open(phase1_5_file, 'w') as f:
            json.dump(phase1_5_results, f, indent=2)

        logger.info(f"\nPhase 1.5 complete. Results saved to: {phase1_5_file}")

        return phase1_5_results


def main():
    """Test k-fold orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1: K-Fold Cross-Validation with Iterative Refinement")
    parser.add_argument("train_dir", type=Path, help="Training data directory")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--n-folds", type=int, default=KFOLD_DEFAULTS['n_folds'], help="Number of folds")
    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="nim", help="LLM platform")
    parser.add_argument("--seed", type=int, default=KFOLD_DEFAULTS['random_seed'], help="Random seed")
    parser.add_argument("--workers", type=int, default=KFOLD_DEFAULTS['workers'], help="Parallel workers")
    parser.add_argument("--issue-types", nargs="+", help="Specific issue types to process")

    args = parser.parse_args()

    orchestrator = KFoldOrchestrator(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        platform=args.platform,
        random_seed=args.seed,
        workers=args.workers,
        issue_types=args.issue_types
    )

    results = orchestrator.run_phase1()

    print("\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    print(f"Issue Types: {results['overall_summary']['total_issue_types']}")
    print(f"Folds: {results['overall_summary']['total_folds']}")
    print(f"Refinement Steps: {results['overall_summary']['refinement_steps']}")
    print(f"Duration: {results['overall_summary']['duration_seconds']:.1f}s")


if __name__ == "__main__":
    main()