#!/usr/bin/env python3
"""
Refinement Orchestrator - Phase 2 Orchestration

Orchestrates iterative pattern refinement using proper ML pipeline:
1. Start with merged patterns from Phase 1
2. Evaluate on TRAIN (find misclassifications for refinement)
3. Evaluate on VAL (for early stopping)
4. Refine patterns based on TRAIN misclassifications
5. Repeat until VAL F1 stops improving (< 1% for 3 iterations)

Key ML Best Practices:
- Refines on TRAIN data only (avoids data leakage)
- Uses VAL for early stopping criterion (prevents overfitting)
- Tracks both train and val metrics (overfitting detection)
- Test set remains completely held-out until Phase 3
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.pattern_refiner import PatternRefiner
from process_mining.kfold_pattern_learning.fold_evaluator import FoldEvaluator
from process_mining.core.parsers import ValidationEntryParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RefinementOrchestrator:
    """
    Orchestrates Phase 2: Iterative Pattern Refinement.

    Takes merged patterns from Phase 1 and refines them using proper ML pipeline:
    - Refines on TRAIN misclassifications (learning)
    - Validates on VAL for early stopping (prevents overfitting)
    - Tracks both train/val metrics (overfitting detection)
    """

    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        output_dir: Path,
        platform: str = "nim",
        workers: int = 1,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01,
        convergence_patience: int = 3,
        max_misclassified_per_iteration: int = 20,
        eval_sample_size: int = 500
    ):
        """
        Initialize refinement orchestrator.

        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            output_dir: Output directory for Phase 2 results
            platform: LLM platform ("local" or "nim")
            workers: Number of parallel workers for evaluation
            max_iterations: Maximum refinement iterations
            convergence_threshold: F1 improvement threshold (default: 1%)
            convergence_patience: Iterations without improvement before stopping
            max_misclassified_per_iteration: Max misclassified examples per refinement
            eval_sample_size: Max entries to sample for train/val evaluation (default: 500)
                             Set to 0 or None to evaluate all entries (slow but accurate)
        """
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        self.platform = platform
        self.workers = workers
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.max_misclassified_per_iteration = max_misclassified_per_iteration
        self.eval_sample_size = eval_sample_size if eval_sample_size and eval_sample_size > 0 else None

        # Create output directories
        self.phase2_dir = self.output_dir / "phase2_refinement_results"
        self.phase2_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pattern_refiner = PatternRefiner(platform=platform)
        self.fold_evaluator = FoldEvaluator(platform=platform, workers=workers)

        logger.info(f"Initialized RefinementOrchestrator:")
        logger.info(f"  Train dir: {self.train_dir}")
        logger.info(f"  Val dir: {self.val_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Platform: {self.platform}")
        logger.info(f"  Max iterations: {self.max_iterations}")
        logger.info(f"  Convergence threshold: {self.convergence_threshold}")
        logger.info(f"  Eval sample size: {self.eval_sample_size if self.eval_sample_size else 'ALL (no sampling)'}")

    def run_phase2(
        self,
        initial_patterns_by_issue_type: Dict[str, Dict],
        issue_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Run Phase 2: Iterative Pattern Refinement.

        Args:
            initial_patterns_by_issue_type: Dict mapping issue_type -> merged patterns from Phase 1
            issue_types: List of issue types to process (None = all from initial_patterns)

        Returns:
            Dictionary with complete Phase 2 results:
            {
                'metadata': {...},
                'issue_types': {
                    'RESOURCE_LEAK': {
                        'initial_patterns': {...},
                        'iterations': [...],
                        'final_patterns': {...},
                        'convergence_info': {...}
                    },
                    ...
                },
                'overall_summary': {...}
            }
        """
        logger.info("="*80)
        logger.info("PHASE 2: ITERATIVE PATTERN REFINEMENT")
        logger.info("="*80)

        start_time = datetime.now()

        # Determine issue types to process
        if issue_types is None:
            issue_types = list(initial_patterns_by_issue_type.keys())

        logger.info(f"\nProcessing {len(issue_types)} issue types: {issue_types}")

        # Keep train and val separate for proper ML pipeline
        train_files = sorted(self.train_dir.glob("*.txt"))
        val_files = sorted(self.val_dir.glob("*.txt"))

        logger.info(f"Dataset split: {len(train_files)} train + {len(val_files)} val")
        logger.info(f"Strategy: Refine on TRAIN misclassifications, validate on VAL for early stopping")

        # Process each issue type
        results_by_issue_type = {}

        for issue_type in issue_types:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Issue Type: {issue_type}")
            logger.info(f"{'='*80}")

            initial_patterns = initial_patterns_by_issue_type.get(issue_type, {"fp": [], "tp": []})

            # Run iterative refinement
            refinement_results = self._refine_issue_type(
                issue_type=issue_type,
                initial_patterns=initial_patterns,
                train_files=train_files,
                val_files=val_files
            )

            results_by_issue_type[issue_type] = refinement_results

            # Log convergence info
            convergence = refinement_results['convergence_info']
            logger.info(f"\n  Refinement complete:")
            logger.info(f"    Iterations: {convergence['iterations_completed']}")
            logger.info(f"    Converged: {convergence['converged']}")
            logger.info(f"    Reason: {convergence['convergence_reason']}")
            logger.info(f"    Initial F1: {convergence['initial_f1']:.3f}")
            logger.info(f"    Final F1: {convergence['final_f1']:.3f}")
            logger.info(f"    Improvement: {convergence['f1_improvement']:.3f}")

        # Generate overall summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        overall_summary = {
            'total_issue_types': len(issue_types),
            'duration_seconds': duration,
            'avg_iterations': sum(
                results_by_issue_type[it]['convergence_info']['iterations_completed']
                for it in issue_types
            ) / len(issue_types),
            'avg_f1_improvement': sum(
                results_by_issue_type[it]['convergence_info']['f1_improvement']
                for it in issue_types
            ) / len(issue_types)
        }

        # Compile final results
        phase2_results = {
            'metadata': {
                'phase': 2,
                'train_dir': str(self.train_dir),
                'val_dir': str(self.val_dir),
                'platform': self.platform,
                'max_iterations': self.max_iterations,
                'convergence_threshold': self.convergence_threshold,
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration
            },
            'issue_types': results_by_issue_type,
            'overall_summary': overall_summary
        }

        # Save complete Phase 2 results
        phase2_file = self.phase2_dir / "phase2_complete_results.json"
        with open(phase2_file, 'w') as f:
            json.dump(phase2_results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2 COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Avg iterations: {overall_summary['avg_iterations']:.1f}")
        logger.info(f"Avg F1 improvement: {overall_summary['avg_f1_improvement']:.3f}")
        logger.info(f"Results saved to: {self.phase2_dir}")

        return phase2_results

    def _refine_issue_type(
        self,
        issue_type: str,
        initial_patterns: Dict,
        train_files: List[Path],
        val_files: List[Path]
    ) -> Dict:
        """
        Refine patterns for a single issue type.

        Uses train data for refinement and val data for early stopping (proper ML pipeline).

        Args:
            issue_type: Issue type to refine
            initial_patterns: Initial patterns from Phase 1
            train_files: Training files (for refinement)
            val_files: Validation files (for early stopping)

        Returns:
            Dictionary with refinement results for this issue type
        """
        current_patterns = initial_patterns.copy()
        iterations = []

        best_val_f1 = 0.0
        initial_train_f1 = 0.0
        initial_val_f1 = 0.0
        no_improvement_count = 0
        convergence_reason = "Unknown"

        for iteration in range(self.max_iterations):
            logger.info(f"\n  --- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Step A: Evaluate on TRAIN (to find refinement targets)
            logger.info(f"  Evaluating on TRAIN ({len(train_files)} files)...")
            train_eval_result = self.fold_evaluator.evaluate_fold(
                patterns_dict=current_patterns,
                val_files=train_files,
                issue_type=issue_type,
                max_entries=self.eval_sample_size
            )

            train_metrics = train_eval_result.get('metrics', {}).get('overall', {})
            train_f1 = train_metrics.get('f1', 0.0)

            logger.info(f"    Train F1: {train_f1:.3f}")
            logger.info(f"    Train Precision: {train_metrics.get('precision', 0.0):.3f}")
            logger.info(f"    Train Recall: {train_metrics.get('recall', 0.0):.3f}")

            # Step B: Evaluate on VAL (for early stopping)
            logger.info(f"  Evaluating on VAL ({len(val_files)} files)...")
            val_eval_result = self.fold_evaluator.evaluate_fold(
                patterns_dict=current_patterns,
                val_files=val_files,
                issue_type=issue_type,
                max_entries=self.eval_sample_size
            )

            val_metrics = val_eval_result.get('metrics', {}).get('overall', {})
            val_f1 = val_metrics.get('f1', 0.0)

            logger.info(f"    Val F1: {val_f1:.3f}")
            logger.info(f"    Val Precision: {val_metrics.get('precision', 0.0):.3f}")
            logger.info(f"    Val Recall: {val_metrics.get('recall', 0.0):.3f}")

            # Check for overfitting
            if train_f1 - val_f1 > 0.1:
                logger.warning(f"    ⚠️  Potential overfitting detected: Train F1 ({train_f1:.3f}) >> Val F1 ({val_f1:.3f})")

            # Step C: Check convergence based on VAL F1 (early stopping)
            if iteration == 0:
                best_val_f1 = val_f1
                initial_train_f1 = train_f1
                initial_val_f1 = val_f1
            else:
                improvement = val_f1 - best_val_f1

                if improvement > self.convergence_threshold:
                    logger.info(f"    Val F1 Improvement: +{improvement:.3f} (above threshold)")
                    best_val_f1 = val_f1
                    no_improvement_count = 0
                else:
                    logger.info(f"    Val F1 Improvement: +{improvement:.3f} (below threshold)")
                    no_improvement_count += 1

            # Step D: Get misclassified entries from TRAIN (for refinement)
            train_misclassified_entries = self._get_misclassified_entries(
                train_eval_result,
                train_files,
                issue_type
            )

            logger.info(f"    Train Misclassified: {len(train_misclassified_entries)}")

            # Save iteration results (with both train and val metrics)
            iteration_result = {
                'iteration': iteration + 1,
                'train_f1': train_f1,
                'train_precision': train_metrics.get('precision', 0.0),
                'train_recall': train_metrics.get('recall', 0.0),
                'train_accuracy': train_metrics.get('accuracy', 0.0),
                'val_f1': val_f1,
                'val_precision': val_metrics.get('precision', 0.0),
                'val_recall': val_metrics.get('recall', 0.0),
                'val_accuracy': val_metrics.get('accuracy', 0.0),
                'train_misclassified_count': len(train_misclassified_entries),
                'patterns': current_patterns.copy()
            }
            iterations.append(iteration_result)

            # Save iteration patterns
            iter_pattern_file = self.phase2_dir / f"{issue_type}_iteration_{iteration + 1}_patterns.json"
            with open(iter_pattern_file, 'w') as f:
                json.dump(current_patterns, f, indent=2)

            # Check convergence (based on VAL F1 for early stopping)
            if no_improvement_count >= self.convergence_patience:
                logger.info(f"\n  Convergence reached: No VAL F1 improvement for {self.convergence_patience} iterations")
                convergence_reason = f"No VAL F1 improvement for {self.convergence_patience} iterations"
                break

            if len(train_misclassified_entries) == 0:
                logger.info(f"\n  Perfect classification on TRAIN achieved!")
                convergence_reason = "Perfect TRAIN classification"
                break

            # Step E: Refine patterns based on TRAIN misclassifications
            logger.info(f"  Refining patterns based on {len(train_misclassified_entries)} TRAIN misclassified examples...")
            refinements = self.pattern_refiner.refine_patterns(
                current_patterns=current_patterns,
                misclassified_entries=train_misclassified_entries[:self.max_misclassified_per_iteration],
                evaluation_results=train_eval_result.get('results', []),
                issue_type=issue_type,
                max_examples_per_type=self.max_misclassified_per_iteration // 2
            )

            # Apply refinements
            current_patterns = self._apply_refinements(current_patterns, refinements)

            # Save refinements
            refinement_file = self.phase2_dir / f"{issue_type}_iteration_{iteration + 1}_refinements.json"
            with open(refinement_file, 'w') as f:
                json.dump(refinements, f, indent=2)

            # Last iteration check
            if iteration == self.max_iterations - 1:
                logger.info(f"\n  Max iterations reached")
                convergence_reason = "Max iterations reached"

        # Compile results (use VAL F1 for final metrics, as it's the early stopping criterion)
        final_val_f1 = iterations[-1]['val_f1'] if iterations else 0.0
        final_train_f1 = iterations[-1]['train_f1'] if iterations else 0.0

        convergence_info = {
            'converged': no_improvement_count >= self.convergence_patience or (iterations and iterations[-1]['train_misclassified_count'] == 0),
            'convergence_reason': convergence_reason,
            'iterations_completed': len(iterations),
            'initial_train_f1': initial_train_f1 if iterations else 0.0,
            'initial_val_f1': initial_val_f1 if iterations else 0.0,
            'final_train_f1': final_train_f1,
            'final_val_f1': final_val_f1,
            'initial_f1': initial_val_f1 if iterations else 0.0,  # For backward compatibility
            'final_f1': final_val_f1,  # For backward compatibility
            'f1_improvement': final_val_f1 - (initial_val_f1 if iterations else 0.0),
            'overfitting_gap': final_train_f1 - final_val_f1  # Track overfitting
        }

        return {
            'initial_patterns': initial_patterns,
            'iterations': iterations,
            'final_patterns': current_patterns,
            'convergence_info': convergence_info
        }

    def _get_misclassified_entries(
        self,
        eval_result: Dict,
        combined_files: List[Path],
        issue_type: str
    ) -> List:
        """Get misclassified ValidationEntry objects from evaluation results."""
        parser = ValidationEntryParser()

        # Parse all entries
        all_entries = []
        for file in combined_files:
            entries = parser.parse_file(file)
            entries = [e for e in entries if e.issue_type == issue_type]
            all_entries.extend(entries)

        # Get misclassified results
        results = eval_result.get('results', [])
        misclassified_results = [r for r in results if not r.get('correct', True)]

        # Match misclassified results to entries using entry_id
        misclassified_entries = []
        for result in misclassified_results:
            result_entry_id = result.get('entry_id', '')
            for entry in all_entries:
                if entry.entry_id == result_entry_id:
                    misclassified_entries.append(entry)
                    break

        return misclassified_entries

    def _apply_refinements(
        self,
        current_patterns: Dict,
        refinements: Dict
    ) -> Dict:
        """Apply refinement actions to current patterns."""
        patterns = {
            'fp': current_patterns.get('fp', []).copy(),
            'tp': current_patterns.get('tp', []).copy()
        }

        # Apply adds
        for add_action in refinements.get('add', []):
            pattern_type = add_action.get('pattern_type')
            new_pattern = {
                'pattern_id': add_action.get('pattern_id'),
                'group': add_action.get('group'),
                'summary': add_action.get('summary')
            }

            if pattern_type == 'fp':
                patterns['fp'].append(new_pattern)
            elif pattern_type == 'tp':
                patterns['tp'].append(new_pattern)

        # Apply modifications
        for modify_action in refinements.get('modify', []):
            pattern_id = modify_action.get('pattern_id')
            new_summary = modify_action.get('new_summary')

            # Find and update pattern
            for pattern_type in ['fp', 'tp']:
                for pattern in patterns[pattern_type]:
                    if pattern.get('pattern_id') == pattern_id:
                        pattern['summary'] = new_summary
                        break

        # Apply removals
        for remove_action in refinements.get('remove', []):
            pattern_id = remove_action.get('pattern_id')

            # Remove from both fp and tp
            for pattern_type in ['fp', 'tp']:
                patterns[pattern_type] = [
                    p for p in patterns[pattern_type]
                    if p.get('pattern_id') != pattern_id
                ]

        return patterns


def main():
    """Test refinement orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2: Iterative Pattern Refinement")
    parser.add_argument("train_dir", type=Path, help="Training data directory")
    parser.add_argument("val_dir", type=Path, help="Validation data directory")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("phase1_results", type=Path, help="Phase 1 results JSON file")
    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="nim", help="LLM platform")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max refinement iterations")
    parser.add_argument("--issue-types", nargs="+", help="Specific issue types to process")

    args = parser.parse_args()

    # Load Phase 1 results
    with open(args.phase1_results, 'r') as f:
        phase1_results = json.load(f)

    # Extract merged patterns by issue type
    initial_patterns_by_issue_type = {}
    for issue_type, data in phase1_results.get('issue_types', {}).items():
        initial_patterns_by_issue_type[issue_type] = data.get('merged_patterns', {"fp": [], "tp": []})

    # Run Phase 2
    orchestrator = RefinementOrchestrator(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        platform=args.platform,
        workers=args.workers,
        max_iterations=args.max_iterations
    )

    results = orchestrator.run_phase2(
        initial_patterns_by_issue_type=initial_patterns_by_issue_type,
        issue_types=args.issue_types
    )

    print("\n" + "="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    print(f"Issue Types: {results['overall_summary']['total_issue_types']}")
    print(f"Avg Iterations: {results['overall_summary']['avg_iterations']:.1f}")
    print(f"Duration: {results['overall_summary']['duration_seconds']:.1f}s")
    print(f"Avg F1 Improvement: {results['overall_summary']['avg_f1_improvement']:.3f}")


if __name__ == "__main__":
    main()