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
import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Suppress Google Cloud SDK credential warnings
warnings.filterwarnings('ignore', message='Your application has authenticated using end user credentials')

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

    # Class-level lock for main logger
    _main_log_lock = threading.Lock()

    # Class-level lock and counter for progress tracking
    _progress_lock = threading.Lock()
    _progress_callback = None
    _progress_bar = None

    # Class-level LLM call tracking (real-time)
    _llm_lock = threading.Lock()
    _llm_successes = 0
    _llm_failures = 0
    _llm_log_file = None  # Progress log file handle
    _llm_calls_log_file = None  # LLM calls log file handle

    # Class-level error tracking
    _errors_lock = threading.Lock()
    _llm_errors = []  # List of (issue_type, step_idx, operation, error_msg) tuples

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
            workers: Number of parallel workers (applies to both issue-type-level and evaluation-level parallelization)
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

        # Create run-specific log directory under logs/
        from datetime import datetime as dt
        run_timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_dir = Path("logs") / f"run_logs_{run_timestamp}"
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress log file (simple, human-readable progress tracking)
        progress_log_path = self.run_log_dir / "progress.log"
        KFoldOrchestrator._llm_log_file = open(progress_log_path, 'w', buffering=1)  # Line buffered
        KFoldOrchestrator._llm_log_file.write(f"Progress Log - Started at {run_timestamp}\n")
        KFoldOrchestrator._llm_log_file.write("=" * 80 + "\n\n")

        # Initialize LLM calls log file (detailed LLM call tracking)
        llm_calls_log_path = self.run_log_dir / "llm_calls.log"
        KFoldOrchestrator._llm_calls_log_file = open(llm_calls_log_path, 'w', buffering=1)  # Line buffered
        KFoldOrchestrator._llm_calls_log_file.write(f"LLM Calls Log - Started at {run_timestamp}\n")
        KFoldOrchestrator._llm_calls_log_file.write("=" * 80 + "\n\n")

        logger.info(f"  Progress log: {progress_log_path}")
        logger.info(f"  LLM calls log: {llm_calls_log_path}")
        logger.info(f"  Watch progress: tail -f {progress_log_path}")
        logger.info(f"  Watch LLM calls: tail -f {llm_calls_log_path}")

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
        logger.info(f"  Log dir: {self.run_log_dir}")
        logger.info(f"  K-folds: {self.n_folds}")
        logger.info(f"  Platform: {self.platform}")
        logger.info(f"  Workers: {self.workers} (issue-type + evaluation parallelization)")
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

        # Log phase start
        self._log_progress("="*80)
        self._log_progress("PHASE 1 STARTED: K-Fold Cross-Validation with Iterative Refinement")
        self._log_progress("="*80)

        # Step 1: Create k-fold splits
        logger.info("\n[Step 1/4] Creating stratified k-fold splits...")
        self._log_progress("▶ PHASE: Parsing files and creating k-fold splits")
        folds = self.splitter.split(self.train_dir)
        self._log_progress(f"✓ PHASE: Created {len(folds)} folds")

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
        logger.info(f"  Using {self.workers} parallel workers for issue-type-level parallelization")

        # Step 3: Process each issue type with iterative refinement
        results_by_issue_type = {}

        # Validate fold coverage first (quick check)
        valid_issue_types = []
        for issue_type in self.issue_types:
            if self.validate_all_types_per_fold:
                coverage_ok = self._validate_fold_coverage(folds, issue_type)
                if not coverage_ok:
                    logger.error(f"Issue type {issue_type} not in all folds. Skipping.")
                    continue
            valid_issue_types.append(issue_type)

        # Process issue types in parallel
        if self.workers > 1:
            logger.info(f"🔄 Processing {len(valid_issue_types)} issue types with {self.workers} parallel workers...")

            # Calculate total packages to process across all steps
            # Estimate: count files in first fold and multiply by issue types × steps
            sample_fold = folds[0]
            total_files_per_fold = len(sample_fold[0]) + len(sample_fold[1])

            # Total progress units: issue_types × steps × (train_files + val_files)
            # For simplicity, use steps as major unit and files as minor
            total_steps = len(valid_issue_types) * self.n_folds

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Create shared progress bar
                with tqdm(total=total_steps, desc="Processing steps", unit="step") as pbar:
                    # Set the progress bar as class variable so workers can update it
                    KFoldOrchestrator._progress_bar = pbar

                    # Submit all issue types
                    futures = {
                        executor.submit(self._process_issue_type_wrapper, folds, issue_type): issue_type
                        for issue_type in valid_issue_types
                    }

                    # Collect results as they complete
                    for future in as_completed(futures):
                        issue_type = futures[future]
                        try:
                            issue_results = future.result()
                            results_by_issue_type[issue_type] = issue_results

                            # Show final F1 score when issue type completes
                            final_f1 = issue_results.get('final_metrics', {}).get('f1_score', 0.0)
                            pbar.set_postfix_str(f"✅ {issue_type} complete (F1: {final_f1:.3f})")
                        except Exception as e:
                            logger.error(f"✗ Failed {issue_type}: {e}")
                            results_by_issue_type[issue_type] = {'error': str(e)}
                            pbar.set_postfix_str(f"❌ {issue_type} failed")

                    # Clear progress bar reference
                    KFoldOrchestrator._progress_bar = None
        else:
            logger.info(f"🔄 Processing {len(valid_issue_types)} issue types sequentially...")

            for issue_type in tqdm(valid_issue_types, desc="Processing issue types"):
                try:
                    issue_results = self._process_issue_type_wrapper(folds, issue_type)
                    results_by_issue_type[issue_type] = issue_results
                except Exception as e:
                    logger.error(f"✗ Failed {issue_type}: {e}")
                    results_by_issue_type[issue_type] = {'error': str(e)}

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

        # Report any LLM errors that occurred
        self._report_llm_errors()

        return phase1_results

    def _process_issue_type_wrapper(self, folds: List, issue_type: str) -> Dict:
        """
        Wrapper for parallel processing of issue types.

        All detailed logging is suppressed - only results are returned.
        Progress is shown via tqdm progress bar.
        """
        # Get root logger and save its current handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]

        # Remove all handlers from root logger to suppress detailed logging
        for handler in original_handlers:
            root_logger.removeHandler(handler)

        # Create a null handler to discard all detailed logs
        null_handler = logging.NullHandler()
        root_logger.addHandler(null_handler)

        try:
            result = self._run_iterative_refinement(folds, issue_type, logger)
            return result
        except Exception as e:
            # Log error to main log (restore handlers temporarily)
            root_logger.removeHandler(null_handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)

            logger.error(f"Error processing {issue_type}: {e}", exc_info=True)
            raise
        finally:
            # Remove null handler
            root_logger.removeHandler(null_handler)

            # Restore original handlers to root logger
            for handler in original_handlers:
                root_logger.addHandler(handler)

    def _run_iterative_refinement(self, folds: List, issue_type: str, issue_logger: logging.Logger) -> Dict:
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
            issue_logger.info(f"\n  --- Step {step_idx}/{len(refinement_steps)-1} ---")
            issue_logger.info(f"  Train folds: {train_fold_indices}, Val fold: {val_fold_idx}")

            # Get train and val files
            train_files = []
            for fold_idx in train_fold_indices:
                train_files.extend(folds[fold_idx][0])  # Get train portion of each fold
                train_files.extend(folds[fold_idx][1])  # Get val portion of each fold (all files)

            val_files = list(folds[val_fold_idx][0]) + list(folds[val_fold_idx][1])  # All files from val fold

            if step_idx == 0:
                # Step 0: Learn patterns from scratch
                issue_logger.info(f"  Learning initial patterns from {len(train_files)} training files...")
                self._log_progress(f"▶ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | LEARNING patterns from {len(train_files)} files")

                current_patterns = self.pattern_learner.learn_patterns(
                    train_files=train_files,
                    issue_type=issue_type
                )

                self._log_progress(f"✓ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | Pattern learning complete")

                # Save initial patterns
                pattern_file = self.phase1_dir / f"{issue_type}_step_0_patterns.json"
                with open(pattern_file, 'w') as f:
                    json.dump(current_patterns, f, indent=2)

            else:
                # Steps 1-2: Refine existing patterns
                issue_logger.info(f"  Refining patterns using {len(train_files)} training files...")
                self._log_progress(f"▶ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | EVALUATING on {len(train_files)} training files")

                # Evaluate on train files to find misclassifications
                train_eval = self.fold_evaluator.evaluate_fold(
                    patterns_dict=current_patterns,
                    val_files=train_files,
                    issue_type=issue_type
                )

                # Extract misclassified entries
                misclassified = self._extract_misclassified(train_eval, max_entries=self.max_misclassified_per_step)

                if misclassified:
                    issue_logger.info(f"  Found {len(misclassified)} misclassifications, refining patterns...")
                    self._log_progress(f"▶ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | REFINING patterns ({len(misclassified)} misclassifications)")

                    # Refine patterns
                    refinements = self.pattern_refiner.refine_patterns(
                        current_patterns=current_patterns,
                        misclassified_entries=misclassified,
                        issue_type=issue_type
                    )

                    # Apply refinements
                    current_patterns = self._apply_refinements(current_patterns, refinements)

                    self._log_progress(f"✓ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | Pattern refinement complete")

                    # Save refinement details
                    refinement_file = self.phase1_dir / f"{issue_type}_step_{step_idx}_refinements.json"
                    with open(refinement_file, 'w') as f:
                        json.dump(refinements, f, indent=2)
                else:
                    issue_logger.info(f"  No misclassifications found, patterns unchanged")
                    self._log_progress(f"✓ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | No refinements needed")

                # Save refined patterns
                pattern_file = self.phase1_dir / f"{issue_type}_step_{step_idx}_patterns.json"
                with open(pattern_file, 'w') as f:
                    json.dump(current_patterns, f, indent=2)

            # Evaluate on validation fold
            issue_logger.info(f"  Evaluating on {len(val_files)} validation files...")
            self._log_progress(f"▶ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | VALIDATING on {len(val_files)} files")

            val_eval = self.fold_evaluator.evaluate_fold(
                patterns_dict=current_patterns,
                val_files=val_files,
                issue_type=issue_type
            )

            metrics = val_eval.get('metrics', {})
            f1_score = metrics.get('f1_score', 0.0)
            issue_logger.info(f"  Validation Metrics:")
            issue_logger.info(f"    F1: {f1_score:.3f}")
            issue_logger.info(f"    Precision: {metrics.get('precision', 0.0):.3f}")
            issue_logger.info(f"    Recall: {metrics.get('recall', 0.0):.3f}")

            self._log_progress(f"✓ {issue_type} | Step {step_idx}/{len(refinement_steps)-1} | Validation complete (F1: {f1_score:.3f})")

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

            # Report progress to main thread
            self._report_step_progress(issue_type, step_idx, len(refinement_steps), f1_score)

        # Return results with final patterns
        self._log_progress(f"✅ {issue_type} | ALL STEPS COMPLETE")
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
            # Setup per-issue-type logger (append mode)
            issue_logger = self._setup_issue_logger(issue_type, phase_name="PHASE 1.5")

            logger.info(f"\nEvaluating final patterns for {issue_type}...")
            issue_logger.info("\n" + "="*80)
            issue_logger.info("PHASE 1.5: VALIDATION SET EVALUATION")
            issue_logger.info("="*80)

            # Temporarily replace global logger
            original_logger = globals()['logger']
            globals()['logger'] = issue_logger

            try:
                # Evaluate on validation set with sampling
                eval_result = self.fold_evaluator.evaluate_fold(
                    patterns_dict=final_patterns,
                    val_files=val_files,
                    issue_type=issue_type,
                    max_entries=max_eval_samples
                )

                metrics = eval_result.get('metrics', {}).get('overall', {})
                issue_logger.info(f"Final Pattern Metrics:")
                issue_logger.info(f"  F1: {metrics.get('f1', 0.0):.3f}")
                issue_logger.info(f"  Precision: {metrics.get('precision', 0.0):.3f}")
                issue_logger.info(f"  Recall: {metrics.get('recall', 0.0):.3f}")

                # Save results
                results_by_issue_type[issue_type] = eval_result

                # Save per-issue-type evaluation
                eval_file = self.phase1_dir / f"{issue_type}_final_evaluation.json"
                with open(eval_file, 'w') as f:
                    json.dump(eval_result, f, indent=2)

            finally:
                # Restore original logger
                globals()['logger'] = original_logger
                self._cleanup_issue_logger(issue_logger)

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

    def _setup_issue_logger(self, issue_type: str, phase_name: str = "") -> logging.Logger:
        """
        Setup or get per-issue-type logger (append mode for continuation).

        Args:
            issue_type: Issue type name
            phase_name: Optional phase name for logging header

        Returns:
            Configured logger instance
        """
        log_file = self.run_log_dir / f"{issue_type}.log"

        # Create issue-specific logger
        issue_logger = logging.getLogger(f"kfold.{issue_type}.{phase_name}")
        issue_logger.setLevel(logging.INFO)
        issue_logger.propagate = False

        # Add file handler in append mode
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        issue_logger.addHandler(file_handler)

        return issue_logger

    def _cleanup_issue_logger(self, issue_logger: logging.Logger) -> None:
        """
        Clean up issue-specific logger handlers.

        Args:
            issue_logger: Logger to clean up
        """
        for handler in issue_logger.handlers[:]:
            handler.close()
            issue_logger.removeHandler(handler)

    @classmethod
    def _report_step_progress(cls, issue_type: str, step_idx: int, total_steps: int, f1_score: float) -> None:
        """
        Report step progress to main thread (thread-safe).

        Updates the shared progress bar and writes detailed progress to stderr.

        Args:
            issue_type: Issue type being processed
            step_idx: Current step index (0-based)
            total_steps: Total number of steps
            f1_score: F1 score for this step
        """
        import sys
        with cls._progress_lock:
            # Get current LLM stats
            successes, failures = cls._get_llm_stats()
            total_llm = successes + failures

            # Update progress bar if available
            if hasattr(cls, '_progress_bar') and cls._progress_bar is not None:
                if total_llm > 0:
                    cls._progress_bar.set_postfix_str(f"{issue_type} Step {step_idx}/{total_steps-1} (F1: {f1_score:.3f}, LLM: {successes}/{total_llm})")
                else:
                    cls._progress_bar.set_postfix_str(f"{issue_type} Step {step_idx}/{total_steps-1} (F1: {f1_score:.3f})")
                cls._progress_bar.update(1)

            # Also write detailed message to stderr
            if total_llm > 0:
                progress_msg = f"  → {issue_type}: Step {step_idx}/{total_steps-1} complete (F1: {f1_score:.3f}, LLM: {successes}/{total_llm} ✅)\n"
            else:
                progress_msg = f"  → {issue_type}: Step {step_idx}/{total_steps-1} complete (F1: {f1_score:.3f})\n"
            sys.stderr.write(progress_msg)
            sys.stderr.flush()

    @classmethod
    def _report_activity(cls, issue_type: str, step_idx: int, activity: str) -> None:
        """
        Report detailed activity progress (thread-safe).

        Args:
            issue_type: Issue type being processed
            step_idx: Current step index
            activity: Activity description
        """
        import sys
        with cls._progress_lock:
            activity_msg = f"    ⚙️  {issue_type} [Step {step_idx}]: {activity}\n"
            sys.stderr.write(activity_msg)
            sys.stderr.flush()

    @classmethod
    def _log_progress(cls, message: str) -> None:
        """Write a progress message to the progress log (thread-safe)."""
        with cls._llm_lock:
            if cls._llm_log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cls._llm_log_file.write(f"[{timestamp}] {message}\n")

    @classmethod
    def _record_llm_success(cls, issue_type: str = "unknown", operation: str = "classify", context: str = "") -> None:
        """Record a successful LLM API call (thread-safe)."""
        with cls._llm_lock:
            cls._llm_successes += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Write to progress log (simple format)
            if cls._llm_log_file:
                cls._llm_log_file.write(f"[{timestamp}] ✅ SUCCESS | {issue_type} | {operation} | Total: {cls._llm_successes}/{cls._llm_successes + cls._llm_failures}\n")

            # Write to detailed LLM calls log
            if cls._llm_calls_log_file:
                context_str = f" | Context: {context}" if context else ""
                cls._llm_calls_log_file.write(f"[{timestamp}] ✅ SUCCESS\n")
                cls._llm_calls_log_file.write(f"  Issue Type: {issue_type}\n")
                cls._llm_calls_log_file.write(f"  Operation: {operation}{context_str}\n")
                cls._llm_calls_log_file.write(f"  Running Total: {cls._llm_successes} success / {cls._llm_failures} failures\n")
                cls._llm_calls_log_file.write("-" * 80 + "\n\n")

    @classmethod
    def _record_llm_failure(cls, issue_type: str = "unknown", operation: str = "classify", error: str = "", context: str = "") -> None:
        """Record a failed LLM API call (thread-safe)."""
        with cls._llm_lock:
            cls._llm_failures += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Write to progress log (simple format)
            if cls._llm_log_file:
                error_msg = f" | Error: {error[:100]}" if error else ""
                cls._llm_log_file.write(f"[{timestamp}] ❌ FAILURE | {issue_type} | {operation}{error_msg} | Total: {cls._llm_successes}/{cls._llm_successes + cls._llm_failures}\n")

            # Write to detailed LLM calls log
            if cls._llm_calls_log_file:
                context_str = f" | Context: {context}" if context else ""
                cls._llm_calls_log_file.write(f"[{timestamp}] ❌ FAILURE\n")
                cls._llm_calls_log_file.write(f"  Issue Type: {issue_type}\n")
                cls._llm_calls_log_file.write(f"  Operation: {operation}{context_str}\n")
                if error:
                    cls._llm_calls_log_file.write(f"  Error: {error}\n")
                cls._llm_calls_log_file.write(f"  Running Total: {cls._llm_successes} success / {cls._llm_failures} failures\n")
                cls._llm_calls_log_file.write("-" * 80 + "\n\n")

    @classmethod
    def _get_llm_stats(cls) -> tuple:
        """Get current LLM statistics (thread-safe)."""
        with cls._llm_lock:
            return (cls._llm_successes, cls._llm_failures)

    def _report_llm_errors(self) -> None:
        """
        Report LLM call statistics from logs.

        Scans logs for successful and failed LLM calls.
        """
        import re

        error_patterns = [
            r'(?i)error.*vertex.*api',
            r'(?i)error.*anthropic',
            r'(?i)rate.*limit',
            r'(?i)quota.*exceeded',
            r'(?i)429',
            r'(?i)failed to.*pattern',
            r'(?i)json.*decode.*error'
        ]

        success_patterns = [
            r'Successfully generated patterns',
            r'Successfully refined patterns',
            r'Pattern learning complete',
            r'Refinement complete',
            r'Classified \d+ entries'
        ]

        main_log = self.run_log_dir / "main.log"
        llm_errors = []
        llm_successes = 0

        # Check main log
        if main_log.exists():
            with open(main_log, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Check for errors
                    for pattern in error_patterns:
                        if re.search(pattern, line):
                            llm_errors.append(('main.log', line_num, line.strip()))
                            break
                    # Check for successes
                    for pattern in success_patterns:
                        if re.search(pattern, line):
                            llm_successes += 1
                            break

        # Check issue-specific logs
        for log_file in self.run_log_dir.glob("*.log"):
            if log_file.name == "main.log":
                continue

            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Check for errors
                    for pattern in error_patterns:
                        if re.search(pattern, line):
                            llm_errors.append((log_file.name, line_num, line.strip()))
                            break
                    # Check for successes
                    for pattern in success_patterns:
                        if re.search(pattern, line):
                            llm_successes += 1
                            break

        # Report statistics
        total_calls = llm_successes + len(llm_errors)

        logger.info(f"\n📊 LLM Call Statistics:")
        logger.info(f"  Total LLM calls: {total_calls}")
        logger.info(f"  ✅ Successful: {llm_successes}")
        logger.info(f"  ❌ Failed: {len(llm_errors)}")

        if total_calls > 0:
            success_rate = (llm_successes / total_calls) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")

        if llm_errors:
            logger.warning(f"\n⚠️  LLM Error Details ({len(llm_errors)} errors):")
            for log_file, line_num, line in llm_errors[:10]:  # Show first 10
                logger.warning(f"  {log_file}:{line_num} - {line[:100]}")
            if len(llm_errors) > 10:
                logger.warning(f"  ... and {len(llm_errors) - 10} more errors")
            logger.warning(f"  Check logs in: {self.run_log_dir}")


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