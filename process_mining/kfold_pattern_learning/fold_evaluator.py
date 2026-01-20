#!/usr/bin/env python3
"""
Fold Evaluator - Evaluate Patterns on Held-Out Fold

Evaluates learned patterns on a validation fold using the existing
pattern_classifier infrastructure.
"""

import sys
import json
import tempfile
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.evaluators import PatternEvaluator
from process_mining.core.parsers import ValidationEntryParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FoldEvaluator:
    """
    Evaluates patterns on a held-out validation fold.

    Reuses the existing PatternEvaluator infrastructure but provides
    a simplified interface for k-fold validation.
    """

    def __init__(
        self,
        platform: str = "nim",
        workers: int = 1,
        verbose: bool = False
    ):
        """
        Initialize fold evaluator.

        Args:
            platform: LLM platform ("local" or "nim")
            workers: Number of parallel workers for evaluation
            verbose: Enable verbose logging
        """
        self.platform = platform
        self.workers = workers
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def evaluate_fold(
        self,
        patterns_dict: Dict,
        val_files: List[Path],
        issue_type: str,
        max_entries: int = None
    ) -> Dict:
        """
        Evaluate patterns on validation files.

        Args:
            patterns_dict: Pattern dictionary {"fp": [...], "tp": [...]}
            val_files: List of validation .txt files
            issue_type: Issue type these patterns are for
            max_entries: Optional max entries to sample (stratified by FP/TP)
                        If None, evaluates all entries

        Returns:
            Dictionary with metrics:
            {
                'issue_type': str,
                'total_entries': int,
                'fp_count': int,
                'tp_count': int,
                'metrics': {
                    'precision': float,
                    'recall': float,
                    'f1': float,
                    'accuracy': float,
                    ...
                },
                'confusion_matrix': {...},
                'results': [...]  # per-entry results
            }
        """
        if max_entries:
            logger.info(f"Evaluating patterns for {issue_type} on {len(val_files)} files (sampling max {max_entries} entries)...")
        else:
            logger.info(f"Evaluating patterns for {issue_type} on {len(val_files)} validation files...")

        # Parse and collect all entries first
        parser = ValidationEntryParser()
        all_entries = []
        for val_file in val_files:
            entries = parser.parse_file(val_file)
            # Filter to this issue type
            entries = [e for e in entries if e.issue_type == issue_type]
            all_entries.extend(entries)

        # Apply stratified sampling if max_entries specified
        if max_entries and len(all_entries) > max_entries:
            sampled_entries = self._stratified_sample(all_entries, max_entries)
            logger.info(f"  Sampled {len(sampled_entries)} / {len(all_entries)} entries (stratified by FP/TP)")
        else:
            sampled_entries = all_entries

        # Create temporary directory for pattern file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            pattern_file = temp_dir_path / f"{issue_type}_patterns.json"

            # Write patterns to temporary file
            with open(pattern_file, 'w') as f:
                json.dump(patterns_dict, f, indent=2)

            # Create temporary directory for validation data
            val_dir = temp_dir_path / "validation"
            val_dir.mkdir()

            # Write sampled entries to temp file
            temp_val_file = val_dir / "sampled_entries.txt"
            self._write_entries_to_file(sampled_entries, temp_val_file)

            # Create output file path
            output_file = temp_dir_path / "evaluation_results.json"

            # Run evaluation using PatternEvaluator
            evaluator = PatternEvaluator(
                validation_data_dir=val_dir,
                output_file=output_file,
                patterns_dir=temp_dir_path,  # Directory containing pattern file
                verbose=self.verbose,
                workers=self.workers,
                platform=self.platform,
                baseline_mode=False
            )

            # Run evaluation
            results = evaluator.run(dry_run=False)

            # Extract metrics
            metrics = results.get('metrics', {})

            # Count entries from sampled set
            fp_count = sum(1 for e in sampled_entries if 'FALSE' in e.ground_truth_classification)
            tp_count = len(sampled_entries) - fp_count

            # Compile fold results
            fold_results = {
                'issue_type': issue_type,
                'total_entries': len(sampled_entries),
                'fp_count': fp_count,
                'tp_count': tp_count,
                'metrics': metrics,
                'results': results.get('results', [])
            }

            logger.info(f"  Evaluated {len(sampled_entries)} entries for {issue_type}")
            logger.info(f"  F1: {metrics.get('f1_score', 0.0):.3f}")
            logger.info(f"  Precision: {metrics.get('precision', 0.0):.3f}")
            logger.info(f"  Recall: {metrics.get('recall', 0.0):.3f}")

            return fold_results

    def _stratified_sample(self, all_entries: List, max_entries: int) -> List:
        """
        Stratified sampling preserving FP/TP ratio.

        Args:
            all_entries: List of ValidationEntry objects
            max_entries: Maximum entries to sample

        Returns:
            Sampled list of ValidationEntry objects preserving FP/TP ratio
        """
        import random

        if not all_entries or len(all_entries) <= max_entries:
            return all_entries

        # Separate into FP and TP
        fp_entries = [e for e in all_entries if 'FALSE' in e.ground_truth_classification]
        tp_entries = [e for e in all_entries if 'TRUE' in e.ground_truth_classification]

        total_count = len(all_entries)
        fp_ratio = len(fp_entries) / total_count if total_count > 0 else 0.5

        # Calculate target counts maintaining FP/TP ratio
        target_fp_count = int(max_entries * fp_ratio)
        target_tp_count = max_entries - target_fp_count

        # Ensure at least 1 of each if they exist
        if len(fp_entries) > 0 and target_fp_count == 0:
            target_fp_count = 1
            target_tp_count = max_entries - 1
        if len(tp_entries) > 0 and target_tp_count == 0:
            target_tp_count = 1
            target_fp_count = max_entries - 1

        # Random sample maintaining ratio
        sampled_fp = random.sample(fp_entries, min(target_fp_count, len(fp_entries)))
        sampled_tp = random.sample(tp_entries, min(target_tp_count, len(tp_entries)))

        sampled = sampled_fp + sampled_tp
        random.shuffle(sampled)

        logger.debug(f"    Stratified sampling: {len(sampled)} / {total_count} entries "
                    f"(FP: {len(sampled_fp)}/{len(fp_entries)}, "
                    f"TP: {len(sampled_tp)}/{len(tp_entries)}, "
                    f"FP ratio: {fp_ratio:.2f})")

        return sampled

    def _write_entries_to_file(self, entries: List, file_path: Path):
        """
        Write ValidationEntry objects to file in correct format.

        Args:
            entries: List of ValidationEntry objects
            file_path: Path to output file
        """
        if not entries:
            # Write empty file
            file_path.write_text("")
            return

        # Group entries by package for proper formatting
        package_name = entries[0].package_name if entries else "sampled"

        with open(file_path, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"GROUND-TRUTH ENTRIES FOR: {package_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Package: {package_name}\n")
            f.write(f"Total Entries: {len(entries)}\n\n")

            # Write each entry
            for i, entry in enumerate(entries, 1):
                f.write("---\n")
                f.write(f"Entry #{i}:\n")
                f.write(f"Issue Type: {entry.issue_type}\n")
                f.write(f"CWE: {entry.cwe}\n")
                f.write("\n")
                f.write("Error Trace:\n")
                f.write(f"{entry.error_trace}\n")
                f.write("\n")
                # Extract file path and write source code
                # Assume source_code contains the full formatted code
                f.write(f"Source Code:\n")
                f.write(f"```c\n")
                f.write(f"{entry.source_code}\n")
                f.write(f"```\n")
                f.write("\n")
                f.write(f"Ground Truth Classification: {entry.ground_truth_classification}\n")
                f.write(f"Human Expert Justification: {entry.ground_truth_justification}\n")

    def evaluate_fold_multi_issue_types(
        self,
        patterns_by_issue_type: Dict[str, Dict],
        val_files: List[Path]
    ) -> Dict:
        """
        Evaluate patterns for multiple issue types on same validation files.

        Args:
            patterns_by_issue_type: Dict mapping issue_type -> {"fp": [...], "tp": [...]}
            val_files: List of validation .txt files

        Returns:
            Dictionary with per-issue-type results:
            {
                'RESOURCE_LEAK': {...},
                'UNINIT': {...},
                ...
                'overall_metrics': {...}
            }
        """
        logger.info(f"Evaluating {len(patterns_by_issue_type)} issue types on {len(val_files)} validation files...")

        results = {}
        all_entries_count = 0
        all_correct_count = 0

        for issue_type, patterns in patterns_by_issue_type.items():
            fold_results = self.evaluate_fold(patterns, val_files, issue_type)
            results[issue_type] = fold_results

            all_entries_count += fold_results['total_entries']
            # Count correct predictions
            correct = sum(1 for r in fold_results['results'] if r.get('correct', False))
            all_correct_count += correct

        # Calculate overall metrics
        overall_accuracy = all_correct_count / all_entries_count if all_entries_count > 0 else 0.0

        results['overall_metrics'] = {
            'total_entries': all_entries_count,
            'total_correct': all_correct_count,
            'accuracy': overall_accuracy,
            'issue_types_evaluated': len(patterns_by_issue_type)
        }

        logger.info(f"Overall accuracy across all issue types: {overall_accuracy:.3f}")

        return results


def main():
    """Test fold evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test fold evaluation")
    parser.add_argument("patterns_file", type=Path, help="Pattern JSON file")
    parser.add_argument("val_dir", type=Path, help="Validation directory")
    parser.add_argument("issue_type", type=str, help="Issue type")
    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="nim", help="LLM platform")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load patterns
    with open(args.patterns_file, 'r') as f:
        patterns = json.load(f)

    # Get validation files
    val_files = sorted(args.val_dir.glob("*.txt"))
    if not val_files:
        logger.error(f"No .txt files found in {args.val_dir}")
        return

    # Evaluate
    evaluator = FoldEvaluator(
        platform=args.platform,
        workers=args.workers,
        verbose=args.verbose
    )

    results = evaluator.evaluate_fold(patterns, val_files, args.issue_type)

    print("\n" + "="*80)
    print("FOLD EVALUATION RESULTS")
    print("="*80)
    print(f"Issue Type: {results['issue_type']}")
    print(f"Total Entries: {results['total_entries']}")
    print(f"FP: {results['fp_count']}, TP: {results['tp_count']}")
    print(f"\nMetrics:")
    metrics = results['metrics'].get('overall', {})
    print(f"  Precision: {metrics.get('precision', 0.0):.3f}")
    print(f"  Recall: {metrics.get('recall', 0.0):.3f}")
    print(f"  F1: {metrics.get('f1', 0.0):.3f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0.0):.3f}")


if __name__ == "__main__":
    main()