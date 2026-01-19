#!/usr/bin/env python3
"""
Test Evaluator - Phase 3 Final Evaluation

Evaluates refined patterns on the held-out test set and generates
comprehensive performance metrics and analysis.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.fold_evaluator import FoldEvaluator
from process_mining.core.parsers import ValidationEntryParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestEvaluator:
    """
    Evaluates refined patterns on held-out test set (Phase 3).

    Provides comprehensive metrics, confusion matrices, and
    per-issue-type breakdown for final evaluation.
    """

    def __init__(
        self,
        test_dir: Path,
        output_dir: Path,
        platform: str = "nim",
        workers: int = 1
    ):
        """
        Initialize test evaluator.

        Args:
            test_dir: Test data directory
            output_dir: Output directory for Phase 3 results
            platform: LLM platform ("local" or "nim")
            workers: Number of parallel workers for evaluation
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.platform = platform
        self.workers = workers

        # Create output directories
        self.phase3_dir = self.output_dir / "phase3_test_results"
        self.phase3_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.fold_evaluator = FoldEvaluator(platform=platform, workers=workers)

        logger.info(f"Initialized TestEvaluator:")
        logger.info(f"  Test dir: {self.test_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Platform: {self.platform}")

    def run_phase3(
        self,
        final_patterns_by_issue_type: Dict[str, Dict],
        issue_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Run Phase 3: Final Test Evaluation.

        Args:
            final_patterns_by_issue_type: Dict mapping issue_type -> refined patterns from Phase 2
            issue_types: List of issue types to evaluate (None = all)

        Returns:
            Dictionary with complete Phase 3 results:
            {
                'metadata': {...},
                'issue_types': {
                    'RESOURCE_LEAK': {
                        'patterns': {...},
                        'metrics': {...},
                        'test_set_stats': {...},
                        'misclassified_examples': [...]
                    },
                    ...
                },
                'overall_summary': {...}
            }
        """
        logger.info("="*80)
        logger.info("PHASE 3: FINAL TEST EVALUATION")
        logger.info("="*80)

        start_time = datetime.now()

        # Determine issue types to process
        if issue_types is None:
            issue_types = list(final_patterns_by_issue_type.keys())

        logger.info(f"\nEvaluating {len(issue_types)} issue types: {issue_types}")

        # Get test files
        test_files = sorted(self.test_dir.glob("*.txt"))
        if not test_files:
            logger.error(f"No test files found in {self.test_dir}")
            return {}

        logger.info(f"Test set: {len(test_files)} files")

        # Process each issue type
        results_by_issue_type = {}

        for issue_type in issue_types:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating Issue Type: {issue_type}")
            logger.info(f"{'='*80}")

            final_patterns = final_patterns_by_issue_type.get(issue_type, {"fp": [], "tp": []})

            # Evaluate on test set
            test_results = self._evaluate_issue_type(
                issue_type=issue_type,
                patterns=final_patterns,
                test_files=test_files
            )

            results_by_issue_type[issue_type] = test_results

            # Log metrics
            metrics = test_results['metrics']
            logger.info(f"\n  Test Set Metrics:")
            logger.info(f"    F1: {metrics['f1']:.3f}")
            logger.info(f"    Precision: {metrics['precision']:.3f}")
            logger.info(f"    Recall: {metrics['recall']:.3f}")
            logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")

        # Generate overall summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        overall_summary = self._generate_overall_summary(
            results_by_issue_type,
            issue_types
        )
        overall_summary['duration_seconds'] = duration

        # Compile final results
        phase3_results = {
            'metadata': {
                'phase': 3,
                'test_dir': str(self.test_dir),
                'platform': self.platform,
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration
            },
            'issue_types': results_by_issue_type,
            'overall_summary': overall_summary
        }

        # Save complete Phase 3 results
        phase3_file = self.phase3_dir / "phase3_complete_results.json"
        with open(phase3_file, 'w') as f:
            json.dump(phase3_results, f, indent=2)

        # Generate human-readable report
        self._generate_report(phase3_results)

        logger.info(f"\n{'='*80}")
        logger.info("PHASE 3 COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Overall F1: {overall_summary['avg_f1']:.3f}")
        logger.info(f"Overall Precision: {overall_summary['avg_precision']:.3f}")
        logger.info(f"Overall Recall: {overall_summary['avg_recall']:.3f}")
        logger.info(f"Results saved to: {self.phase3_dir}")

        return phase3_results

    def _evaluate_issue_type(
        self,
        issue_type: str,
        patterns: Dict,
        test_files: List[Path]
    ) -> Dict:
        """
        Evaluate patterns for a single issue type on test set.

        Args:
            issue_type: Issue type to evaluate
            patterns: Patterns to evaluate
            test_files: Test files

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"  Evaluating on {len(test_files)} test files...")

        # Evaluate using FoldEvaluator
        eval_result = self.fold_evaluator.evaluate_fold(
            patterns_dict=patterns,
            val_files=test_files,
            issue_type=issue_type
        )

        # Parse test set statistics
        parser = ValidationEntryParser()
        all_entries = []
        for file in test_files:
            entries = parser.parse_file(file)
            entries = [e for e in entries if e.issue_type == issue_type]
            all_entries.extend(entries)

        fp_count = sum(1 for e in all_entries if 'FALSE' in e.ground_truth_classification)
        tp_count = len(all_entries) - fp_count

        # Get misclassified examples
        results = eval_result.get('results', [])
        misclassified_results = [r for r in results if not r.get('correct', True)]

        # Extract misclassified examples with details
        misclassified_examples = []
        for result in misclassified_results[:10]:  # Limit to 10 examples
            example = {
                'file_name': result.get('file_name'),
                'finding_id': result.get('finding_id'),
                'ground_truth': result.get('ground_truth'),
                'predicted': result.get('predicted_class'),
                'justification': result.get('justification', '')
            }
            misclassified_examples.append(example)

        # Compile results
        metrics = eval_result.get('metrics', {}).get('overall', {})

        test_results = {
            'patterns': patterns,
            'metrics': {
                'f1': metrics.get('f1', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'accuracy': metrics.get('accuracy', 0.0)
            },
            'test_set_stats': {
                'total_entries': len(all_entries),
                'fp_count': fp_count,
                'tp_count': tp_count,
                'test_files_count': len(test_files)
            },
            'confusion_matrix': eval_result.get('metrics', {}).get('confusion_matrix', {}),
            'misclassified_count': len(misclassified_results),
            'misclassified_examples': misclassified_examples,
            'full_evaluation': eval_result
        }

        # Save issue-type specific results
        issue_type_file = self.phase3_dir / f"{issue_type}_test_results.json"
        with open(issue_type_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        return test_results

    def _generate_overall_summary(
        self,
        results_by_issue_type: Dict,
        issue_types: List[str]
    ) -> Dict:
        """Generate overall summary across all issue types."""
        if not issue_types:
            return {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_accuracy = 0.0
        total_entries = 0
        total_misclassified = 0

        for issue_type in issue_types:
            results = results_by_issue_type[issue_type]
            metrics = results['metrics']

            total_f1 += metrics['f1']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_accuracy += metrics['accuracy']
            total_entries += results['test_set_stats']['total_entries']
            total_misclassified += results['misclassified_count']

        n = len(issue_types)

        return {
            'total_issue_types': n,
            'total_test_entries': total_entries,
            'total_misclassified': total_misclassified,
            'avg_f1': total_f1 / n,
            'avg_precision': total_precision / n,
            'avg_recall': total_recall / n,
            'avg_accuracy': total_accuracy / n,
            'per_issue_type_summary': {
                issue_type: {
                    'f1': results_by_issue_type[issue_type]['metrics']['f1'],
                    'entries': results_by_issue_type[issue_type]['test_set_stats']['total_entries'],
                    'misclassified': results_by_issue_type[issue_type]['misclassified_count']
                }
                for issue_type in issue_types
            }
        }

    def _generate_report(self, phase3_results: Dict):
        """Generate human-readable markdown report."""
        report_file = self.phase3_dir / "phase3_report.md"

        with open(report_file, 'w') as f:
            f.write("# Phase 3: Final Test Evaluation Report\n\n")

            # Metadata
            metadata = phase3_results['metadata']
            f.write("## Metadata\n\n")
            f.write(f"- **Test Directory**: {metadata['test_dir']}\n")
            f.write(f"- **Platform**: {metadata['platform']}\n")
            f.write(f"- **Timestamp**: {metadata['timestamp']}\n")
            f.write(f"- **Duration**: {metadata['duration_seconds']:.1f}s\n\n")

            # Overall Summary
            summary = phase3_results['overall_summary']
            f.write("## Overall Summary\n\n")
            f.write(f"- **Issue Types**: {summary['total_issue_types']}\n")
            f.write(f"- **Total Test Entries**: {summary['total_test_entries']}\n")
            f.write(f"- **Total Misclassified**: {summary['total_misclassified']}\n")
            f.write(f"- **Average F1**: {summary['avg_f1']:.3f}\n")
            f.write(f"- **Average Precision**: {summary['avg_precision']:.3f}\n")
            f.write(f"- **Average Recall**: {summary['avg_recall']:.3f}\n")
            f.write(f"- **Average Accuracy**: {summary['avg_accuracy']:.3f}\n\n")

            # Per-Issue-Type Results
            f.write("## Per-Issue-Type Results\n\n")
            f.write("| Issue Type | F1 | Precision | Recall | Accuracy | Test Entries | Misclassified |\n")
            f.write("|------------|-----|-----------|--------|----------|--------------|---------------|\n")

            for issue_type, results in phase3_results['issue_types'].items():
                metrics = results['metrics']
                stats = results['test_set_stats']
                misclassified = results['misclassified_count']

                f.write(f"| {issue_type} | {metrics['f1']:.3f} | {metrics['precision']:.3f} | "
                       f"{metrics['recall']:.3f} | {metrics['accuracy']:.3f} | "
                       f"{stats['total_entries']} | {misclassified} |\n")

            f.write("\n")

            # Detailed Per-Issue-Type Analysis
            f.write("## Detailed Analysis\n\n")

            for issue_type, results in phase3_results['issue_types'].items():
                f.write(f"### {issue_type}\n\n")

                metrics = results['metrics']
                stats = results['test_set_stats']

                f.write(f"**Test Set Statistics**:\n")
                f.write(f"- Total Entries: {stats['total_entries']}\n")
                f.write(f"- False Positives: {stats['fp_count']}\n")
                f.write(f"- True Positives: {stats['tp_count']}\n")
                f.write(f"- Test Files: {stats['test_files_count']}\n\n")

                f.write(f"**Metrics**:\n")
                f.write(f"- F1 Score: {metrics['f1']:.3f}\n")
                f.write(f"- Precision: {metrics['precision']:.3f}\n")
                f.write(f"- Recall: {metrics['recall']:.3f}\n")
                f.write(f"- Accuracy: {metrics['accuracy']:.3f}\n\n")

                # Pattern counts
                patterns = results['patterns']
                f.write(f"**Pattern Counts**:\n")
                f.write(f"- FP Patterns: {len(patterns.get('fp', []))}\n")
                f.write(f"- TP Patterns: {len(patterns.get('tp', []))}\n\n")

                # Misclassified examples
                if results['misclassified_examples']:
                    f.write(f"**Sample Misclassified Examples** (showing up to 10):\n\n")
                    for idx, example in enumerate(results['misclassified_examples'], 1):
                        f.write(f"{idx}. **{example['file_name']}** (Finding: {example['finding_id']})\n")
                        f.write(f"   - Ground Truth: {example['ground_truth']}\n")
                        f.write(f"   - Predicted: {example['predicted']}\n")
                        f.write(f"   - Justification: {example['justification'][:200]}...\n\n")

                f.write("\n")

        logger.info(f"  Generated report: {report_file}")


def main():
    """Test evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3: Final Test Evaluation")
    parser.add_argument("test_dir", type=Path, help="Test data directory")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("phase2_results", type=Path, help="Phase 2 results JSON file")
    parser.add_argument("--platform", "-p", choices=["local", "nim"],
                       default="nim", help="LLM platform")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--issue-types", nargs="+", help="Specific issue types to evaluate")

    args = parser.parse_args()

    # Load Phase 2 results
    with open(args.phase2_results, 'r') as f:
        phase2_results = json.load(f)

    # Extract final patterns by issue type
    final_patterns_by_issue_type = {}
    for issue_type, data in phase2_results.get('issue_types', {}).items():
        final_patterns_by_issue_type[issue_type] = data.get('final_patterns', {"fp": [], "tp": []})

    # Run Phase 3
    evaluator = TestEvaluator(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        platform=args.platform,
        workers=args.workers
    )

    results = evaluator.run_phase3(
        final_patterns_by_issue_type=final_patterns_by_issue_type,
        issue_types=args.issue_types
    )

    print("\n" + "="*80)
    print("PHASE 3 SUMMARY")
    print("="*80)
    print(f"Issue Types: {results['overall_summary']['total_issue_types']}")
    print(f"Total Test Entries: {results['overall_summary']['total_test_entries']}")
    print(f"Overall F1: {results['overall_summary']['avg_f1']:.3f}")
    print(f"Overall Precision: {results['overall_summary']['avg_precision']:.3f}")
    print(f"Overall Recall: {results['overall_summary']['avg_recall']:.3f}")


if __name__ == "__main__":
    main()