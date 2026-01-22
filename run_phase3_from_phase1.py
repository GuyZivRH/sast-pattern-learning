#!/usr/bin/env python3
"""
Run Phase 3 Evaluation on Phase 1 Step 2 Patterns

This script evaluates existing Phase 1 step_2 patterns on the held-out test set
using NIM platform (or any other platform) for parallel execution.

Usage:
    python run_phase3_from_phase1.py \\
        --phase1-dir /path/to/run_20260120_200345 \\
        --test-dir data/test \\
        --output-dir outputs/phase3_results \\
        --platform nim \\
        --workers 10
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from process_mining.kfold_pattern_learning.test_evaluator import TestEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable verbose HTTP logging (too much detail)
# If you want to see HTTP requests, set these to DEBUG:
# logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger("openai").setLevel(logging.DEBUG)


def load_phase1_step2_patterns(phase1_dir: Path) -> Dict[str, Dict]:
    """
    Load all step_2 pattern files from Phase 1 output directory.

    Args:
        phase1_dir: Directory containing Phase 1 results (e.g., run_20260120_200345)

    Returns:
        Dictionary mapping issue_type -> {'fp': [...], 'tp': [...]}
    """
    logger.info(f"Loading Phase 1 step_2 patterns from: {phase1_dir}")

    # Find all step_2_patterns.json files
    pattern_files = sorted(phase1_dir.glob("*_step_2_patterns.json"))

    if not pattern_files:
        logger.error(f"No *_step_2_patterns.json files found in {phase1_dir}")
        return {}

    logger.info(f"Found {len(pattern_files)} pattern files")

    patterns_by_issue_type = {}

    for pattern_file in pattern_files:
        # Extract issue type from filename (e.g., "COMPILER_WARNING_step_2_patterns.json")
        issue_type = pattern_file.stem.replace("_step_2_patterns", "")

        logger.info(f"  Loading {issue_type}: {pattern_file.name}")

        with open(pattern_file, 'r') as f:
            patterns = json.load(f)

        # Ensure patterns have fp and tp lists
        if not isinstance(patterns, dict):
            logger.warning(f"  Unexpected pattern format for {issue_type}, skipping")
            continue

        # Validate structure
        fp_patterns = patterns.get('fp', [])
        tp_patterns = patterns.get('tp', [])

        logger.info(f"    FP patterns: {len(fp_patterns)}, TP patterns: {len(tp_patterns)}")

        patterns_by_issue_type[issue_type] = {
            'fp': fp_patterns,
            'tp': tp_patterns
        }

    logger.info(f"\nLoaded patterns for {len(patterns_by_issue_type)} issue types")
    return patterns_by_issue_type


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 evaluation on Phase 1 step_2 patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 3 with NIM platform (parallel execution)
  python run_phase3_from_phase1.py \\
      --phase1-dir outputs/test_run/run_20260120_200345 \\
      --test-dir data/test \\
      --output-dir outputs/phase3_results \\
      --platform nim \\
      --workers 10

  # Run Phase 3 with local platform
  python run_phase3_from_phase1.py \\
      --phase1-dir outputs/test_run/run_20260120_200345 \\
      --test-dir data/test \\
      --output-dir outputs/phase3_results \\
      --platform local \\
      --workers 4

  # Run Phase 3 for specific issue types only
  python run_phase3_from_phase1.py \\
      --phase1-dir outputs/test_run/run_20260120_200345 \\
      --test-dir data/test \\
      --output-dir outputs/phase3_results \\
      --platform nim \\
      --workers 10 \\
      --issue-types COMPILER_WARNING COPY_PASTE_ERROR
        """
    )

    parser.add_argument(
        "--phase1-dir",
        type=Path,
        required=True,
        help="Phase 1 output directory containing *_step_2_patterns.json files"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Test data directory (e.g., data/test)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Phase 3 results"
    )
    parser.add_argument(
        "--platform",
        choices=["local", "nim", "vertex"],
        default="nim",
        help="LLM platform to use (default: nim)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--issue-types",
        nargs="+",
        help="Specific issue types to evaluate (default: all found in phase1-dir)"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run in baseline mode (no patterns) for comparison"
    )

    args = parser.parse_args()

    # Validate directories
    if not args.phase1_dir.exists():
        logger.error(f"Phase 1 directory not found: {args.phase1_dir}")
        sys.exit(1)

    if not args.test_dir.exists():
        logger.error(f"Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    if args.baseline:
        logger.info("PHASE 3 BASELINE EVALUATION (NO PATTERNS)")
    else:
        logger.info("PHASE 3 EVALUATION - FROM PHASE 1 STEP 2 PATTERNS")
    logger.info("="*80)
    logger.info(f"Phase 1 directory: {args.phase1_dir}")
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Platform: {args.platform}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Baseline mode: {args.baseline}")
    logger.info("="*80)

    # Load Phase 1 step_2 patterns (or create empty patterns for baseline)
    if args.baseline:
        logger.info("\nRunning in BASELINE mode - no patterns will be used")
        # Load pattern files to get issue types, but use empty patterns
        patterns_by_issue_type = load_phase1_step2_patterns(args.phase1_dir)
        if not patterns_by_issue_type:
            logger.error("No pattern files found to determine issue types. Exiting.")
            sys.exit(1)
        # Replace all patterns with empty lists for baseline
        logger.info("Replacing all patterns with empty lists for baseline comparison")
        for issue_type in patterns_by_issue_type:
            patterns_by_issue_type[issue_type] = {'fp': [], 'tp': []}
    else:
        patterns_by_issue_type = load_phase1_step2_patterns(args.phase1_dir)
        if not patterns_by_issue_type:
            logger.error("No patterns loaded. Exiting.")
            sys.exit(1)

    # Filter to specific issue types if requested
    if args.issue_types:
        logger.info(f"\nFiltering to requested issue types: {args.issue_types}")
        patterns_by_issue_type = {
            it: patterns_by_issue_type[it]
            for it in args.issue_types
            if it in patterns_by_issue_type
        }

        if not patterns_by_issue_type:
            logger.error("None of the requested issue types were found in patterns. Exiting.")
            sys.exit(1)

    # Initialize TestEvaluator
    evaluator = TestEvaluator(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        platform=args.platform,
        workers=args.workers
    )

    # Run Phase 3 evaluation
    results = evaluator.run_phase3(
        final_patterns_by_issue_type=patterns_by_issue_type,
        issue_types=list(patterns_by_issue_type.keys())
    )

    # Print summary
    print("\n" + "="*80)
    print("PHASE 3 SUMMARY")
    print("="*80)
    print(f"Issue Types: {results['overall_summary']['total_issue_types']}")
    print(f"Total Test Entries: {results['overall_summary']['total_test_entries']}")
    print(f"Total Misclassified: {results['overall_summary']['total_misclassified']}")
    print(f"Overall F1: {results['overall_summary']['avg_f1']:.3f}")
    print(f"Overall Precision: {results['overall_summary']['avg_precision']:.3f}")
    print(f"Overall Recall: {results['overall_summary']['avg_recall']:.3f}")
    print(f"Overall Accuracy: {results['overall_summary']['avg_accuracy']:.3f}")
    print("\nPer-Issue-Type Results:")
    print("-"*80)

    for issue_type in sorted(results['issue_types'].keys()):
        metrics = results['issue_types'][issue_type]['metrics']
        stats = results['issue_types'][issue_type]['test_set_stats']
        misclass = results['issue_types'][issue_type]['misclassified_count']

        print(f"{issue_type:25s}: F1={metrics['f1']:.3f}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"Acc={metrics['accuracy']:.3f} "
              f"({stats['total_entries']} entries, {misclass} misclassified)")

    print("="*80)
    print(f"Results saved to: {args.output_dir}/phase3_test_results/")
    print("="*80)


if __name__ == "__main__":
    main()