#!/usr/bin/env python3
"""
Run Phase 2 from Existing Step 2 Patterns

Loads step_2_patterns.json files from a completed Phase 1 run and
runs Phase 2 iterative refinement.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from process_mining.kfold_pattern_learning.refinement_orchestrator import RefinementOrchestrator
from process_mining.kfold_pattern_learning.config import PHASE2_EXCLUDE_ISSUE_TYPES, REFINEMENT_DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 from existing step_2 patterns")
    parser.add_argument("phase1_run_dir", type=Path,
                       help="Phase 1 run directory containing *_step_2_patterns.json files")
    parser.add_argument("--train-dir", type=Path, required=True,
                       help="Training data directory")
    parser.add_argument("--val-dir", type=Path, required=True,
                       help="Validation data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_run"),
                       help="Output directory for Phase 2 results")
    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="vertex", help="LLM platform")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--max-iterations", type=int,
                       default=REFINEMENT_DEFAULTS['max_iterations'],
                       help="Maximum refinement iterations")
    parser.add_argument("--eval-sample-size", type=int,
                       default=REFINEMENT_DEFAULTS['eval_sample_size'],
                       help="Max samples for train/val evaluation (0 for no sampling)")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("PHASE 2: Running from Existing Step 2 Patterns")
    logger.info("="*80)
    logger.info(f"Phase 1 run directory: {args.phase1_run_dir}")
    logger.info(f"Train directory: {args.train_dir}")
    logger.info(f"Val directory: {args.val_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Platform: {args.platform}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Eval sample size: {args.eval_sample_size}")
    logger.info("="*80 + "\n")

    # Find all step_2_patterns.json files
    step_2_patterns = list(args.phase1_run_dir.glob("*_step_2_patterns.json"))

    if not step_2_patterns:
        logger.error(f"No step_2_patterns.json files found in {args.phase1_run_dir}")
        logger.error("Expected files like: RESOURCE_LEAK_step_2_patterns.json")
        sys.exit(1)

    # Load patterns by issue type
    initial_patterns_by_issue_type = {}
    issue_types = []

    logger.info(f"Loading step_2 patterns from {len(step_2_patterns)} issue types:")
    for pattern_file in sorted(step_2_patterns):
        issue_type = pattern_file.stem.replace("_step_2_patterns", "")

        # Skip excluded issue types (from config)
        if issue_type in PHASE2_EXCLUDE_ISSUE_TYPES:
            logger.info(f"  ⊘ {issue_type}: SKIPPED (excluded in config)")
            continue

        with open(pattern_file, 'r') as f:
            patterns = json.load(f)

        fp_count = len(patterns.get('fp', []))
        tp_count = len(patterns.get('tp', []))

        initial_patterns_by_issue_type[issue_type] = patterns
        issue_types.append(issue_type)

        logger.info(f"  ✓ {issue_type}: {fp_count} FP + {tp_count} TP patterns")

    if PHASE2_EXCLUDE_ISSUE_TYPES:
        logger.info(f"\nTotal: {len(issue_types)} issue types loaded (excluded: {', '.join(PHASE2_EXCLUDE_ISSUE_TYPES)})\n")
    else:
        logger.info(f"\nTotal: {len(issue_types)} issue types loaded\n")

    # Create RefinementOrchestrator
    orchestrator = RefinementOrchestrator(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        platform=args.platform,
        workers=args.workers,
        max_iterations=args.max_iterations,
        convergence_threshold=REFINEMENT_DEFAULTS['convergence_threshold'],
        convergence_patience=REFINEMENT_DEFAULTS['convergence_patience'],
        max_misclassified_per_iteration=REFINEMENT_DEFAULTS['max_misclassified_per_iteration'],
        eval_sample_size=args.eval_sample_size if args.eval_sample_size > 0 else None
    )

    # Run Phase 2
    logger.info("Starting Phase 2 iterative refinement...\n")
    phase2_results = orchestrator.run_phase2(
        initial_patterns_by_issue_type=initial_patterns_by_issue_type,
        issue_types=issue_types
    )

    # Print summary
    overall = phase2_results.get('overall_summary', {})

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2 SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Issue Types Processed: {overall.get('total_issue_types', 0)}")
    logger.info(f"Average Iterations: {overall.get('avg_iterations', 0):.1f}")
    logger.info(f"Average F1 Improvement: {overall.get('avg_f1_improvement', 0):.3f}")
    logger.info(f"Duration: {overall.get('duration_seconds', 0):.1f}s")

    logger.info(f"\nPer-Issue-Type Results:")
    for issue_type in sorted(issue_types):
        data = phase2_results['issue_types'][issue_type]
        conv = data.get('convergence_info', {})
        logger.info(f"\n  {issue_type}:")
        logger.info(f"    Iterations: {conv.get('iterations_completed', 0)}")
        logger.info(f"    Initial F1: {conv.get('initial_f1', 0):.3f}")
        logger.info(f"    Final F1: {conv.get('final_f1', 0):.3f}")
        logger.info(f"    Improvement: {conv.get('f1_improvement', 0):.3f}")
        logger.info(f"    Converged: {conv.get('converged', False)}")
        logger.info(f"    Reason: {conv.get('convergence_reason', 'Unknown')}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to: {orchestrator.phase2_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()