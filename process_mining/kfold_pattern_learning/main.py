#!/usr/bin/env python3
"""
Main CLI - K-Fold Pattern Learning Pipeline

Unified entry point for the complete 3-phase pattern learning pipeline:
- Phase 1: K-Fold Cross-Validation on training data
- Phase 2: Iterative Refinement on train+val combined
- Phase 3: Final Evaluation on test set
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.kfold_orchestrator import KFoldOrchestrator
from process_mining.kfold_pattern_learning.refinement_orchestrator import RefinementOrchestrator
from process_mining.kfold_pattern_learning.test_evaluator import TestEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PatternLearningPipeline:
    """
    Complete K-Fold Pattern Learning Pipeline.

    Orchestrates all 3 phases of pattern learning:
    1. K-Fold CV on training data
    2. Iterative refinement on train+val
    3. Final evaluation on test set
    """

    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        test_dir: Path,
        output_dir: Path,
        platform: str = "nim",
        n_folds: int = 5,
        max_refinement_iterations: int = 10,
        workers: int = 1,
        random_seed: int = 42,
        eval_sample_size: int = 500,
        issue_types: Optional[List[str]] = None
    ):
        """
        Initialize pattern learning pipeline.

        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            test_dir: Test data directory
            output_dir: Output directory for all results
            platform: LLM platform ("local" or "nim")
            n_folds: Number of folds for k-fold CV
            max_refinement_iterations: Max iterations for Phase 2
            workers: Number of parallel workers
            random_seed: Random seed for reproducibility
            eval_sample_size: Max entries to sample for Phase 2 train/val evaluation (default: 500)
            issue_types: List of issue types to process (None = auto-detect)
        """
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.platform = platform
        self.n_folds = n_folds
        self.max_refinement_iterations = max_refinement_iterations
        self.workers = workers
        self.random_seed = random_seed
        self.eval_sample_size = eval_sample_size
        self.issue_types = issue_types

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("K-FOLD PATTERN LEARNING PIPELINE")
        logger.info("="*80)
        logger.info(f"Train directory: {self.train_dir}")
        logger.info(f"Validation directory: {self.val_dir}")
        logger.info(f"Test directory: {self.test_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Platform: {self.platform}")
        logger.info(f"K-folds: {self.n_folds}")
        logger.info(f"Max refinement iterations: {self.max_refinement_iterations}")

    def run_full_pipeline(self) -> Dict:
        """
        Run complete 3-phase pipeline.

        Returns:
            Dictionary with results from all 3 phases
        """
        start_time = datetime.now()

        # Phase 1: K-Fold Cross-Validation
        logger.info("\n" + "="*80)
        logger.info("STARTING PHASE 1: K-FOLD CROSS-VALIDATION")
        logger.info("="*80)

        phase1_results = self.run_phase1()

        # Phase 1.5: Evaluate final patterns on validation set
        logger.info("\n" + "="*80)
        logger.info("STARTING PHASE 1.5: FINAL PATTERN EVALUATION")
        logger.info("="*80)

        phase1_5_results = self.run_phase1_5(phase1_results)

        # Phase 2: Iterative Refinement
        logger.info("\n" + "="*80)
        logger.info("STARTING PHASE 2: ITERATIVE REFINEMENT")
        logger.info("="*80)

        phase2_results = self.run_phase2(phase1_results)

        # Phase 3: Final Test Evaluation
        logger.info("\n" + "="*80)
        logger.info("STARTING PHASE 3: FINAL TEST EVALUATION")
        logger.info("="*80)

        phase3_results = self.run_phase3(phase2_results)

        # Compile complete results
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        complete_results = {
            'metadata': {
                'pipeline': 'k-fold-pattern-learning',
                'train_dir': str(self.train_dir),
                'val_dir': str(self.val_dir),
                'test_dir': str(self.test_dir),
                'output_dir': str(self.output_dir),
                'platform': self.platform,
                'n_folds': self.n_folds,
                'max_refinement_iterations': self.max_refinement_iterations,
                'random_seed': self.random_seed,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration
            },
            'phase1_results': phase1_results,
            'phase2_results': phase2_results,
            'phase3_results': phase3_results,
            'pipeline_summary': self._generate_pipeline_summary(
                phase1_results,
                phase2_results,
                phase3_results
            )
        }

        # Save complete results
        complete_file = self.output_dir / "complete_pipeline_results.json"
        with open(complete_file, 'w') as f:
            json.dump(complete_results, f, indent=2)

        # Generate final report
        self._generate_final_report(complete_results)

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        logger.info(f"Results saved to: {self.output_dir}")

        return complete_results

    def run_phase1(self) -> Dict:
        """Run Phase 1: K-Fold Cross-Validation with Iterative Refinement."""
        orchestrator = KFoldOrchestrator(
            train_dir=self.train_dir,
            output_dir=self.output_dir,
            n_folds=self.n_folds,
            platform=self.platform,
            random_seed=self.random_seed,
            workers=self.workers,
            issue_types=self.issue_types
        )

        return orchestrator.run_phase1()

    def run_phase1_5(self, phase1_results: Dict) -> Dict:
        """Run Phase 1.5: Evaluate final refined patterns on validation set."""
        # Extract final patterns from Phase 1
        final_patterns_by_issue_type = {}
        for issue_type, data in phase1_results.get('issue_types', {}).items():
            final_patterns_by_issue_type[issue_type] = data.get('final_patterns', {"fp": [], "tp": []})

        orchestrator = KFoldOrchestrator(
            train_dir=self.train_dir,
            output_dir=self.output_dir,
            n_folds=self.n_folds,
            platform=self.platform,
            random_seed=self.random_seed,
            workers=self.workers,
            issue_types=self.issue_types
        )

        return orchestrator.run_phase1_5(
            final_patterns_by_issue_type=final_patterns_by_issue_type,
            val_dir=self.val_dir
        )

    def run_phase2(self, phase1_results: Dict) -> Dict:
        """Run Phase 2: Further Iterative Refinement on train+val."""
        # Extract final patterns from Phase 1 (patterns_2)
        initial_patterns_by_issue_type = {}
        for issue_type, data in phase1_results.get('issue_types', {}).items():
            initial_patterns_by_issue_type[issue_type] = data.get('final_patterns', {"fp": [], "tp": []})

        orchestrator = RefinementOrchestrator(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            output_dir=self.output_dir,
            platform=self.platform,
            workers=self.workers,
            max_iterations=self.max_refinement_iterations,
            eval_sample_size=self.eval_sample_size
        )

        return orchestrator.run_phase2(
            initial_patterns_by_issue_type=initial_patterns_by_issue_type,
            issue_types=self.issue_types
        )

    def run_phase3(self, phase2_results: Dict) -> Dict:
        """Run Phase 3: Final Test Evaluation."""
        # Extract final patterns by issue type
        final_patterns_by_issue_type = {}
        for issue_type, data in phase2_results.get('issue_types', {}).items():
            final_patterns_by_issue_type[issue_type] = data.get('final_patterns', {"fp": [], "tp": []})

        evaluator = TestEvaluator(
            test_dir=self.test_dir,
            output_dir=self.output_dir,
            platform=self.platform,
            workers=self.workers
        )

        return evaluator.run_phase3(
            final_patterns_by_issue_type=final_patterns_by_issue_type,
            issue_types=self.issue_types
        )

    def _generate_pipeline_summary(
        self,
        phase1_results: Dict,
        phase2_results: Dict,
        phase3_results: Dict
    ) -> Dict:
        """Generate summary across all phases."""
        summary = {}

        # Phase 1 summary
        phase1_summary = phase1_results.get('overall_summary', {})
        summary['phase1'] = {
            'k_folds': phase1_summary.get('total_folds', 0),
            'avg_f1_across_folds': phase1_summary.get('avg_f1_across_issue_types', 0.0),
            'duration_seconds': phase1_summary.get('duration_seconds', 0.0)
        }

        # Phase 2 summary
        phase2_summary = phase2_results.get('overall_summary', {})
        summary['phase2'] = {
            'avg_iterations': phase2_summary.get('avg_iterations', 0.0),
            'avg_f1_improvement': phase2_summary.get('avg_f1_improvement', 0.0),
            'duration_seconds': phase2_summary.get('duration_seconds', 0.0)
        }

        # Phase 3 summary
        phase3_summary = phase3_results.get('overall_summary', {})
        summary['phase3'] = {
            'test_entries': phase3_summary.get('total_test_entries', 0),
            'final_f1': phase3_summary.get('avg_f1', 0.0),
            'final_precision': phase3_summary.get('avg_precision', 0.0),
            'final_recall': phase3_summary.get('avg_recall', 0.0),
            'duration_seconds': phase3_summary.get('duration_seconds', 0.0)
        }

        # Overall metrics
        summary['overall'] = {
            'total_issue_types': phase3_summary.get('total_issue_types', 0),
            'total_duration_seconds': (
                summary['phase1']['duration_seconds'] +
                summary['phase2']['duration_seconds'] +
                summary['phase3']['duration_seconds']
            ),
            'final_test_f1': summary['phase3']['final_f1']
        }

        return summary

    def _generate_final_report(self, complete_results: Dict):
        """Generate final markdown report."""
        report_file = self.output_dir / "final_report.md"

        with open(report_file, 'w') as f:
            f.write("# K-Fold Pattern Learning Pipeline - Final Report\n\n")

            # Metadata
            metadata = complete_results['metadata']
            f.write("## Pipeline Configuration\n\n")
            f.write(f"- **Train Directory**: {metadata['train_dir']}\n")
            f.write(f"- **Validation Directory**: {metadata['val_dir']}\n")
            f.write(f"- **Test Directory**: {metadata['test_dir']}\n")
            f.write(f"- **Platform**: {metadata['platform']}\n")
            f.write(f"- **K-Folds**: {metadata['n_folds']}\n")
            f.write(f"- **Max Refinement Iterations**: {metadata['max_refinement_iterations']}\n")
            f.write(f"- **Random Seed**: {metadata['random_seed']}\n")
            f.write(f"- **Start Time**: {metadata['start_time']}\n")
            f.write(f"- **Total Duration**: {metadata['total_duration_seconds']:.1f}s ({metadata['total_duration_seconds']/60:.1f} min)\n\n")

            # Pipeline Summary
            summary = complete_results['pipeline_summary']
            f.write("## Pipeline Summary\n\n")

            f.write("### Phase 1: K-Fold Cross-Validation\n")
            f.write(f"- K-Folds: {summary['phase1']['k_folds']}\n")
            f.write(f"- Average F1 Across Folds: {summary['phase1']['avg_f1_across_folds']:.3f}\n")
            f.write(f"- Duration: {summary['phase1']['duration_seconds']:.1f}s\n\n")

            f.write("### Phase 2: Iterative Refinement\n")
            f.write(f"- Average Iterations: {summary['phase2']['avg_iterations']:.1f}\n")
            f.write(f"- Average F1 Improvement: {summary['phase2']['avg_f1_improvement']:.3f}\n")
            f.write(f"- Duration: {summary['phase2']['duration_seconds']:.1f}s\n\n")

            f.write("### Phase 3: Final Test Evaluation\n")
            f.write(f"- Test Entries: {summary['phase3']['test_entries']}\n")
            f.write(f"- Final F1: {summary['phase3']['final_f1']:.3f}\n")
            f.write(f"- Final Precision: {summary['phase3']['final_precision']:.3f}\n")
            f.write(f"- Final Recall: {summary['phase3']['final_recall']:.3f}\n")
            f.write(f"- Duration: {summary['phase3']['duration_seconds']:.1f}s\n\n")

            # Per-Issue-Type Summary
            f.write("## Per-Issue-Type Summary\n\n")
            f.write("| Issue Type | Phase 1 F1 | Phase 2 Iterations | Phase 2 Improvement | Final Test F1 |\n")
            f.write("|------------|------------|--------------------|--------------------|---------------|\n")

            phase1_results = complete_results['phase1_results']
            phase2_results = complete_results['phase2_results']
            phase3_results = complete_results['phase3_results']

            for issue_type in phase3_results.get('issue_types', {}).keys():
                phase1_f1 = phase1_results['issue_types'][issue_type]['final_metrics'].get('f1', 0.0)
                phase2_iters = phase2_results['issue_types'][issue_type]['convergence_info']['iterations_completed']
                phase2_improvement = phase2_results['issue_types'][issue_type]['convergence_info']['f1_improvement']
                phase3_f1 = phase3_results['issue_types'][issue_type]['metrics']['f1']

                f.write(f"| {issue_type} | {phase1_f1:.3f} | {phase2_iters} | +{phase2_improvement:.3f} | {phase3_f1:.3f} |\n")

            f.write("\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            f.write(f"- Total Issue Types Processed: {summary['overall']['total_issue_types']}\n")
            f.write(f"- Final Average Test F1: **{summary['phase3']['final_f1']:.3f}**\n")
            f.write(f"- Total Pipeline Duration: {summary['overall']['total_duration_seconds']:.1f}s ({summary['overall']['total_duration_seconds']/60:.1f} min)\n\n")

            # Output Files
            f.write("## Output Files\n\n")
            f.write("### Phase 1 (K-Fold Cross-Validation)\n")
            f.write(f"- Results: `{self.output_dir}/phase1_kfold_results/`\n")
            f.write(f"  - Per-fold patterns and evaluations\n")
            f.write(f"  - Merged patterns per issue type\n")
            f.write(f"  - Complete Phase 1 results: `phase1_complete_results.json`\n\n")

            f.write("### Phase 2 (Iterative Refinement)\n")
            f.write(f"- Results: `{self.output_dir}/phase2_refinement_results/`\n")
            f.write(f"  - Per-iteration patterns and refinements\n")
            f.write(f"  - Final refined patterns per issue type\n")
            f.write(f"  - Complete Phase 2 results: `phase2_complete_results.json`\n\n")

            f.write("### Phase 3 (Final Test Evaluation)\n")
            f.write(f"- Results: `{self.output_dir}/phase3_test_results/`\n")
            f.write(f"  - Per-issue-type test results\n")
            f.write(f"  - Complete Phase 3 results: `phase3_complete_results.json`\n")
            f.write(f"  - Test evaluation report: `phase3_report.md`\n\n")

            f.write("### Complete Pipeline Results\n")
            f.write(f"- `{self.output_dir}/complete_pipeline_results.json`\n")
            f.write(f"- `{self.output_dir}/final_report.md` (this file)\n\n")

        logger.info(f"Generated final report: {report_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="K-Fold Pattern Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py \\
    --train-dir /path/to/train \\
    --val-dir /path/to/val \\
    --test-dir /path/to/test \\
    --output-dir /path/to/output

  # Run with specific issue types
  python main.py \\
    --train-dir /path/to/train \\
    --val-dir /path/to/val \\
    --test-dir /path/to/test \\
    --output-dir /path/to/output \\
    --issue-types RESOURCE_LEAK UNINIT

  # Run individual phases
  python main.py --phase 1 --train-dir /path/to/train --output-dir /path/to/output
  python main.py --phase 2 --train-dir /path/to/train --val-dir /path/to/val --output-dir /path/to/output
  python main.py --phase 3 --test-dir /path/to/test --output-dir /path/to/output
        """
    )

    parser.add_argument("--train-dir", type=Path, required=True,
                       help="Training data directory")
    parser.add_argument("--val-dir", type=Path,
                       help="Validation data directory (required for full pipeline or phase 2/3)")
    parser.add_argument("--test-dir", type=Path,
                       help="Test data directory (required for full pipeline or phase 3)")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for results")

    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="nim", help="LLM platform")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Number of folds for k-fold CV")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Max refinement iterations for Phase 2")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--eval-sample-size", type=int, default=500,
                       help="Max entries to sample for Phase 2 train/val evaluation (default: 500, set to 0 for no sampling)")
    parser.add_argument("--issue-types", nargs="+",
                       help="Specific issue types to process (default: auto-detect)")

    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                       help="Run specific phase only (default: run all phases)")

    args = parser.parse_args()

    # Validate arguments
    if args.phase is None:
        # Full pipeline requires all directories
        if not args.val_dir or not args.test_dir:
            parser.error("Full pipeline requires --val-dir and --test-dir")
    elif args.phase in [2, 3]:
        if not args.val_dir:
            parser.error("Phase 2 and 3 require --val-dir")
        if args.phase == 3 and not args.test_dir:
            parser.error("Phase 3 requires --test-dir")

    # Create pipeline
    pipeline = PatternLearningPipeline(
        train_dir=args.train_dir,
        val_dir=args.val_dir or args.train_dir,  # Fallback for phase 1 only
        test_dir=args.test_dir or args.train_dir,  # Fallback for phase 1/2 only
        output_dir=args.output_dir,
        platform=args.platform,
        n_folds=args.n_folds,
        max_refinement_iterations=args.max_iterations,
        workers=args.workers,
        random_seed=args.seed,
        eval_sample_size=args.eval_sample_size,
        issue_types=args.issue_types
    )

    # Run requested phase(s)
    if args.phase == 1:
        results = pipeline.run_phase1()
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE")
        print(f"Avg F1: {results['overall_summary']['avg_f1_across_issue_types']:.3f}")
    elif args.phase == 2:
        # Load phase 1 results
        phase1_file = args.output_dir / "phase1_kfold_results" / "phase1_complete_results.json"
        if not phase1_file.exists():
            parser.error(f"Phase 1 results not found: {phase1_file}. Run phase 1 first.")
        with open(phase1_file, 'r') as f:
            phase1_results = json.load(f)
        results = pipeline.run_phase2(phase1_results)
        print("\n" + "="*80)
        print("PHASE 2 COMPLETE")
        print(f"Avg F1 Improvement: {results['overall_summary']['avg_f1_improvement']:.3f}")
    elif args.phase == 3:
        # Load phase 2 results
        phase2_file = args.output_dir / "phase2_refinement_results" / "phase2_complete_results.json"
        if not phase2_file.exists():
            parser.error(f"Phase 2 results not found: {phase2_file}. Run phases 1 and 2 first.")
        with open(phase2_file, 'r') as f:
            phase2_results = json.load(f)
        results = pipeline.run_phase3(phase2_results)
        print("\n" + "="*80)
        print("PHASE 3 COMPLETE")
        print(f"Final F1: {results['overall_summary']['avg_f1']:.3f}")
    else:
        # Full pipeline
        results = pipeline.run_full_pipeline()
        print("\n" + "="*80)
        print("FULL PIPELINE COMPLETE")
        print(f"Final Test F1: {results['pipeline_summary']['phase3']['final_f1']:.3f}")


if __name__ == "__main__":
    main()