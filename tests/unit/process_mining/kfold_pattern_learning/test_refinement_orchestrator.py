"""
Unit tests for RefinementOrchestrator.

Tests Phase 2 iterative refinement including:
- Proper train/val separation
- Early stopping based on val F1
- Overfitting detection
- Pattern refinement application
- Convergence criteria
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.refinement_orchestrator import RefinementOrchestrator


class TestRefinementOrchestrator:
    """Test suite for RefinementOrchestrator."""

    def test_init(self, sample_train_val_test_dirs, temp_dir):
        """Test RefinementOrchestrator initialization."""
        orchestrator = RefinementOrchestrator(
            train_dir=sample_train_val_test_dirs['train'],
            val_dir=sample_train_val_test_dirs['val'],
            output_dir=temp_dir / "output",
            platform="nim",
            workers=2,
            max_iterations=5,
            eval_sample_size=100
        )

        assert orchestrator.train_dir == sample_train_val_test_dirs['train']
        assert orchestrator.val_dir == sample_train_val_test_dirs['val']
        assert orchestrator.platform == "nim"
        assert orchestrator.workers == 2
        assert orchestrator.max_iterations == 5
        assert orchestrator.eval_sample_size == 100
        assert orchestrator.phase2_dir.exists()

    def test_apply_refinements_add(self):
        """Test applying ADD refinements."""
        orchestrator = RefinementOrchestrator(
            train_dir=Path("/tmp/train"),
            val_dir=Path("/tmp/val"),
            output_dir=Path("/tmp/output")
        )

        current_patterns = {
            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'existing'}],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'existing'}]
        }

        refinements = {
            'add': [
                {
                    'pattern_type': 'fp',
                    'pattern_id': 'FP-002',
                    'group': 'B',
                    'summary': 'new FP pattern'
                },
                {
                    'pattern_type': 'tp',
                    'pattern_id': 'TP-002',
                    'group': 'B',
                    'summary': 'new TP pattern'
                }
            ],
            'modify': [],
            'remove': []
        }

        updated = orchestrator._apply_refinements(current_patterns, refinements)

        assert len(updated['fp']) == 2
        assert len(updated['tp']) == 2
        assert any(p['pattern_id'] == 'FP-002' for p in updated['fp'])
        assert any(p['pattern_id'] == 'TP-002' for p in updated['tp'])

    def test_apply_refinements_modify(self):
        """Test applying MODIFY refinements."""
        orchestrator = RefinementOrchestrator(
            train_dir=Path("/tmp/train"),
            val_dir=Path("/tmp/val"),
            output_dir=Path("/tmp/output")
        )

        current_patterns = {
            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'old summary'}],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'old summary'}]
        }

        refinements = {
            'add': [],
            'modify': [
                {
                    'pattern_id': 'FP-001',
                    'new_summary': 'updated FP summary'
                }
            ],
            'remove': []
        }

        updated = orchestrator._apply_refinements(current_patterns, refinements)

        fp_pattern = next(p for p in updated['fp'] if p['pattern_id'] == 'FP-001')
        assert fp_pattern['summary'] == 'updated FP summary'

    def test_apply_refinements_remove(self):
        """Test applying REMOVE refinements."""
        orchestrator = RefinementOrchestrator(
            train_dir=Path("/tmp/train"),
            val_dir=Path("/tmp/val"),
            output_dir=Path("/tmp/output")
        )

        current_patterns = {
            'fp': [
                {'pattern_id': 'FP-001', 'group': 'A', 'summary': 'keep'},
                {'pattern_id': 'FP-002', 'group': 'B', 'summary': 'remove'}
            ],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'keep'}]
        }

        refinements = {
            'add': [],
            'modify': [],
            'remove': [
                {'pattern_id': 'FP-002'}
            ]
        }

        updated = orchestrator._apply_refinements(current_patterns, refinements)

        assert len(updated['fp']) == 1
        assert updated['fp'][0]['pattern_id'] == 'FP-001'
        assert not any(p['pattern_id'] == 'FP-002' for p in updated['fp'])

    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.FoldEvaluator')
    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.PatternRefiner')
    def test_refine_issue_type_convergence(self, mock_refiner_class, mock_evaluator_class,
                                           sample_train_val_test_dirs, temp_dir):
        """Test refinement converges based on val F1."""
        # Setup mocks
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator

        # Simulate convergence: val F1 stops improving after iteration 2
        # Each iteration needs train eval with some misclassifications, then val eval
        mock_evaluator.evaluate_fold.side_effect = [
            # Iteration 1: Train eval (has misclassifications)
            {'metrics': {'overall': {'f1': 0.70, 'precision': 0.65, 'recall': 0.75}},
             'results': [
                 {'ground_truth': 'TRUE_POSITIVE', 'predicted_class': 'FALSE_POSITIVE', 'correct': False, 'entry_id': 'id1'},
                 {'ground_truth': 'FALSE_POSITIVE', 'predicted_class': 'TRUE_POSITIVE', 'correct': False, 'entry_id': 'id2'},
             ]},
            # Iteration 1: Val eval
            {'metrics': {'overall': {'f1': 0.65, 'precision': 0.60, 'recall': 0.70}}, 'results': []},
            # Iteration 2: Train eval (still has misclassifications)
            {'metrics': {'overall': {'f1': 0.80, 'precision': 0.75, 'recall': 0.85}},
             'results': [
                 {'ground_truth': 'TRUE_POSITIVE', 'predicted_class': 'FALSE_POSITIVE', 'correct': False, 'entry_id': 'id3'},
             ]},
            # Iteration 2: Val eval (improvement)
            {'metrics': {'overall': {'f1': 0.75, 'precision': 0.70, 'recall': 0.80}}, 'results': []},
            # Iteration 3: Train eval (still has misclassifications)
            {'metrics': {'overall': {'f1': 0.85, 'precision': 0.80, 'recall': 0.90}},
             'results': [
                 {'ground_truth': 'FALSE_POSITIVE', 'predicted_class': 'TRUE_POSITIVE', 'correct': False, 'entry_id': 'id4'},
             ]},
            # Iteration 3: Val eval (no improvement - below threshold)
            {'metrics': {'overall': {'f1': 0.755, 'precision': 0.71, 'recall': 0.80}}, 'results': []},
            # Iteration 4: Train eval (still has misclassifications)
            {'metrics': {'overall': {'f1': 0.90, 'precision': 0.85, 'recall': 0.95}},
             'results': [
                 {'ground_truth': 'TRUE_POSITIVE', 'predicted_class': 'FALSE_POSITIVE', 'correct': False, 'entry_id': 'id5'},
             ]},
            # Iteration 4: Val eval (still no improvement)
            {'metrics': {'overall': {'f1': 0.75, 'precision': 0.70, 'recall': 0.80}}, 'results': []},
        ]

        mock_refiner = Mock()
        mock_refiner.refine_patterns.return_value = {'add': [], 'modify': [], 'remove': []}
        mock_refiner_class.return_value = mock_refiner

        orchestrator = RefinementOrchestrator(
            train_dir=sample_train_val_test_dirs['train'],
            val_dir=sample_train_val_test_dirs['val'],
            output_dir=temp_dir / "output",
            max_iterations=10,
            convergence_threshold=0.01,
            convergence_patience=3,
            eval_sample_size=None  # No sampling for test
        )

        initial_patterns = {
            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
        }

        train_files = list(sample_train_val_test_dirs['train'].glob("*.txt"))
        val_files = list(sample_train_val_test_dirs['val'].glob("*.txt"))

        result = orchestrator._refine_issue_type(
            issue_type='RESOURCE_LEAK',
            initial_patterns=initial_patterns,
            train_files=train_files,
            val_files=val_files
        )

        # Should converge (either due to no improvement or perfect classification)
        assert result['convergence_info']['converged']
        # Convergence can happen due to various reasons
        assert 'convergence_reason' in result['convergence_info']
        assert len(result['convergence_info']['convergence_reason']) > 0

    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.FoldEvaluator')
    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.PatternRefiner')
    def test_overfitting_detection(self, mock_refiner_class, mock_evaluator_class,
                                   sample_train_val_test_dirs, temp_dir):
        """Test overfitting detection when train F1 >> val F1."""
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator

        # Simulate overfitting: train F1 much higher than val F1
        mock_evaluator.evaluate_fold.side_effect = [
            # Train eval: high F1
            {'metrics': {'overall': {'f1': 0.95, 'precision': 0.92, 'recall': 0.98}}, 'results': []},
            # Val eval: much lower F1 (gap > 0.1)
            {'metrics': {'overall': {'f1': 0.70, 'precision': 0.65, 'recall': 0.75}}, 'results': []},
        ]

        mock_refiner = Mock()
        mock_refiner.refine_patterns.return_value = {'add': [], 'modify': [], 'remove': []}
        mock_refiner_class.return_value = mock_refiner

        orchestrator = RefinementOrchestrator(
            train_dir=sample_train_val_test_dirs['train'],
            val_dir=sample_train_val_test_dirs['val'],
            output_dir=temp_dir / "output",
            max_iterations=1,  # Only 1 iteration for this test
            eval_sample_size=None
        )

        initial_patterns = {'fp': [], 'tp': []}
        train_files = list(sample_train_val_test_dirs['train'].glob("*.txt"))
        val_files = list(sample_train_val_test_dirs['val'].glob("*.txt"))

        result = orchestrator._refine_issue_type(
            issue_type='RESOURCE_LEAK',
            initial_patterns=initial_patterns,
            train_files=train_files,
            val_files=val_files
        )

        # Check overfitting gap is tracked
        assert 'overfitting_gap' in result['convergence_info']
        assert result['convergence_info']['overfitting_gap'] > 0.1

    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.FoldEvaluator')
    def test_eval_sample_size_passed_to_evaluator(self, mock_evaluator_class,
                                                   sample_train_val_test_dirs, temp_dir):
        """Test that eval_sample_size is passed to fold evaluator."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_fold.return_value = {
            'metrics': {'overall': {'f1': 0.80, 'precision': 0.75, 'recall': 0.85}},
            'results': []
        }
        mock_evaluator_class.return_value = mock_evaluator

        orchestrator = RefinementOrchestrator(
            train_dir=sample_train_val_test_dirs['train'],
            val_dir=sample_train_val_test_dirs['val'],
            output_dir=temp_dir / "output",
            max_iterations=1,
            eval_sample_size=500  # Set sample size
        )

        initial_patterns = {'fp': [], 'tp': []}
        train_files = list(sample_train_val_test_dirs['train'].glob("*.txt"))
        val_files = list(sample_train_val_test_dirs['val'].glob("*.txt"))

        with patch.object(orchestrator.pattern_refiner, 'refine_patterns', return_value={'add': [], 'modify': [], 'remove': []}):
            orchestrator._refine_issue_type(
                issue_type='RESOURCE_LEAK',
                initial_patterns=initial_patterns,
                train_files=train_files,
                val_files=val_files
            )

        # Verify evaluate_fold was called with max_entries=500
        calls = mock_evaluator.evaluate_fold.call_args_list
        for call_args in calls:
            assert call_args[1]['max_entries'] == 500

    def test_run_phase2_integration(self, sample_phase1_results, sample_train_val_test_dirs, temp_dir):
        """Test full Phase 2 run with mocked components."""
        with patch('process_mining.kfold_pattern_learning.refinement_orchestrator.FoldEvaluator') as mock_eval_class, \
             patch('process_mining.kfold_pattern_learning.refinement_orchestrator.PatternRefiner') as mock_refiner_class:

            # Setup mocks
            mock_evaluator = Mock()
            mock_evaluator.evaluate_fold.return_value = {
                'metrics': {'overall': {'f1': 0.80, 'precision': 0.75, 'recall': 0.85}},
                'results': []
            }
            mock_eval_class.return_value = mock_evaluator

            mock_refiner = Mock()
            mock_refiner.refine_patterns.return_value = {'add': [], 'modify': [], 'remove': []}
            mock_refiner_class.return_value = mock_refiner

            orchestrator = RefinementOrchestrator(
                train_dir=sample_train_val_test_dirs['train'],
                val_dir=sample_train_val_test_dirs['val'],
                output_dir=temp_dir / "output",
                max_iterations=2
            )

            initial_patterns_by_issue_type = sample_phase1_results['issue_types']['RESOURCE_LEAK']['merged_patterns']
            initial_patterns_by_issue_type = {'RESOURCE_LEAK': initial_patterns_by_issue_type}

            results = orchestrator.run_phase2(
                initial_patterns_by_issue_type=initial_patterns_by_issue_type,
                issue_types=['RESOURCE_LEAK']
            )

            # Verify results structure
            assert 'metadata' in results
            assert 'issue_types' in results
            assert 'overall_summary' in results
            assert 'RESOURCE_LEAK' in results['issue_types']

            # Verify output files created
            assert (temp_dir / "output" / "phase2_refinement_results" / "phase2_complete_results.json").exists()