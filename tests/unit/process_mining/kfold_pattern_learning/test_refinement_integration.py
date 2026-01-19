"""
Integration tests for RefinementOrchestrator.

Tests end-to-end Phase 2 refinement including:
- Misclassified entry matching by entry_id
- Real file parsing and evaluation flow
- Convergence behavior with actual data
- Entry_id and predicted_classification requirements
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.refinement_orchestrator import RefinementOrchestrator
from process_mining.core.data_models import ValidationEntry


class TestRefinementIntegration:
    """Integration test suite for Phase 2 refinement."""

    def test_get_misclassified_entries_matches_by_entry_id(self, temp_dir, sample_validation_entry_text):
        """Test that misclassified entries are matched by entry_id correctly."""
        # Create test files
        train_dir = temp_dir / "train"
        val_dir = temp_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        (train_dir / "package1.txt").write_text(sample_validation_entry_text)

        orchestrator = RefinementOrchestrator(
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=temp_dir / "output",
            max_iterations=1,
            eval_sample_size=None
        )

        # Mock evaluation result with specific entry_ids that match our sample data
        eval_result = {
            'metrics': {'overall': {'f1': 0.5}},
            'results': [
                {
                    'entry_id': 'test-package_entry_1',  # Matches first entry in sample
                    'ground_truth': 'TRUE_POSITIVE',
                    'predicted_class': 'FALSE_POSITIVE',
                    'correct': False
                },
                {
                    'entry_id': 'test-package_entry_2',  # Matches second entry
                    'ground_truth': 'FALSE_POSITIVE',
                    'predicted_class': 'TRUE_POSITIVE',
                    'correct': False
                },
                {
                    'entry_id': 'nonexistent_entry',  # Should not match anything
                    'ground_truth': 'TRUE_POSITIVE',
                    'predicted_class': 'FALSE_POSITIVE',
                    'correct': False
                }
            ]
        }

        combined_files = list(train_dir.glob("*.txt"))

        misclassified = orchestrator._get_misclassified_entries(
            eval_result,
            combined_files,
            'RESOURCE_LEAK'
        )

        # Should find 2 entries (not 3, because one has non-matching entry_id)
        assert len(misclassified) == 2
        # Verify they are actual ValidationEntry objects
        assert all(isinstance(e, ValidationEntry) for e in misclassified)
        # Verify entry_ids match
        entry_ids = [e.entry_id for e in misclassified]
        assert 'test-package_entry_1' in entry_ids
        assert 'test-package_entry_2' in entry_ids

    def test_get_misclassified_entries_requires_entry_id(self, temp_dir, sample_validation_entry_text):
        """Test that matching fails gracefully if entry_id is missing."""
        train_dir = temp_dir / "train"
        val_dir = temp_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        (train_dir / "package1.txt").write_text(sample_validation_entry_text)

        orchestrator = RefinementOrchestrator(
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=temp_dir / "output",
            max_iterations=1,
            eval_sample_size=None
        )

        # Evaluation result missing entry_id
        eval_result = {
            'metrics': {'overall': {'f1': 0.5}},
            'results': [
                {
                    # Missing 'entry_id'
                    'ground_truth': 'TRUE_POSITIVE',
                    'predicted_class': 'FALSE_POSITIVE',
                    'correct': False
                }
            ]
        }

        combined_files = list(train_dir.glob("*.txt"))

        misclassified = orchestrator._get_misclassified_entries(
            eval_result,
            combined_files,
            'RESOURCE_LEAK'
        )

        # Should return empty list (no match possible without entry_id)
        assert len(misclassified) == 0

    def test_get_misclassified_entries_filters_by_correct_flag(self, temp_dir, sample_validation_entry_text):
        """Test that only incorrect predictions are returned."""
        train_dir = temp_dir / "train"
        val_dir = temp_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        (train_dir / "package1.txt").write_text(sample_validation_entry_text)

        orchestrator = RefinementOrchestrator(
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=temp_dir / "output",
            max_iterations=1,
            eval_sample_size=None
        )

        eval_result = {
            'metrics': {'overall': {'f1': 0.5}},
            'results': [
                {
                    'entry_id': 'test-package_entry_1',
                    'ground_truth': 'TRUE_POSITIVE',
                    'predicted_class': 'TRUE_POSITIVE',
                    'correct': True  # Correct prediction - should NOT be returned
                },
                {
                    'entry_id': 'test-package_entry_2',
                    'ground_truth': 'FALSE_POSITIVE',
                    'predicted_class': 'TRUE_POSITIVE',
                    'correct': False  # Incorrect - SHOULD be returned
                }
            ]
        }

        combined_files = list(train_dir.glob("*.txt"))

        misclassified = orchestrator._get_misclassified_entries(
            eval_result,
            combined_files,
            'RESOURCE_LEAK'
        )

        # Should only return the misclassified entry
        assert len(misclassified) == 1
        assert misclassified[0].entry_id == 'test-package_entry_2'

    def test_get_misclassified_entries_filters_by_issue_type(self, temp_dir):
        """Test that entries are filtered by issue type."""
        train_dir = temp_dir / "train"
        val_dir = temp_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # Create file with mixed issue types
        mixed_content = """================================================================================
GROUND-TRUTH ENTRIES FOR: test-pkg
================================================================================

Package: test-pkg
Total Entries: 2

---
Entry #1:
Issue Type: RESOURCE_LEAK
CWE: CWE-401

Error Trace:
test.c:100:2: leak

Source Code (test.c):
```c
int test() { return 0; }
```

Ground Truth Classification: TRUE_POSITIVE
Human Expert Justification: test

---
Entry #2:
Issue Type: NULL_DEREFERENCE
CWE: CWE-476

Error Trace:
test.c:200:2: null deref

Source Code (test.c):
```c
void test2() { int *p = NULL; *p = 1; }
```

Ground Truth Classification: TRUE_POSITIVE
Human Expert Justification: test
"""
        (train_dir / "mixed.txt").write_text(mixed_content)

        orchestrator = RefinementOrchestrator(
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=temp_dir / "output",
            max_iterations=1,
            eval_sample_size=None
        )

        eval_result = {
            'metrics': {'overall': {'f1': 0.5}},
            'results': [
                {
                    'entry_id': 'test-pkg_entry_1',
                    'ground_truth': 'TRUE_POSITIVE',
                    'predicted_class': 'FALSE_POSITIVE',
                    'correct': False
                },
                {
                    'entry_id': 'test-pkg_entry_2',
                    'ground_truth': 'TRUE_POSITIVE',
                    'predicted_class': 'FALSE_POSITIVE',
                    'correct': False
                }
            ]
        }

        combined_files = list(train_dir.glob("*.txt"))

        # Get only RESOURCE_LEAK misclassified
        misclassified = orchestrator._get_misclassified_entries(
            eval_result,
            combined_files,
            'RESOURCE_LEAK'
        )

        # Should only return the RESOURCE_LEAK entry, not NULL_DEREFERENCE
        assert len(misclassified) == 1
        assert misclassified[0].issue_type == 'RESOURCE_LEAK'
        assert misclassified[0].entry_id == 'test-pkg_entry_1'

    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.FoldEvaluator')
    @patch('process_mining.kfold_pattern_learning.refinement_orchestrator.PatternRefiner')
    def test_refine_issue_type_end_to_end_convergence(
        self,
        mock_refiner_class,
        mock_evaluator_class,
        temp_dir,
        sample_validation_entry_text
    ):
        """Test end-to-end refinement with real convergence behavior."""
        # Setup test data
        train_dir = temp_dir / "train"
        val_dir = temp_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        (train_dir / "pkg1.txt").write_text(sample_validation_entry_text)
        (val_dir / "pkg2.txt").write_text(sample_validation_entry_text)

        # Mock evaluator to simulate improving F1 then plateauing
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator

        mock_evaluator.evaluate_fold.side_effect = [
            # Iteration 1: Train
            {
                'metrics': {'overall': {'f1': 0.60}},
                'results': [
                    {'entry_id': 'test-package_entry_1', 'ground_truth': 'TRUE_POSITIVE',
                     'predicted_class': 'FALSE_POSITIVE', 'correct': False}
                ]
            },
            # Iteration 1: Val
            {'metrics': {'overall': {'f1': 0.55}}, 'results': []},
            # Iteration 2: Train (improved)
            {
                'metrics': {'overall': {'f1': 0.75}},
                'results': [
                    {'entry_id': 'test-package_entry_2', 'ground_truth': 'FALSE_POSITIVE',
                     'predicted_class': 'TRUE_POSITIVE', 'correct': False}
                ]
            },
            # Iteration 2: Val (improved)
            {'metrics': {'overall': {'f1': 0.70}}, 'results': []},
            # Iteration 3: Train
            {
                'metrics': {'overall': {'f1': 0.80}},
                'results': [
                    {'entry_id': 'test-package_entry_1', 'ground_truth': 'TRUE_POSITIVE',
                     'predicted_class': 'FALSE_POSITIVE', 'correct': False}
                ]
            },
            # Iteration 3: Val (no improvement - triggers convergence patience)
            {'metrics': {'overall': {'f1': 0.705}}, 'results': []},
            # Iteration 4: Train
            {
                'metrics': {'overall': {'f1': 0.82}},
                'results': []
            },
            # Iteration 4: Val (still no improvement - convergence reached)
            {'metrics': {'overall': {'f1': 0.70}}, 'results': []},
        ]

        mock_refiner = Mock()
        mock_refiner.refine_patterns.return_value = {'add': [], 'modify': [], 'remove': []}
        mock_refiner_class.return_value = mock_refiner

        orchestrator = RefinementOrchestrator(
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=temp_dir / "output",
            max_iterations=5,
            convergence_threshold=0.01,
            convergence_patience=2,
            eval_sample_size=None
        )

        initial_patterns = {
            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
        }

        result = orchestrator._refine_issue_type(
            issue_type='RESOURCE_LEAK',
            initial_patterns=initial_patterns,
            train_files=list(train_dir.glob("*.txt")),
            val_files=list(val_dir.glob("*.txt"))
        )

        # Verify convergence
        assert result['convergence_info']['converged']
        assert result['convergence_info']['iterations_completed'] >= 2

        # Verify iterations were recorded
        assert len(result['iterations']) >= 2

        # Verify val F1 was tracked
        assert 'final_patterns' in result
        assert result['final_patterns'] is not None

    def test_apply_refinements_all_actions(self):
        """Test that all refinement actions (add/modify/remove) work correctly."""
        orchestrator = RefinementOrchestrator(
            train_dir=Path("/tmp"),
            val_dir=Path("/tmp"),
            output_dir=Path("/tmp"),
            max_iterations=1,
            eval_sample_size=None
        )

        current_patterns = {
            'fp': [
                {'pattern_id': 'FP-001', 'group': 'A', 'summary': 'old summary 1'},
                {'pattern_id': 'FP-002', 'group': 'B', 'summary': 'will be removed'}
            ],
            'tp': [
                {'pattern_id': 'TP-001', 'group': 'A', 'summary': 'old summary 2'}
            ]
        }

        refinements = {
            'add': [
                {
                    'pattern_type': 'fp',
                    'pattern_id': 'FP-003',
                    'group': 'C',
                    'summary': 'new FP pattern'
                },
                {
                    'pattern_type': 'tp',
                    'pattern_id': 'TP-002',
                    'group': 'B',
                    'summary': 'new TP pattern'
                }
            ],
            'modify': [
                {
                    'pattern_id': 'FP-001',
                    'new_summary': 'updated FP summary'
                },
                {
                    'pattern_id': 'TP-001',
                    'new_summary': 'updated TP summary'
                }
            ],
            'remove': [
                {'pattern_id': 'FP-002'}
            ]
        }

        result = orchestrator._apply_refinements(current_patterns, refinements)

        # Check adds
        assert len(result['fp']) == 2  # FP-001 (modified), FP-003 (added) - FP-002 removed
        assert len(result['tp']) == 2  # TP-001 (modified), TP-002 (added)

        # Check modifications
        fp_001 = next((p for p in result['fp'] if p['pattern_id'] == 'FP-001'), None)
        assert fp_001 is not None
        assert fp_001['summary'] == 'updated FP summary'

        tp_001 = next((p for p in result['tp'] if p['pattern_id'] == 'TP-001'), None)
        assert tp_001 is not None
        assert tp_001['summary'] == 'updated TP summary'

        # Check removal
        fp_002 = next((p for p in result['fp'] if p['pattern_id'] == 'FP-002'), None)
        assert fp_002 is None  # Should be removed

        # Check additions
        fp_003 = next((p for p in result['fp'] if p['pattern_id'] == 'FP-003'), None)
        assert fp_003 is not None
        assert fp_003['summary'] == 'new FP pattern'

        tp_002 = next((p for p in result['tp'] if p['pattern_id'] == 'TP-002'), None)
        assert tp_002 is not None
        assert tp_002['summary'] == 'new TP pattern'