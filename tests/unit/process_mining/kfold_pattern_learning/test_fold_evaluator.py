"""
Unit tests for FoldEvaluator.

Tests pattern evaluation on validation folds including:
- Basic evaluation with patterns
- Stratified sampling
- Entry file writing
- Metrics calculation
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.fold_evaluator import FoldEvaluator
from process_mining.core.data_models import ValidationEntry


class TestFoldEvaluator:
    """Test suite for FoldEvaluator."""

    def test_init(self):
        """Test FoldEvaluator initialization."""
        evaluator = FoldEvaluator(platform="nim", workers=2, verbose=True)

        assert evaluator.platform == "nim"
        assert evaluator.workers == 2
        assert evaluator.verbose is True

    def test_stratified_sample_preserves_ratio(self):
        """Test that stratified sampling preserves FP/TP ratio."""
        evaluator = FoldEvaluator()

        # Create sample entries with known FP/TP ratio
        entries = []
        for i in range(60):  # 60 FP
            entry = Mock()
            entry.ground_truth_classification = "FALSE_POSITIVE"
            entries.append(entry)

        for i in range(40):  # 40 TP
            entry = Mock()
            entry.ground_truth_classification = "TRUE_POSITIVE"
            entries.append(entry)

        # Original ratio: 60/100 = 0.6 FP
        sampled = evaluator._stratified_sample(entries, max_entries=50)

        # Check sampled size
        assert len(sampled) == 50

        # Check FP/TP ratio preserved (should be ~30 FP, ~20 TP)
        fp_count = sum(1 for e in sampled if 'FALSE' in e.ground_truth_classification)
        tp_count = len(sampled) - fp_count

        # Allow 10% tolerance
        assert 27 <= fp_count <= 33  # Target: 30
        assert 17 <= tp_count <= 23  # Target: 20

    def test_stratified_sample_no_sampling_when_below_max(self):
        """Test no sampling when entries <= max_entries."""
        evaluator = FoldEvaluator()

        entries = []
        for i in range(30):
            entry = Mock()
            entry.ground_truth_classification = "FALSE_POSITIVE"
            entries.append(entry)

        sampled = evaluator._stratified_sample(entries, max_entries=50)

        # Should return all entries
        assert len(sampled) == 30
        assert sampled == entries

    def test_stratified_sample_handles_edge_cases(self):
        """Test stratified sampling handles edge cases."""
        evaluator = FoldEvaluator()

        # Case 1: All FP
        all_fp = []
        for i in range(100):
            entry = Mock()
            entry.ground_truth_classification = "FALSE_POSITIVE"
            all_fp.append(entry)

        sampled_fp = evaluator._stratified_sample(all_fp, max_entries=50)
        assert len(sampled_fp) == 50
        assert all('FALSE' in e.ground_truth_classification for e in sampled_fp)

        # Case 2: All TP
        all_tp = []
        for i in range(100):
            entry = Mock()
            entry.ground_truth_classification = "TRUE_POSITIVE"
            all_tp.append(entry)

        sampled_tp = evaluator._stratified_sample(all_tp, max_entries=50)
        assert len(sampled_tp) == 50
        assert all('TRUE' in e.ground_truth_classification for e in sampled_tp)

    def test_write_entries_to_file(self, temp_dir):
        """Test writing ValidationEntry objects to file."""
        evaluator = FoldEvaluator()

        # Create sample entries
        entries = [
            ValidationEntry(
                entry_id="test1",
                package_name="pkg1",
                issue_type="RESOURCE_LEAK",
                cwe="CWE-401",
                error_trace="trace1",
                source_code="code1",
                ground_truth_classification="TRUE_POSITIVE",
                ground_truth_justification="just1"
            ),
            ValidationEntry(
                entry_id="test2",
                package_name="pkg2",
                issue_type="RESOURCE_LEAK",
                cwe="CWE-401",
                error_trace="trace2",
                source_code="code2",
                ground_truth_classification="FALSE_POSITIVE",
                ground_truth_justification="just2"
            )
        ]

        output_file = temp_dir / "test_entries.txt"
        evaluator._write_entries_to_file(entries, output_file)

        # Verify file was created
        assert output_file.exists()

        # Verify content
        content = output_file.read_text()
        assert "RESOURCE_LEAK" in content
        assert "CWE-401" in content
        assert "Ground Truth Classification: TRUE_POSITIVE" in content
        assert "Ground Truth Classification: FALSE_POSITIVE" in content
        assert "---" in content  # Separator between entries
        assert "Entry #1" in content
        assert "Entry #2" in content

    @patch('process_mining.kfold_pattern_learning.fold_evaluator.PatternEvaluator')
    def test_evaluate_fold_with_sampling(self, mock_evaluator_class, temp_dir, sample_validation_file):
        """Test evaluate_fold with sampling enabled."""
        # Setup mock
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.run.return_value = {
            'metrics': {
                'overall': {
                    'f1': 0.85,
                    'precision': 0.80,
                    'recall': 0.90
                }
            },
            'results': []
        }
        mock_evaluator_class.return_value = mock_evaluator_instance

        evaluator = FoldEvaluator(platform="nim")

        patterns = {
            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
        }

        result = evaluator.evaluate_fold(
            patterns_dict=patterns,
            val_files=[sample_validation_file],
            issue_type="RESOURCE_LEAK",
            max_entries=1  # Sample to 1 entry
        )

        # Verify result structure
        assert 'metrics' in result
        assert 'total_entries' in result
        assert 'fp_count' in result
        assert 'tp_count' in result

        # Verify PatternEvaluator was called
        assert mock_evaluator_instance.run.called

    @patch('process_mining.kfold_pattern_learning.fold_evaluator.PatternEvaluator')
    def test_evaluate_fold_without_sampling(self, mock_evaluator_class, temp_dir, sample_validation_file):
        """Test evaluate_fold without sampling (max_entries=None)."""
        # Setup mock
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.run.return_value = {
            'metrics': {
                'overall': {
                    'f1': 0.85,
                    'precision': 0.80,
                    'recall': 0.90
                }
            },
            'results': []
        }
        mock_evaluator_class.return_value = mock_evaluator_instance

        evaluator = FoldEvaluator(platform="nim")

        patterns = {
            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
        }

        result = evaluator.evaluate_fold(
            patterns_dict=patterns,
            val_files=[sample_validation_file],
            issue_type="RESOURCE_LEAK",
            max_entries=None  # No sampling
        )

        # Verify result structure
        assert 'metrics' in result
        assert result['total_entries'] == 2  # Both entries from sample file

    def test_evaluate_fold_multi_issue_types(self, sample_validation_file):
        """Test evaluating multiple issue types."""
        evaluator = FoldEvaluator(platform="nim")

        with patch.object(evaluator, 'evaluate_fold') as mock_evaluate:
            mock_evaluate.return_value = {
                'issue_type': 'RESOURCE_LEAK',
                'total_entries': 100,
                'metrics': {'overall': {'f1': 0.85}},
                'results': []
            }

            patterns_by_issue_type = {
                'RESOURCE_LEAK': {'fp': [], 'tp': []},
                'UNINIT': {'fp': [], 'tp': []}
            }

            results = evaluator.evaluate_fold_multi_issue_types(
                patterns_by_issue_type=patterns_by_issue_type,
                val_files=[sample_validation_file]
            )

            # Verify results for both issue types
            assert 'RESOURCE_LEAK' in results
            assert 'UNINIT' in results
            assert 'overall_metrics' in results
            assert mock_evaluate.call_count == 2  # Once per issue type