"""
Unit tests for StratifiedKFoldSplitter.

Tests stratified k-fold splitting including:
- Strata assignment based on size and FP ratio
- Even distribution across folds
- Reproducibility with seed
- Edge cases (small datasets, extreme ratios)
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.stratified_kfold import StratifiedKFoldSplitter


class TestStratifiedKFoldSplitter:
    """Test suite for StratifiedKFoldSplitter."""

    def test_init(self):
        """Test splitter initialization."""
        splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)

        assert splitter.n_splits == 5
        assert splitter.random_state == 42
        assert splitter.shuffle == True

    def test_init_validates_n_splits(self):
        """Test that n_splits is validated."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            StratifiedKFoldSplitter(n_splits=1)

    def test_assign_size_category_small(self):
        """Test size category assignment for small files."""
        splitter = StratifiedKFoldSplitter(n_splits=5)

        assert splitter._assign_size_category(3) == 'small'
        assert splitter._assign_size_category(5) == 'small'

    def test_assign_size_category_medium(self):
        """Test size category assignment for medium files."""
        splitter = StratifiedKFoldSplitter(n_splits=5)

        assert splitter._assign_size_category(6) == 'medium'
        assert splitter._assign_size_category(15) == 'medium'
        assert splitter._assign_size_category(20) == 'medium'

    def test_assign_size_category_large(self):
        """Test size category assignment for large files."""
        splitter = StratifiedKFoldSplitter(n_splits=5)

        assert splitter._assign_size_category(21) == 'large'
        assert splitter._assign_size_category(100) == 'large'

    def test_assign_fp_bucket_low(self):
        """Test FP bucket assignment for low FP ratio."""
        splitter = StratifiedKFoldSplitter(n_splits=5)

        assert splitter._assign_fp_bucket(0.0) == 'low'
        assert splitter._assign_fp_bucket(0.2) == 'low'
        assert splitter._assign_fp_bucket(0.24) == 'low'

    def test_assign_fp_bucket_medium(self):
        """Test FP bucket assignment for medium FP ratio."""
        splitter = StratifiedKFoldSplitter(n_splits=5)

        assert splitter._assign_fp_bucket(0.25) == 'medium'
        assert splitter._assign_fp_bucket(0.5) == 'medium'
        assert splitter._assign_fp_bucket(0.74) == 'medium'

    def test_assign_fp_bucket_high(self):
        """Test FP bucket assignment for high FP ratio."""
        splitter = StratifiedKFoldSplitter(n_splits=5)

        assert splitter._assign_fp_bucket(0.75) == 'high'
        assert splitter._assign_fp_bucket(0.9) == 'high'
        assert splitter._assign_fp_bucket(1.0) == 'high'

    def test_split_creates_correct_number_of_folds(self, sample_train_val_test_dirs):
        """Test that split creates the correct number of folds."""
        splitter = StratifiedKFoldSplitter(n_splits=3, random_state=42)

        folds = splitter.split(sample_train_val_test_dirs['train'])

        assert len(folds) == 3

    def test_split_each_fold_has_train_and_val(self, sample_train_val_test_dirs):
        """Test that each fold has both train and val files."""
        splitter = StratifiedKFoldSplitter(n_splits=3, random_state=42)

        folds = splitter.split(sample_train_val_test_dirs['train'])

        for train_files, val_files in folds:
            assert len(train_files) > 0, "Each fold should have train files"
            assert len(val_files) > 0, "Each fold should have val files"

    def test_split_reproducible_with_same_seed(self, sample_train_val_test_dirs):
        """Test that splitting is reproducible with same seed."""
        splitter1 = StratifiedKFoldSplitter(n_splits=3, random_state=42)
        folds1 = splitter1.split(sample_train_val_test_dirs['train'])

        splitter2 = StratifiedKFoldSplitter(n_splits=3, random_state=42)
        folds2 = splitter2.split(sample_train_val_test_dirs['train'])

        # Should produce identical splits
        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            assert set(train1) == set(train2), "Train sets should be identical with same seed"
            assert set(val1) == set(val2), "Val sets should be identical with same seed"

    def test_split_different_with_different_seed(self, temp_dir, sample_validation_entry_text):
        """Test that different seeds produce different splits with enough files."""
        # Create more files to ensure different shuffling
        test_dir = temp_dir / "test_data"
        test_dir.mkdir()

        for i in range(12):  # Create 12 files for better shuffle detection
            (test_dir / f"file_{i}.txt").write_text(sample_validation_entry_text)

        splitter1 = StratifiedKFoldSplitter(n_splits=3, random_state=42)
        folds1 = splitter1.split(test_dir)

        splitter2 = StratifiedKFoldSplitter(n_splits=3, random_state=999)
        folds2 = splitter2.split(test_dir)

        # At least one fold should be different
        different = False
        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            if set(train1) != set(train2) or set(val1) != set(val2):
                different = True
                break

        assert different, "Different seeds should produce different splits"

    def test_split_no_file_reuse_in_val_sets(self, sample_train_val_test_dirs):
        """Test that no file appears in multiple val sets."""
        splitter = StratifiedKFoldSplitter(n_splits=3, random_state=42)
        folds = splitter.split(sample_train_val_test_dirs['train'])

        # Collect all val files across folds
        all_val_files = []
        for train_files, val_files in folds:
            all_val_files.extend(val_files)

        # No duplicates in val sets across folds
        assert len(all_val_files) == len(set(all_val_files)), \
            "Files should not appear in multiple val sets"

    def test_split_all_files_used(self, sample_train_val_test_dirs):
        """Test that all files are used across all folds."""
        data_dir = sample_train_val_test_dirs['train']
        all_files = set(data_dir.glob("*.txt"))

        splitter = StratifiedKFoldSplitter(n_splits=3, random_state=42)
        folds = splitter.split(data_dir)

        # Collect all files used
        all_used_files = set()
        for train_files, val_files in folds:
            all_used_files.update(train_files)
            all_used_files.update(val_files)

        assert all_used_files == all_files, "All files should be used across folds"

    def test_split_handles_empty_directory(self, temp_dir):
        """Test that split handles empty directory gracefully."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        splitter = StratifiedKFoldSplitter(n_splits=3, random_state=42)

        with pytest.raises(ValueError, match="No .txt files found"):
            splitter.split(empty_dir)

    def test_split_handles_nonexistent_directory(self):
        """Test that split handles nonexistent directory gracefully."""
        splitter = StratifiedKFoldSplitter(n_splits=3, random_state=42)

        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            splitter.split(Path("/nonexistent/directory"))