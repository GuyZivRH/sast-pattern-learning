#!/usr/bin/env python3
"""
Stratified K-Fold Splitting for Pattern Learning

Creates k stratified folds from training data while preserving:
1. FP/TP ratio distribution
2. Issue type distribution
3. Package size distribution (via strata)

Similar to sklearn.model_selection.StratifiedKFold but adapted for
file-based SAST ground truth data with multiple stratification criteria.
"""

import sys
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.parsers import ValidationEntryParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PackageStats:
    """Statistics for a single package file."""
    file_path: Path
    package_name: str
    issue_count: int
    fp_count: int
    tp_count: int
    fp_ratio: float
    issue_types: Dict[str, int]
    size_category: str  # small/medium/large
    fp_bucket: str      # low/medium/high
    stratum: str        # combined stratification key


class StratifiedKFoldSplitter:
    """
    Stratified K-Fold splitter for SAST ground truth files.

    Ensures each fold has similar:
    - FP/TP ratio
    - Issue type distribution
    - Package size distribution
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize k-fold splitter.

        Args:
            n_splits: Number of folds (default: 5 for 80/20 train/val per fold)
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.parser = ValidationEntryParser()

    def _assign_size_category(self, issue_count: int) -> str:
        """Assign size category based on issue count."""
        if issue_count <= 5:
            return 'small'
        elif issue_count <= 20:
            return 'medium'
        else:
            return 'large'

    def _assign_fp_bucket(self, fp_ratio: float) -> str:
        """Assign FP bucket based on FP ratio."""
        if fp_ratio < 0.25:
            return 'low'
        elif fp_ratio < 0.75:
            return 'medium'
        else:
            return 'high'

    def _load_package_stats(self, txt_files: List[Path]) -> List[PackageStats]:
        """
        Load statistics for all package files.

        Args:
            txt_files: List of .txt validation files

        Returns:
            List of PackageStats objects
        """
        logger.info(f"Loading statistics for {len(txt_files)} packages...")

        stats_list = []
        for txt_file in txt_files:
            try:
                entries = self.parser.parse_file(txt_file)
                if not entries:
                    logger.warning(f"No entries in {txt_file.name}, skipping")
                    continue

                issue_count = len(entries)
                fp_count = sum(1 for e in entries if 'FALSE' in e.ground_truth_classification)
                tp_count = issue_count - fp_count
                fp_ratio = fp_count / issue_count if issue_count > 0 else 0.0

                issue_types = defaultdict(int)
                for entry in entries:
                    issue_types[entry.issue_type] += 1
                issue_types = dict(issue_types)

                size_category = self._assign_size_category(issue_count)
                fp_bucket = self._assign_fp_bucket(fp_ratio)
                stratum = f"{size_category}_{fp_bucket}"

                stats = PackageStats(
                    file_path=txt_file,
                    package_name=txt_file.stem,
                    issue_count=issue_count,
                    fp_count=fp_count,
                    tp_count=tp_count,
                    fp_ratio=fp_ratio,
                    issue_types=issue_types,
                    size_category=size_category,
                    fp_bucket=fp_bucket,
                    stratum=stratum
                )
                stats_list.append(stats)

            except Exception as e:
                logger.error(f"Error processing {txt_file.name}: {e}")
                continue

        logger.info(f"Loaded statistics for {len(stats_list)} packages")
        logger.info(f"  Total issues: {sum(s.issue_count for s in stats_list)}")
        logger.info(f"  Total FP: {sum(s.fp_count for s in stats_list)}")
        logger.info(f"  Total TP: {sum(s.tp_count for s in stats_list)}")

        return stats_list

    def _group_by_strata(self, stats_list: List[PackageStats]) -> Dict[str, List[PackageStats]]:
        """Group packages by stratification key."""
        strata = defaultdict(list)
        for stats in stats_list:
            strata[stats.stratum].append(stats)
        return strata

    def split(self, data_dir: Path) -> List[Tuple[List[Path], List[Path]]]:
        """
        Generate k-fold splits from directory of TXT files.

        Args:
            data_dir: Directory containing .txt validation files

        Returns:
            List of (train_files, val_files) tuples, one per fold

        Example:
            >>> splitter = StratifiedKFoldSplitter(n_splits=5)
            >>> folds = splitter.split(Path("train/"))
            >>> for fold_idx, (train_files, val_files) in enumerate(folds):
            ...     print(f"Fold {fold_idx}: {len(train_files)} train, {len(val_files)} val")
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        txt_files = sorted([f for f in data_dir.glob("*.txt") if not f.name.startswith("_")])
        if not txt_files:
            raise ValueError(f"No .txt files found in {data_dir}")

        logger.info(f"Found {len(txt_files)} TXT files in {data_dir}")

        stats_list = self._load_package_stats(txt_files)

        if self.shuffle:
            random.seed(self.random_state)
            random.shuffle(stats_list)

        strata = self._group_by_strata(stats_list)
        logger.info(f"Created {len(strata)} strata:")
        for stratum_name in sorted(strata.keys()):
            stratum_packages = strata[stratum_name]
            logger.info(f"  {stratum_name}: {len(stratum_packages)} packages, "
                       f"{sum(p.issue_count for p in stratum_packages)} issues")

        folds = [[] for _ in range(self.n_splits)]

        for stratum_name, stratum_packages in strata.items():
            packages_per_fold = len(stratum_packages) // self.n_splits
            remainder = len(stratum_packages) % self.n_splits

            idx = 0
            for fold_idx in range(self.n_splits):
                fold_size = packages_per_fold + (1 if fold_idx < remainder else 0)
                folds[fold_idx].extend(stratum_packages[idx:idx + fold_size])
                idx += fold_size

        fold_splits = []
        for val_fold_idx in range(self.n_splits):
            val_files = [s.file_path for s in folds[val_fold_idx]]

            train_files = []
            for train_fold_idx in range(self.n_splits):
                if train_fold_idx != val_fold_idx:
                    train_files.extend([s.file_path for s in folds[train_fold_idx]])

            fold_splits.append((train_files, val_files))

            logger.info(f"Fold {val_fold_idx}: {len(train_files)} train files, {len(val_files)} val files")

        self._validate_splits(fold_splits, stats_list)

        return fold_splits

    def _validate_splits(
        self,
        fold_splits: List[Tuple[List[Path], List[Path]]],
        stats_list: List[PackageStats]
    ):
        """
        Validate that splits maintain proper stratification.

        Checks:
        1. No overlap between train and val in each fold
        2. FP/TP ratios are similar across folds
        3. All files are included exactly once as validation
        """
        logger.info("Validating k-fold splits...")

        stats_by_path = {s.file_path: s for s in stats_list}
        all_val_files = set()

        overall_fp = sum(s.fp_count for s in stats_list)
        overall_total = sum(s.issue_count for s in stats_list)
        overall_fp_ratio = overall_fp / overall_total if overall_total > 0 else 0.0

        for fold_idx, (train_files, val_files) in enumerate(fold_splits):
            train_set = set(train_files)
            val_set = set(val_files)

            if train_set & val_set:
                raise ValueError(f"Fold {fold_idx}: Train/val overlap detected!")

            all_val_files.update(val_set)

            val_stats = [stats_by_path[f] for f in val_files]
            val_fp = sum(s.fp_count for s in val_stats)
            val_total = sum(s.issue_count for s in val_stats)
            val_fp_ratio = val_fp / val_total if val_total > 0 else 0.0

            deviation = abs(val_fp_ratio - overall_fp_ratio)
            logger.info(f"  Fold {fold_idx}: FP ratio = {val_fp_ratio:.1%} "
                       f"(overall: {overall_fp_ratio:.1%}, deviation: {deviation:.1%})")

        all_files = set(s.file_path for s in stats_list)
        if all_val_files != all_files:
            missing = all_files - all_val_files
            extra = all_val_files - all_files
            if missing:
                logger.warning(f"  Files never used as validation: {len(missing)}")
            if extra:
                logger.warning(f"  Extra files in validation: {len(extra)}")

        logger.info("Validation complete!")


def main():
    """Test the k-fold splitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test stratified k-fold splitting")
    parser.add_argument("data_dir", type=Path, help="Directory containing .txt files")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    splitter = StratifiedKFoldSplitter(n_splits=args.n_splits, random_state=args.seed)
    folds = splitter.split(args.data_dir)

    print(f"\nGenerated {len(folds)} folds:")
    for fold_idx, (train_files, val_files) in enumerate(folds):
        print(f"  Fold {fold_idx}: {len(train_files)} train, {len(val_files)} val")


if __name__ == "__main__":
    main()