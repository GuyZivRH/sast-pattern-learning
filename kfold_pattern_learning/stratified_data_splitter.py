#!/usr/bin/env python3
"""
Stratified Data Splitter - File-Level Stratification

Splits data files while maintaining:
1. Issue type distribution across train/val/test
2. TP/FP class balance across splits
3. All issue types in each split (when possible)

This replaces package-level splitting which caused severe data imbalance.
"""

import sys
import random
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.parsers import ValidationEntryParser, ValidationEntry
from process_mining.kfold_pattern_learning.config import TOP_10_ISSUE_TYPES, SPLIT_RATIOS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StratifiedDataSplitter:
    """
    File-level stratified data splitter.

    Ensures:
    - All issue types appear in train/val/test
    - TP/FP balance is maintained
    - Top 10 issue types are prioritized for pattern learning
    """

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        top_n_only: bool = True
    ):
        """
        Initialize stratified splitter.

        Args:
            source_dir: Source directory with .txt files
            output_dir: Output directory for train/val/test splits
            train_ratio: Fraction for training (default: 0.6)
            val_ratio: Fraction for validation (default: 0.2)
            test_ratio: Fraction for test (default: 0.2)
            random_seed: Random seed for reproducibility
            top_n_only: If True, only include top 10 issue types
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.top_n_only = top_n_only

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"

        random.seed(random_seed)

        logger.info(f"Initialized StratifiedDataSplitter:")
        logger.info(f"  Source: {self.source_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        logger.info(f"  Top 10 only: {top_n_only}")

    def split_data(self) -> Dict:
        """
        Perform stratified split of data files.

        Returns:
            Statistics about the split
        """
        logger.info("="*80)
        logger.info("STRATIFIED DATA SPLITTING")
        logger.info("="*80)

        # Parse all entries from source files
        logger.info("\nParsing source files...")
        entries_by_file = self._parse_source_files()

        # Group entries by (issue_type, classification)
        logger.info("\nGrouping by issue_type and classification...")
        grouped = self._group_entries(entries_by_file)

        # Perform stratified split
        logger.info("\nPerforming stratified split...")
        train_files, val_files, test_files = self._stratified_split(grouped)

        # Copy files to output directories
        logger.info("\nCopying files to output directories...")
        self._copy_files(train_files, val_files, test_files)

        # Generate statistics
        stats = self._generate_statistics(train_files, val_files, test_files, entries_by_file)

        logger.info("\n" + "="*80)
        logger.info("SPLIT COMPLETE")
        logger.info("="*80)

        return stats

    def _parse_source_files(self) -> Dict[Path, List[ValidationEntry]]:
        """Parse all source .txt files and return entries grouped by file."""
        parser = ValidationEntryParser()
        entries_by_file = {}

        txt_files = sorted(self.source_dir.glob("*.txt"))
        logger.info(f"  Found {len(txt_files)} .txt files")

        for txt_file in txt_files:
            try:
                entries = parser.parse_file(txt_file)

                # Filter to top 10 if requested
                if self.top_n_only:
                    entries = [e for e in entries if e.issue_type in TOP_10_ISSUE_TYPES]

                if entries:
                    entries_by_file[txt_file] = entries
            except Exception as e:
                logger.warning(f"  Error parsing {txt_file.name}: {e}")
                continue

        total_entries = sum(len(entries) for entries in entries_by_file.values())
        logger.info(f"  Parsed {total_entries} entries from {len(entries_by_file)} files")

        return entries_by_file

    def _group_entries(self, entries_by_file: Dict[Path, List[ValidationEntry]]) -> Dict[Tuple[str, str], List[Path]]:
        """
        Group files by (issue_type, classification).

        Returns:
            Dict mapping (issue_type, classification) -> list of file paths
        """
        grouped = defaultdict(list)

        for file_path, entries in entries_by_file.items():
            # Group this file's entries by (issue_type, classification)
            file_groups = defaultdict(int)
            for entry in entries:
                classification = 'TP' if 'TRUE' in entry.ground_truth_classification else 'FP'
                key = (entry.issue_type, classification)
                file_groups[key] += 1

            # Assign file to its dominant group
            if file_groups:
                dominant_key = max(file_groups.items(), key=lambda x: x[1])[0]
                grouped[dominant_key].append(file_path)

        logger.info(f"  Created {len(grouped)} stratification groups")

        return grouped

    def _stratified_split(
        self,
        grouped: Dict[Tuple[str, str], List[Path]]
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Perform stratified split maintaining (issue_type, classification) distribution.

        Returns:
            (train_files, val_files, test_files)
        """
        train_files = []
        val_files = []
        test_files = []

        for (issue_type, classification), files in sorted(grouped.items()):
            # Shuffle files in this group
            files = list(files)
            random.shuffle(files)

            n = len(files)
            train_size = max(1, int(n * self.train_ratio))
            val_size = max(1, int(n * self.val_ratio))

            # Split files
            group_train = files[:train_size]
            group_val = files[train_size:train_size + val_size]
            group_test = files[train_size + val_size:]

            # Ensure at least 1 file in each split if possible
            if n >= 3 and not group_test:
                # Move one from train to test
                group_test = [group_train.pop()]

            train_files.extend(group_train)
            val_files.extend(group_val)
            test_files.extend(group_test)

            logger.info(f"  {issue_type:<30} {classification:<5} {len(files):>4} files → "
                       f"train={len(group_train)}, val={len(group_val)}, test={len(group_test)}")

        logger.info(f"\n  Total: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

        return train_files, val_files, test_files

    def _copy_files(self, train_files: List[Path], val_files: List[Path], test_files: List[Path]):
        """Copy files to output directories."""
        # Create output directories
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for src_file in train_files:
            shutil.copy2(src_file, self.train_dir / src_file.name)

        for src_file in val_files:
            shutil.copy2(src_file, self.val_dir / src_file.name)

        for src_file in test_files:
            shutil.copy2(src_file, self.test_dir / src_file.name)

        logger.info(f"  Copied files to:")
        logger.info(f"    Train: {self.train_dir} ({len(train_files)} files)")
        logger.info(f"    Val: {self.val_dir} ({len(val_files)} files)")
        logger.info(f"    Test: {self.test_dir} ({len(test_files)} files)")

    def _generate_statistics(
        self,
        train_files: List[Path],
        val_files: List[Path],
        test_files: List[Path],
        entries_by_file: Dict[Path, List[ValidationEntry]]
    ) -> Dict:
        """Generate statistics about the split."""
        def analyze_files(files):
            stats = defaultdict(lambda: {'TP': 0, 'FP': 0})
            for file_path in files:
                if file_path in entries_by_file:
                    for entry in entries_by_file[file_path]:
                        classification = 'TP' if 'TRUE' in entry.ground_truth_classification else 'FP'
                        stats[entry.issue_type][classification] += 1
            return stats

        train_stats = analyze_files(train_files)
        val_stats = analyze_files(val_files)
        test_stats = analyze_files(test_files)

        # Get all issue types
        all_types = set(train_stats.keys()) | set(val_stats.keys()) | set(test_stats.keys())

        # Check coverage
        in_all_three = [t for t in all_types if t in train_stats and t in val_stats and t in test_stats]

        logger.info(f"\nIssue Type Coverage:")
        logger.info(f"  Total unique issue types: {len(all_types)}")
        logger.info(f"  In all three splits: {len(in_all_three)}")

        if len(in_all_three) < len(all_types):
            missing = all_types - set(in_all_three)
            logger.warning(f"  Missing from some splits: {missing}")

        return {
            'train': {'files': len(train_files), 'stats': dict(train_stats)},
            'val': {'files': len(val_files), 'stats': dict(val_stats)},
            'test': {'files': len(test_files), 'stats': dict(test_stats)},
            'coverage': {
                'total_types': len(all_types),
                'in_all_three': len(in_all_three),
                'issue_types': sorted(all_types)
            }
        }


def main():
    """CLI for stratified data splitting."""
    import argparse

    parser = argparse.ArgumentParser(description="Stratified data splitter")
    parser.add_argument("source_dir", type=Path, help="Source directory with .txt files")
    parser.add_argument("output_dir", type=Path, help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.6, help="Train ratio (default: 0.6)")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio (default: 0.2)")
    parser.add_argument("--test", type=float, default=0.2, help="Test ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--all-types", action="store_true", help="Include all issue types (not just top 10)")

    args = parser.parse_args()

    splitter = StratifiedDataSplitter(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed,
        top_n_only=not args.all_types
    )

    stats = splitter.split_data()

    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    print(f"\nTrain: {stats['train']['files']} files")
    print(f"Val:   {stats['val']['files']} files")
    print(f"Test:  {stats['test']['files']} files")
    print(f"\nCoverage: {stats['coverage']['in_all_three']}/{stats['coverage']['total_types']} issue types in all splits")


if __name__ == "__main__":
    main()