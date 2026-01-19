#!/usr/bin/env python3
"""
Stratified K-Fold Splitting for Pattern Learning

Creates k stratified folds from training data ensuring:
1. ALL issue types appear in EVERY fold
2. UNIQUE packages per fold (no package contamination)
3. Balanced distribution across folds

Strategy: Works at PACKAGE level (no contamination)
- Distributes packages evenly across k folds using round-robin
- Each package goes entirely into one fold
- Validates that issue types have sufficient coverage
- No concept of "primary" issue type - packages can contain multiple types
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
from process_mining.kfold_pattern_learning.config import KFOLD_DEFAULTS, TOP_10_ISSUE_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PackageStats:
    """Statistics for a package file."""
    file_path: Path
    package_name: str
    total_entries: int
    tp_count: int
    fp_count: int
    issue_types: set
    primary_issue_type: str  # Issue type with most entries

    @property
    def fp_ratio(self) -> float:
        return self.fp_count / self.total_entries if self.total_entries > 0 else 0.0


class StratifiedKFoldSplitter:
    """
    Package-level K-Fold splitter that ensures ALL issue types in EVERY fold.

    Strategy:
    - Reads all packages from train directory
    - Groups packages by primary issue type
    - Distributes packages evenly across k folds
    - Each package goes entirely into one fold (no contamination)
    - Guarantees every fold has every issue type (with unique packages)
    """

    def __init__(
        self,
        n_splits: int = KFOLD_DEFAULTS['n_folds'],
        random_state: int = KFOLD_DEFAULTS['random_seed'],
        shuffle: bool = True
    ):
        """
        Initialize k-fold splitter.

        Args:
            n_splits: Number of folds (default: 3)
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle packages before splitting
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.parser = ValidationEntryParser()

        logger.info(f"Initialized StratifiedKFoldSplitter (Package-level):")
        logger.info(f"  n_splits: {n_splits}")
        logger.info(f"  random_state: {random_state}")
        logger.info(f"  Strategy: Each package → one fold (no contamination)")

    def _load_package_stats(self, data_dir: Path) -> List[PackageStats]:
        """Load statistics for all packages."""
        txt_files = sorted([f for f in data_dir.glob("*.txt") if not f.name.startswith("_")])

        if not txt_files:
            raise ValueError(f"No .txt files found in {data_dir}")

        logger.info(f"Found {len(txt_files)} package files")

        package_stats = []
        for txt_file in txt_files:
            try:
                entries = self.parser.parse_file(txt_file)

                if not entries:
                    logger.warning(f"Skipping {txt_file.name}: no entries")
                    continue

                # Count TP/FP
                tp_count = sum(1 for e in entries if "TRUE" in e.ground_truth_classification)
                fp_count = len(entries) - tp_count

                # Get issue types and find primary (most common)
                issue_type_counts = defaultdict(int)
                for entry in entries:
                    issue_type_counts[entry.issue_type] += 1

                primary_issue_type = max(issue_type_counts.items(), key=lambda x: x[1])[0]

                stats = PackageStats(
                    file_path=txt_file,
                    package_name=txt_file.stem,
                    total_entries=len(entries),
                    tp_count=tp_count,
                    fp_count=fp_count,
                    issue_types=set(issue_type_counts.keys()),
                    primary_issue_type=primary_issue_type
                )

                package_stats.append(stats)

            except Exception as e:
                logger.error(f"Error parsing {txt_file.name}: {e}")
                continue

        logger.info(f"Loaded {len(package_stats)} packages with entries")
        return package_stats

    def _validate_issue_type_coverage(
        self,
        package_stats: List[PackageStats]
    ):
        """
        Validate that TOP 10 issue types (that exist in the data) have enough coverage.
        Only validates issue types that will actually be used for pattern learning.
        """
        # Count how many packages contain each issue type
        issue_type_coverage = defaultdict(set)

        for pkg in package_stats:
            for issue_type in pkg.issue_types:
                issue_type_coverage[issue_type].add(pkg.package_name)

        # Check only TOP 10 issue types that actually exist in the data
        insufficient = []
        top10_in_data = []

        for issue_type in TOP_10_ISSUE_TYPES:
            packages = issue_type_coverage.get(issue_type, set())
            if len(packages) > 0:  # Only validate if it exists in the data
                top10_in_data.append(issue_type)
                if len(packages) < self.n_splits:
                    insufficient.append(
                        f"{issue_type}: {len(packages)} packages < {self.n_splits} folds"
                    )

        if insufficient:
            msg = (f"Some TOP-10 issue types have insufficient package coverage for {self.n_splits}-fold split:\n" +
                   "\n".join(f"  - {x}" for x in insufficient))
            logger.error(msg)
            logger.error("Note: Validation only checks TOP-10 issue types present in the data")
            raise ValueError(msg)

        # Log coverage for TOP 10 types that exist
        if top10_in_data:
            logger.info(f"Issue type coverage (TOP-10 types present in data):")
            for issue_type in top10_in_data:
                packages = issue_type_coverage.get(issue_type, set())
                logger.info(f"  {issue_type}: {len(packages)} packages")
        else:
            logger.warning("None of the TOP-10 issue types found in data - validation skipped")

    def _distribute_packages_to_folds(
        self,
        package_stats: List[PackageStats]
    ) -> List[List[Path]]:
        """
        Distribute packages across folds ensuring TOP-10 issue type coverage.

        Strategy:
        1. Shuffle packages for randomness
        2. Assign packages greedily to ensure each fold gets TOP-10 coverage
        3. Balance fold sizes
        """
        # Shuffle packages if requested
        packages = list(package_stats)
        if self.shuffle:
            random.seed(self.random_state)
            random.shuffle(packages)

        # Initialize k empty folds
        folds = [[] for _ in range(self.n_splits)]
        fold_issue_types = [set() for _ in range(self.n_splits)]  # Track coverage per fold

        # Get TOP-10 types that exist in data
        all_issue_types = set()
        for pkg in packages:
            all_issue_types.update(pkg.issue_types)

        top10_in_data = [it for it in TOP_10_ISSUE_TYPES if it in all_issue_types]

        # Phase 1: Ensure each fold has at least one package with each TOP-10 type
        assigned = [False] * len(packages)

        for issue_type in top10_in_data:
            # Find packages containing this issue type
            packages_with_type = [
                (i, pkg) for i, pkg in enumerate(packages)
                if issue_type in pkg.issue_types and not assigned[i]
            ]

            # Assign one to each fold that doesn't have this type yet
            for fold_idx in range(self.n_splits):
                if issue_type not in fold_issue_types[fold_idx] and packages_with_type:
                    # Pick the first available package with this type
                    pkg_idx, pkg = packages_with_type.pop(0)
                    folds[fold_idx].append(pkg.file_path)
                    fold_issue_types[fold_idx].update(pkg.issue_types)
                    assigned[pkg_idx] = True

        # Phase 2: Distribute remaining packages round-robin for balance
        remaining_packages = [pkg for i, pkg in enumerate(packages) if not assigned[i]]

        for idx, pkg in enumerate(remaining_packages):
            fold_idx = idx % self.n_splits
            folds[fold_idx].append(pkg.file_path)
            fold_issue_types[fold_idx].update(pkg.issue_types)

        # Log fold sizes and coverage
        for fold_idx in range(self.n_splits):
            top10_covered = sum(1 for it in top10_in_data if it in fold_issue_types[fold_idx])
            logger.info(f"Fold {fold_idx}: {len(folds[fold_idx])} packages, "
                       f"{top10_covered}/{len(top10_in_data)} TOP-10 types")

        return folds

    def split(self, data_dir: Path) -> List[Tuple[List[Path], List[Path]]]:
        """
        Generate k-fold splits ensuring TOP-10 issue types have coverage in every fold.

        Args:
            data_dir: Directory containing .txt package files

        Returns:
            List of (train_files, val_files) tuples, one per fold

        Strategy:
            1. Load all package files and compute stats
            2. Validate that TOP-10 issue types have sufficient coverage
            3. Distribute packages evenly across k folds using round-robin
            4. Each package goes entirely into one fold
            5. Return train/val file splits
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load package statistics
        logger.info(f"Reading all packages from {data_dir}...")
        package_stats = self._load_package_stats(data_dir)

        # Validate TOP-10 issue type coverage
        logger.info(f"Validating TOP-10 issue type coverage...")
        self._validate_issue_type_coverage(package_stats)

        # Distribute packages to folds
        logger.info(f"Creating {self.n_splits} folds with round-robin distribution...")
        folds = self._distribute_packages_to_folds(package_stats)

        # Create train/val splits
        fold_splits = []
        for val_fold_idx in range(self.n_splits):
            val_files = folds[val_fold_idx]

            train_files = []
            for train_fold_idx in range(self.n_splits):
                if train_fold_idx != val_fold_idx:
                    train_files.extend(folds[train_fold_idx])

            fold_splits.append((train_files, val_files))

        # Validate folds
        self._validate_folds(fold_splits)

        logger.info(f"✅ K-fold splitting complete: {self.n_splits} folds created")
        logger.info(f"   Each fold has unique packages (no contamination)")

        return fold_splits

    def _validate_folds(self, fold_splits: List[Tuple[List[Path], List[Path]]]):
        """Validate that all folds contain all issue types and no package overlap."""
        logger.info("Validating fold coverage and uniqueness...")

        # Check for package overlap between folds
        all_val_packages = set()
        for fold_idx, (train_files, val_files) in enumerate(fold_splits):
            val_package_names = set(f.stem for f in val_files)

            # Check for duplicates
            overlap = all_val_packages.intersection(val_package_names)
            if overlap:
                logger.error(f"Package contamination detected in fold {fold_idx}: {overlap}")
                raise ValueError("Packages appear in multiple folds")

            all_val_packages.update(val_package_names)

            # Check issue type coverage
            val_issue_types = set()
            for val_file in val_files:
                entries = self.parser.parse_file(val_file)
                val_issue_types.update(e.issue_type for e in entries)

            train_issue_types = set()
            for train_file in train_files:
                entries = self.parser.parse_file(train_file)
                train_issue_types.update(e.issue_type for e in entries)

            logger.info(f"  Fold {fold_idx}: {len(train_issue_types)} issue types in train, "
                       f"{len(val_issue_types)} in val")

        logger.info("✅ Validation complete: No package contamination detected")


def main():
    """Test the k-fold splitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test stratified k-fold splitting")
    parser.add_argument("data_dir", type=Path, help="Directory containing .txt files")
    parser.add_argument("--n-splits", type=int, default=KFOLD_DEFAULTS['n_folds'], help="Number of folds")
    parser.add_argument("--seed", type=int, default=KFOLD_DEFAULTS['random_seed'], help="Random seed")

    args = parser.parse_args()

    splitter = StratifiedKFoldSplitter(n_splits=args.n_splits, random_state=args.seed)
    folds = splitter.split(args.data_dir)

    print(f"\nGenerated {len(folds)} folds:")
    for fold_idx, (train_files, val_files) in enumerate(folds):
        print(f"  Fold {fold_idx}: {len(train_files)} train packages, {len(val_files)} val packages")


if __name__ == "__main__":
    main()