#!/usr/bin/env python3
"""
Create a small sample dataset from full_pattern_data for end-to-end testing.

Creates train/val/test splits with 10 files each.
"""
import shutil
import random
from pathlib import Path


def create_sample_dataset(
    source_dir: Path,
    output_dir: Path,
    n_train: int = 10,
    n_val: int = 10,
    n_test: int = 10,
    random_seed: int = 42
):
    """
    Create sample dataset with stratified split.

    Args:
        source_dir: Directory with full dataset (.txt files)
        output_dir: Output directory for sample dataset
        n_train: Number of files for train set
        n_val: Number of files for val set
        n_test: Number of files for test set
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)

    # Get all .txt files
    all_files = list(source_dir.glob("*.txt"))

    if len(all_files) < (n_train + n_val + n_test):
        raise ValueError(
            f"Not enough files. Need {n_train + n_val + n_test}, "
            f"found {len(all_files)}"
        )

    # Shuffle files
    random.shuffle(all_files)

    # Split
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:n_train + n_val + n_test]

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"Creating sample dataset in: {output_dir}")
    print(f"  Train: {n_train} files")
    print(f"  Val: {n_val} files")
    print(f"  Test: {n_test} files")
    print()

    print("Train files:")
    for f in train_files:
        dest = train_dir / f.name
        shutil.copy2(f, dest)
        print(f"  - {f.name}")

    print(f"\nVal files:")
    for f in val_files:
        dest = val_dir / f.name
        shutil.copy2(f, dest)
        print(f"  - {f.name}")

    print(f"\nTest files:")
    for f in test_files:
        dest = test_dir / f.name
        shutil.copy2(f, dest)
        print(f"  - {f.name}")

    print(f"\nSample dataset created successfully!")
    print(f"Total: {len(train_files) + len(val_files) + len(test_files)} files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create sample dataset for e2e testing"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / "full_pattern_data",
        help="Source directory with full dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "sample_pattern_data",
        help="Output directory for sample dataset"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10,
        help="Number of files for train set"
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=10,
        help="Number of files for val set"
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=10,
        help="Number of files for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    create_sample_dataset(
        source_dir=args.source,
        output_dir=args.output,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        random_seed=args.seed
    )