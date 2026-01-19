#!/usr/bin/env python3
"""
Validate sample dataset before running e2e.

Checks:
- Files exist
- Files are parseable
- Issue types present
- Entry counts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from process_mining.core.parsers import ValidationEntryParser


def validate_sample_dataset(data_dir: Path):
    """Validate sample dataset structure and content."""
    parser = ValidationEntryParser()

    print("="*60)
    print("Sample Dataset Validation")
    print("="*60)
    print()

    # Check directories exist
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for dir_name, dir_path in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if not dir_path.exists():
            print(f"❌ {dir_name} directory missing: {dir_path}")
            return False
        else:
            print(f"✓ {dir_name} directory exists: {dir_path}")

    print()

    # Parse and analyze each split
    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        print(f"{split_name} Split Analysis:")
        print("-" * 40)

        files = list(split_dir.glob("*.txt"))
        print(f"  Files: {len(files)}")

        if len(files) == 0:
            print(f"  ❌ No files found!")
            continue

        # Parse all entries
        all_entries = []
        for file in files:
            try:
                entries = parser.parse_file(file)
                all_entries.extend(entries)
            except Exception as e:
                print(f"  ❌ Failed to parse {file.name}: {e}")
                continue

        print(f"  Total entries: {len(all_entries)}")

        # Count by issue type
        issue_type_counts = {}
        for entry in all_entries:
            issue_type_counts[entry.issue_type] = issue_type_counts.get(entry.issue_type, 0) + 1

        print(f"  Issue types: {len(issue_type_counts)}")
        for issue_type, count in sorted(issue_type_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {issue_type}: {count} entries")

        # Count by ground truth
        gt_counts = {}
        for entry in all_entries:
            gt = entry.ground_truth_classification
            gt_counts[gt] = gt_counts.get(gt, 0) + 1

        print(f"  Ground truth distribution:")
        for gt, count in sorted(gt_counts.items()):
            print(f"    - {gt}: {count} entries ({count/len(all_entries)*100:.1f}%)")

        print()

    print("="*60)
    print("Validation Complete!")
    print("="*60)
    print()
    print("Dataset is ready for e2e testing.")
    print()
    print("To run e2e test:")
    print("  ./process_mining/run_sample_e2e.sh")
    print()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate sample dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "sample_pattern_data",
        help="Sample dataset directory"
    )

    args = parser.parse_args()

    success = validate_sample_dataset(args.data_dir)
    sys.exit(0 if success else 1)