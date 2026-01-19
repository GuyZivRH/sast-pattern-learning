#!/usr/bin/env python3
"""Check distribution of issue types across train/val/test datasets."""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from process_mining.core.parsers import ValidationEntryParser

def analyze_dataset(data_dir, label):
    """Analyze issue type distribution in a dataset directory."""
    parser = ValidationEntryParser()
    issue_type_counts = defaultdict(int)

    txt_files = list(Path(data_dir).glob("*.txt"))

    for txt_file in txt_files:
        try:
            entries = parser.parse_file(txt_file)
            for entry in entries:
                issue_type_counts[entry.issue_type] += 1
        except Exception as e:
            print(f"Error parsing {txt_file.name}: {e}")
            continue

    return issue_type_counts, len(txt_files)

def main():
    base_dir = Path("process_mining/full_pattern_data")

    print("="*80)
    print("ISSUE TYPE DISTRIBUTION ACROSS DATASETS")
    print("="*80)
    print()

    # Analyze each dataset
    train_counts, train_files = analyze_dataset(base_dir / "train", "Train")
    val_counts, val_files = analyze_dataset(base_dir / "validation", "Validation")
    test_counts, test_files = analyze_dataset(base_dir / "test", "Test")

    # Get all unique issue types
    all_issue_types = set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())

    print(f"Dataset files: Train={train_files}, Validation={val_files}, Test={test_files}")
    print(f"Total unique issue types: {len(all_issue_types)}")
    print()

    # Check coverage
    print("COVERAGE ANALYSIS:")
    print("-"*80)

    in_all_three = []
    in_train_only = []
    in_val_only = []
    in_test_only = []
    in_train_val = []
    in_train_test = []
    in_val_test = []

    for issue_type in sorted(all_issue_types):
        in_train = issue_type in train_counts
        in_val = issue_type in val_counts
        in_test = issue_type in test_counts

        if in_train and in_val and in_test:
            in_all_three.append(issue_type)
        elif in_train and not in_val and not in_test:
            in_train_only.append(issue_type)
        elif not in_train and in_val and not in_test:
            in_val_only.append(issue_type)
        elif not in_train and not in_val and in_test:
            in_test_only.append(issue_type)
        elif in_train and in_val and not in_test:
            in_train_val.append(issue_type)
        elif in_train and not in_val and in_test:
            in_train_test.append(issue_type)
        elif not in_train and in_val and in_test:
            in_val_test.append(issue_type)

    print(f"Present in ALL THREE datasets: {len(in_all_three)}")
    print(f"Train + Val only (missing Test): {len(in_train_val)}")
    print(f"Train + Test only (missing Val): {len(in_train_test)}")
    print(f"Val + Test only (missing Train): {len(in_val_test)}")
    print(f"Train only: {len(in_train_only)}")
    print(f"Validation only: {len(in_val_only)}")
    print(f"Test only: {len(in_test_only)}")
    print()

    # Detailed breakdown
    print("="*80)
    print("DETAILED DISTRIBUTION")
    print("="*80)
    print(f"{'Issue Type':<35} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-"*80)

    for issue_type in sorted(all_issue_types):
        train_count = train_counts.get(issue_type, 0)
        val_count = val_counts.get(issue_type, 0)
        test_count = test_counts.get(issue_type, 0)
        total = train_count + val_count + test_count

        # Flag missing datasets
        flag = ""
        if train_count == 0:
            flag += " [NO TRAIN]"
        if val_count == 0:
            flag += " [NO VAL]"
        if test_count == 0:
            flag += " [NO TEST]"

        print(f"{issue_type:<35} {train_count:>10} {val_count:>10} {test_count:>10} {total:>10}{flag}")

    print()
    print("="*80)

    if in_train_val:
        print()
        print("WARNING: Issue types in Train+Val but MISSING from Test:")
        for it in in_train_val:
            print(f"  - {it}")

    if in_train_test:
        print()
        print("WARNING: Issue types in Train+Test but MISSING from Validation:")
        for it in in_train_test:
            print(f"  - {it}")

    if in_val_test:
        print()
        print("WARNING: Issue types in Val+Test but MISSING from Train:")
        for it in in_val_test:
            print(f"  - {it}")

if __name__ == '__main__':
    main()