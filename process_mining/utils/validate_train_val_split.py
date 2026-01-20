#!/usr/bin/env python3
"""
Validate train/val split for package contamination and issue type coverage.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from process_mining.core.parsers import ValidationEntryParser


def validate_split(train_dir: Path, val_dir: Path):
    """Validate train/val split."""
    parser = ValidationEntryParser()

    # Get package names from each split
    train_files = sorted([f for f in train_dir.glob("*.txt") if not f.name.startswith("_")])
    val_files = sorted([f for f in val_dir.glob("*.txt") if not f.name.startswith("_")])

    train_packages = set(f.stem for f in train_files)
    val_packages = set(f.stem for f in val_files)

    print("="*80)
    print("PACKAGE CONTAMINATION CHECK")
    print("="*80)
    print(f"Train packages: {len(train_packages)}")
    print(f"Val packages: {len(val_packages)}")

    # Check for contamination
    overlap = train_packages.intersection(val_packages)

    if overlap:
        print(f"\n❌ CONTAMINATION DETECTED: {len(overlap)} packages in both splits!")
        print(f"Overlapping packages: {sorted(overlap)[:10]}...")
    else:
        print(f"\n✅ NO CONTAMINATION: All packages are unique")

    print("\n" + "="*80)
    print("ISSUE TYPE COVERAGE CHECK")
    print("="*80)

    # Get issue types from each split
    train_issue_types = defaultdict(int)
    val_issue_types = defaultdict(int)

    print("\nAnalyzing train set...")
    for txt_file in train_files:
        entries = parser.parse_file(txt_file)
        for entry in entries:
            train_issue_types[entry.issue_type] += 1

    print(f"Analyzing val set...")
    for txt_file in val_files:
        entries = parser.parse_file(txt_file)
        for entry in entries:
            val_issue_types[entry.issue_type] += 1

    # Compare issue types
    train_types = set(train_issue_types.keys())
    val_types = set(val_issue_types.keys())

    all_types = train_types.union(val_types)
    only_train = train_types - val_types
    only_val = val_types - train_types
    both = train_types.intersection(val_types)

    print(f"\nTotal unique issue types: {len(all_types)}")
    print(f"Issue types in BOTH splits: {len(both)}")
    print(f"Issue types ONLY in train: {len(only_train)}")
    print(f"Issue types ONLY in val: {len(only_val)}")

    if only_train:
        print(f"\n⚠️  Issue types missing from val:")
        for it in sorted(only_train):
            print(f"  - {it}: {train_issue_types[it]} entries (train only)")

    if only_val:
        print(f"\n⚠️  Issue types missing from train:")
        for it in sorted(only_val):
            print(f"  - {it}: {val_issue_types[it]} entries (val only)")

    if not only_train and not only_val:
        print(f"\n✅ ALL ISSUE TYPES appear in both train and val")

    # Show distribution for common issue types
    print("\n" + "="*80)
    print("ISSUE TYPE DISTRIBUTION (both splits)")
    print("="*80)

    # Sort by total count
    all_counts = {}
    for it in both:
        all_counts[it] = train_issue_types[it] + val_issue_types[it]

    sorted_types = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Issue Type':<30} {'Train':>10} {'Val':>10} {'Total':>10} {'Train %':>10}")
    print("-"*80)

    for issue_type, total in sorted_types[:20]:
        train_count = train_issue_types[issue_type]
        val_count = val_issue_types[issue_type]
        train_pct = train_count / total * 100 if total > 0 else 0

        print(f"{issue_type:<30} {train_count:>10} {val_count:>10} {total:>10} {train_pct:>9.1f}%")

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    issues = []

    if overlap:
        issues.append(f"❌ Package contamination: {len(overlap)} packages overlap")
    else:
        print("✅ No package contamination")

    if only_train or only_val:
        issues.append(f"❌ Issue type coverage incomplete: {len(only_train)} train-only, {len(only_val)} val-only")
    else:
        print("✅ All issue types in both splits")

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n🎉 VALIDATION PASSED: Split is valid!")
        return True


if __name__ == "__main__":
    train_dir = Path("../../data/train")
    val_dir = Path("../../data/val")

    if not train_dir.exists():
        print(f"Error: {train_dir} not found")
        sys.exit(1)

    if not val_dir.exists():
        print(f"Error: {val_dir} not found")
        sys.exit(1)

    valid = validate_split(train_dir, val_dir)
    sys.exit(0 if valid else 1)