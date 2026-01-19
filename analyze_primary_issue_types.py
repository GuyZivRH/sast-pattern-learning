#!/usr/bin/env python3
"""
Analyze primary issue type distribution in training data.
Shows which issue types would fail k-fold validation.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))

from process_mining.core.parsers import ValidationEntryParser


def analyze_primary_issue_types(data_dir: Path, n_folds: int = 3):
    """Analyze primary issue type distribution."""
    parser = ValidationEntryParser()

    # Load all packages
    txt_files = sorted(data_dir.glob("*.txt"))
    print(f"Total packages: {len(txt_files)}\n")

    # Determine primary issue type for each package
    primary_issue_types = []

    for txt_file in txt_files:
        entries = parser.parse_file(txt_file)

        if not entries:
            continue

        # Count entries per issue type
        issue_type_counts = Counter()
        for entry in entries:
            issue_type_counts[entry.issue_type] += 1

        # Get primary issue type (most entries)
        primary_issue_type = max(issue_type_counts.items(), key=lambda x: x[1])[0]
        primary_issue_types.append(primary_issue_type)

    # Count packages per primary issue type
    primary_dist = Counter(primary_issue_types)

    print("="*80)
    print("PRIMARY ISSUE TYPE DISTRIBUTION")
    print("="*80)
    print(f"(Each package assigned to its most common issue type)\n")

    # Sort by count
    sorted_types = sorted(primary_dist.items(), key=lambda x: x[1], reverse=True)

    insufficient = []
    sufficient = []

    for issue_type, count in sorted_types:
        status = "✅" if count >= n_folds else "❌"
        print(f"{status} {issue_type:30s}: {count:3d} packages")

        if count < n_folds:
            insufficient.append((issue_type, count))
        else:
            sufficient.append((issue_type, count))

    print(f"\n{'='*80}")
    print(f"SUMMARY FOR {n_folds}-FOLD SPLIT")
    print(f"{'='*80}")
    print(f"Issue types with sufficient packages (>= {n_folds}): {len(sufficient)}")
    print(f"Issue types with insufficient packages (< {n_folds}): {len(insufficient)}")

    if insufficient:
        print(f"\n⚠️  These {len(insufficient)} issue types will cause validation to fail:")
        for issue_type, count in insufficient:
            print(f"  - {issue_type}: {count} packages < {n_folds} folds")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        data_dir = Path(sys.argv[1])
        n_folds = int(sys.argv[2])
    elif len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        n_folds = 3
    else:
        data_dir = Path("data/train")
        n_folds = 3

    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        sys.exit(1)

    analyze_primary_issue_types(data_dir, n_folds)