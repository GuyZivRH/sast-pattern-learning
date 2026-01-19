#!/usr/bin/env python3
"""
Analyze issue type coverage across packages.
Shows how many packages contain each issue type (regardless of primary).
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from process_mining.core.parsers import ValidationEntryParser


def analyze_issue_type_coverage(data_dir: Path, n_folds: int = 3):
    """Analyze which packages contain which issue types."""
    parser = ValidationEntryParser()

    # Load all packages
    txt_files = sorted(data_dir.glob("*.txt"))
    print(f"Total packages: {len(txt_files)}\n")

    # Track which packages contain which issue types
    issue_type_packages = defaultdict(set)

    for txt_file in txt_files:
        entries = parser.parse_file(txt_file)

        if not entries:
            continue

        # Get all issue types in this package
        issue_types_in_package = set(e.issue_type for e in entries)

        for issue_type in issue_types_in_package:
            issue_type_packages[issue_type].add(txt_file.stem)

    print("="*80)
    print("ISSUE TYPE COVERAGE (packages containing each issue type)")
    print("="*80)
    print(f"(Regardless of whether it's the primary type)\n")

    # Sort by number of packages
    sorted_types = sorted(issue_type_packages.items(), key=lambda x: len(x[1]), reverse=True)

    insufficient = []
    sufficient = []

    for issue_type, packages in sorted_types:
        count = len(packages)
        status = "✅" if count >= n_folds else "❌"
        print(f"{status} {issue_type:30s}: {count:3d} packages")

        if count < n_folds:
            insufficient.append((issue_type, count))
        else:
            sufficient.append((issue_type, count))

    print(f"\n{'='*80}")
    print(f"SUMMARY FOR {n_folds}-FOLD SPLIT")
    print(f"{'='*80}")
    print(f"Issue types with sufficient coverage (>= {n_folds}): {len(sufficient)}")
    print(f"Issue types with insufficient coverage (< {n_folds}): {len(insufficient)}")

    if insufficient:
        print(f"\n⚠️  These {len(insufficient)} issue types don't appear in enough packages:")
        for issue_type, count in insufficient:
            print(f"  - {issue_type}: {count} packages < {n_folds} folds")

    # Check TOP 10 specifically
    TOP_10 = [
        'RESOURCE_LEAK', 'OVERRUN', 'UNINIT', 'INTEGER_OVERFLOW',
        'USE_AFTER_FREE', 'CPPCHECK_WARNING', 'BUFFER_SIZE',
        'VARARGS', 'COMPILER_WARNING', 'COPY_PASTE_ERROR'
    ]

    print(f"\n{'='*80}")
    print("TOP 10 ISSUE TYPES (for pattern learning)")
    print("="*80)

    top10_ok = []
    top10_failed = []

    for issue_type in TOP_10:
        packages = issue_type_packages.get(issue_type, set())
        count = len(packages)
        status = "✅" if count >= n_folds else "❌"
        print(f"{status} {issue_type:30s}: {count:3d} packages")

        if count >= n_folds:
            top10_ok.append(issue_type)
        else:
            top10_failed.append(issue_type)

    print(f"\nTOP 10 summary: {len(top10_ok)}/{len(TOP_10)} have sufficient coverage")

    if top10_failed:
        print(f"⚠️  TOP 10 types that will fail: {', '.join(top10_failed)}")


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

    analyze_issue_type_coverage(data_dir, n_folds)