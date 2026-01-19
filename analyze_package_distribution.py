#!/usr/bin/env python3
"""
Analyze package distribution for stratified splitting.

Checks if package-level stratified splitting is feasible by analyzing:
- Number of packages per issue type
- FP/TP ratio distribution
- Strata sizes
"""

import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from process_mining.core.parsers import ValidationEntryParser


@dataclass
class PackageStats:
    """Statistics for a package."""
    package_name: str
    total_entries: int
    tp_count: int
    fp_count: int
    issue_types: set

    @property
    def fp_ratio(self) -> float:
        return self.fp_count / self.total_entries if self.total_entries > 0 else 0.0


def analyze_data_distribution(data_dir: Path):
    """Analyze package distribution for stratification."""
    parser = ValidationEntryParser()

    # Load all packages
    txt_files = sorted(data_dir.glob("*.txt"))
    print(f"Total packages: {len(txt_files)}")

    # Analyze each package
    package_stats = []
    issue_type_packages = defaultdict(list)

    for txt_file in txt_files:
        entries = parser.parse_file(txt_file)

        if not entries:
            continue

        tp_count = sum(1 for e in entries if "TRUE" in e.ground_truth_classification)
        fp_count = len(entries) - tp_count
        issue_types = set(e.issue_type for e in entries)

        stats = PackageStats(
            package_name=txt_file.stem,
            total_entries=len(entries),
            tp_count=tp_count,
            fp_count=fp_count,
            issue_types=issue_types
        )

        package_stats.append(stats)

        # Track which packages contain which issue types
        for issue_type in issue_types:
            issue_type_packages[issue_type].append(stats)

    print(f"Packages with entries: {len(package_stats)}\n")

    # Analyze issue type distribution
    print("="*80)
    print("ISSUE TYPE DISTRIBUTION")
    print("="*80)

    issue_types_sorted = sorted(issue_type_packages.items(),
                                 key=lambda x: len(x[1]),
                                 reverse=True)

    for issue_type, packages in issue_types_sorted[:15]:
        total_entries = sum(p.total_entries for p in packages
                          if issue_type in p.issue_types)
        print(f"{issue_type:25s}: {len(packages):3d} packages, {total_entries:5d} total entries")

    print(f"\nTotal unique issue types: {len(issue_type_packages)}")

    # Analyze FP ratio distribution
    print("\n" + "="*80)
    print("FP RATIO DISTRIBUTION (all packages)")
    print("="*80)

    fp_buckets = {
        'low (< 0.25)': [],
        'medium (0.25-0.75)': [],
        'high (> 0.75)': []
    }

    for pkg in package_stats:
        if pkg.fp_ratio < 0.25:
            fp_buckets['low (< 0.25)'].append(pkg)
        elif pkg.fp_ratio < 0.75:
            fp_buckets['medium (0.25-0.75)'].append(pkg)
        else:
            fp_buckets['high (> 0.75)'].append(pkg)

    for bucket_name, packages in fp_buckets.items():
        print(f"{bucket_name:25s}: {len(packages):3d} packages")

    # Analyze top 10 issue types for stratification
    print("\n" + "="*80)
    print("TOP 10 ISSUE TYPES - STRATIFICATION ANALYSIS")
    print("="*80)

    TOP_10 = [
        'RESOURCE_LEAK', 'OVERRUN', 'UNINIT', 'INTEGER_OVERFLOW',
        'USE_AFTER_FREE', 'CPPCHECK_WARNING', 'BUFFER_SIZE',
        'VARARGS', 'COMPILER_WARNING', 'COPY_PASTE_ERROR'
    ]

    for issue_type in TOP_10:
        packages = issue_type_packages.get(issue_type, [])
        if not packages:
            print(f"\n{issue_type}: NO PACKAGES")
            continue

        # Analyze FP ratio distribution for this issue type
        low = sum(1 for p in packages if p.fp_ratio < 0.25)
        medium = sum(1 for p in packages if 0.25 <= p.fp_ratio < 0.75)
        high = sum(1 for p in packages if p.fp_ratio >= 0.75)

        print(f"\n{issue_type}:")
        print(f"  Total packages: {len(packages)}")
        print(f"  FP ratio buckets: low={low}, medium={medium}, high={high}")

        # Check if we can do 60/20/20 split
        min_for_split = max(low, medium, high)
        can_split = len(packages) >= 15  # Need at least ~15 for meaningful split

        if can_split:
            print(f"  ✅ Can stratify (60/20/20 split feasible)")
        else:
            print(f"  ❌ Cannot stratify (only {len(packages)} packages, need ~15+)")

    # Overall assessment
    print("\n" + "="*80)
    print("STRATIFICATION FEASIBILITY")
    print("="*80)

    # Count how many top 10 issue types have enough packages
    feasible_count = sum(
        1 for it in TOP_10
        if len(issue_type_packages.get(it, [])) >= 15
    )

    print(f"Top 10 issue types with enough packages: {feasible_count}/10")

    # Recommend strata
    print("\nRecommended stratification approach:")
    print("  Strata: (issue_type, fp_ratio_bucket)")
    print("  FP buckets: low (<0.25), medium (0.25-0.75), high (>0.75)")
    print("  Split: 60% train, 20% val, 20% test")
    print("\nNote: Package-level splitting ensures no contamination between splits")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("data/full_pattern_data")

    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        sys.exit(1)

    analyze_data_distribution(data_dir)