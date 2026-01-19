#!/usr/bin/env python3
"""Check which issue types had no refinements and correlate with pattern usage."""

import json
from pathlib import Path
from collections import defaultdict

# Get issue types with no refinements
phase2_dir = Path("process_mining/full_output/phase2_refinement_results")
no_refinement_types = []

for refine_file in phase2_dir.glob("*_iteration_1_refinements.json"):
    with open(refine_file) as f:
        data = json.load(f)
        summary = data.get('refinement_metadata', {}).get('summary', '')
        if 'No misclassified' in summary:
            issue_type = refine_file.stem.replace('_iteration_1_refinements', '')
            no_refinement_types.append(issue_type)

print(f"Found {len(no_refinement_types)} issue types with no refinements needed")
print()

# Get pattern citation data from Phase 1
phase1_dir = Path("process_mining/full_output/phase1_kfold_results")
citation_stats = defaultdict(lambda: {'total': 0, 'cited': 0})

for eval_file in phase1_dir.glob("*_evaluation.json"):
    with open(eval_file) as f:
        data = json.load(f)
        issue_type = data.get('issue_type', '')

        for entry in data.get('results', []):
            citation_stats[issue_type]['total'] += 1
            if entry.get('cited_patterns', []):
                citation_stats[issue_type]['cited'] += 1

# Analyze correlation
print("Issue Types with NO REFINEMENTS NEEDED:")
print("=" * 80)
print(f"{'Issue Type':<35} {'Citation Rate':>15} {'Entries':>10}")
print("-" * 80)

no_refine_with_patterns = []
no_refine_without_patterns = []

for issue_type in sorted(no_refinement_types):
    stats = citation_stats.get(issue_type, {'total': 0, 'cited': 0})
    total = stats['total']
    cited = stats['cited']
    rate = (cited / total * 100) if total > 0 else 0

    print(f"{issue_type:<35} {rate:>14.1f}% {total:>10}")

    if rate > 0:
        no_refine_with_patterns.append(issue_type)
    else:
        no_refine_without_patterns.append(issue_type)

print()
print("=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Total issue types with no refinements needed: {len(no_refinement_types)}")
print(f"  - Used patterns (citation rate > 0%): {len(no_refine_with_patterns)}")
print(f"  - Did NOT use patterns (0% citations): {len(no_refine_without_patterns)}")
print()

if no_refine_with_patterns:
    print("Issue types that got perfect accuracy WITH pattern usage:")
    for it in no_refine_with_patterns:
        rate = (citation_stats[it]['cited'] / citation_stats[it]['total'] * 100)
        print(f"  - {it}: {rate:.1f}% citation rate")
    print()

if no_refine_without_patterns:
    print("Issue types that got perfect accuracy WITHOUT using patterns:")
    for it in no_refine_without_patterns:
        print(f"  - {it}: 0% citation rate (LLM worked without patterns)")