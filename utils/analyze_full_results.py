#!/usr/bin/env python3
"""Analyze pattern citation and accuracy from full dataset k-fold results."""

import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_results():
    results_dir = Path("process_mining/full_output/phase1_kfold_results")

    # Track overall metrics
    total_entries = 0
    entries_with_citations = 0

    # Track by issue type
    issue_type_stats = defaultdict(lambda: {
        'total': 0,
        'cited': 0,
        'correct_total': 0,
        'correct_with_citation': 0,
        'correct_without_citation': 0,
        'citations': []
    })

    # Process all evaluation files
    eval_files = sorted(results_dir.glob("*_evaluation.json"))

    for eval_file in eval_files:
        # Extract issue type from filename
        filename = eval_file.stem
        parts = filename.split('_fold_')
        if len(parts) != 2:
            continue
        issue_type = parts[0]

        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
        except:
            continue

        # Process each entry
        entries = data.get('results', [])
        for entry in entries:
            total_entries += 1
            stats = issue_type_stats[issue_type]
            stats['total'] += 1

            cited_patterns = entry.get('cited_patterns', [])
            predicted = entry.get('predicted_classification', '')
            actual = entry.get('ground_truth_classification', '')

            is_correct = entry.get('correct', predicted == actual)
            has_citation = len(cited_patterns) > 0

            if has_citation:
                entries_with_citations += 1
                stats['cited'] += 1
                stats['citations'].extend(cited_patterns)

            if is_correct:
                stats['correct_total'] += 1
                if has_citation:
                    stats['correct_with_citation'] += 1
                else:
                    stats['correct_without_citation'] += 1

    # Print overall summary
    print("=" * 80)
    print("FULL DATASET PATTERN CITATION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total entries evaluated: {total_entries}")
    print(f"Entries with pattern citations: {entries_with_citations}")
    citation_rate = (entries_with_citations / total_entries * 100) if total_entries > 0 else 0
    print(f"Overall pattern citation rate: {citation_rate:.1f}%")
    print()

    # Calculate overall accuracy
    total_correct = sum(s['correct_total'] for s in issue_type_stats.values())
    total_correct_with = sum(s['correct_with_citation'] for s in issue_type_stats.values())
    total_correct_without = sum(s['correct_without_citation'] for s in issue_type_stats.values())

    overall_acc = (total_correct / total_entries * 100) if total_entries > 0 else 0
    acc_with = (total_correct_with / entries_with_citations * 100) if entries_with_citations > 0 else 0
    acc_without = (total_correct_without / (total_entries - entries_with_citations) * 100) if (total_entries - entries_with_citations) > 0 else 0

    print(f"Overall accuracy: {overall_acc:.1f}%")
    print(f"Accuracy WITH pattern citations: {acc_with:.1f}% ({total_correct_with}/{entries_with_citations})")
    print(f"Accuracy WITHOUT pattern citations: {acc_without:.1f}% ({total_correct_without}/{total_entries - entries_with_citations})")
    print()

    # Print by issue type (sorted by citation rate)
    print("=" * 80)
    print("PATTERN CITATION BY ISSUE TYPE")
    print("=" * 80)
    print()
    print(f"{'Issue Type':<30} {'Total':>8} {'Cited':>8} {'Rate':>8} {'Acc w/':>8} {'Acc w/o':>8}")
    print("-" * 80)

    sorted_types = sorted(issue_type_stats.items(),
                         key=lambda x: (x[1]['cited'] / x[1]['total'] if x[1]['total'] > 0 else 0),
                         reverse=True)

    for issue_type, stats in sorted_types:
        total = stats['total']
        cited = stats['cited']
        rate = (cited / total * 100) if total > 0 else 0

        acc_with = (stats['correct_with_citation'] / cited * 100) if cited > 0 else 0
        without_count = total - cited
        acc_without = (stats['correct_without_citation'] / without_count * 100) if without_count > 0 else 0

        print(f"{issue_type:<30} {total:>8} {cited:>8} {rate:>7.1f}% {acc_with:>7.1f}% {acc_without:>7.1f}%")

    print()
    print("=" * 80)

    # Show top/bottom performers
    print()
    print("TOP 5 ISSUE TYPES BY CITATION RATE:")
    for issue_type, stats in sorted_types[:5]:
        total = stats['total']
        cited = stats['cited']
        rate = (cited / total * 100) if total > 0 else 0
        print(f"  {issue_type}: {rate:.1f}% ({cited}/{total})")

    print()
    print("BOTTOM 5 ISSUE TYPES BY CITATION RATE:")
    for issue_type, stats in sorted_types[-5:]:
        total = stats['total']
        cited = stats['cited']
        rate = (cited / total * 100) if total > 0 else 0
        print(f"  {issue_type}: {rate:.1f}% ({cited}/{total})")

if __name__ == '__main__':
    analyze_results()