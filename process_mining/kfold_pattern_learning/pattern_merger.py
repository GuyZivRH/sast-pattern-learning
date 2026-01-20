#!/usr/bin/env python3
"""
Pattern Merger - LLM-based Pattern Deduplication

Merges patterns from multiple folds using LLM to deduplicate and
consolidate overlapping patterns.
"""

import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.classifiers import (
    PatternBasedClassifier,
    PLATFORM_CONFIGS
)
from process_mining.kfold_pattern_learning.config import LLM_DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PatternMerger:
    """
    Merges patterns from k folds using LLM-based deduplication.

    Takes k pattern files and consolidates them into a single
    deduplicated set while preserving all unique insights.
    """

    MERGE_PROMPT_TEMPLATE = """# Pattern Merging Task

You are consolidating patterns learned from {k} independent folds into a single deduplicated pattern set for issue type: **{issue_type}**

## Input Patterns

You have {k} pattern files, each containing FP and TP patterns:

{pattern_files_content}

## Your Task

Merge these patterns into a single deduplicated set following these rules:

### Deduplication Rules

1. **Identify Duplicates**: If 2+ patterns describe the SAME code idiom/vulnerability, merge them into ONE pattern
2. **Preserve Unique Patterns**: Keep all patterns that describe DISTINCT code patterns
3. **Combine Insights**: When merging duplicates, combine the best insights from all versions
4. **Keep Best Description**: Use the most comprehensive and clear summary
5. **Renumber Pattern IDs**: Ensure final pattern IDs are sequential (FP-001, FP-002, ..., TP-001, TP-002, ...)
6. **Reorganize Groups**: Update group labels to be consistent (A, B, C, D, ...)

### Quality Criteria

- **Completeness**: Don't lose any unique insights from the k folds
- **Clarity**: Each merged pattern should be self-contained and clear
- **No Redundancy**: Eliminate true duplicates (same code idiom described differently)
- **Consistency**: Pattern IDs and group labels should follow conventions

## Output Format

Generate a JSON file with this EXACT structure:

```json
{{
  "fp": [
    {{
      "pattern_id": "{issue_type}-FP-001",
      "group": "A: Pattern Category Name",
      "summary": "Merged summary combining insights from multiple folds..."
    }}
  ],
  "tp": [
    {{
      "pattern_id": "{issue_type}-TP-001",
      "group": "A: Vulnerability Category",
      "summary": "Merged summary..."
    }}
  ],
  "merge_metadata": {{
    "total_input_patterns_fp": <count>,
    "total_input_patterns_tp": <count>,
    "merged_fp": <count>,
    "merged_tp": <count>,
    "deduplication_summary": "Brief explanation of what was merged"
  }}
}}
```

## Important Notes

- Output ONLY the JSON block
- Ensure valid JSON syntax
- Pattern IDs must be sequential starting from 001
- Group labels must start with letters (A:, B:, C:, ...)
- Include merge_metadata to explain deduplication decisions
"""

    def __init__(
        self,
        platform: str = LLM_DEFAULTS['platform'],
        model: Optional[str] = None,
        max_tokens: int = LLM_DEFAULTS['max_tokens'],
        temperature: float = LLM_DEFAULTS['temperature']
    ):
        """
        Initialize pattern merger.

        Args:
            platform: LLM platform ("local" or "nim")
            model: Optional model override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        if platform not in PLATFORM_CONFIGS:
            raise ValueError(f"Unknown platform: {platform}")

        self.platform = platform
        config = PLATFORM_CONFIGS[platform]
        self.model = model or config["model"]
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Reuse PatternBasedClassifier's LLM infrastructure
        self.classifier = PatternBasedClassifier(
            platform=platform,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            baseline_mode=True
        )

        logger.info(f"Initialized PatternMerger: {platform} / {self.model}")

    def merge_patterns(
        self,
        fold_patterns: List[Dict],
        issue_type: str
    ) -> Dict:
        """
        Merge patterns from multiple folds.

        Args:
            fold_patterns: List of pattern dicts [{"fp": [...], "tp": [...]}, ...]
            issue_type: Issue type being merged

        Returns:
            Merged pattern dictionary {"fp": [...], "tp": [...], "merge_metadata": {...}}
        """
        k = len(fold_patterns)
        logger.info(f"Merging patterns from {k} folds for {issue_type}...")

        # Count input patterns
        total_fp = sum(len(p.get('fp', [])) for p in fold_patterns)
        total_tp = sum(len(p.get('tp', [])) for p in fold_patterns)
        logger.info(f"  Input: {total_fp} FP patterns, {total_tp} TP patterns across {k} folds")

        # Build merge prompt
        prompt = self._build_merge_prompt(fold_patterns, issue_type, k)

        # Call LLM
        try:
            logger.info(f"  Calling LLM to merge patterns...")

            self.classifier._wait_for_rate_limit()

            # Platform-aware API call
            if self.classifier.platform == "vertex":
                response = self.classifier.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
            else:
                # OpenAI-compatible API (local, nim)
                response = self.classifier.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content

            if response_text is None:
                raise ValueError("LLM returned empty response")

            # Parse response
            merged_patterns = self._parse_merge_response(response_text, issue_type)

            # Validate merge
            merged_fp = len(merged_patterns.get('fp', []))
            merged_tp = len(merged_patterns.get('tp', []))

            logger.info(f"  Merged to: {merged_fp} FP patterns, {merged_tp} TP patterns")

            if 'merge_metadata' in merged_patterns:
                metadata = merged_patterns['merge_metadata']
                logger.info(f"  Deduplication: {metadata.get('deduplication_summary', 'N/A')}")

            return merged_patterns

        except Exception as e:
            logger.error(f"Error merging patterns for {issue_type}: {e}")
            # Fallback: return first fold's patterns
            logger.warning("Falling back to first fold's patterns")
            return fold_patterns[0] if fold_patterns else {"fp": [], "tp": []}

    def _build_merge_prompt(
        self,
        fold_patterns: List[Dict],
        issue_type: str,
        k: int
    ) -> str:
        """Build prompt for pattern merging."""
        # Format each fold's patterns
        pattern_files_content = []
        for fold_idx, patterns in enumerate(fold_patterns, 1):
            fold_content = f"### Fold {fold_idx} Patterns\n\n"
            fold_content += "```json\n"
            fold_content += json.dumps(patterns, indent=2)
            fold_content += "\n```\n"
            pattern_files_content.append(fold_content)

        pattern_files_text = "\n".join(pattern_files_content)

        # Replace placeholders
        prompt = self.MERGE_PROMPT_TEMPLATE.format(
            k=k,
            issue_type=issue_type,
            pattern_files_content=pattern_files_text
        )

        return prompt

    def _parse_merge_response(
        self,
        response_text: str,
        issue_type: str
    ) -> Dict:
        """Parse LLM response to extract merged patterns."""
        # Extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                logger.error("No JSON found in merge response")
                return {"fp": [], "tp": []}

        try:
            merged = json.loads(json_str)

            # Validate structure
            if not isinstance(merged, dict):
                logger.error(f"Expected dict, got {type(merged)}")
                return {"fp": [], "tp": []}

            if "fp" not in merged:
                merged["fp"] = []
            if "tp" not in merged:
                merged["tp"] = []

            return merged

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse merge JSON: {e}")
            return {"fp": [], "tp": []}


def main():
    """Test pattern merger."""
    import argparse

    parser = argparse.ArgumentParser(description="Test pattern merging")
    parser.add_argument("pattern_files", nargs="+", type=Path,
                       help="Pattern JSON files from different folds")
    parser.add_argument("issue_type", type=str, help="Issue type")
    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="nim", help="LLM platform")
    parser.add_argument("--output", "-o", type=Path, help="Output merged patterns file")

    args = parser.parse_args()

    # Load pattern files
    fold_patterns = []
    for pattern_file in args.pattern_files:
        with open(pattern_file, 'r') as f:
            patterns = json.load(f)
            fold_patterns.append(patterns)

    # Merge
    merger = PatternMerger(platform=args.platform)
    merged = merger.merge_patterns(fold_patterns, args.issue_type)

    print("\n" + "="*80)
    print("MERGED PATTERNS")
    print("="*80)
    print(json.dumps(merged, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(merged, f, indent=2)
        logger.info(f"Saved merged patterns to: {args.output}")


if __name__ == "__main__":
    main()