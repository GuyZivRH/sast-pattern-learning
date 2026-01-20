#!/usr/bin/env python3
"""
Pattern Refiner - LLM-based Pattern Refinement

Analyzes misclassified examples and generates refinement suggestions:
- Add new patterns for uncovered cases
- Modify existing patterns for clarity/accuracy
- Remove low-utility patterns
"""

import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.classifiers import (
    PatternBasedClassifier,
    PLATFORM_CONFIGS
)
from process_mining.core.parsers import ValidationEntry
from process_mining.kfold_pattern_learning.config import LLM_DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PatternRefiner:
    """
    Refines patterns based on misclassified examples.

    Analyzes false negatives and false positives from pattern evaluation
    and generates LLM-based refinement suggestions.
    """

    REFINEMENT_PROMPT_TEMPLATE = """# Pattern Refinement Task

You are refining patterns for issue type: **{issue_type}**

## Current Patterns

### False Positive (FP) Patterns
```json
{fp_patterns}
```

### True Positive (TP) Patterns
```json
{tp_patterns}
```

## Misclassified Examples

We have {total_misclassified} misclassified examples:
- {fn_count} False Negatives (real vulnerabilities missed)
- {fp_missed_count} False Positives (safe code incorrectly flagged)

### False Negatives (Real Vulnerabilities Missed)
{fn_examples}

### False Positives Not Caught (Safe Code Incorrectly Flagged)
{fp_missed_examples}

## Your Task

Analyze the misclassified examples and suggest refinements:

### Refinement Actions

1. **ADD**: Create new patterns to cover misclassified cases not addressed by existing patterns
2. **MODIFY**: Update existing patterns to improve clarity, accuracy, or coverage
3. **REMOVE**: Remove patterns that have low utility or are redundant

### Guidelines

- **For False Negatives**: Add TP patterns describing the vulnerability patterns we're missing
- **For FP Missed**: Add FP patterns describing safe code idioms we're not recognizing
- **For Modifications**: Improve pattern descriptions to be clearer and more precise
- **For Removals**: Only suggest if a pattern is truly redundant or misleading

### Quality Criteria

- New patterns must be DISTINCT from existing ones
- Modified patterns must preserve the pattern_id
- Each pattern should be concise (3-4 sentences max)
- Focus on actionable code patterns with clear indicators

## Output Format

Generate a JSON file with this EXACT structure:

```json
{{
  "add": [
    {{
      "pattern_type": "fp" or "tp",
      "pattern_id": "{issue_type}-FP-XXX" or "{issue_type}-TP-XXX",
      "group": "X: Category Name",
      "summary": "Pattern description...",
      "justification": "Why this pattern is needed based on misclassified examples"
    }}
  ],
  "modify": [
    {{
      "pattern_id": "{issue_type}-FP-001",
      "new_summary": "Updated pattern description...",
      "justification": "Why this modification improves the pattern"
    }}
  ],
  "remove": [
    {{
      "pattern_id": "{issue_type}-TP-002",
      "justification": "Why this pattern should be removed"
    }}
  ],
  "refinement_metadata": {{
    "total_misclassified_analyzed": {total_misclassified},
    "fn_count": {fn_count},
    "fp_missed_count": {fp_missed_count},
    "patterns_added": <count>,
    "patterns_modified": <count>,
    "patterns_removed": <count>,
    "summary": "Brief explanation of refinement strategy"
  }}
}}
```

## Important Notes

- Output ONLY the JSON block
- Ensure valid JSON syntax
- Pattern IDs for new patterns must follow sequential numbering
- Be conservative with removals - only remove truly redundant patterns
- Justify all refinement actions based on misclassified examples
- If no refinements are needed, return empty arrays for add/modify/remove
"""

    def __init__(
        self,
        platform: str = LLM_DEFAULTS['platform'],
        model: Optional[str] = None,
        max_tokens: int = LLM_DEFAULTS['max_tokens'],
        temperature: float = LLM_DEFAULTS['temperature']
    ):
        """
        Initialize pattern refiner.

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

        logger.info(f"Initialized PatternRefiner: {platform} / {self.model}")

    def refine_patterns(
        self,
        current_patterns: Dict,
        misclassified_entries: List[ValidationEntry],
        evaluation_results: List[Dict],
        issue_type: str,
        max_examples_per_type: int = 20
    ) -> Dict:
        """
        Generate pattern refinement suggestions.

        Args:
            current_patterns: Current pattern dict {"fp": [...], "tp": [...]}
            misclassified_entries: List of misclassified ValidationEntry objects
            evaluation_results: List of evaluation results with predictions
            issue_type: Issue type being refined
            max_examples_per_type: Max examples to include per type (FN/FP)

        Returns:
            Refinement actions: {"add": [...], "modify": [...], "remove": [...], "refinement_metadata": {...}}
        """
        logger.info(f"Refining patterns for {issue_type} based on {len(misclassified_entries)} misclassified examples...")

        # Categorize misclassifications
        fn_entries, fp_missed_entries = self._categorize_misclassifications(
            misclassified_entries,
            evaluation_results
        )

        logger.info(f"  False Negatives: {len(fn_entries)}")
        logger.info(f"  FP Missed: {len(fp_missed_entries)}")

        # Limit examples to avoid token overflow
        fn_entries = fn_entries[:max_examples_per_type]
        fp_missed_entries = fp_missed_entries[:max_examples_per_type]

        # Build refinement prompt
        prompt = self._build_refinement_prompt(
            current_patterns,
            fn_entries,
            fp_missed_entries,
            issue_type
        )

        # Call LLM
        try:
            logger.info(f"  Calling LLM to generate refinement suggestions...")

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
            refinements = self._parse_refinement_response(response_text, issue_type)

            # Log refinement summary
            metadata = refinements.get('refinement_metadata', {})
            logger.info(f"  Refinement suggestions:")
            logger.info(f"    Patterns to add: {metadata.get('patterns_added', 0)}")
            logger.info(f"    Patterns to modify: {metadata.get('patterns_modified', 0)}")
            logger.info(f"    Patterns to remove: {metadata.get('patterns_removed', 0)}")

            return refinements

        except Exception as e:
            logger.error(f"Error refining patterns for {issue_type}: {e}")
            # Fallback: return empty refinements
            logger.warning("Returning empty refinements")
            return {
                "add": [],
                "modify": [],
                "remove": [],
                "refinement_metadata": {
                    "error": str(e)
                }
            }

    def _categorize_misclassifications(
        self,
        misclassified_entries: List[ValidationEntry],
        evaluation_results: List[Dict]
    ) -> tuple[List[ValidationEntry], List[ValidationEntry]]:
        """
        Categorize misclassifications into FN and FP missed.

        Args:
            misclassified_entries: List of misclassified entries
            evaluation_results: List of evaluation results

        Returns:
            (fn_entries, fp_missed_entries)
        """
        # Create mapping from entry_id to prediction
        entry_to_prediction = {}
        for result in evaluation_results:
            result_entry_id = result.get('entry_id', '')
            predicted_classification = result.get('predicted_classification', '')
            if result_entry_id:
                entry_to_prediction[result_entry_id] = predicted_classification

        fn_entries = []
        fp_missed_entries = []

        for entry in misclassified_entries:
            ground_truth = entry.ground_truth_classification
            predicted = entry_to_prediction.get(entry.entry_id, '')

            # False Negative: Ground truth is TP, but we predicted FP
            if 'TRUE' in ground_truth and 'FALSE' in predicted:
                fn_entries.append(entry)
            # FP Missed: Ground truth is FP, but we predicted TP
            elif 'FALSE' in ground_truth and 'TRUE' in predicted:
                fp_missed_entries.append(entry)

        return fn_entries, fp_missed_entries

    def _build_refinement_prompt(
        self,
        current_patterns: Dict,
        fn_entries: List[ValidationEntry],
        fp_missed_entries: List[ValidationEntry],
        issue_type: str
    ) -> str:
        """Build prompt for pattern refinement."""
        # Format current patterns
        fp_patterns = json.dumps(current_patterns.get('fp', []), indent=2)
        tp_patterns = json.dumps(current_patterns.get('tp', []), indent=2)

        # Format FN examples
        fn_examples = self._format_examples(fn_entries, "False Negative")

        # Format FP missed examples
        fp_missed_examples = self._format_examples(fp_missed_entries, "FP Missed")

        total_misclassified = len(fn_entries) + len(fp_missed_entries)

        # Replace placeholders
        prompt = self.REFINEMENT_PROMPT_TEMPLATE.format(
            issue_type=issue_type,
            fp_patterns=fp_patterns,
            tp_patterns=tp_patterns,
            total_misclassified=total_misclassified,
            fn_count=len(fn_entries),
            fp_missed_count=len(fp_missed_entries),
            fn_examples=fn_examples,
            fp_missed_examples=fp_missed_examples
        )

        return prompt

    def _format_examples(
        self,
        entries: List[ValidationEntry],
        example_type: str
    ) -> str:
        """Format examples for prompt."""
        if not entries:
            return f"No {example_type} examples.\n"

        formatted = []
        for idx, entry in enumerate(entries, 1):
            formatted.append(f"#### Example {idx}\n")
            formatted.append(f"**Entry ID**: {entry.entry_id}\n")
            formatted.append(f"**Package**: {entry.package_name}\n")
            formatted.append(f"**File**: {entry.file_path}:{entry.line_number}\n")
            formatted.append(f"**Issue Type**: {entry.issue_type}\n")
            formatted.append(f"**CWE**: {entry.cwe}\n")
            formatted.append(f"**Ground Truth**: {entry.ground_truth_classification}\n")
            formatted.append(f"**Error Trace**:\n```\n{entry.error_trace}\n```\n")
            formatted.append(f"**Source Code**:\n```c\n{entry.source_code}\n```\n")
            formatted.append(f"**Ground Truth Justification**:\n{entry.ground_truth_justification}\n\n")

        return "\n".join(formatted)

    def _parse_refinement_response(
        self,
        response_text: str,
        issue_type: str
    ) -> Dict:
        """Parse LLM response to extract refinement actions."""
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
                logger.error("No JSON found in refinement response")
                return {"add": [], "modify": [], "remove": [], "refinement_metadata": {}}

        try:
            refinements = json.loads(json_str)

            # Validate structure
            if not isinstance(refinements, dict):
                logger.error(f"Expected dict, got {type(refinements)}")
                return {"add": [], "modify": [], "remove": [], "refinement_metadata": {}}

            # Ensure required fields
            if "add" not in refinements:
                refinements["add"] = []
            if "modify" not in refinements:
                refinements["modify"] = []
            if "remove" not in refinements:
                refinements["remove"] = []

            # Update metadata counts if missing
            if "refinement_metadata" not in refinements:
                refinements["refinement_metadata"] = {}

            refinements["refinement_metadata"].update({
                "patterns_added": len(refinements["add"]),
                "patterns_modified": len(refinements["modify"]),
                "patterns_removed": len(refinements["remove"])
            })

            return refinements

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse refinement JSON: {e}")
            return {"add": [], "modify": [], "remove": [], "refinement_metadata": {}}


def main():
    """Test pattern refiner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test pattern refinement")
    parser.add_argument("patterns_file", type=Path, help="Current patterns JSON file")
    parser.add_argument("val_dir", type=Path, help="Validation directory")
    parser.add_argument("issue_type", type=str, help="Issue type")
    parser.add_argument("--platform", "-p", choices=["local", "nim", "vertex"],
                       default="nim", help="LLM platform")
    parser.add_argument("--output", "-o", type=Path, help="Output refinements file")

    args = parser.parse_args()

    # Load patterns
    with open(args.patterns_file, 'r') as f:
        patterns = json.load(f)

    # For testing, we would need actual misclassified entries
    # This is a placeholder
    logger.info("Note: This is a test script. In real usage, pass misclassified entries from evaluation.")

    # Create dummy misclassified entries for demo
    misclassified_entries = []
    evaluation_results = []

    # Refine
    refiner = PatternRefiner(platform=args.platform)
    refinements = refiner.refine_patterns(
        current_patterns=patterns,
        misclassified_entries=misclassified_entries,
        evaluation_results=evaluation_results,
        issue_type=args.issue_type
    )

    print("\n" + "="*80)
    print("PATTERN REFINEMENTS")
    print("="*80)
    print(json.dumps(refinements, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(refinements, f, indent=2)
        logger.info(f"Saved refinements to: {args.output}")


if __name__ == "__main__":
    main()