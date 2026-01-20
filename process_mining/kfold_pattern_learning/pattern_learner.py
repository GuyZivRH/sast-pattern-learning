#!/usr/bin/env python3
"""
Pattern Learner - LLM-based Pattern Extraction

Generates concise FP/TP patterns from human-annotated SAST findings
using LLM (via OpenAI-compatible API).

Supports:
- NVIDIA NIM platform
- Local vLLM deployment
- Issue-type-specific pattern learning
- Batch processing of validation entries
"""

import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.parsers import (
    ValidationEntryParser,
    ValidationEntry
)
from process_mining.core.classifiers import (
    PatternBasedClassifier,
    PLATFORM_CONFIGS
)
from process_mining.kfold_pattern_learning.config import (
    SAMPLING_DEFAULTS,
    LLM_DEFAULTS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Learns concise FP/TP patterns from annotated SAST findings using LLM.

    Uses the same LLM infrastructure as PatternBasedClassifier but with
    a different prompt template focused on pattern extraction rather than
    classification.
    """

    def __init__(
        self,
        platform: str = LLM_DEFAULTS['platform'],
        model: Optional[str] = None,
        max_tokens: int = LLM_DEFAULTS['max_tokens'],
        temperature: float = LLM_DEFAULTS['temperature'],
        prompt_template_path: Optional[Path] = None
    ):
        """
        Initialize pattern learner.

        Args:
            platform: LLM platform ("local" or "nim")
            model: Optional model override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)
            prompt_template_path: Path to prompt template (default: auto-detect)
        """
        if platform not in PLATFORM_CONFIGS:
            raise ValueError(f"Unknown platform: {platform}")

        self.platform = platform
        config = PLATFORM_CONFIGS[platform]
        self.model = model or config["model"]
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Reuse PatternBasedClassifier's LLM client infrastructure
        self.classifier = PatternBasedClassifier(
            platform=platform,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            baseline_mode=True  # Don't load patterns
        )

        # Load prompt template
        if prompt_template_path is None:
            prompt_template_path = Path(__file__).parent.parent / "prompts" / "kfold_pattern_learning_prompt.md"

        if not prompt_template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()

        logger.info(f"Initialized PatternLearner: {platform} / {self.model}")
        logger.info(f"Loaded prompt template from: {prompt_template_path}")

    def learn_patterns(
        self,
        train_files: List[Path],
        issue_type: str,
        max_tp_per_batch: int = SAMPLING_DEFAULTS['max_tp_per_batch'],
        max_fp_per_batch: int = SAMPLING_DEFAULTS['max_fp_per_batch']
    ) -> Dict:
        """
        Learn patterns from training files for a specific issue type.

        Uses balanced sampling: 3 TP + 3 FP to ensure both classes are represented.

        Args:
            train_files: List of .txt validation files
            issue_type: Issue type to learn patterns for (e.g., "RESOURCE_LEAK")
            max_tp_per_batch: Max TP entries to send to LLM (default: 3)
            max_fp_per_batch: Max FP entries to send to LLM (default: 3)

        Returns:
            Dictionary with {"fp": [...], "tp": [...]} pattern structure
        """
        logger.info(f"Learning patterns for {issue_type}...")
        logger.info(f"  Processing {len(train_files)} training files")
        logger.info(f"  Balanced sampling: {max_tp_per_batch} TP + {max_fp_per_batch} FP")
        print(f"DEBUG [{issue_type}]: Starting pattern learning with {len(train_files)} files", flush=True)

        # Parse all entries from training files
        parser = ValidationEntryParser()
        all_entries = []

        # Report first few files being parsed
        file_names = [f.stem for f in train_files[:5]]
        if len(train_files) > 5:
            logger.info(f"  Parsing {len(train_files)} files: {', '.join(file_names)}... (+{len(train_files)-5} more)")
        else:
            logger.info(f"  Parsing {len(train_files)} files: {', '.join(file_names)}")

        print(f"DEBUG [{issue_type}]: Parsing {len(train_files)} files...", flush=True)
        for idx, txt_file in enumerate(train_files):
            try:
                entries = parser.parse_file(txt_file)
                all_entries.extend(entries)
                if (idx + 1) % 50 == 0:
                    print(f"DEBUG [{issue_type}]: Parsed {idx + 1}/{len(train_files)} files, {len(all_entries)} entries so far", flush=True)
            except Exception as e:
                logger.warning(f"Failed to parse {txt_file.name}: {e}")
                continue

        print(f"DEBUG [{issue_type}]: File parsing complete, {len(all_entries)} total entries", flush=True)

        # Filter to specific issue type
        issue_entries = [e for e in all_entries if e.issue_type == issue_type]

        if not issue_entries:
            logger.warning(f"No entries found for issue type: {issue_type}")
            print(f"DEBUG [{issue_type}]: No entries found - returning empty patterns", flush=True)
            return {"fp": [], "tp": []}

        logger.info(f"  Found {len(issue_entries)} total entries for {issue_type}")
        print(f"DEBUG [{issue_type}]: Found {len(issue_entries)} entries for this issue type", flush=True)

        # Separate TP and FP
        tp_entries = [e for e in issue_entries if 'TRUE' in e.ground_truth_classification]
        fp_entries = [e for e in issue_entries if 'FALSE' in e.ground_truth_classification]

        logger.info(f"  Available: {len(tp_entries)} TP, {len(fp_entries)} FP")

        # Process ALL data in batches
        # Strategy: TP used once (with replacement if < 3), FP in sliding batches of 3
        import random

        all_patterns = {"fp": [], "tp": []}

        # Handle TP: process once (they're typically fewer)
        sampled_tp = self._sample_with_replacement(tp_entries, max_tp_per_batch)

        # Process FP in batches of 3
        fp_batches = []
        fp_idx = 0
        while fp_idx < len(fp_entries):
            batch = fp_entries[fp_idx:fp_idx + max_fp_per_batch]

            # If batch < 3, sample with replacement to reach 3
            if len(batch) < max_fp_per_batch and len(fp_entries) > 0:
                extras_needed = max_fp_per_batch - len(batch)
                extras = random.choices(fp_entries, k=extras_needed)
                batch = batch + extras

            fp_batches.append(batch)
            fp_idx += max_fp_per_batch

        # If no FP entries, create one empty batch to still extract TP patterns
        if not fp_batches:
            fp_batches = [[]]

        logger.info(f"  Processing {len(fp_batches)} batches (TP used in each)")

        # Process each batch
        print(f"DEBUG [{issue_type}]: Processing {len(fp_batches)} batches...", flush=True)
        for batch_idx, fp_batch in enumerate(fp_batches):
            # Combine TP + FP for this batch
            batch_entries = sampled_tp + fp_batch
            random.shuffle(batch_entries)

            logger.info(f"  Batch {batch_idx + 1}/{len(fp_batches)}: {len(sampled_tp)} TP + {len(fp_batch)} FP")
            print(f"DEBUG [{issue_type}]: Batch {batch_idx + 1}/{len(fp_batches)} - {len(sampled_tp)} TP + {len(fp_batch)} FP", flush=True)

            # Build learning prompt
            prompt = self._build_learning_prompt(batch_entries, issue_type)
            print(f"DEBUG [{issue_type}]: Built prompt, length: {len(prompt)} chars", flush=True)

            # Call LLM
            try:
                logger.info(f"    Calling LLM to generate patterns...")
                print(f"DEBUG [{issue_type}]: Calling LLM API (platform: {self.classifier.platform})...", flush=True)

                # Respect rate limits
                self.classifier._wait_for_rate_limit()

                # Platform-aware API call
                if self.classifier.platform == "vertex":
                    print(f"DEBUG [{issue_type}]: Making Vertex AI API call...", flush=True)
                    response = self.classifier.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_text = response.content[0].text
                    print(f"DEBUG [{issue_type}]: Vertex AI response received, length: {len(response_text)} chars", flush=True)
                else:
                    # OpenAI-compatible API (local, nim)
                    print(f"DEBUG [{issue_type}]: Making OpenAI-compatible API call...", flush=True)
                    response = self.classifier.client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_text = response.choices[0].message.content
                    print(f"DEBUG [{issue_type}]: API response received, length: {len(response_text)} chars", flush=True)

                if response_text is None:
                    raise ValueError("LLM returned empty response")

                # Parse response
                batch_patterns = self._parse_pattern_response(response_text, issue_type)

                # Accumulate patterns
                all_patterns["fp"].extend(batch_patterns.get("fp", []))
                all_patterns["tp"].extend(batch_patterns.get("tp", []))

                logger.info(f"    Generated {len(batch_patterns.get('fp', []))} FP, {len(batch_patterns.get('tp', []))} TP patterns")
                print(f"DEBUG [{issue_type}]: Batch {batch_idx + 1} complete - {len(batch_patterns.get('fp', []))} FP, {len(batch_patterns.get('tp', []))} TP patterns", flush=True)

            except Exception as e:
                logger.error(f"    Error calling LLM for batch {batch_idx + 1}: {e}")
                print(f"DEBUG [{issue_type}]: ERROR in batch {batch_idx + 1}: {e}", flush=True)
                continue

        # Deduplicate and renumber patterns
        patterns = self._deduplicate_patterns(all_patterns, issue_type)

        logger.info(f"  Generated {len(patterns.get('fp', []))} FP patterns, "
                   f"{len(patterns.get('tp', []))} TP patterns")

        return patterns

    def _sample_with_replacement(
        self,
        entries: List[ValidationEntry],
        target_size: int
    ) -> List[ValidationEntry]:
        """
        Sample entries with replacement if insufficient.

        If entries < target_size, take all + sample remainder with replacement.
        Otherwise, sample without replacement.

        Args:
            entries: List of entries to sample from
            target_size: Target number of samples

        Returns:
            List of sampled entries
        """
        import random

        if not entries:
            return []

        if len(entries) >= target_size:
            # Enough entries - sample without replacement
            return random.sample(entries, target_size)
        else:
            # Insufficient entries - take all + sample extras with replacement
            all_entries = entries.copy()
            remainder = target_size - len(entries)
            sampled_extra = random.choices(entries, k=remainder)  # with replacement
            logger.info(f"    Used sampling with replacement: {len(entries)} → {target_size} (added {remainder} duplicates)")
            return all_entries + sampled_extra

    def _build_learning_prompt(
        self,
        entries: List[ValidationEntry],
        issue_type: str
    ) -> str:
        """
        Build prompt for pattern learning.

        Args:
            entries: List of validation entries for one issue type
            issue_type: The issue type being analyzed

        Returns:
            Complete prompt string
        """
        # Format entries as in validation TXT format
        entries_text = []
        for idx, entry in enumerate(entries, 1):
            entry_block = f"""---
Entry #{idx}:
Package: {entry.package_name}
Issue Type: {entry.issue_type}
CWE: {entry.cwe}
File: {entry.file_path}:{entry.line_number}

Error Trace:
{entry.error_trace}

Source Code:
```c
{entry.source_code}
```

Ground Truth Classification: {entry.ground_truth_classification}
Human Expert Justification: {entry.ground_truth_justification}"""

            if entry.analyst_comment and entry.analyst_comment.strip():
                entry_block += f"\nAnalyst Comment: {entry.analyst_comment}"

            entry_block += "\n--------------------------------------------\n"
            entries_text.append(entry_block)

        # Replace placeholders in template
        prompt = self.prompt_template.replace("{issue_type}", issue_type)
        prompt += "\n\n## Provided Entries\n\n"
        prompt += "\n".join(entries_text)
        prompt += "\n\n## Your Task\n\n"
        prompt += f"Analyze the {len(entries)} entries above and generate concise FP/TP patterns in JSON format.\n"
        prompt += "Remember: Output ONLY the JSON block with no additional text.\n"

        return prompt

    def _deduplicate_patterns(self, patterns: Dict, issue_type: str) -> Dict:
        """
        Deduplicate patterns and renumber IDs.

        Args:
            patterns: Dictionary with {"fp": [...], "tp": [...]}
            issue_type: Issue type for ID prefix

        Returns:
            Deduplicated patterns with sequential IDs
        """
        deduplicated = {"fp": [], "tp": []}

        for category in ["fp", "tp"]:
            seen_summaries = set()
            category_prefix = "FP" if category == "fp" else "TP"

            for idx, pattern in enumerate(patterns.get(category, []), 1):
                # Simple deduplication by summary text
                summary = pattern.get("summary", "")
                if summary and summary not in seen_summaries:
                    seen_summaries.add(summary)

                    # Renumber pattern ID
                    pattern["pattern_id"] = f"{issue_type}-{category_prefix}-{idx:03d}"
                    deduplicated[category].append(pattern)

        return deduplicated

    def _parse_pattern_response(
        self,
        response_text: str,
        issue_type: str
    ) -> Dict:
        """
        Parse LLM response to extract patterns.

        Args:
            response_text: Raw LLM response
            issue_type: Issue type (for validation)

        Returns:
            Dictionary with {"fp": [...], "tp": [...]}
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                logger.error("No JSON found in LLM response")
                logger.debug(f"Response: {response_text[:500]}")
                return {"fp": [], "tp": []}

        try:
            patterns = json.loads(json_str)

            # Validate structure
            if not isinstance(patterns, dict):
                logger.error(f"Expected dict, got {type(patterns)}")
                return {"fp": [], "tp": []}

            if "fp" not in patterns:
                logger.warning("No 'fp' key in patterns, adding empty list")
                patterns["fp"] = []

            if "tp" not in patterns:
                logger.warning("No 'tp' key in patterns, adding empty list")
                patterns["tp"] = []

            # Validate pattern structure
            for category in ["fp", "tp"]:
                if not isinstance(patterns[category], list):
                    logger.error(f"'{category}' should be a list")
                    patterns[category] = []

                for i, pattern in enumerate(patterns[category]):
                    if not isinstance(pattern, dict):
                        logger.warning(f"Pattern {i} in '{category}' is not a dict, skipping")
                        continue

                    # Ensure required fields
                    required_fields = ["pattern_id", "group", "summary"]
                    for field in required_fields:
                        if field not in pattern:
                            logger.warning(f"Pattern {i} in '{category}' missing '{field}' field")

            return patterns

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"JSON string: {json_str[:500]}")
            return {"fp": [], "tp": []}


def main():
    """Test pattern learning on a small sample."""
    import argparse

    parser = argparse.ArgumentParser(description="Test pattern learning")
    parser.add_argument("train_dir", type=Path, help="Directory with training .txt files")
    parser.add_argument("issue_type", type=str, help="Issue type to learn patterns for")
    parser.add_argument("--platform", "-p", choices=list(PLATFORM_CONFIGS.keys()),
                       default="nim", help="LLM platform")
    parser.add_argument("--max-entries", type=int, default=20,
                       help="Max entries to use (for testing)")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")

    args = parser.parse_args()

    learner = PatternLearner(platform=args.platform)

    train_files = sorted(args.train_dir.glob("*.txt"))
    if not train_files:
        logger.error(f"No .txt files found in {args.train_dir}")
        return

    patterns = learner.learn_patterns(
        train_files=train_files,
        issue_type=args.issue_type,
        max_entries_per_issue_type=args.max_entries
    )

    print("\n" + "="*80)
    print("GENERATED PATTERNS")
    print("="*80)
    print(json.dumps(patterns, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(patterns, f, indent=2)
        logger.info(f"Saved patterns to: {args.output}")


if __name__ == "__main__":
    main()