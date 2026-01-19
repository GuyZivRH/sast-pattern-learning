"""
Unit tests for PatternRefiner.

Tests pattern refinement including:
- JSON parsing from LLM responses (both fenced and raw)
- Fallback handling for malformed JSON
- Refinement structure validation
"""
import pytest
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.kfold_pattern_learning.pattern_refiner import PatternRefiner


class TestPatternRefiner:
    """Test suite for PatternRefiner."""

    def test_parse_refinement_response_fenced_json(self):
        """Test parsing JSON from fenced code blocks."""
        refiner = PatternRefiner(platform="nim")

        response = """Here are the refinements:

```json
{
  "add": [
    {
      "pattern_type": "fp",
      "pattern_id": "FP-003",
      "group": "C: New Group",
      "summary": "New FP pattern"
    }
  ],
  "modify": [],
  "remove": []
}
```

That's all!"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        assert 'add' in result
        assert 'modify' in result
        assert 'remove' in result
        assert len(result['add']) == 1
        assert result['add'][0]['pattern_id'] == 'FP-003'

    def test_parse_refinement_response_fenced_json_no_lang_tag(self):
        """Test parsing JSON from fenced code blocks without json tag."""
        refiner = PatternRefiner(platform="nim")

        response = """```
{
  "add": [],
  "modify": [
    {
      "pattern_id": "TP-001",
      "new_summary": "Updated summary"
    }
  ],
  "remove": []
}
```"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        assert 'modify' in result
        assert len(result['modify']) == 1
        assert result['modify'][0]['pattern_id'] == 'TP-001'

    def test_parse_refinement_response_raw_json(self):
        """Test parsing raw JSON (fallback when no fenced block)."""
        refiner = PatternRefiner(platform="nim")

        response = """{
  "add": [],
  "modify": [],
  "remove": [
    {
      "pattern_id": "FP-002"
    }
  ]
}"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        assert 'remove' in result
        assert len(result['remove']) == 1
        assert result['remove'][0]['pattern_id'] == 'FP-002'

    def test_parse_refinement_response_no_json(self):
        """Test handling of response with no JSON."""
        refiner = PatternRefiner(platform="nim")

        response = "Sorry, I cannot provide refinements right now."

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        # Should return empty refinements structure
        assert result == {
            "add": [],
            "modify": [],
            "remove": [],
            "refinement_metadata": {}
        }

    def test_parse_refinement_response_malformed_json(self):
        """Test handling of malformed JSON."""
        refiner = PatternRefiner(platform="nim")

        response = """```json
{
  "add": [
    {"pattern_id": "FP-003", "group": "C"}  // missing closing bracket
}
```"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        # Should return empty refinements on parse error
        assert result == {
            "add": [],
            "modify": [],
            "remove": [],
            "refinement_metadata": {}
        }

    def test_parse_refinement_response_missing_fields(self):
        """Test that missing add/modify/remove fields are added."""
        refiner = PatternRefiner(platform="nim")

        # Response with only 'add' field
        response = """```json
{
  "add": [
    {
      "pattern_type": "tp",
      "pattern_id": "TP-005",
      "group": "D",
      "summary": "Test"
    }
  ]
}
```"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        # Should have all required fields
        assert 'add' in result
        assert 'modify' in result
        assert 'remove' in result
        assert result['modify'] == []
        assert result['remove'] == []

    def test_parse_refinement_response_adds_metadata(self):
        """Test that refinement_metadata is added with counts."""
        refiner = PatternRefiner(platform="nim")

        response = """```json
{
  "add": [{"pattern_type": "fp", "pattern_id": "FP-1", "group": "A", "summary": "x"}],
  "modify": [{"pattern_id": "TP-1", "new_summary": "y"}, {"pattern_id": "TP-2", "new_summary": "z"}],
  "remove": []
}
```"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        assert 'refinement_metadata' in result
        assert result['refinement_metadata']['patterns_added'] == 1
        assert result['refinement_metadata']['patterns_modified'] == 2
        assert result['refinement_metadata']['patterns_removed'] == 0

    def test_parse_refinement_response_non_dict(self):
        """Test handling when JSON is not a dict."""
        refiner = PatternRefiner(platform="nim")

        response = """```json
["not", "a", "dict"]
```"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        # Should return empty refinements for non-dict
        assert result == {
            "add": [],
            "modify": [],
            "remove": [],
            "refinement_metadata": {}
        }

    def test_parse_refinement_response_complex_json(self):
        """Test parsing complex refinement with all action types."""
        refiner = PatternRefiner(platform="nim")

        response = """```json
{
  "add": [
    {
      "pattern_type": "fp",
      "pattern_id": "RESOURCE_LEAK-FP-010",
      "group": "E: Complex Case",
      "summary": "Resource freed in destructor"
    }
  ],
  "modify": [
    {
      "pattern_id": "RESOURCE_LEAK-TP-003",
      "new_summary": "Updated: Resource leaked on error path"
    }
  ],
  "remove": [
    {
      "pattern_id": "RESOURCE_LEAK-FP-002"
    }
  ],
  "refinement_metadata": {
    "reason": "Improved pattern coverage"
  }
}
```"""

        result = refiner._parse_refinement_response(response, "RESOURCE_LEAK")

        assert len(result['add']) == 1
        assert len(result['modify']) == 1
        assert len(result['remove']) == 1
        assert result['add'][0]['pattern_id'] == 'RESOURCE_LEAK-FP-010'
        assert result['modify'][0]['new_summary'].startswith('Updated:')
        assert result['remove'][0]['pattern_id'] == 'RESOURCE_LEAK-FP-002'
        assert 'refinement_metadata' in result