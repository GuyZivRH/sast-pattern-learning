"""
Unit tests for ValidationEntry and ValidationEntryParser.

Tests entry parsing and masking including:
- Parsing validation .txt files
- ValidationEntry creation
- Ground truth masking
- Multiple entries per file
"""
import pytest
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.core.data_models import ValidationEntry
from process_mining.core.parsers import ValidationEntryParser


class TestValidationEntry:
    """Test suite for ValidationEntry."""

    def test_init(self):
        """Test ValidationEntry initialization."""
        entry = ValidationEntry(
            entry_id="test/file.c:100-RESOURCE_LEAK-001",
            package_name="test-pkg",
            issue_type="RESOURCE_LEAK",
            cwe="CWE-401",
            error_trace="test trace",
            source_code="test code",
            ground_truth_classification="TRUE_POSITIVE",
            ground_truth_justification="test justification"
        )

        assert entry.entry_id == "test/file.c:100-RESOURCE_LEAK-001"
        assert entry.package_name == "test-pkg"
        assert entry.issue_type == "RESOURCE_LEAK"
        assert entry.ground_truth_classification == "TRUE_POSITIVE"

    def test_to_dict_includes_ground_truth(self):
        """Test to_dict includes ground truth."""
        entry = ValidationEntry(
            entry_id="test-id",
            package_name="test-pkg",
            issue_type="RESOURCE_LEAK",
            cwe="CWE-401",
            error_trace="trace",
            source_code="code",
            ground_truth_classification="TRUE_POSITIVE",
            ground_truth_justification="justification"
        )

        d = entry.to_dict()

        assert "ground_truth_classification" in d
        assert "ground_truth_justification" in d
        assert d["ground_truth_classification"] == "TRUE_POSITIVE"

    def test_get_masked_entry_excludes_ground_truth(self):
        """Test get_masked_entry excludes ground truth fields."""
        entry = ValidationEntry(
            entry_id="test-id",
            package_name="test-pkg",
            issue_type="RESOURCE_LEAK",
            cwe="CWE-401",
            error_trace="trace",
            source_code="code",
            ground_truth_classification="TRUE_POSITIVE",
            ground_truth_justification="This is the answer"
        )

        masked = entry.get_masked_entry()

        # Should include these fields
        assert "entry_id" in masked
        assert "package_name" in masked
        assert "issue_type" in masked
        assert "cwe" in masked
        assert "error_trace" in masked
        assert "source_code" in masked

        # Should NOT include ground truth
        assert "ground_truth_classification" not in masked
        assert "ground_truth_justification" not in masked

    def test_get_masked_entry_preserves_data_integrity(self):
        """Test masked entry preserves all non-GT data."""
        entry = ValidationEntry(
            entry_id="test-id",
            package_name="test-pkg",
            issue_type="RESOURCE_LEAK",
            cwe="CWE-401",
            error_trace="complex trace here",
            source_code="int main() { return 0; }",
            ground_truth_classification="TRUE_POSITIVE",
            ground_truth_justification="secret answer"
        )

        masked = entry.get_masked_entry()

        # Verify data is intact
        assert masked["entry_id"] == "test-id"
        assert masked["package_name"] == "test-pkg"
        assert masked["issue_type"] == "RESOURCE_LEAK"
        assert masked["error_trace"] == "complex trace here"
        assert masked["source_code"] == "int main() { return 0; }"


class TestValidationEntryParser:
    """Test suite for ValidationEntryParser."""

    def test_parse_single_entry(self, temp_dir):
        """Test parsing a file with single entry."""
        content = """================================================================================
GROUND-TRUTH ENTRIES FOR: test-pkg
================================================================================

Package: test-pkg
Total Entries: 1

---
Entry #1:
Issue Type: RESOURCE_LEAK
CWE: CWE-401

Error Trace:
test.c:100:2: test trace

Source Code (test.c):
```c
int test() {
    return 0;
}
```

Ground Truth Classification: TRUE_POSITIVE

Human Expert Justification: test justification
"""
        test_file = temp_dir / "test.txt"
        test_file.write_text(content)

        parser = ValidationEntryParser()
        entries = parser.parse_file(test_file)

        assert len(entries) == 1
        assert entries[0].package_name == "test-pkg"
        assert entries[0].issue_type == "RESOURCE_LEAK"
        assert entries[0].ground_truth_classification == "TRUE_POSITIVE"

    def test_parse_multiple_entries(self, sample_validation_file):
        """Test parsing a file with multiple entries."""
        parser = ValidationEntryParser()
        entries = parser.parse_file(sample_validation_file)

        # Sample file has 2 entries
        assert len(entries) == 2
        assert entries[0].entry_id == "test-package_entry_1"
        assert entries[1].entry_id == "test-package_entry_2"
        assert entries[0].issue_type == "RESOURCE_LEAK"
        assert entries[1].issue_type == "RESOURCE_LEAK"

    def test_parse_preserves_order(self, sample_validation_file):
        """Test that entries are parsed in order."""
        parser = ValidationEntryParser()
        entries = parser.parse_file(sample_validation_file)

        # First entry should be entry_1, second should be entry_2
        assert entries[0].entry_id == "test-package_entry_1"
        assert entries[1].entry_id == "test-package_entry_2"

    def test_parse_directory(self, sample_train_val_test_dirs):
        """Test parsing all files in a directory."""
        parser = ValidationEntryParser()
        entries = parser.parse_directory(sample_train_val_test_dirs['train'])

        # Should find entries from all files in train directory
        assert len(entries) > 0

        # All entries should be ValidationEntry objects
        assert all(isinstance(e, ValidationEntry) for e in entries)

    def test_parse_filters_by_issue_type(self, sample_validation_file):
        """Test filtering entries by issue type."""
        parser = ValidationEntryParser()
        all_entries = parser.parse_file(sample_validation_file)

        # Filter to RESOURCE_LEAK
        leak_entries = [e for e in all_entries if e.issue_type == "RESOURCE_LEAK"]

        # All sample entries are RESOURCE_LEAK
        assert len(leak_entries) == len(all_entries)

    def test_parse_handles_missing_file(self):
        """Test parser handles missing file gracefully."""
        parser = ValidationEntryParser()

        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.txt"))

    def test_parse_skips_entries_without_source_code(self, temp_dir):
        """Test parser skips entries missing source code."""
        content = """================================================================================
GROUND-TRUTH ENTRIES FOR: test-pkg
================================================================================

Package: test-pkg
Total Entries: 2

---
Entry #1:
Issue Type: RESOURCE_LEAK
CWE: CWE-401

Error Trace:
test.c:100:2: test trace

Source Code (test.c):
```c
source code not available
```

Ground Truth Classification: TRUE_POSITIVE

Human Expert Justification: test justification

---
Entry #2:
Issue Type: RESOURCE_LEAK
CWE: CWE-401

Error Trace:
test.c:200:2: test trace

Source Code (test.c):
```c
void test() {
    int x = 0;
}
```

Ground Truth Classification: FALSE_POSITIVE

Human Expert Justification: test justification
"""
        test_file = temp_dir / "test.txt"
        test_file.write_text(content)

        parser = ValidationEntryParser()
        entries = parser.parse_file(test_file)

        # Should only parse the second entry (with source code)
        assert len(entries) == 1
        assert entries[0].entry_id == "test-pkg_entry_2"
        assert parser.skipped_no_source_code == 1

    def test_masked_entries_consistent_across_phases(self, sample_validation_file):
        """Test that masked entries are identical regardless of when they're created."""
        parser = ValidationEntryParser()
        entries = parser.parse_file(sample_validation_file)

        # Get masked entry twice
        masked1 = entries[0].get_masked_entry()
        masked2 = entries[0].get_masked_entry()

        # Should be identical
        assert masked1 == masked2

        # Should not contain ground truth
        assert "ground_truth_classification" not in masked1
        assert "ground_truth_justification" not in masked1