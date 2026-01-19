from typing import Dict


class ValidationEntry:
    """Represents a single SAST finding entry from validation data."""

    def __init__(
        self,
        entry_id: str,
        package_name: str,
        issue_type: str,
        cwe: str,
        error_trace: str,
        source_code: str,
        ground_truth_classification: str,
        ground_truth_justification: str,
        analyst_comment: str = "",
        file_path: str = "",
        line_number: int = 0
    ):
        self.entry_id = entry_id
        self.package_name = package_name
        self.issue_type = issue_type
        self.cwe = cwe
        self.error_trace = error_trace
        self.source_code = source_code
        self.ground_truth_classification = ground_truth_classification
        self.ground_truth_justification = ground_truth_justification
        self.analyst_comment = analyst_comment
        self.file_path = file_path
        self.line_number = line_number

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "entry_id": self.entry_id,
            "package_name": self.package_name,
            "issue_type": self.issue_type,
            "cwe": self.cwe,
            "error_trace": self.error_trace,
            "source_code": self.source_code,
            "ground_truth_classification": self.ground_truth_classification,
            "ground_truth_justification": self.ground_truth_justification,
            "analyst_comment": self.analyst_comment,
            "file_path": self.file_path,
            "line_number": self.line_number
        }

    def get_masked_entry(self) -> Dict:
        """
        Get entry with ground truth masked for LLM classification.

        Returns entry without classification and justification - only the
        information an LLM would use to make a prediction.
        """
        return {
            "entry_id": self.entry_id,
            "package_name": self.package_name,
            "issue_type": self.issue_type,
            "cwe": self.cwe,
            "error_trace": self.error_trace,
            "source_code": self.source_code,
            "file_path": self.file_path,
            "line_number": self.line_number
        }


class ClassificationResult:
    """Result of LLM classification for a single entry."""

    def __init__(
        self,
        predicted_classification: str,
        predicted_justification: str,
        cited_patterns: list,
        exploitability_assessment: str = "",
        raw_response: str = ""
    ):
        self.predicted_classification = predicted_classification
        self.predicted_justification = predicted_justification
        self.cited_patterns = cited_patterns
        self.exploitability_assessment = exploitability_assessment
        self.raw_response = raw_response

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "predicted_classification": self.predicted_classification,
            "predicted_justification": self.predicted_justification,
            "cited_patterns": self.cited_patterns,
            "exploitability_assessment": self.exploitability_assessment
        }