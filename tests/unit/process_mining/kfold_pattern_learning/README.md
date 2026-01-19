# Process Mining V1 Test Suite

Comprehensive unit tests for the k-fold pattern learning pipeline (`process_mining/v1/kfold_pattern_learning`).

## Test Coverage

### Core Components

1. **`test_entry_parser.py`** - ValidationEntry and ValidationEntryParser
   - Entry parsing from .txt files
   - Ground truth masking (critical for ML pipeline integrity)
   - Multi-entry file handling
   - Directory parsing

2. **`test_fold_evaluator.py`** - FoldEvaluator
   - Pattern evaluation on validation folds
   - Stratified sampling (FP/TP ratio preservation)
   - Entry file writing
   - Metrics calculation
   - Multi-issue-type evaluation

3. **`test_refinement_orchestrator.py`** - RefinementOrchestrator (Phase 2)
   - Train/val separation (no data leakage)
   - Early stopping based on val F1
   - Overfitting detection (train F1 vs val F1)
   - Pattern refinement (add/modify/remove)
   - Convergence criteria
   - Sampling parameter passing

4. **`test_stratified_kfold.py`** - StratifiedKFoldSplitter
   - Strata assignment (size × FP ratio)
   - Balanced fold distribution
   - Reproducibility with seed
   - Edge case handling

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/unit/process_mining/ -v

# With coverage
pytest tests/unit/process_mining/ --cov=process_mining.v1 --cov-report=html
```

### Run Specific Test Files

```bash
# Entry parser tests
pytest tests/unit/process_mining/test_entry_parser.py -v

# Fold evaluator tests
pytest tests/unit/process_mining/test_fold_evaluator.py -v

# Refinement orchestrator tests
pytest tests/unit/process_mining/test_refinement_orchestrator.py -v

# Stratified k-fold tests
pytest tests/unit/process_mining/test_stratified_kfold.py -v
```

### Run Specific Test Cases

```bash
# Test ground truth masking
pytest tests/unit/process_mining/test_entry_parser.py::TestValidationEntry::test_get_masked_entry_excludes_ground_truth -v

# Test stratified sampling
pytest tests/unit/process_mining/test_fold_evaluator.py::TestFoldEvaluator::test_stratified_sample_preserves_ratio -v

# Test overfitting detection
pytest tests/unit/process_mining/test_refinement_orchestrator.py::TestRefinementOrchestrator::test_overfitting_detection -v
```

### Run with Markers (if configured)

```bash
# Run only fast tests
pytest tests/unit/process_mining/ -m "not slow" -v

# Run integration-like tests
pytest tests/unit/process_mining/ -k "integration" -v
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

### Data Fixtures
- `sample_validation_entry_text` - Sample .txt file content
- `sample_validation_file` - Temporary .txt file
- `sample_patterns` - Pattern dictionary
- `sample_train_val_test_dirs` - Complete train/val/test structure

### Mock Fixtures
- `mock_llm_response` - Mock LLM pattern generation
- `mock_pattern_learner` - Mock PatternLearner
- `mock_fold_evaluator` - Mock FoldEvaluator
- `mock_pattern_refiner` - Mock PatternRefiner

### Phase Result Fixtures
- `sample_phase1_results` - Complete Phase 1 output
- `sample_phase2_results` - Complete Phase 2 output
- `sample_evaluation_result` - Pattern evaluation metrics

## Critical Tests for ML Pipeline Integrity

### 1. Ground Truth Masking
**Why Critical**: Ensures LLM never sees answers during inference.

```bash
pytest tests/unit/process_mining/test_entry_parser.py::TestValidationEntry::test_get_masked_entry_excludes_ground_truth -v
```

**What it checks**:
- `ground_truth_classification` excluded from masked entry
- `ground_truth_justification` excluded from masked entry
- All other data preserved

### 2. Train/Val Separation
**Why Critical**: Prevents data leakage in Phase 2 refinement.

```bash
pytest tests/unit/process_mining/test_refinement_orchestrator.py::TestRefinementOrchestrator::test_refine_issue_type_convergence -v
```

**What it checks**:
- Train data used for refinement only
- Val data used for early stopping only
- No cross-contamination

### 3. Stratified Sampling
**Why Critical**: Maintains representative metrics when sampling.

```bash
pytest tests/unit/process_mining/test_fold_evaluator.py::TestFoldEvaluator::test_stratified_sample_preserves_ratio -v
```

**What it checks**:
- FP/TP ratio preserved (±10% tolerance)
- Sample size correct
- Edge cases handled

### 4. Overfitting Detection
**Why Critical**: Identifies when patterns memorize train data.

```bash
pytest tests/unit/process_mining/test_refinement_orchestrator.py::TestRefinementOrchestrator::test_overfitting_detection -v
```

**What it checks**:
- Train F1 vs Val F1 gap tracked
- Warning when gap > 0.1
- Convergence info includes overfitting_gap

## Test Data Requirements

Tests use mock data and temporary files. No real SAST data required.

**Sample validation entry format**:
```
Entry ID: package/file.c:line-ISSUE_TYPE-NNN
Package Name: package-name
Issue Type: RESOURCE_LEAK
CWE: CWE-401
Error Trace: [trace here]
Source Code: [code here]
Ground Truth Classification: TRUE_POSITIVE
Ground Truth Justification: [justification here]
---
```

## Expected Test Runtime

| Test File | Tests | Runtime |
|-----------|-------|---------|
| test_entry_parser.py | 15 | ~2s |
| test_fold_evaluator.py | 10 | ~3s |
| test_refinement_orchestrator.py | 8 | ~4s |
| test_stratified_kfold.py | 12 | ~3s |
| **Total** | **45** | **~12s** |

## Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/tests.yml
- name: Run Process Mining Tests
  run: |
    pytest tests/unit/process_mining/ \
      --cov=process_mining.v1 \
      --cov-report=xml \
      --junitxml=test-results/process_mining.xml
```

## Adding New Tests

### Test File Template

```python
"""
Unit tests for NewComponent.

Tests new component including:
- Feature 1
- Feature 2
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from process_mining.v1.kfold_pattern_learning.new_component import NewComponent


class TestNewComponent:
    """Test suite for NewComponent."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        component = NewComponent()
        result = component.do_something()
        assert result is not None
```

### Using Fixtures

```python
def test_with_fixture(self, sample_validation_file, temp_dir):
    """Test using shared fixtures."""
    # sample_validation_file is ready to use
    # temp_dir is a clean temporary directory
    pass
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Ensure you're running from project root
cd /Users/gziv/Dev/sast-ai-workflow

# Install in editable mode
pip install -e .
```

### Fixture Not Found

If fixture is not found:
1. Check `conftest.py` in same directory
2. Check parent `conftest.py` files
3. Ensure fixture name matches exactly

### Mock Not Working

If mocks aren't intercepting calls:
1. Verify import path in `@patch` decorator
2. Check patch is applied before test runs
3. Use `patch.object()` for instance methods

## Contributing

When adding tests:
1. Follow existing naming conventions
2. Add docstrings explaining what test verifies
3. Use appropriate fixtures
4. Test both success and failure cases
5. Include edge cases
6. Update this README if adding new test files

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures guide](https://docs.pytest.org/en/stable/fixture.html)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)