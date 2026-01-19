# Testing Guide - Process Mining V1

Comprehensive testing setup for the k-fold pattern learning pipeline.

## Setup

### 1. Install Package in Development Mode

```bash
# From project root
cd /Users/gziv/Dev/sast-ai-workflow

# Install in editable mode (if pyproject.toml is configured)
pip install -e .

# OR ensure process_mining is in PYTHONPATH
export PYTHONPATH="/Users/gziv/Dev/sast-ai-workflow:$PYTHONPATH"
```

### 2. Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### Quick Start

```bash
# Run all process_mining tests
pytest tests/unit/process_mining/ -v

# Run with coverage
pytest tests/unit/process_mining/ --cov=process_mining.v1 --cov-report=html

# Run specific test file
pytest tests/unit/process_mining/test_entry_parser.py -v

# Run specific test
pytest tests/unit/process_mining/test_entry_parser.py::TestValidationEntry::test_get_masked_entry_excludes_ground_truth -v
```

### Using Test Runner Script

```bash
cd tests/unit/process_mining

# Run all tests
./run_tests.sh

# Run with coverage
./run_tests.sh --coverage

# Run specific test file
./run_tests.sh --test test_entry_parser.py

# Verbose output
./run_tests.sh --verbose
```

## Test Structure

```
tests/unit/process_mining/
├── __init__.py
├── conftest.py                           # Shared fixtures
├── README.md                             # Test documentation
├── TESTING_GUIDE.md                      # This file
├── run_tests.sh                          # Test runner script
├── test_entry_parser.py                  # Entry parsing & masking tests
├── test_fold_evaluator.py                # Evaluation & sampling tests
├── test_refinement_orchestrator.py       # Phase 2 refinement tests
└── test_stratified_kfold.py              # K-fold splitting tests
```

## Critical Tests

### 1. Ground Truth Masking (ML Integrity)

**Why**: Ensures LLM never sees answers during inference.

```bash
pytest tests/unit/process_mining/test_entry_parser.py::TestValidationEntry::test_get_masked_entry_excludes_ground_truth -v
```

**What it verifies**:
- ✅ `ground_truth_classification` excluded from masked entry
- ✅ `ground_truth_justification` excluded from masked entry
- ✅ All other fields (error_trace, source_code, etc.) preserved

### 2. Stratified Sampling (Performance Optimization)

**Why**: Maintains representative metrics when sampling for speed.

```bash
pytest tests/unit/process_mining/test_fold_evaluator.py::TestFoldEvaluator::test_stratified_sample_preserves_ratio -v
```

**What it verifies**:
- ✅ FP/TP ratio preserved (±10% tolerance)
- ✅ Sample size matches max_entries parameter
- ✅ Edge cases (all FP, all TP, small datasets) handled

### 3. Train/Val Separation (No Data Leakage)

**Why**: Prevents data leakage in Phase 2 refinement.

```bash
pytest tests/unit/process_mining/test_refinement_orchestrator.py::TestRefinementOrchestrator::test_refine_issue_type_convergence -v
```

**What it verifies**:
- ✅ Train data used for pattern refinement only
- ✅ Val data used for early stopping only
- ✅ Convergence based on val F1, not train F1

### 4. Overfitting Detection

**Why**: Identifies when patterns memorize training data.

```bash
pytest tests/unit/process_mining/test_refinement_orchestrator.py::TestRefinementOrchestrator::test_overfitting_detection -v
```

**What it verifies**:
- ✅ Train F1 vs Val F1 gap tracked
- ✅ Warning logged when gap > 0.1
- ✅ `overfitting_gap` included in convergence_info

## Troubleshooting

### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'process_mining'`

**Solution**:
```bash
# Option 1: Install in editable mode
pip install -e .

# Option 2: Set PYTHONPATH
export PYTHONPATH="/Users/gziv/Dev/sast-ai-workflow:$PYTHONPATH"
pytest tests/unit/process_mining/ -v

# Option 3: Run from project root with PYTHONPATH
cd /Users/gziv/Dev/sast-ai-workflow
PYTHONPATH=. pytest tests/unit/process_mining/ -v
```

### Fixture Not Found

**Problem**: `fixture 'sample_validation_file' not found`

**Solution**:
- Check `conftest.py` exists in `tests/unit/process_mining/`
- Ensure fixture name matches exactly
- Run from project root, not test directory

### Import Errors in Test Files

**Problem**: Tests can't import from `process_mining.v1`

**Solution**:
The `conftest.py` adds project root to sys.path automatically. If still failing:

```python
# Add to top of test file
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
```

## Test Coverage Goals

| Component | Target Coverage | Current |
|-----------|----------------|---------|
| entry_parser | 95% | TBD |
| fold_evaluator | 85% | TBD |
| refinement_orchestrator | 80% | TBD |
| stratified_kfold | 90% | TBD |

Check coverage:
```bash
pytest tests/unit/process_mining/ --cov=process_mining.v1 --cov-report=term-missing
```

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Run Process Mining V1 Tests
  run: |
    export PYTHONPATH="${GITHUB_WORKSPACE}:$PYTHONPATH"
    pytest tests/unit/process_mining/ \
      --cov=process_mining.v1 \
      --cov-report=xml \
      --cov-report=term \
      --junitxml=test-results/process_mining.xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    flags: process_mining_v1
```

## Adding New Tests

### 1. Create Test File

```python
"""
Unit tests for NewComponent.

Tests new component including:
- Feature 1
- Feature 2
"""
import pytest
from pathlib import Path

# Import will work because conftest.py adds project root to path
from process_mining.v1.kfold_pattern_learning.new_component import NewComponent


class TestNewComponent:
    """Test suite for NewComponent."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        component = NewComponent()
        assert component is not None
```

### 2. Use Shared Fixtures

```python
def test_with_fixtures(self, sample_validation_file, temp_dir):
    """Test using shared fixtures from conftest.py."""
    # Fixtures are automatically available
    assert sample_validation_file.exists()
    assert temp_dir.is_dir()
```

### 3. Add Custom Fixtures

```python
# In conftest.py
@pytest.fixture
def my_custom_fixture():
    """My custom test fixture."""
    return {"key": "value"}
```

## Best Practices

1. **Test Naming**: `test_<what>_<condition>` (e.g., `test_stratified_sample_preserves_ratio`)
2. **One Assert Per Test**: Focus on one behavior per test
3. **Use Fixtures**: Reuse test data via fixtures
4. **Mock External Dependencies**: Use `@patch` for LLM calls, file I/O
5. **Test Edge Cases**: Empty lists, None values, extreme ratios
6. **Docstrings**: Explain what each test verifies
7. **Arrange-Act-Assert**: Structure tests clearly

## Example Test Session

```bash
$ cd /Users/gziv/Dev/sast-ai-workflow
$ export PYTHONPATH=".:$PYTHONPATH"
$ pytest tests/unit/process_mining/ -v

============================= test session starts ==============================
tests/unit/process_mining/test_entry_parser.py::TestValidationEntry::test_init PASSED
tests/unit/process_mining/test_entry_parser.py::TestValidationEntry::test_get_masked_entry_excludes_ground_truth PASSED
tests/unit/process_mining/test_fold_evaluator.py::TestFoldEvaluator::test_stratified_sample_preserves_ratio PASSED
tests/unit/process_mining/test_refinement_orchestrator.py::TestRefinementOrchestrator::test_overfitting_detection PASSED
...
============================== 45 passed in 12.34s ===============================
```

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-cov](https://pytest-cov.readthedocs.io/)