"""
Pytest fixtures for process_mining.v1 tests.

Provides test data, mock objects, and utilities for testing
the k-fold pattern learning pipeline.
"""
import pytest
import json
import tempfile
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_validation_entry_text():
    """Sample validation entry in .txt format matching actual parser expectations."""
    return """================================================================================
GROUND-TRUTH ENTRIES FOR: test-package
================================================================================

Package: test-package
Total Entries: 2

---
Entry #1:
Issue Type: RESOURCE_LEAK
CWE: CWE-401

Error Trace:
test.c:100:2: file = fopen("test.txt", "r")
test.c:105:2: return -1 without closing file

Source Code (test.c):
```c
int test() {
    FILE *file = fopen("test.txt", "r");
    if (!file) return -1; /* missing fclose */
    return 0;
}
```

Ground Truth Classification: TRUE_POSITIVE

Human Expert Justification: File handle 'file' is not closed before function returns at line 105, causing a resource leak.

---
Entry #2:
Issue Type: RESOURCE_LEAK
CWE: CWE-401

Error Trace:
test.c:200:2: ptr = malloc(100)
test.c:205:2: return

Source Code (test.c):
```c
void test2() {
    char *ptr = malloc(100);
    if (ptr) {
        free(ptr);
    }
    return;
}
```

Ground Truth Classification: FALSE_POSITIVE

Human Expert Justification: Memory allocated at line 200 is properly freed at line 203 before function returns.
"""


@pytest.fixture
def sample_validation_file(temp_dir, sample_validation_entry_text):
    """Create a sample validation .txt file."""
    val_file = temp_dir / "test_package.txt"
    val_file.write_text(sample_validation_entry_text)
    return val_file


@pytest.fixture
def sample_patterns():
    """Sample pattern dictionary."""
    return {
        "fp": [
            {
                "pattern_id": "RESOURCE_LEAK-FP-001",
                "group": "A: Proper Cleanup",
                "summary": "Resource is properly freed before function returns. Key indicators: free() call in all code paths, resource released before return."
            },
            {
                "pattern_id": "RESOURCE_LEAK-FP-002",
                "group": "B: Transfer of Ownership",
                "summary": "Resource ownership is transferred to another function or data structure. Key indicators: pointer assigned to struct field, returned to caller."
            }
        ],
        "tp": [
            {
                "pattern_id": "RESOURCE_LEAK-TP-001",
                "group": "A: Missing Cleanup",
                "summary": "Resource is not freed before function returns. Key indicators: no free() call, resource goes out of scope without cleanup."
            },
            {
                "pattern_id": "RESOURCE_LEAK-TP-002",
                "group": "B: Error Path Leak",
                "summary": "Resource is freed in normal path but not in error paths. Key indicators: free() only on success path, early returns without cleanup."
            }
        ]
    }


@pytest.fixture
def sample_patterns_file(temp_dir, sample_patterns):
    """Create a sample patterns JSON file."""
    pattern_file = temp_dir / "RESOURCE_LEAK_patterns.json"
    with open(pattern_file, 'w') as f:
        json.dump(sample_patterns, f, indent=2)
    return pattern_file


@pytest.fixture
def sample_fold_split():
    """Sample fold split for k-fold testing."""
    return {
        'fold_0': {
            'train': ['file1.txt', 'file2.txt', 'file3.txt'],
            'val': ['file4.txt']
        },
        'fold_1': {
            'train': ['file1.txt', 'file2.txt', 'file4.txt'],
            'val': ['file3.txt']
        }
    }


@pytest.fixture
def sample_evaluation_result():
    """Sample evaluation result from PatternEvaluator."""
    return {
        'metrics': {
            'overall': {
                'f1': 0.85,
                'precision': 0.80,
                'recall': 0.90,
                'accuracy': 0.88
            },
            'confusion_matrix': {
                'TP': 45,
                'TN': 38,
                'FP': 5,
                'FN': 12
            }
        },
        'results': [
            {
                'entry_id': 'test-package/test.c:100-RESOURCE_LEAK-001',
                'ground_truth': 'TRUE_POSITIVE',
                'predicted_class': 'TRUE_POSITIVE',
                'correct': True,
                'justification': 'Pattern RESOURCE_LEAK-TP-001 matches...'
            },
            {
                'entry_id': 'test-package/test.c:200-RESOURCE_LEAK-002',
                'ground_truth': 'FALSE_POSITIVE',
                'predicted_class': 'FALSE_POSITIVE',
                'correct': True,
                'justification': 'Pattern RESOURCE_LEAK-FP-001 matches...'
            }
        ]
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for pattern learning."""
    return {
        "fp": [
            {
                "pattern_id": "RESOURCE_LEAK-FP-001",
                "group": "A: Test Group",
                "summary": "Test FP pattern summary"
            }
        ],
        "tp": [
            {
                "pattern_id": "RESOURCE_LEAK-TP-001",
                "group": "A: Test Group",
                "summary": "Test TP pattern summary"
            }
        ]
    }


@pytest.fixture
def mock_pattern_learner(mock_llm_response):
    """Mock PatternLearner that returns sample patterns."""
    from unittest.mock import Mock
    learner = Mock()
    learner.learn_patterns.return_value = mock_llm_response
    return learner


@pytest.fixture
def mock_fold_evaluator(sample_evaluation_result):
    """Mock FoldEvaluator that returns sample metrics."""
    from unittest.mock import Mock
    evaluator = Mock()
    evaluator.evaluate_fold.return_value = sample_evaluation_result
    return evaluator


@pytest.fixture
def mock_pattern_refiner():
    """Mock PatternRefiner for testing refinement."""
    from unittest.mock import Mock
    refiner = Mock()
    refiner.refine_patterns.return_value = {
        'add': [
            {
                'pattern_type': 'fp',
                'pattern_id': 'RESOURCE_LEAK-FP-003',
                'group': 'C: New Pattern',
                'summary': 'New pattern from refinement'
            }
        ],
        'modify': [],
        'remove': []
    }
    return refiner


@pytest.fixture
def sample_train_val_test_dirs(temp_dir, sample_validation_entry_text):
    """Create sample train/val/test directory structure."""
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    test_dir = temp_dir / "test"

    train_dir.mkdir()
    val_dir.mkdir()
    test_dir.mkdir()

    # Create sample files in each directory
    for i in range(3):
        (train_dir / f"train_file_{i}.txt").write_text(sample_validation_entry_text)

    for i in range(1):
        (val_dir / f"val_file_{i}.txt").write_text(sample_validation_entry_text)

    for i in range(1):
        (test_dir / f"test_file_{i}.txt").write_text(sample_validation_entry_text)

    return {
        'train': train_dir,
        'val': val_dir,
        'test': test_dir
    }


@pytest.fixture
def sample_phase1_results():
    """Sample Phase 1 k-fold results."""
    return {
        'metadata': {
            'phase': 1,
            'n_folds': 2,
            'train_dir': '/path/to/train',
            'platform': 'nim'
        },
        'issue_types': {
            'RESOURCE_LEAK': {
                'folds': [
                    {
                        'fold': 0,
                        'train_files': ['file1.txt', 'file2.txt'],
                        'val_files': ['file3.txt'],
                        'patterns': {
                            'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
                            'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
                        },
                        'metrics': {'f1': 0.75, 'precision': 0.70, 'recall': 0.80}
                    }
                ],
                'merged_patterns': {
                    'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
                    'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
                },
                'avg_metrics': {'f1': 0.75, 'precision': 0.70, 'recall': 0.80}
            }
        },
        'overall_summary': {
            'total_folds': 2,
            'total_issue_types': 1,
            'avg_f1_across_issue_types': 0.75
        }
    }


@pytest.fixture
def sample_phase2_results():
    """Sample Phase 2 refinement results."""
    return {
        'metadata': {
            'phase': 2,
            'train_dir': '/path/to/train',
            'val_dir': '/path/to/val',
            'platform': 'nim'
        },
        'issue_types': {
            'RESOURCE_LEAK': {
                'initial_patterns': {
                    'fp': [{'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'}],
                    'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
                },
                'iterations': [
                    {
                        'iteration': 1,
                        'train_f1': 0.80,
                        'val_f1': 0.75,
                        'train_misclassified_count': 10
                    }
                ],
                'final_patterns': {
                    'fp': [
                        {'pattern_id': 'FP-001', 'group': 'A', 'summary': 'test'},
                        {'pattern_id': 'FP-002', 'group': 'B', 'summary': 'refined'}
                    ],
                    'tp': [{'pattern_id': 'TP-001', 'group': 'A', 'summary': 'test'}]
                },
                'convergence_info': {
                    'converged': True,
                    'iterations_completed': 3,
                    'initial_val_f1': 0.70,
                    'final_val_f1': 0.80,
                    'f1_improvement': 0.10
                }
            }
        },
        'overall_summary': {
            'total_issue_types': 1,
            'avg_iterations': 3.0,
            'avg_f1_improvement': 0.10
        }
    }