#!/bin/bash
# Test runner for Process Mining V1 tests
# Usage: ./run_tests.sh [options]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/../../../.."

# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo -e "${YELLOW}Running Process Mining V1 K-Fold Pattern Learning Tests${NC}"
echo "========================================================="
echo

# Parse arguments
COVERAGE=false
VERBOSE=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --test|-t)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [options]"
            echo
            echo "Options:"
            echo "  --coverage, -c       Generate coverage report"
            echo "  --verbose, -v        Verbose output"
            echo "  --test FILE, -t FILE Run specific test file"
            echo "  --help, -h           Show this help"
            echo
            echo "Examples:"
            echo "  ./run_tests.sh                        # Run all tests"
            echo "  ./run_tests.sh --coverage             # Run with coverage"
            echo "  ./run_tests.sh -t test_entry_parser.py # Run specific test"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/unit/process_mining/kfold_pattern_learning/"

if [ -n "$SPECIFIC_TEST" ]; then
    PYTEST_CMD="pytest tests/unit/process_mining/kfold_pattern_learning/$SPECIFIC_TEST"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=process_mining.kfold_pattern_learning --cov-report=html --cov-report=term"
    echo -e "${YELLOW}Running with coverage analysis...${NC}"
fi

# Run tests
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo

if $PYTEST_CMD; then
    echo
    echo -e "${GREEN}✓ All tests passed!${NC}"

    if [ "$COVERAGE" = true ]; then
        echo
        echo -e "${YELLOW}Coverage report generated: htmlcov/index.html${NC}"
        echo "Open with: open htmlcov/index.html"
    fi

    exit 0
else
    echo
    echo -e "${RED}✗ Tests failed!${NC}"
    exit 1
fi