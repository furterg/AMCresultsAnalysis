# Testing Documentation - AMC Report Generator

This document provides comprehensive information about the test suite for the AMC (Auto Multiple Choice) Report Generator.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Suite Structure](#test-suite-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Statistics

- **Total Tests:** 358
- **Code Coverage:** 96.17%
- **Test Success Rate:** 100% (357 passing, 1 skipped)
- **Framework:** pytest 8.4+
- **Execution Time:** ~2 minutes (full suite)

### Test Philosophy

The test suite follows a **comprehensive testing approach**:

1. **Unit Tests** (338 tests) - Fast, isolated tests of individual functions
2. **Integration Tests** (20 tests) - Complete workflow validation with real data
3. **Edge Case Tests** (6 tests) - Coverage of error paths and rare scenarios

---

## Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov

# Verify installation
pytest --version
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest -v

# Run with coverage report
pytest --cov --cov-report=html --cov-report=term

# Run only fast tests (skip integration)
pytest -m "not integration"
```

### View Coverage Report

```bash
# Generate HTML coverage report
pytest --cov --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Test Suite Structure

### Test Files Overview

| File | Tests | Purpose | Speed |
|------|-------|---------|-------|
| `test_amcreport.py` | 15 | Core ExamData class functionality | Fast |
| `test_charts.py` | 46 | Chart generation and visualization | Medium |
| `test_claude_analyzer.py` | 59 | AI analysis integration | Fast |
| `test_error_handling.py` | 62 | Error conditions and edge cases | Fast |
| `test_exceptions.py` | 5 | Custom exception classes | Fast |
| `test_pdf_report.py` | 63 | PDF report generation | Medium |
| `test_pydantic_settings.py` | 42 | Configuration validation | Fast |
| `test_report.py` | 10 | Report utility functions | Fast |
| `test_settings_project.py` | 30 | Settings and project management | Fast |
| `test_utilities.py` | 18 | Utility functions | Fast |
| `test_integration.py` | 20 | End-to-end workflow validation | Slow |
| `test_coverage_edge_cases.py` | 6 | Additional coverage improvements | Fast |

### Test Categories

#### Unit Tests (338 tests)
Fast, isolated tests using mocks and fixtures. Run these during development.

**Key Areas:**
- Database loading and parsing
- Statistical calculations
- Chart generation logic
- PDF report formatting
- Configuration validation
- Error handling

#### Integration Tests (20 tests - marked with `@pytest.mark.integration`)
Slow tests using real exam databases. Run before commits or in CI.

**Coverage:**
- Complete database → analysis → charts → PDF workflow
- Real data validation
- Cross-component interactions
- Performance benchmarks
- Data consistency checks

#### Edge Case Tests (6 tests)
Tests for uncommon scenarios and error paths.

**Coverage:**
- Logging system initialization
- Chart coloring algorithms
- Directory fallback mechanisms

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run specific test file
pytest test_amcreport.py -v

# Run specific test class
pytest test_charts.py::TestChartsInitialization -v

# Run specific test
pytest test_amcreport.py::test_exam_data_initialization -v
```

### Using Markers

```bash
# Run only integration tests
pytest -m integration -v

# Run only unit tests (skip integration)
pytest -m "not integration" -v

# Run only slow tests
pytest -m slow -v

# Combine markers
pytest -m "integration and not slow" -v
```

### Coverage Options

```bash
# Basic coverage
pytest --cov

# Coverage with HTML report
pytest --cov --cov-report=html

# Coverage for specific modules
pytest --cov=amcreport --cov=report --cov=settings

# Show missing lines
pytest --cov --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov --cov-fail-under=95
```

### Filtering Output

```bash
# Quiet mode (less verbose)
pytest -q

# Only show failures
pytest --tb=short

# No output capture (see print statements)
pytest -s

# Show extra test summary
pytest -ra
```

### Debugging Tests

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace

# Show print statements
pytest -s

# Verbose logging
pytest --log-cli-level=DEBUG
```

---

## Test Coverage

### Current Coverage by Module

| Module | Coverage | Lines | Missed |
|--------|----------|-------|--------|
| **amcreport.py** | 93.58% | 483 | 31 |
| **report.py** | 99.30% | 430 | 3 |
| **settings.py** | 94.44% | 54 | 3 |
| **Overall** | **96.17%** | **967** | **37** |

### Uncovered Code

The remaining 4% consists of:
- Error handling paths (difficult to trigger)
- Debug logging functions (uses global state)
- Visual styling code (matplotlib internals)
- Rare database schema edge cases

### Coverage Configuration

Coverage is configured in `.coveragerc`:

```ini
[run]
source = .
omit = test_*.py, conftest.py

[report]
precision = 2
show_missing = True
exclude_lines =
    pragma: no cover
    if __name__ == .__main__.:
    @abstractmethod
```

### Improving Coverage

To identify uncovered code:

```bash
# Show missing lines
coverage report -m --include="amcreport.py"

# Generate HTML report for detailed view
pytest --cov --cov-report=html
open htmlcov/index.html
```

---

## Writing New Tests

### Test Structure

Follow this template for new test files:

```python
"""
Brief description of what this test file covers.

Test organization:
- TestClassName1: Description
- TestClassName2: Description
"""
import pytest
from amcreport import YourModule

class TestYourFeature:
    """Test suite for your feature."""

    def test_basic_functionality(self):
        """Test the basic happy path."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = your_function(input_data)

        # Assert
        assert result == expected_value

    def test_error_handling(self):
        """Test that errors are handled correctly."""
        with pytest.raises(ExpectedError):
            your_function(invalid_input)
```

### Best Practices

1. **Use Descriptive Names**
   ```python
   # Good
   def test_exam_data_calculates_difficulty_correctly(self):

   # Bad
   def test_calc(self):
   ```

2. **Follow AAA Pattern** (Arrange, Act, Assert)
   ```python
   def test_marks_calculation(self):
       # Arrange - setup test data
       exam_data = create_exam_data()

       # Act - execute the code
       marks = exam_data.calculate_marks()

       # Assert - verify results
       assert len(marks) == expected_count
   ```

3. **Use Fixtures for Common Setup**
   ```python
   @pytest.fixture
   def sample_exam_data():
       """Provide sample exam data for tests."""
       return ExamData(test_project_path)

   def test_with_fixture(sample_exam_data):
       assert sample_exam_data.number_of_examinees > 0
   ```

4. **Test One Thing Per Test**
   ```python
   # Good - focused test
   def test_difficulty_calculation(self):
       difficulty = calculate_difficulty(scores)
       assert 0 <= difficulty <= 1

   # Bad - testing multiple things
   def test_everything(self):
       difficulty = calculate_difficulty(scores)
       assert 0 <= difficulty <= 1
       discrimination = calculate_discrimination(scores)
       assert discrimination > 0
   ```

5. **Use Parametrize for Multiple Cases**
   ```python
   @pytest.mark.parametrize("threshold,expected", [
       (10, True),
       (99, False),
       (100, False),
   ])
   def test_threshold_validation(threshold, expected):
       result = is_valid_threshold(threshold)
       assert result == expected
   ```

### Mocking Guidelines

```python
from unittest.mock import Mock, patch

# Mock external dependencies
@patch('amcreport.ExamData._get_database')
def test_with_mock(mock_db):
    mock_db.return_value = create_fake_db()
    # Test code here

# Mock objects
def test_with_mock_object():
    mock_exam = Mock()
    mock_exam.questions = pd.DataFrame({'q': [1, 2, 3]})
    result = process_exam(mock_exam)
    assert result is not None
```

### Adding Integration Tests

Mark tests that use real data:

```python
import pytest

@pytest.mark.integration
class TestRealDataWorkflow:
    """Integration tests using real exam databases."""

    def test_complete_workflow(self):
        """Test full pipeline with real data."""
        # Uses actual data/scoring.sqlite and data/capture.sqlite
        exam_data = ExamData(os.getcwd())
        charts = Charts(project, exam_data)
        # Verify complete workflow
```

---

## Test Configuration Files

### pytest.ini

Main pytest configuration:
- Test discovery patterns
- Custom marker registration
- Warning filters
- Default options

### .coveragerc

Coverage tool configuration:
- Source paths
- Files to omit
- Report formatting
- Exclusion rules

### conftest.py

Shared fixtures and test utilities:
- Common test data generators
- Reusable fixtures
- Test helpers

---

## Continuous Integration

### Pre-Commit Checks

Run fast tests before committing:

```bash
# Quick validation (unit tests only)
pytest -m "not integration" -x --tb=short

# With coverage
pytest -m "not integration" --cov --cov-fail-under=95
```

### Pre-Push Checks

Run full test suite before pushing:

```bash
# All tests including integration
pytest -v

# With coverage report
pytest --cov --cov-report=html --cov-report=term
```

### CI Pipeline Recommendations

For GitHub Actions, GitLab CI, or similar:

```yaml
# Example CI configuration
test:
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest -v --cov --cov-report=xml
    - coverage report --fail-under=95
```

---

## Troubleshooting

### Common Issues

#### Tests Fail with "Database Not Found"

**Problem:** Integration tests can't find `data/scoring.sqlite` or `data/capture.sqlite`

**Solution:**
```bash
# Ensure you're in the project root directory
cd /path/to/AMCresultsAnalysis

# Or skip integration tests
pytest -m "not integration"
```

#### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'amcreport'`

**Solution:**
```bash
# Install package in development mode
pip install -e .

# Or ensure PYTHONPATH is set
export PYTHONPATH=/path/to/AMCresultsAnalysis:$PYTHONPATH
```

#### Slow Test Execution

**Problem:** Tests take too long

**Solution:**
```bash
# Run only fast unit tests
pytest -m "not integration"

# Use multiple processes (if pytest-xdist installed)
pytest -n auto

# Stop on first failure
pytest -x
```

#### Coverage Not Generated

**Problem:** Coverage report is empty or missing

**Solution:**
```bash
# Verify pytest-cov is installed
pip install pytest-cov

# Check .coveragerc exists
ls -la .coveragerc

# Run with explicit coverage options
pytest --cov=. --cov-report=term
```

#### Matplotlib/Chart Tests Fail

**Problem:** Chart generation tests fail on headless servers

**Solution:**
```python
# Tests already use Agg backend
import matplotlib
matplotlib.use('Agg')

# Or skip chart tests in CI
pytest -m "not charts"
```

---

## Maintenance

### Updating Tests After Code Changes

When modifying code:

1. **Update corresponding tests first** (TDD approach)
2. **Run affected tests** to verify changes
3. **Update integration tests** if workflow changed
4. **Check coverage** to ensure no regression
5. **Update documentation** if test behavior changed

### Adding New Features

When adding new functionality:

1. Write tests **before** implementing (TDD)
2. Add both unit tests and integration tests
3. Ensure coverage stays above 95%
4. Document new test markers if needed
5. Update this TESTING.md if needed

### Periodic Review

Quarterly maintenance tasks:

- [ ] Review and remove obsolete tests
- [ ] Update deprecated pytest features
- [ ] Verify all markers are still relevant
- [ ] Check for new testing best practices
- [ ] Update dependency versions
- [ ] Review coverage gaps

---

## Resources

### Documentation

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)

### Tools

- **pytest** - Test framework
- **pytest-cov** - Coverage plugin
- **coverage.py** - Coverage measurement
- **pytest-xdist** - Parallel test execution (optional)
- **pytest-benchmark** - Performance benchmarking (optional)

### Contact

For questions about the test suite, consult:
- This documentation
- Test file docstrings
- Code comments in complex tests

---

## Appendix: Test Markers Reference

### Built-in Markers

- `@pytest.mark.skip` - Skip test unconditionally
- `@pytest.mark.skipif(condition)` - Skip test conditionally
- `@pytest.mark.parametrize` - Run test with multiple parameters
- `@pytest.mark.xfail` - Expect test to fail

### Custom Markers

- `@pytest.mark.integration` - Integration tests (20 tests)
- `@pytest.mark.slow` - Slow-running tests (future use)
- `@pytest.mark.requires_data` - Tests requiring real exam data (future use)

### Example Usage

```python
@pytest.mark.integration
def test_full_pipeline():
    """Integration test - uses real databases."""
    pass

@pytest.mark.slow
@pytest.mark.requires_data
def test_large_dataset():
    """Test with large dataset - runs slowly."""
    pass

@pytest.mark.parametrize("input,expected", [(1, 2), (3, 4)])
def test_increment(input, expected):
    """Test with multiple inputs."""
    assert increment(input) == expected
```

---

**Last Updated:** 2025-11-09
**Test Suite Version:** 1.0
**Coverage:** 96.17%
