"""
Pytest tests for Charts class.

These tests validate chart generation functionality, ensuring that
statistical visualizations are created correctly without errors.
"""
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from amcreport import Charts, ExamProject, ExamData


@pytest.fixture
def mock_exam_project(tmp_path):
    """
    Create a mock ExamProject for testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Mock ExamProject with necessary attributes
    """
    project = Mock(spec=ExamProject)
    project.path = str(tmp_path)
    project.name = "Test Exam"
    project.company_name = "Test University"
    project.company_url = "https://test.edu"
    project.threshold = 99
    return project


@pytest.fixture
def mock_exam_data_large():
    """
    Create mock ExamData with enough students for all charts.

    Returns:
        Mock ExamData with 200 students (above threshold)
    """
    data = Mock(spec=ExamData)

    # General statistics
    data.number_of_examinees = 200
    data.threshold = 99

    # Create realistic DataFrames
    data.marks = pd.DataFrame({
        'mark': np.random.uniform(7, 20, 200),
        'student': range(200)
    })

    data.questions = pd.DataFrame({
        'difficulty': np.random.uniform(0.3, 0.9, 40),
        'discrimination': np.random.uniform(-0.1, 0.6, 40),
        'correlation': np.random.uniform(-0.1, 0.7, 40),
        'presented': [200] * 40,
        'cancelled': np.random.randint(0, 10, 40),
        'replied': np.random.randint(150, 200, 40),
        'correct': np.random.randint(100, 190, 40),
        'empty': np.random.randint(0, 20, 40),
        'error': np.random.randint(0, 5, 40),
    })

    data.general_stats = {
        'Mean': 14.5,
        'Number of examinees': 200,
    }

    return data


@pytest.fixture
def mock_exam_data_small():
    """
    Create mock ExamData with insufficient students for discrimination charts.

    Returns:
        Mock ExamData with 50 students (below threshold)
    """
    data = Mock(spec=ExamData)

    # General statistics
    data.number_of_examinees = 50
    data.threshold = 99

    # Create realistic DataFrames
    data.marks = pd.DataFrame({
        'mark': np.random.uniform(7, 20, 50),
        'student': range(50)
    })

    data.questions = pd.DataFrame({
        'difficulty': np.random.uniform(0.3, 0.9, 40),
        'correlation': np.random.uniform(-0.1, 0.7, 40),
        'presented': [50] * 40,
        'cancelled': np.random.randint(0, 5, 40),
        'replied': np.random.randint(35, 50, 40),
        'correct': np.random.randint(20, 48, 40),
        'empty': np.random.randint(0, 10, 40),
        'error': np.random.randint(0, 3, 40),
    })

    data.general_stats = {
        'Mean': 13.2,
        'Number of examinees': 50,
    }

    return data


class TestChartsInitialization:
    """Test Charts class initialization."""

    def test_charts_initialization_success(self, mock_exam_project, mock_exam_data_large):
        """Test that Charts initializes without errors."""
        charts = Charts(mock_exam_project, mock_exam_data_large)

        assert charts.exam == mock_exam_project
        assert charts.data == mock_exam_data_large
        assert charts.image_path == os.path.join(mock_exam_project.path, 'img')

    def test_img_directory_created(self, mock_exam_project, mock_exam_data_large):
        """Test that img directory is created during initialization."""
        charts = Charts(mock_exam_project, mock_exam_data_large)

        img_path = Path(mock_exam_project.path) / 'img'
        assert img_path.exists(), "img directory should be created"
        assert img_path.is_dir(), "img path should be a directory"

    def test_mark_bins_calculated(self, mock_exam_project, mock_exam_data_large):
        """Test that mark bins are calculated correctly."""
        charts = Charts(mock_exam_project, mock_exam_data_large)

        expected_bins = int(
            mock_exam_data_large.marks['mark'].max() -
            mock_exam_data_large.marks['mark'].min()
        )
        assert charts.mark_bins == expected_bins


class TestChartFileCreation:
    """Test that chart image files are created."""

    def test_all_charts_created_large_dataset(self, mock_exam_project, mock_exam_data_large):
        """
        Test that all chart files are created with large dataset.

        With 200 students (above threshold), all charts should be created.
        """
        Charts(mock_exam_project, mock_exam_data_large)

        img_path = Path(mock_exam_project.path) / 'img'

        # Expected chart files for large dataset
        expected_files = [
            'marks.png',
            'difficulty.png',
            'discrimination.png',  # Only with enough students
            'discrimination_vs_difficulty.png',  # Only with enough students
            'item_correlation.png',
            'question_columns.png',
        ]

        for filename in expected_files:
            file_path = img_path / filename
            assert file_path.exists(), f"{filename} should be created"
            assert file_path.stat().st_size > 0, f"{filename} should not be empty"

    def test_limited_charts_created_small_dataset(self, mock_exam_project, mock_exam_data_small):
        """
        Test that discrimination charts are NOT created with small dataset.

        With 50 students (below threshold), discrimination charts should be skipped.
        """
        Charts(mock_exam_project, mock_exam_data_small)

        img_path = Path(mock_exam_project.path) / 'img'

        # Charts that should be created
        expected_files = [
            'marks.png',
            'difficulty.png',
            'item_correlation.png',
            'question_columns.png',
        ]

        for filename in expected_files:
            file_path = img_path / filename
            assert file_path.exists(), f"{filename} should be created even with small dataset"

        # Charts that should NOT be created
        not_expected_files = [
            'discrimination.png',
            'discrimination_vs_difficulty.png',
        ]

        for filename in not_expected_files:
            file_path = img_path / filename
            assert not file_path.exists(), f"{filename} should NOT be created with small dataset"

    def test_marks_histogram_created(self, mock_exam_project, mock_exam_data_large):
        """Test that marks histogram is created."""
        Charts(mock_exam_project, mock_exam_data_large)

        marks_file = Path(mock_exam_project.path) / 'img' / 'marks.png'
        assert marks_file.exists()

    def test_difficulty_histogram_created(self, mock_exam_project, mock_exam_data_large):
        """Test that difficulty histogram is created."""
        Charts(mock_exam_project, mock_exam_data_large)

        difficulty_file = Path(mock_exam_project.path) / 'img' / 'difficulty.png'
        assert difficulty_file.exists()

    def test_correlation_histogram_created(self, mock_exam_project, mock_exam_data_large):
        """Test that correlation histogram is created."""
        Charts(mock_exam_project, mock_exam_data_large)

        correlation_file = Path(mock_exam_project.path) / 'img' / 'item_correlation.png'
        assert correlation_file.exists()

    def test_bar_chart_created(self, mock_exam_project, mock_exam_data_large):
        """Test that bar chart is created."""
        Charts(mock_exam_project, mock_exam_data_large)

        bar_chart_file = Path(mock_exam_project.path) / 'img' / 'question_columns.png'
        assert bar_chart_file.exists()


class TestConditionalChartCreation:
    """Test conditional chart creation based on student count."""

    def test_discrimination_chart_only_above_threshold(self, mock_exam_project):
        """Test discrimination charts are only created above threshold."""
        # Test with data above threshold
        data_large = Mock(spec=ExamData)
        data_large.number_of_examinees = 150
        data_large.threshold = 99
        data_large.marks = pd.DataFrame({'mark': np.random.uniform(7, 20, 150)})
        data_large.questions = pd.DataFrame({
            'difficulty': np.random.uniform(0.3, 0.9, 40),
            'discrimination': np.random.uniform(-0.1, 0.6, 40),
            'correlation': np.random.uniform(-0.1, 0.7, 40),
            'presented': [150] * 40,
            'cancelled': [2] * 40,
            'replied': [145] * 40,
            'correct': [120] * 40,
            'empty': [5] * 40,
            'error': [1] * 40,
        })
        data_large.general_stats = {'Mean': 14.0, 'Number of examinees': 150}

        Charts(mock_exam_project, data_large)

        discrimination_file = Path(mock_exam_project.path) / 'img' / 'discrimination.png'
        assert discrimination_file.exists(), "Discrimination chart should exist above threshold"

    def test_threshold_boundary_exact(self, mock_exam_project):
        """Test chart creation at exact threshold (99 students)."""
        # Test with data at threshold boundary
        data_threshold = Mock(spec=ExamData)
        data_threshold.number_of_examinees = 99
        data_threshold.threshold = 99
        data_threshold.marks = pd.DataFrame({'mark': np.random.uniform(7, 20, 99)})
        data_threshold.questions = pd.DataFrame({
            'difficulty': np.random.uniform(0.3, 0.9, 40),
            'correlation': np.random.uniform(-0.1, 0.7, 40),
            'presented': [99] * 40,
            'cancelled': [2] * 40,
            'replied': [95] * 40,
            'correct': [80] * 40,
            'empty': [5] * 40,
            'error': [1] * 40,
        })
        data_threshold.general_stats = {'Mean': 13.5, 'Number of examinees': 99}

        Charts(mock_exam_project, data_threshold)

        discrimination_file = Path(mock_exam_project.path) / 'img' / 'discrimination.png'
        # At threshold (not above), discrimination should NOT be created
        assert not discrimination_file.exists(), "Discrimination chart should NOT exist at exact threshold"


class TestChartDataColumns:
    """Test handling of different data columns."""

    def test_actual_data_columns_filtered(self, mock_exam_project, mock_exam_data_large):
        """Test that actual_data_columns only includes existing columns."""
        charts = Charts(mock_exam_project, mock_exam_data_large)

        # All columns should exist in our mock data
        expected_columns = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error']

        for col in charts.actual_data_columns:
            assert col in expected_columns
            assert col in mock_exam_data_large.questions.columns

    def test_missing_optional_columns_handled(self, mock_exam_project):
        """Test that missing optional columns don't cause errors."""
        # Create data with only required columns
        data_minimal = Mock(spec=ExamData)
        data_minimal.number_of_examinees = 50
        data_minimal.threshold = 99
        data_minimal.marks = pd.DataFrame({'mark': np.random.uniform(7, 20, 50)})
        data_minimal.questions = pd.DataFrame({
            'difficulty': np.random.uniform(0.3, 0.9, 40),
            'correlation': np.random.uniform(-0.1, 0.7, 40),
            'presented': [50] * 40,
            'correct': [30] * 40,
        })
        data_minimal.general_stats = {'Mean': 13.0, 'Number of examinees': 50}

        # Should not raise an error
        charts = Charts(mock_exam_project, data_minimal)

        # actual_data_columns should only contain existing columns
        assert 'presented' in charts.actual_data_columns
        assert 'correct' in charts.actual_data_columns
        assert 'cancelled' not in charts.actual_data_columns  # Missing from data


class TestChartErrorHandling:
    """Test error handling in chart creation."""

    def test_difficulty_vs_discrimination_no_data(self, mock_exam_project):
        """Test error handling when data is empty."""
        data_empty = Mock(spec=ExamData)
        data_empty.number_of_examinees = 150
        data_empty.threshold = 99
        data_empty.marks = pd.DataFrame({'mark': np.random.uniform(7, 20, 150)})
        # Empty questions DataFrame
        data_empty.questions = pd.DataFrame()
        data_empty.general_stats = {'Mean': 14.0, 'Number of examinees': 150}

        # Should raise KeyError when trying to access missing columns
        with pytest.raises(KeyError, match="difficulty"):
            Charts(mock_exam_project, data_empty)

    def test_difficulty_vs_discrimination_missing_columns(self, mock_exam_project):
        """Test error handling when required columns are missing."""
        data_missing_cols = Mock(spec=ExamData)
        data_missing_cols.number_of_examinees = 150
        data_missing_cols.threshold = 99
        data_missing_cols.marks = pd.DataFrame({'mark': np.random.uniform(7, 20, 150)})
        # Missing 'discrimination' column
        data_missing_cols.questions = pd.DataFrame({
            'difficulty': np.random.uniform(0.3, 0.9, 40),
            'correlation': np.random.uniform(-0.1, 0.7, 40),
        })
        data_missing_cols.general_stats = {'Mean': 14.0, 'Number of examinees': 150}

        # Should raise KeyError when accessing missing 'discrimination' column
        with pytest.raises(KeyError, match="discrimination"):
            Charts(mock_exam_project, data_missing_cols)


class TestChartWithRealData:
    """Test Charts with real ExamData fixture."""

    def test_charts_with_real_exam_data(self, exam_data, tmp_path):
        """
        Test Charts creation with real ExamData from fixture.

        This is an integration test using actual database data.
        """
        # Create a mock project pointing to tmp_path
        project = Mock(spec=ExamProject)
        project.path = str(tmp_path)
        project.name = "Integration Test"
        project.company_name = "Test"
        project.company_url = "test.com"
        project.threshold = exam_data.threshold

        # Create charts (should not raise)
        charts = Charts(project, exam_data)

        # Verify img directory exists
        img_path = tmp_path / 'img'
        assert img_path.exists()

        # Verify at least some charts were created
        chart_files = list(img_path.glob('*.png'))
        assert len(chart_files) > 0, "At least some chart files should be created"


class TestChartImageProperties:
    """Test properties of generated chart images."""

    def test_chart_files_are_png(self, mock_exam_project, mock_exam_data_large):
        """Test that all generated files are PNG images."""
        Charts(mock_exam_project, mock_exam_data_large)

        img_path = Path(mock_exam_project.path) / 'img'

        for chart_file in img_path.glob('*'):
            assert chart_file.suffix == '.png', f"{chart_file.name} should be a PNG file"

    def test_chart_files_non_empty(self, mock_exam_project, mock_exam_data_large):
        """Test that generated chart files are not empty."""
        Charts(mock_exam_project, mock_exam_data_large)

        img_path = Path(mock_exam_project.path) / 'img'

        for chart_file in img_path.glob('*.png'):
            file_size = chart_file.stat().st_size
            assert file_size > 1000, f"{chart_file.name} should be larger than 1KB (got {file_size} bytes)"

    def test_img_directory_only_contains_png(self, mock_exam_project, mock_exam_data_large):
        """Test that img directory only contains PNG files."""
        Charts(mock_exam_project, mock_exam_data_large)

        img_path = Path(mock_exam_project.path) / 'img'

        all_files = list(img_path.glob('*'))
        png_files = list(img_path.glob('*.png'))

        # Filter out any directories
        all_files = [f for f in all_files if f.is_file()]

        assert len(all_files) == len(png_files), "img directory should only contain PNG files"
