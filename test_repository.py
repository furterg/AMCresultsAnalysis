"""
Tests for the exam repository module.

This module tests the functionality of storing exam metrics in various
repository backends (Airtable, Baserow).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
from datetime import datetime
import tempfile
import os
import sys

from repository import (
    ExamMetrics,
    RepositoryError,
    ExamRepository,
    AirtableBackend,
    BaserowBackend,
    create_exam_metrics_from_data
)
from settings import AMCSettings


@pytest.fixture
def temp_projects_dir():
    """Create a temporary directory for testing, clean up after test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass  # Directory may not be empty, which is fine


class TestExamMetrics:
    """Test the ExamMetrics data class."""

    def test_exam_metrics_initialization(self):
        """Test basic initialization of ExamMetrics."""
        metrics = ExamMetrics(
            project_name="202511-MATH101",
            analysis_date="2025-11-11",
            num_students=50,
            num_questions=20,
            avg_grade=15.5,
            median_grade=16.0,
            std_dev_grade=2.3,
            min_grade=8.0,
            max_grade=20.0
        )

        assert metrics.project_name == "202511-MATH101"
        assert metrics.num_students == 50
        assert metrics.num_questions == 20
        assert metrics.avg_grade == 15.5

    def test_project_name_parsing_valid_format(self):
        """Test parsing of project name in YYYYMM-Course format."""
        metrics = ExamMetrics(
            project_name="202511-BIO205_Final",
            analysis_date="2025-11-11",
            num_students=30,
            num_questions=15,
            avg_grade=14.0,
            median_grade=14.5,
            std_dev_grade=2.0,
            min_grade=9.0,
            max_grade=18.0
        )

        assert metrics.year == "2025"
        assert metrics.month == "11"
        assert metrics.course_code == "BIO205_Final"

    def test_project_name_parsing_invalid_format(self):
        """Test parsing of project name with non-standard format."""
        metrics = ExamMetrics(
            project_name="RandomProjectName",
            analysis_date="2025-11-11",
            num_students=25,
            num_questions=10,
            avg_grade=12.0,
            median_grade=12.0,
            std_dev_grade=1.5,
            min_grade=8.0,
            max_grade=16.0
        )

        assert metrics.year is None
        assert metrics.month is None
        assert metrics.course_code == "RandomProjectName"

    def test_to_dict_with_all_fields(self):
        """Test conversion to dictionary with all fields populated."""
        metrics = ExamMetrics(
            project_name="202511-MATH101",
            analysis_date="2025-11-11",
            num_students=50,
            num_questions=20,
            avg_grade=15.567,
            median_grade=16.123,
            std_dev_grade=2.345,
            min_grade=8.0,
            max_grade=20.0,
            avg_difficulty=0.678,
            avg_discrimination=0.456,
            avg_correlation=0.234,
            pass_rate=0.850,
            cronbach_alpha=0.789
        )

        result = metrics.to_dict()

        # Check basic fields
        assert result['project_name'] == "202511-MATH101"
        assert result['num_students'] == 50
        assert result['num_questions'] == 20

        # Check rounding (2 decimals for grades)
        assert result['avg_grade'] == 15.57
        assert result['median_grade'] == 16.12
        assert result['std_dev_grade'] == 2.35

        # Check rounding (3 decimals for metrics)
        assert result['avg_difficulty'] == 0.678
        assert result['avg_discrimination'] == 0.456
        assert result['pass_rate'] == 0.850

    def test_to_dict_with_none_values(self):
        """Test conversion to dictionary with None values for optional fields."""
        metrics = ExamMetrics(
            project_name="202511-MATH101",
            analysis_date="2025-11-11",
            num_students=10,
            num_questions=5,
            avg_grade=12.0,
            median_grade=12.0,
            std_dev_grade=1.0,
            min_grade=10.0,
            max_grade=14.0,
            avg_difficulty=None,
            avg_discrimination=None,
            avg_correlation=None
        )

        result = metrics.to_dict()

        assert result['avg_difficulty'] is None
        assert result['avg_discrimination'] is None
        assert result['avg_correlation'] is None


class TestAirtableBackend:
    """Test the Airtable repository backend (with mocks)."""

    @patch('repository.AirtableBackend._ensure_table_setup')
    @patch('pyairtable.Api')
    def test_airtable_backend_initialization(self, mock_api, mock_ensure_table, temp_projects_dir):
        """Test initialization of Airtable backend."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='airtable',
            airtable_api_key='test_key',
            airtable_base_id='test_base'
        )

        # Mock table setup to return True (table is ready)
        mock_ensure_table.return_value = True

        # Mock the API and table
        mock_table = MagicMock()
        mock_api.return_value.table.return_value = mock_table

        backend = AirtableBackend(settings)

        assert backend.settings == settings
        mock_api.assert_called_once_with('test_key')

    @patch('repository.AirtableBackend._ensure_table_setup')
    @patch('pyairtable.Api')
    def test_airtable_save_new_record(self, mock_api, mock_ensure_table, temp_projects_dir):
        """Test saving a new record to Airtable."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='airtable',
            airtable_api_key='test_key',
            airtable_base_id='test_base'
        )

        # Mock table setup
        mock_ensure_table.return_value = True

        # Setup mock table
        mock_table = MagicMock()
        mock_table.all.return_value = []  # No existing records
        mock_api.return_value.table.return_value = mock_table

        backend = AirtableBackend(settings)

        metrics = ExamMetrics(
            project_name="202511-TEST",
            analysis_date="2025-11-11",
            num_students=30,
            num_questions=15,
            avg_grade=14.0,
            median_grade=14.0,
            std_dev_grade=2.0,
            min_grade=10.0,
            max_grade=18.0
        )

        result = backend.save(metrics)

        assert result is True
        mock_table.create.assert_called_once()

    @patch('repository.AirtableBackend._ensure_table_setup')
    @patch('pyairtable.Api')
    def test_airtable_save_update_existing(self, mock_api, mock_ensure_table, temp_projects_dir):
        """Test updating an existing record in Airtable."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='airtable',
            airtable_api_key='test_key',
            airtable_base_id='test_base'
        )

        # Mock table setup
        mock_ensure_table.return_value = True

        # Setup mock table with existing record
        mock_table = MagicMock()
        existing_record = {'id': 'rec123', 'fields': {'project_name': '202511-TEST'}}
        mock_table.all.return_value = [existing_record]
        mock_api.return_value.table.return_value = mock_table

        backend = AirtableBackend(settings)

        metrics = ExamMetrics(
            project_name="202511-TEST",
            analysis_date="2025-11-11",
            num_students=35,
            num_questions=15,
            avg_grade=15.0,
            median_grade=15.0,
            std_dev_grade=2.0,
            min_grade=11.0,
            max_grade=19.0
        )

        result = backend.save(metrics)

        assert result is True
        mock_table.update.assert_called_once_with('rec123', metrics.to_dict())


class TestBaserowBackend:
    """Test the Baserow repository backend (with mocks)."""

    def test_baserow_backend_initialization(self, temp_projects_dir):
        """Test initialization of Baserow backend."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='baserow',
            baserow_api_key='test_key',
            baserow_database_id='123',
            baserow_table_id='456'
        )

        # Mock the Baserow import inside the BaserowBackend.__init__
        mock_baserow_module = MagicMock()
        mock_client = MagicMock()
        mock_baserow_module.Baserow.return_value = mock_client

        with patch.dict('sys.modules', {'pybaserow': mock_baserow_module}):
            backend = BaserowBackend(settings)

            assert backend.settings == settings
            assert backend.client == mock_client
            mock_baserow_module.Baserow.assert_called_once_with(token='test_key')

    def test_baserow_save_new_record(self, temp_projects_dir):
        """Test saving a new record to Baserow."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='baserow',
            baserow_api_key='test_key',
            baserow_database_id='123',
            baserow_table_id='456'
        )

        # Mock the Baserow import
        mock_baserow_module = MagicMock()
        mock_client = MagicMock()
        mock_client.get_rows.return_value = []  # No existing records
        mock_baserow_module.Baserow.return_value = mock_client

        with patch.dict('sys.modules', {'pybaserow': mock_baserow_module}):
            backend = BaserowBackend(settings)

            metrics = ExamMetrics(
                project_name="202511-TEST",
                analysis_date="2025-11-11",
                num_students=30,
                num_questions=15,
                avg_grade=14.0,
                median_grade=14.0,
                std_dev_grade=2.0,
                min_grade=10.0,
                max_grade=18.0
            )

            result = backend.save(metrics)

            assert result is True
            mock_client.add_row.assert_called_once()


class TestExamRepository:
    """Test the main ExamRepository class."""

    def test_repository_disabled_by_default(self, temp_projects_dir):
        """Test that repository is disabled when backend is 'none'."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='none'
        )

        repo = ExamRepository(settings)

        assert not repo.is_enabled()

    @patch('repository.AirtableBackend')
    def test_repository_enabled_with_airtable(self, mock_backend, temp_projects_dir):
        """Test that repository is enabled with Airtable backend."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='airtable',
            airtable_api_key='test_key',
            airtable_base_id='test_base'
        )

        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance

        repo = ExamRepository(settings)

        assert repo.is_enabled()

    def test_repository_save_when_disabled(self, temp_projects_dir):
        """Test saving metrics when repository is disabled."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='none'
        )

        repo = ExamRepository(settings)

        metrics = ExamMetrics(
            project_name="202511-TEST",
            analysis_date="2025-11-11",
            num_students=30,
            num_questions=15,
            avg_grade=14.0,
            median_grade=14.0,
            std_dev_grade=2.0,
            min_grade=10.0,
            max_grade=18.0
        )

        result = repo.save_exam_metrics(metrics)

        assert result is False


class TestCreateExamMetricsFromData:
    """Test the create_exam_metrics_from_data function."""

    def test_create_metrics_from_exam_data(self):
        """Test creating ExamMetrics from ExamData instance."""
        # Create mock ExamData
        mock_exam_data = MagicMock()

        # Mock general_stats dictionary
        mock_exam_data.general_stats = {
            'Number of examinees': 50,
            'Number of questions': 20,
            'Mean': 15.5,
            'Median': 16.0,
            'Standard deviation': 2.3,
            'Minimum achieved mark': 8.0,
            'Maximum achieved mark': 20.0
        }

        # Mock marks DataFrame
        mock_exam_data.marks = pd.DataFrame({
            'student': list(range(50)),
            'mark': [12.0] * 25 + [15.0] * 25  # 50 students
        })

        # Mock questions DataFrame
        mock_exam_data.questions = pd.DataFrame({
            'difficulty': [0.6, 0.7, 0.8],
            'discrimination': [0.3, 0.4, 0.5],
            'correlation': [0.2, 0.3, 0.4]
        })

        # Mock scores DataFrame for Cronbach's alpha calculation
        scores_data = []
        for student in range(50):
            for question in range(3):
                scores_data.append({
                    'student': student,
                    'question': question,
                    'score': 1.0 if student % 2 == question % 2 else 0.0
                })
        mock_exam_data.scores = pd.DataFrame(scores_data)

        metrics = create_exam_metrics_from_data(
            project_name="202511-MATH101",
            exam_data=mock_exam_data,
            pass_threshold=10.0
        )

        assert metrics.project_name == "202511-MATH101"
        assert metrics.num_students == 50
        assert metrics.num_questions == 20
        assert metrics.avg_grade == 15.5
        assert metrics.median_grade == 16.0
        assert metrics.avg_difficulty == pytest.approx(0.7, rel=0.01)
        assert metrics.pass_rate == 1.0  # All students passed (all >=10)

    def test_create_metrics_without_psychometric_data(self):
        """Test creating metrics when psychometric columns are missing."""
        mock_exam_data = MagicMock()

        mock_exam_data.general_stats = {
            'Number of examinees': 30,
            'Number of questions': 10,
            'Mean': 12.0,
            'Median': 12.5,
            'Standard deviation': 1.8,
            'Minimum achieved mark': 8.0,
            'Maximum achieved mark': 16.0
        }

        mock_exam_data.marks = pd.DataFrame({
            'student': list(range(30)),
            'mark': [11.0] * 30
        })

        # No difficulty, discrimination, or correlation columns
        mock_exam_data.questions = pd.DataFrame({
            'question': [1, 2, 3]
        })

        mock_exam_data.scores = pd.DataFrame({
            'student': [1, 2],
            'question': [1, 2],
            'score': [1.0, 0.5]
        })

        metrics = create_exam_metrics_from_data(
            project_name="202511-TEST",
            exam_data=mock_exam_data
        )

        assert metrics.avg_difficulty is None
        assert metrics.avg_discrimination is None
        assert metrics.avg_correlation is None


class TestErrorHandling:
    """Test error handling in the repository module."""

    @patch('pyairtable.Api', side_effect=ImportError("No module named 'pyairtable'"))
    def test_repository_error_on_missing_package(self, mock_api, temp_projects_dir):
        """Test that RepositoryError is raised when pyairtable is not installed."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='airtable',
            airtable_api_key='test_key',
            airtable_base_id='test_base'
        )

        with pytest.raises(RepositoryError, match="pyairtable package not installed"):
            AirtableBackend(settings)

    @patch('repository.AirtableBackend')
    def test_repository_graceful_degradation_on_init_failure(self, mock_backend, temp_projects_dir):
        """Test that ExamRepository handles backend initialization failure gracefully."""
        settings = AMCSettings(
            projects_dir=temp_projects_dir,
            repository_backend='airtable',
            airtable_api_key='test_key',
            airtable_base_id='test_base'
        )

        # Make backend initialization fail
        mock_backend.side_effect = Exception("Connection failed")

        # Should not raise, but disable the repository
        repo = ExamRepository(settings)

        assert not repo.is_enabled()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
