"""
Integration tests for AMC Report Generator.

These tests verify that multiple components work together correctly in the
complete workflow: database loading → analysis → charts → PDF generation.

Unlike unit tests (which test individual functions with mocks), integration
tests use real data and verify the entire pipeline.

Run with: pytest test_integration.py -v
Run only unit tests: pytest -m "not integration"
Run only integration tests: pytest -m integration
"""
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from amcreport import ExamData, Charts, Settings, ExamProject
from report import generate_pdf_report

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestDatabaseToAnalysisPipeline:
    """Test the complete database loading and analysis pipeline."""

    def test_load_real_databases(self):
        """Test loading actual SQLite databases from data/ directory."""
        # Get current directory (should have data/ subdirectory)
        project_path = os.getcwd()

        # Verify databases exist
        scoring_db = Path(project_path) / 'data' / 'scoring.sqlite'
        capture_db = Path(project_path) / 'data' / 'capture.sqlite'

        assert scoring_db.exists(), "scoring.sqlite not found in data/"
        assert capture_db.exists(), "capture.sqlite not found in data/"

        # Load real data
        exam_data = ExamData(project_path)

        # Verify data was loaded
        assert exam_data.number_of_examinees > 0, "No examinees loaded"
        assert len(exam_data.questions) > 0, "No questions loaded"
        assert len(exam_data.marks) > 0, "No marks loaded"

        # Verify data structure
        assert 'difficulty' in exam_data.questions.columns
        assert 'discrimination' in exam_data.questions.columns
        assert 'correlation' in exam_data.questions.columns

    def test_exam_data_statistics_calculation(self):
        """Test that statistics are calculated correctly from real data."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Verify general stats exist
        assert 'Mean' in exam_data.general_stats
        assert 'Standard deviation' in exam_data.general_stats
        assert 'Number of examinees' in exam_data.general_stats

        # Verify stats are reasonable
        assert exam_data.general_stats['Mean'] > 0
        assert exam_data.general_stats['Standard deviation'] >= 0
        assert exam_data.general_stats['Number of examinees'] == exam_data.number_of_examinees

    def test_question_metrics_calculation(self):
        """Test that question-level metrics are calculated."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Check difficulty values are in valid range [0, 1]
        assert exam_data.questions['difficulty'].min() >= 0
        assert exam_data.questions['difficulty'].max() <= 1

        # Check correlation values are in valid range [-1, 1]
        # (may have NaN for questions with no variance)
        valid_correlations = exam_data.questions['correlation'].dropna()
        if len(valid_correlations) > 0:
            assert valid_correlations.min() >= -1
            assert valid_correlations.max() <= 1

    def test_items_dataframe_populated(self):
        """Test that items (answer choices) are loaded."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Verify items exist
        assert len(exam_data.items) > 0, "No items loaded"

        # Verify items structure
        assert 'answer' in exam_data.items.columns
        assert 'correct' in exam_data.items.columns
        assert 'ticked' in exam_data.items.columns


class TestChartsIntegration:
    """Test chart generation with real exam data."""

    def test_charts_created_from_real_data(self, tmp_path):
        """Test that charts are created from actual exam data."""
        # Load real exam data
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Create temporary project directory
        temp_project_path = tmp_path / "test_project"
        temp_project_path.mkdir()

        # Mock ExamProject
        mock_project = type('obj', (object,), {'path': str(temp_project_path)})()

        # Generate charts
        Charts(mock_project, exam_data)

        # Verify chart files were created
        img_path = temp_project_path / 'img'
        assert img_path.exists(), "img directory not created"

        # Check required charts exist
        expected_charts = [
            'marks.png',
            'difficulty.png',
            'item_correlation.png',
            'question_columns.png',
        ]

        for chart_name in expected_charts:
            chart_file = img_path / chart_name
            assert chart_file.exists(), f"{chart_name} not created"
            assert chart_file.stat().st_size > 100, f"{chart_name} is too small"

    def test_discrimination_charts_when_above_threshold(self, tmp_path):
        """Test that discrimination charts are only created when examinees > threshold."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        temp_project_path = tmp_path / "test_project"
        temp_project_path.mkdir()

        mock_project = type('obj', (object,), {'path': str(temp_project_path)})()

        Charts(mock_project, exam_data)

        img_path = temp_project_path / 'img'
        discrimination_chart = img_path / 'discrimination.png'

        # Check if discrimination chart exists based on threshold
        if exam_data.number_of_examinees > exam_data.threshold:
            assert discrimination_chart.exists(), \
                f"Discrimination chart should exist (n={exam_data.number_of_examinees} > {exam_data.threshold})"
        else:
            assert not discrimination_chart.exists(), \
                f"Discrimination chart should NOT exist (n={exam_data.number_of_examinees} <= {exam_data.threshold})"


class TestCompleteReportGeneration:
    """Test the complete PDF report generation workflow."""

    @patch.dict(os.environ, {'AMC_ENABLE_AI_ANALYSIS': 'false'})
    def test_end_to_end_report_generation(self, tmp_path):
        """Test complete workflow: database → analysis → charts → PDF."""
        # Load real exam data
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Create temporary output directory
        temp_project_path = tmp_path / "test_exam"
        temp_project_path.mkdir()

        # Generate charts
        mock_project = type('obj', (object,), {'path': str(temp_project_path)})()
        Charts(mock_project, exam_data)

        # Prepare parameters for PDF generation
        params = {
            'project_name': 'Integration Test Exam',
            'project_path': str(temp_project_path),
            'company_name': 'Test University',
            'company_url': 'test.edu',
            'stats': exam_data.general_stats,
            'questions': exam_data.questions,
            'items': exam_data.items,
            'threshold': exam_data.threshold,
            'definitions': exam_data.definitions,
            'findings': exam_data.findings,
            'blurb': 'Integration test analysis.',
            'correction': 'Manual corrections: 0%',
            'palette': {
                'heading_1': (0, 102, 204, 255),
                'heading_2': (102, 153, 204, 255),
                'heading_3': (153, 204, 255, 255),
                'white': (255, 255, 255, 0),
                'yellow': (255, 255, 0, 0),
                'red': (255, 0, 0, 0),
                'green': (0, 255, 0, 0),
                'grey': (128, 128, 128, 0),
                'blue': (0, 0, 255, 0),
            },
        }

        # Mock logo check to avoid logo file requirement
        with patch('report.os.path.isfile', return_value=False):
            # Generate PDF report
            report_path = generate_pdf_report(params)

        # Verify PDF was created
        assert os.path.exists(report_path), "PDF report not created"
        assert report_path.endswith('.pdf'), "Generated file is not a PDF"

        # Verify PDF has reasonable size (should be > 10KB for real data)
        pdf_size = os.path.getsize(report_path)
        assert pdf_size > 10000, f"PDF too small ({pdf_size} bytes), likely incomplete"

    @patch.dict(os.environ, {'AMC_ENABLE_AI_ANALYSIS': 'false'})
    def test_report_with_all_sections(self, tmp_path):
        """Test that all report sections are included."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        temp_project_path = tmp_path / "full_report_test"
        temp_project_path.mkdir()

        # Generate charts
        mock_project = type('obj', (object,), {'path': str(temp_project_path)})()
        Charts(mock_project, exam_data)

        # Create comprehensive findings dictionary
        findings = exam_data.findings

        params = {
            'project_name': 'Full Report Test',
            'project_path': str(temp_project_path),
            'company_name': 'Complete Test',
            'company_url': 'complete.test',
            'stats': exam_data.general_stats,
            'questions': exam_data.questions,
            'items': exam_data.items,
            'threshold': exam_data.threshold,
            'definitions': exam_data.definitions,
            'findings': findings,
            'blurb': 'Complete report with all sections.',
            'correction': 'Manual corrections were performed on selected answers.',
            'palette': {
                'heading_1': (0, 0, 0, 255),
                'heading_2': (50, 50, 50, 255),
                'heading_3': (100, 100, 100, 255),
                'white': (255, 255, 255, 0),
                'yellow': (255, 255, 0, 0),
                'red': (255, 0, 0, 0),
                'green': (0, 255, 0, 0),
                'grey': (128, 128, 128, 0),
                'blue': (0, 0, 255, 0),
            },
        }

        with patch('report.os.path.isfile', return_value=False):
            report_path = generate_pdf_report(params)

        # Verify report exists and has content
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 10000


class TestSettingsAndProjectIntegration:
    """Test Settings and ExamProject integration with real configuration."""

    @patch.dict(os.environ, {'AMC_ENABLE_AI_ANALYSIS': 'false'})
    def test_settings_loads_from_env(self):
        """Test that Settings loads configuration from .env file."""
        from settings import get_settings

        settings = get_settings()

        # Verify settings loaded
        assert settings.projects_dir is not None
        assert settings.student_threshold > 0
        assert isinstance(settings.company_name, str)

    @pytest.mark.skip(reason="Memory intensive - copies entire databases, causes OOM on some systems")
    @patch.dict(os.environ, {'AMC_ENABLE_AI_ANALYSIS': 'false'})
    @patch('builtins.input', return_value='0')  # Select first project
    def test_exam_project_with_settings(self, mock_input, tmp_path):
        """Test ExamProject initialization with Settings."""
        # Create a temporary projects directory
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        # Create _Archive directory that ExamProject expects
        (projects_dir / "_Archive").mkdir()

        # Create a test project with data
        test_project = projects_dir / "TestProject"
        test_project.mkdir()

        # Copy real databases to test project
        real_data_dir = Path(os.getcwd()) / 'data'
        test_data_dir = test_project / 'data'
        shutil.copytree(real_data_dir, test_data_dir)

        # Mock settings
        from settings import AMCSettings
        mock_settings = AMCSettings(
            projects_dir=projects_dir,
            student_threshold=99,
            company_name="Test",
            company_url="test.com",
            enable_ai_analysis=False,
        )

        settings = Settings(settings=mock_settings)

        # This should successfully create an ExamProject
        project = ExamProject(settings)

        assert project.name is not None
        assert project.path is not None


class TestErrorHandlingInPipeline:
    """Test error handling in the integrated workflow."""

    def test_missing_database_in_workflow(self, tmp_path):
        """Test that missing databases are handled gracefully."""
        # Create directory without databases
        empty_project = tmp_path / "empty_project"
        empty_project.mkdir()

        # Should raise appropriate error
        from amcreport import DatabaseError
        with pytest.raises(DatabaseError):
            ExamData(str(empty_project))

    def test_corrupted_data_handling(self, tmp_path):
        """Test handling of corrupted data in the pipeline."""
        # This is already tested in test_error_handling.py
        # but we verify it in integration context
        project_path = tmp_path / "corrupt"
        project_path.mkdir()
        data_dir = project_path / "data"
        data_dir.mkdir()

        # Create corrupted database
        (data_dir / "scoring.sqlite").write_text("not a database")
        (data_dir / "capture.sqlite").write_text("not a database")

        # Should raise database error
        with pytest.raises(Exception):  # Could be DatabaseError or pd.errors.DatabaseError
            ExamData(str(project_path))


class TestDataConsistency:
    """Test data consistency throughout the pipeline."""

    def test_examinees_count_consistency(self):
        """Test that examinee count is consistent across all data structures."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Number should be consistent
        assert exam_data.number_of_examinees == exam_data.general_stats['Number of examinees']
        assert len(exam_data.marks) == exam_data.number_of_examinees

    def test_questions_count_consistency(self):
        """Test that question count is consistent."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        questions_count = len(exam_data.questions)

        # Should match the general stats
        assert questions_count == exam_data.general_stats['Number of questions']

    def test_marks_range_validity(self):
        """Test that all marks are within valid range."""
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # All marks should be between min and max
        min_mark = exam_data.general_stats['Minimum achieved mark']
        max_mark = exam_data.general_stats['Maximum achieved mark']

        assert exam_data.marks['mark'].min() >= min_mark
        assert exam_data.marks['mark'].max() <= max_mark

        # Max should not exceed maximum possible
        max_possible = exam_data.general_stats['Maximum possible mark']
        assert max_mark <= max_possible


class TestPerformance:
    """Test performance of the integrated workflow."""

    def test_complete_workflow_completes_reasonably_fast(self, tmp_path):
        """Test that complete workflow finishes in reasonable time."""
        import time

        start_time = time.time()

        # Run complete workflow
        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        temp_project_path = tmp_path / "perf_test"
        temp_project_path.mkdir()

        mock_project = type('obj', (object,), {'path': str(temp_project_path)})()
        Charts(mock_project, exam_data)

        elapsed = time.time() - start_time

        # Should complete in under 30 seconds for reasonable dataset
        assert elapsed < 30, f"Workflow took too long: {elapsed:.2f}s"

    def test_memory_usage_reasonable(self):
        """Test that memory usage doesn't explode."""
        import sys

        project_path = os.getcwd()
        exam_data = ExamData(project_path)

        # Get approximate size of data structures
        # This is a rough check - real datasets shouldn't be gigabytes
        marks_size = sys.getsizeof(exam_data.marks)
        questions_size = sys.getsizeof(exam_data.questions)

        # These should be reasonable (< 10MB for typical exam data)
        assert marks_size < 10_000_000, "Marks DataFrame unexpectedly large"
        assert questions_size < 10_000_000, "Questions DataFrame unexpectedly large"


@pytest.fixture(scope="module")
def real_exam_data():
    """
    Module-scoped fixture providing real exam data for tests.

    This is loaded once per module to improve test performance.
    """
    project_path = os.getcwd()
    return ExamData(project_path)


class TestWithSharedData:
    """Tests using the shared exam data fixture for efficiency."""

    def test_data_loaded_once(self, real_exam_data):
        """Verify the shared fixture works."""
        assert real_exam_data.number_of_examinees > 0

    def test_marks_statistics(self, real_exam_data):
        """Test marks statistics from shared data."""
        mean = real_exam_data.marks['mark'].mean()
        std = real_exam_data.marks['mark'].std()

        # Should match general_stats
        assert abs(mean - real_exam_data.general_stats['Mean']) < 0.01
        assert abs(std - real_exam_data.general_stats['Standard deviation']) < 0.01

    def test_question_statistics(self, real_exam_data):
        """Test question statistics from shared data."""
        # Average difficulty should be between 0 and 1
        avg_difficulty = real_exam_data.questions['difficulty'].mean()
        assert 0 <= avg_difficulty <= 1

        # Should have some variation
        assert real_exam_data.questions['difficulty'].std() > 0
