"""
Pytest tests for PDF report generation.

These tests validate that PDF reports are generated correctly with proper
structure, content, and formatting.
"""
import os
from unittest.mock import patch

import pandas as pd
import pytest

from report import (
    PDF, to_letter, rnd_float, get_label,
    get_difficulty_label, get_discrimination_label, get_correlation_label,
    _build_report_config, _setup_report_pdf,
    generate_pdf_report
)


class TestToLetter:
    """Test to_letter conversion function."""

    def test_first_letter(self):
        """Test conversion of 1 to A."""
        assert to_letter(1) == 'A'

    def test_second_letter(self):
        """Test conversion of 2 to B."""
        assert to_letter(2) == 'B'

    def test_tenth_letter(self):
        """Test conversion of 10 to J."""
        assert to_letter(10) == 'J'

    def test_26th_letter(self):
        """Test conversion of 26 to Z."""
        assert to_letter(26) == 'Z'

    @pytest.mark.parametrize("value,expected", [
        (1, 'A'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'E'),
        (6, 'F'), (7, 'G'), (8, 'H'), (9, 'I'), (10, 'J')
    ])
    def test_multiple_conversions(self, value, expected):
        """Test multiple value conversions."""
        assert to_letter(value) == expected


class TestRndFloat:
    """Test rnd_float DataFrame rounding function."""

    def test_round_to_2_digits(self):
        """Test rounding floats to 2 decimal places."""
        df = pd.DataFrame({
            'col1': [1.23456, 2.34567, 3.45678],
            'col2': ['text', 'text', 'text']
        })

        result = rnd_float(df, 2)

        assert result['col1'][0] == '1.23'
        assert result['col1'][1] == '2.35'
        assert result['col2'][0] == 'text'  # Non-float unchanged

    def test_round_to_4_digits(self):
        """Test rounding floats to 4 decimal places."""
        df = pd.DataFrame({
            'difficulty': [0.123456789, 0.987654321]
        })

        result = rnd_float(df, 4)

        assert result['difficulty'][0] == '0.1235'
        assert result['difficulty'][1] == '0.9877'

    def test_preserves_trailing_zeros(self):
        """Test that trailing zeros are preserved."""
        df = pd.DataFrame({'col': [1.5, 2.0]})

        result = rnd_float(df, 2)

        assert result['col'][0] == '1.50'
        assert result['col'][1] == '2.00'

    def test_non_float_columns_unchanged(self):
        """Test that non-float columns are not modified."""
        df = pd.DataFrame({
            'title': ['Q001', 'Q002'],
            'count': [100, 200],
            'value': [1.234, 5.678]
        })

        result = rnd_float(df, 2)

        assert result['title'][0] == 'Q001'
        assert result['count'][0] == 100
        assert result['value'][0] == '1.23'


class TestGetLabel:
    """Test get_label dispatcher function."""

    def test_difficulty_dispatch(self):
        """Test that difficulty column is dispatched correctly."""
        label, color = get_label('difficulty', 0.5)
        assert label == 'Intermediate'
        assert color == 'yellow'

    def test_discrimination_dispatch(self):
        """Test that discrimination column is dispatched correctly."""
        label, color = get_label('discrimination', 0.4)
        assert label == 'High'
        assert color == 'green'

    def test_correlation_dispatch(self):
        """Test that correlation column is dispatched correctly."""
        label, color = get_label('correlation', 0.35)
        assert label == 'Strong'
        assert color == 'green'

    def test_unknown_column(self):
        """Test that unknown columns return default values."""
        label, color = get_label('unknown_column', 0.5)
        assert label == '-'
        assert color == 'white'


class TestDifficultyLabels:
    """Test difficulty classification."""

    @pytest.mark.parametrize("value,expected_label,expected_color", [
        (0.0, 'Difficult', 'red'),
        (0.3, 'Difficult', 'red'),
        (0.4, 'Difficult', 'red'),  # Boundary
        (0.41, 'Intermediate', 'yellow'),
        (0.5, 'Intermediate', 'yellow'),
        (0.6, 'Intermediate', 'yellow'),  # Boundary
        (0.61, 'Easy', 'green'),
        (0.8, 'Easy', 'green'),
        (1.0, 'Easy', 'green'),
    ])
    def test_difficulty_classification(self, value, expected_label, expected_color):
        """Test difficulty label classification."""
        label, color = get_difficulty_label(value)
        assert label == expected_label
        assert color == expected_color


class TestDiscriminationLabels:
    """Test discrimination classification."""

    @pytest.mark.parametrize("value,expected_label,expected_color", [
        (-0.5, 'Review!', 'red'),
        (-0.01, 'Review!', 'red'),
        (0.0, 'Low', 'grey'),
        (0.16, 'Low', 'grey'),  # Boundary
        (0.17, 'Moderate', 'yellow'),
        (0.3, 'Moderate', 'yellow'),  # Boundary
        (0.31, 'High', 'green'),
        (0.5, 'High', 'green'),  # Boundary
        (0.51, 'Very high', 'blue'),
        (1.0, 'Very high', 'blue'),
    ])
    def test_discrimination_classification(self, value, expected_label, expected_color):
        """Test discrimination label classification."""
        label, color = get_discrimination_label(value)
        assert label == expected_label
        assert color == expected_color


class TestCorrelationLabels:
    """Test correlation classification."""

    @pytest.mark.parametrize("value,expected_label,expected_color", [
        (-0.5, 'Review!', 'red'),
        (-0.01, 'Review!', 'red'),
        (0.0, 'None', 'white'),
        (0.1, 'None', 'white'),  # Boundary
        (0.11, 'Low', 'grey'),
        (0.2, 'Low', 'grey'),  # Boundary
        (0.21, 'Moderate', 'yellow'),
        (0.3, 'Moderate', 'yellow'),  # Boundary
        (0.31, 'Strong', 'green'),
        (0.45, 'Strong', 'green'),  # Boundary
        (0.46, 'Very strong', 'blue'),
        (1.0, 'Very strong', 'blue'),
    ])
    def test_correlation_classification(self, value, expected_label, expected_color):
        """Test correlation label classification."""
        label, color = get_correlation_label(value)
        assert label == expected_label
        assert color == expected_color


class TestPDFClassInitialization:
    """Test PDF class initialization."""

    def test_pdf_init_basic(self):
        """Test basic PDF initialization."""
        palette = {
            'heading_1': (0, 0, 0, 255),
            'heading_2': (50, 50, 50, 255),
            'white': (255, 255, 255, 0)
        }

        pdf = PDF('Test Project', palette, 'Test Company', 'test.com')

        assert pdf.project == 'Test Project'
        assert pdf.name == 'Test Company'
        assert pdf.url == 'test.com'
        assert pdf.colour_palette == palette
        assert pdf.margin == 10
        assert pdf.ch == 6

    def test_pdf_page_width_calculation(self):
        """Test that page width is calculated correctly."""
        palette = {'heading_1': (0, 0, 0, 255)}
        pdf = PDF('Test', palette, 'Company', 'url')

        # A4 width is 210mm, with 10mm margins on each side
        expected_pw = 210 - 2 * 10
        assert pdf.pw == expected_pw


class TestBuildReportConfig:
    """Test report configuration building."""

    def test_build_config_basic(self, tmp_path):
        """Test basic configuration building."""
        questions = pd.DataFrame({
            'title': ['Q001', 'Q002'],
            'difficulty': [0.5, 0.7],
            'discrimination': [0.3, 0.4],
            'correlation': [0.25, 0.35],
            'presented': [100, 100],
            'cancelled': [0, 0],
            'replied': [100, 100],
            'correct': [50, 70],
            'empty': [0, 0],
            'error': [0, 0],
        })

        items = pd.DataFrame({
            'question': [1, 1, 2, 2],
            'answer': [1, 2, 1, 2],
            'correct': [1, 0, 1, 0],
            'ticked': [70, 30, 80, 20],
            'discrimination': [0.4, 0.1, 0.5, 0.05],
        })

        params = {
            'project_name': 'Test Exam',
            'project_path': str(tmp_path),
            'company_name': 'Test University',
            'company_url': 'test.edu',
            'stats': {'Mean': 14.5, 'Number of examinees': 100},
            'questions': questions,
            'items': items,
            'threshold': 99,
            'definitions': {'Test': 'Definition'},
            'findings': {},
            'blurb': 'Test blurb',
            'correction': 'Correction text',
            'palette': {'white': (255, 255, 255, 0)},
        }

        config = _build_report_config(params)

        assert config['project_name'] == 'Test Exam'
        assert config['company'] == 'Test University'
        assert config['url'] == 'test.edu'
        assert config['threshold'] == 99
        assert 'questions' in config
        assert 'items' in config
        assert 'actual_data_columns' in config
        assert 'actual_analysis_columns' in config

    def test_build_config_filters_missing_columns(self):
        """Test that missing columns are filtered out."""
        questions = pd.DataFrame({
            'title': ['Q001'],
            'difficulty': [0.5],
            'presented': [100],
            'correct': [50],
            # Missing: discrimination, correlation, cancelled, replied, empty, error
        })

        params = {
            'project_name': 'Test',
            'project_path': '/tmp',
            'company_name': '',
            'company_url': '',
            'stats': {},
            'questions': questions,
            'items': pd.DataFrame(),
            'threshold': 99,
            'definitions': {},
            'findings': {},
            'blurb': '',
            'correction': '',
            'palette': {},
        }

        config = _build_report_config(params)

        # Only columns that exist should be in actual_data_columns
        assert 'presented' in config['actual_data_columns']
        assert 'correct' in config['actual_data_columns']
        assert 'cancelled' not in config['actual_data_columns']
        assert 'replied' not in config['actual_data_columns']

        # Only existing analysis columns
        assert 'difficulty' in config['actual_analysis_columns']
        assert 'discrimination' not in config['actual_analysis_columns']
        assert 'correlation' not in config['actual_analysis_columns']


class TestSetupReportPDF:
    """Test PDF setup with title page."""

    @patch('report.os.path.isfile')
    def test_setup_creates_pdf_object(self, mock_isfile, tmp_path):
        """Test that _setup_report_pdf creates a PDF object."""
        mock_isfile.return_value = False  # Simulate missing logo

        config = {
            'project_name': 'Test Exam',
            'company': 'Test Company',
            'url': 'test.com',
            'palette': {
                'heading_1': (0, 0, 0, 255),
                'heading_2': (50, 50, 50, 255),
            }
        }

        pdf = _setup_report_pdf(config)

        assert isinstance(pdf, PDF)
        assert pdf.project == 'Test Exam'
        assert pdf.page_no() >= 1  # At least one page should be added

    @patch('report.os.path.isfile')
    @patch('report.PDF.image')
    def test_setup_adds_logo_when_exists(self, mock_image, mock_isfile, tmp_path):
        """Test that logo is added when it exists."""
        mock_isfile.return_value = True  # Simulate logo exists

        config = {
            'project_name': 'Test',
            'company': 'Company',
            'url': 'url',
            'palette': {
                'heading_1': (0, 0, 0, 255),
                'heading_2': (50, 50, 50, 255),
            }
        }

        pdf = _setup_report_pdf(config)

        # Verify image was called
        mock_image.assert_called_once()


class TestGeneratePDFReport:
    """Test complete PDF report generation."""

    @patch('report.os.path.isfile')
    def test_generate_pdf_creates_file(self, mock_isfile, tmp_path):
        """Test that generate_pdf_report creates a PDF file."""
        mock_isfile.return_value = False  # No logo

        # Create img directory
        img_dir = tmp_path / 'img'
        img_dir.mkdir()

        # Create dummy chart files (includes discrimination charts since examinees > threshold)
        chart_names = ['marks.png', 'difficulty.png', 'discrimination.png',
                      'discrimination_vs_difficulty.png', 'item_correlation.png', 'question_columns.png']
        for chart_name in chart_names:
            chart_file = img_dir / chart_name
            # Create a minimal PNG file (1x1 transparent pixel)
            chart_file.write_bytes(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
                b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            )

        # Create realistic exam data
        questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'difficulty': [0.65, 0.45, 0.80],
            'discrimination': [0.35, 0.42, 0.28],
            'correlation': [0.32, 0.38, 0.25],
            'presented': [100, 100, 100],
            'cancelled': [0, 0, 0],
            'replied': [100, 100, 100],
            'correct': [65, 45, 80],
            'empty': [0, 0, 0],
            'error': [0, 0, 0],
        })

        items = pd.DataFrame({
            'title': ['Q001', 'Q001', 'Q001', 'Q001'],
            'question': [1, 1, 1, 1],
            'answer': [1, 2, 3, 4],
            'correct': [1, 0, 0, 0],
            'ticked': [65, 20, 10, 5],
            'discrimination': [0.35, 0.15, 0.08, 0.05],
        })

        params = {
            'project_name': 'Test Exam',
            'project_path': str(tmp_path),
            'company_name': 'Test University',
            'company_url': 'test.edu',
            'stats': {
                'Number of examinees': 100,
                'Number of questions': 3,
                'Maximum possible mark': 20.0,
                'Minimum achieved mark': 10.5,
                'Maximum achieved mark': 18.2,
                'Mean': 14.5,
                'Median': 14.3,
                'Mode': 15.0,
                'Standard deviation': 2.1,
                'Variance': 4.41,
                'Standard error of mean': 0.21,
                'Standard error of measurement': 0.95,
                'Skew': -0.15,
                'Kurtosis': -0.42,
            },
            'questions': questions,
            'items': items,
            'threshold': 99,
            'definitions': {
                'Difficulty': 'Proportion of students answering correctly',
                'Discrimination': 'How well the question differentiates'
            },
            'findings': {},
            'blurb': 'This is a test exam analysis.',
            'correction': 'Manual corrections: 5%',
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

        report_path = generate_pdf_report(params)

        # Verify PDF was created
        assert os.path.exists(report_path)
        assert report_path.endswith('.pdf')

        # Verify file has content
        assert os.path.getsize(report_path) > 1000  # At least 1KB

    @patch('report.os.path.isfile')
    def test_generate_pdf_returns_correct_path(self, mock_isfile, tmp_path):
        """Test that generate_pdf_report returns the correct path."""
        mock_isfile.return_value = False

        # Create minimal data
        questions = pd.DataFrame({
            'title': ['Q001'],
            'difficulty': [0.5],
            'discrimination': [0.3],
            'correlation': [0.25],
            'presented': [50],
            'cancelled': [0],
            'replied': [50],
            'correct': [25],
            'empty': [0],
            'error': [0],
        })

        items = pd.DataFrame({
            'title': ['Q001'],
            'question': [1],
            'answer': [1],
            'correct': [1],
            'ticked': [25],
            'discrimination': [0.3],
        })

        # Create img directory with minimal charts
        img_dir = tmp_path / 'img'
        img_dir.mkdir()
        for chart in ['marks.png', 'difficulty.png', 'item_correlation.png', 'question_columns.png']:
            (img_dir / chart).write_bytes(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
                b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            )

        params = {
            'project_name': 'TestReport',
            'project_path': str(tmp_path),
            'company_name': 'Test',
            'company_url': 'test.com',
            'stats': {
                'Number of examinees': 50,
                'Number of questions': 1,
                'Maximum possible mark': 20.0,
                'Minimum achieved mark': 10.0,
                'Maximum achieved mark': 18.0,
                'Mean': 14.0,
                'Median': 14.0,
                'Mode': 15.0,
                'Standard deviation': 2.5,
                'Variance': 6.25,
                'Standard error of mean': 0.35,
                'Standard error of measurement': 0.35,
                'Skew': -0.1,
                'Kurtosis': -0.3,
            },
            'questions': questions,
            'items': items,
            'threshold': 99,
            'definitions': {},
            'findings': {},
            'blurb': 'Test',
            'correction': 'Test',
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

        report_path = generate_pdf_report(params)

        expected_path = str(tmp_path / 'TestReport-report.pdf')
        assert report_path == expected_path


class TestPDFErrorHandling:
    """Test error handling in PDF generation."""

    @patch('report.os.path.isfile')
    def test_missing_chart_files_handled(self, mock_isfile, tmp_path):
        """Test that missing chart files are handled gracefully."""
        mock_isfile.return_value = False  # No logo

        # Create img directory but NO chart files
        img_dir = tmp_path / 'img'
        img_dir.mkdir()

        questions = pd.DataFrame({
            'title': ['Q001'],
            'difficulty': [0.5],
            'discrimination': [0.3],
            'correlation': [0.25],
            'presented': [50],
            'cancelled': [0],
            'replied': [50],
            'correct': [25],
            'empty': [0],
            'error': [0],
        })

        items = pd.DataFrame({
            'title': ['Q001'],
            'question': [1],
            'answer': [1],
            'correct': [1],
            'ticked': [25],
            'discrimination': [0.3],
        })

        params = {
            'project_name': 'Test',
            'project_path': str(tmp_path),
            'company_name': '',
            'company_url': '',
            'stats': {
                'Number of examinees': 50,
                'Number of questions': 1,
                'Maximum possible mark': 20.0,
                'Minimum achieved mark': 10.0,
                'Maximum achieved mark': 18.0,
                'Mean': 14.0,
                'Median': 14.0,
                'Mode': 15.0,
                'Standard deviation': 2.5,
                'Variance': 6.25,
                'Standard error of mean': 0.35,
                'Standard error of measurement': 0.35,
                'Skew': -0.1,
                'Kurtosis': -0.3,
            },
            'questions': questions,
            'items': items,
            'threshold': 99,
            'definitions': {},
            'findings': {},
            'blurb': '',
            'correction': '',
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

        # Should raise error when trying to load missing images
        with pytest.raises(FileNotFoundError):
            generate_pdf_report(params)

    @patch('report.os.path.isfile')
    def test_empty_dataframe_handled(self, mock_isfile, tmp_path):
        """Test that empty DataFrames are handled."""
        mock_isfile.return_value = False

        # Create img directory
        img_dir = tmp_path / 'img'
        img_dir.mkdir()

        # Empty DataFrames
        questions = pd.DataFrame(columns=['title', 'difficulty', 'discrimination', 'correlation'])
        items = pd.DataFrame(columns=['title', 'question', 'answer', 'correct', 'ticked'])

        params = {
            'project_name': 'Empty',
            'project_path': str(tmp_path),
            'company_name': '',
            'company_url': '',
            'stats': {
                'Number of examinees': 0,
                'Number of questions': 0,
                'Maximum possible mark': 0.0,
                'Minimum achieved mark': 0.0,
                'Maximum achieved mark': 0.0,
                'Mean': 0.0,
                'Median': 0.0,
                'Mode': 0.0,
                'Standard deviation': 0.0,
                'Variance': 0.0,
                'Standard error of mean': 0.0,
                'Standard error of measurement': 0.0,
                'Skew': 0.0,
                'Kurtosis': 0.0,
            },
            'questions': questions,
            'items': items,
            'threshold': 99,
            'definitions': {},
            'findings': {},
            'blurb': '',
            'correction': '',
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

        # Should raise error due to missing chart files or attempt to process empty data
        with pytest.raises((FileNotFoundError, KeyError, IndexError, ValueError)):
            generate_pdf_report(params)
