"""
Pytest tests for utility functions.

These tests validate helper functions used throughout the AMC Report Generator.
"""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from amcreport import (
    get_list_questions,
    add_blurb_conditionally,
    get_blurb,
    get_correction_text,
    ExamData
)


class TestGetListQuestions:
    """Test get_list_questions function."""

    def test_single_question(self):
        """
        Test formatting a single question.

        Note: In practice, get_list_questions is only called with 2+ items.
        Single items are handled separately in the calling code.
        This test documents the actual behavior with 1 item.
        """
        result = get_list_questions(['Q001'])
        # Actual behavior: returns ' and Q001' for single item
        assert result == ' and Q001'

    def test_two_questions(self):
        """Test formatting two questions."""
        result = get_list_questions(['Q001', 'Q002'])
        assert result == 'Q001 and Q002'

    def test_three_questions(self):
        """Test formatting three questions."""
        result = get_list_questions(['Q001', 'Q002', 'Q003'])
        assert result == 'Q001, Q002 and Q003'

    def test_many_questions(self):
        """Test formatting many questions."""
        result = get_list_questions(['Q001', 'Q002', 'Q003', 'Q004', 'Q005'])
        assert result == 'Q001, Q002, Q003, Q004 and Q005'

    @pytest.mark.parametrize("questions,expected", [
        (['A'], ' and A'),  # Single item edge case (not used in practice)
        (['A', 'B'], 'A and B'),
        (['A', 'B', 'C'], 'A, B and C'),
        (['A', 'B', 'C', 'D'], 'A, B, C and D'),
        (['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'], 'Q1, Q2, Q3, Q4, Q5 and Q6'),
    ])
    def test_various_list_sizes(self, questions, expected):
        """
        Test various list sizes with parametrization.

        Note: Single-item lists return ' and X' but are not used in practice.
        The calling code handles single items separately.
        """
        assert get_list_questions(questions) == expected

    def test_preserves_question_format(self):
        """Test that question formatting is preserved."""
        questions = ['Q001', 'Q010', 'Q100']
        result = get_list_questions(questions)
        assert 'Q001' in result
        assert 'Q010' in result
        assert 'Q100' in result

    def test_with_numeric_strings(self):
        """Test with numeric question identifiers."""
        result = get_list_questions(['1', '2', '3'])
        assert result == '1, 2 and 3'


class TestAddBlurbConditionally:
    """Test add_blurb_conditionally function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003', 'Q004'],
            'cancelled': [10, 50, 5, 80],
            'empty': [5, 10, 40, 15],
            'score': [90, 70, 60, 85],
        })

    def test_no_items_above_threshold(self, sample_dataframe):
        """Test when no items exceed threshold."""
        result = add_blurb_conditionally(
            sample_dataframe,
            'cancelled',
            100,  # Threshold higher than all values
            "Questions {qlist} were cancelled."
        )
        assert result == ''

    def test_single_item_above_threshold(self, sample_dataframe):
        """Test when single item exceeds threshold."""
        result = add_blurb_conditionally(
            sample_dataframe,
            'cancelled',
            60,  # Only Q004 (80) exceeds this
            "Questions {qlist} were cancelled."
        )
        assert result == "Questions Q004 were cancelled."

    def test_multiple_items_above_threshold(self, sample_dataframe):
        """Test when multiple items exceed threshold."""
        result = add_blurb_conditionally(
            sample_dataframe,
            'cancelled',
            30,  # Q002 (50) and Q004 (80) exceed this
            "Questions {qlist} were cancelled."
        )
        assert result == "Questions Q002 and Q004 were cancelled."

    def test_column_not_in_dataframe(self, sample_dataframe):
        """Test when column doesn't exist."""
        result = add_blurb_conditionally(
            sample_dataframe,
            'nonexistent_column',
            50,
            "Questions {qlist} had issues."
        )
        assert result == ''

    def test_message_template_formatting(self, sample_dataframe):
        """Test that message template is correctly formatted."""
        result = add_blurb_conditionally(
            sample_dataframe,
            'empty',
            20,  # Q003 (40) exceeds this
            "- Questions {qlist} have been empty more than threshold.\n"
        )
        assert result.startswith("- Questions")
        assert result.endswith("more than threshold.\n")
        assert "Q003" in result

    def test_sorted_by_title(self, sample_dataframe):
        """Test that results are sorted by title."""
        df = pd.DataFrame({
            'title': ['Q003', 'Q001', 'Q002'],
            'value': [100, 100, 100],
        })
        result = add_blurb_conditionally(
            df,
            'value',
            50,
            "Questions: {qlist}"
        )
        # Should be sorted: Q001, Q002, Q003
        assert result == "Questions: Q001, Q002 and Q003"

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'title': [], 'value': []})
        result = add_blurb_conditionally(
            df,
            'value',
            50,
            "Questions: {qlist}"
        )
        assert result == ''


class TestGetBlurb:
    """Test get_blurb function."""

    @pytest.fixture
    def mock_exam_no_issues(self):
        """Create mock ExamData with no issues."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'presented': [100, 100, 100],
            'cancelled': [0, 0, 0],
            'empty': [0, 0, 0],
            'discrimination': [0.5, 0.4, 0.3],
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'ticked': [50, 50, 50],
        })
        return exam

    @pytest.fixture
    def mock_exam_with_cancelled(self):
        """Create mock ExamData with cancelled questions."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'presented': [100, 100, 100],
            'cancelled': [85, 10, 5],  # Q001 cancelled >80%
            'empty': [0, 0, 0],
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'ticked': [50, 50, 50],
        })
        return exam

    @pytest.fixture
    def mock_exam_with_empty(self):
        """Create mock ExamData with empty questions."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'presented': [100, 100, 100],
            'cancelled': [0, 0, 0],
            'empty': [85, 10, 5],  # Q001 empty >80%
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'ticked': [50, 50, 50],
        })
        return exam

    @pytest.fixture
    def mock_exam_with_negative_discrimination(self):
        """Create mock ExamData with negative discrimination."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'presented': [100, 100, 100],
            'cancelled': [0, 0, 0],
            'empty': [0, 0, 0],
            'discrimination': [-0.2, 0.4, -0.1],  # Q001 and Q003 negative
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'ticked': [50, 50, 50],
        })
        return exam

    @pytest.fixture
    def mock_exam_with_unticked_distractors(self):
        """Create mock ExamData with unticked distractors."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'presented': [100, 100, 100],
            'cancelled': [0, 0, 0],
            'empty': [0, 0, 0],
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003', 'Q001', 'Q002'],
            'ticked': [0, 0, 50, 50, 50],  # Q001 and Q002 have unticked items
        })
        return exam

    def test_no_issues_message(self, mock_exam_no_issues):
        """Test blurb when there are no issues."""
        result = get_blurb(mock_exam_no_issues)
        assert result == "According to the data collected, there are no questions to review based on their performance."

    def test_cancelled_questions_blurb(self, mock_exam_with_cancelled):
        """Test blurb for cancelled questions."""
        result = get_blurb(mock_exam_with_cancelled)
        assert "According to the data collected, the following questions should probably be reviewed:" in result
        assert "Q001" in result
        assert "cancelled more than 80% of the time" in result

    def test_empty_questions_blurb(self, mock_exam_with_empty):
        """Test blurb for empty questions."""
        result = get_blurb(mock_exam_with_empty)
        assert "According to the data collected, the following questions should probably be reviewed:" in result
        assert "Q001" in result
        assert "empty more than 80% of the time" in result

    def test_negative_discrimination_blurb(self, mock_exam_with_negative_discrimination):
        """Test blurb for negative discrimination."""
        result = get_blurb(mock_exam_with_negative_discrimination)
        assert "According to the data collected, the following questions should probably be reviewed:" in result
        assert "Q001 and Q003" in result or ("Q001" in result and "Q003" in result)
        assert "negative discrimination" in result
        assert "incorrect outcome" in result

    def test_unticked_distractors_single(self):
        """Test blurb for single question with unticked distractors."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002'],
            'presented': [100, 100],
            'cancelled': [0, 0],
            'empty': [0, 0],
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002'],
            'ticked': [0, 50],  # Only Q001 has unticked
        })

        result = get_blurb(exam)
        assert "Question Q001 has distractors that have never been chosen" in result

    def test_unticked_distractors_multiple(self, mock_exam_with_unticked_distractors):
        """Test blurb for multiple questions with unticked distractors."""
        result = get_blurb(mock_exam_with_unticked_distractors)
        assert "Questions" in result
        assert "Q001" in result
        assert "Q002" in result
        assert "distractors that have never been chosen" in result

    def test_multiple_issues_combined(self):
        """Test blurb when multiple issues exist."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'presented': [100, 100, 100],
            'cancelled': [85, 0, 0],  # Q001 cancelled
            'empty': [0, 85, 0],  # Q002 empty
            'discrimination': [0.3, 0.4, -0.2],  # Q003 negative
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002', 'Q003'],
            'ticked': [50, 50, 50],
        })

        result = get_blurb(exam)
        assert "Q001" in result  # From cancelled
        assert "Q002" in result  # From empty
        assert "Q003" in result  # From negative discrimination
        assert result.count("-") >= 3  # At least 3 bullet points

    def test_blurb_without_discrimination_column(self):
        """Test blurb when discrimination column is missing."""
        exam = Mock(spec=ExamData)
        exam.questions = pd.DataFrame({
            'title': ['Q001', 'Q002'],
            'presented': [100, 100],
            'cancelled': [85, 0],
            'empty': [0, 0],
            # No discrimination column
        })
        exam.items = pd.DataFrame({
            'title': ['Q001', 'Q002'],
            'ticked': [50, 50],
        })

        result = get_blurb(exam)
        # Should still work, just without discrimination section
        assert "Q001" in result
        assert "cancelled" in result
        # Should not raise an error


class TestGetCorrectionText:
    """Test get_correction_text function."""

    def test_basic_correction_text(self):
        """Test basic correction text generation."""
        df = pd.DataFrame({
            'student': [1, 2, 3, 4, 5],
            'manual': [1, 0, -1, 1, 0],
            'black': [150, 200, 100, 200, 150],  # Threshold is 180
        })

        result = get_correction_text(df)

        assert "5 boxes in total" in result
        assert "This examination is comprised of" in result
        assert "have been manually filled" in result
        assert "have been manually emptied" in result
        assert "have not been changed" in result

    def test_filled_boxes_calculation(self):
        """Test calculation of manually filled boxes."""
        df = pd.DataFrame({
            'student': [1, 2, 3],
            'manual': [1, 1, 0],  # Two with manual=1
            'black': [100, 150, 200],  # First two < 180 (filled)
        })

        result = get_correction_text(df)
        assert "2 (66.67%)" in result  # 2 filled out of 3

    def test_emptied_boxes_calculation(self):
        """Test calculation of manually emptied boxes."""
        df = pd.DataFrame({
            'student': [1, 2, 3],
            'manual': [0, 0, 1],  # Two with manual=0
            'black': [200, 190, 100],  # First two > 180 (emptied)
        })

        result = get_correction_text(df)
        assert "2 (66.67%)" in result  # 2 emptied out of 3

    def test_untouched_boxes_calculation(self):
        """Test calculation of untouched boxes."""
        df = pd.DataFrame({
            'student': [1, 2, 3, 4],
            'manual': [-1, 1, 0, -1],  # Two with manual=-1
            'black': [100, 200, 100, 150],  # manual[1]=1 & black>180, manual[2]=0 & black<180
        })

        result = get_correction_text(df)
        # Untouched: 2 with manual=-1, + 1 (manual=1 & black>180), + 1 (manual=0 & black<180) = 4
        assert "4 (100.00%)" in result

    def test_all_boxes_manually_filled(self):
        """Test when all boxes are manually filled."""
        df = pd.DataFrame({
            'student': [1, 2, 3],
            'manual': [1, 1, 1],
            'black': [100, 120, 150],  # All < 180
        })

        result = get_correction_text(df)
        assert "3 (100.00%)" in result or "100.00%" in result
        assert "0 (0.00%)" in result  # No emptied

    def test_no_manual_corrections(self):
        """Test when no manual corrections were made."""
        df = pd.DataFrame({
            'student': [1, 2, 3],
            'manual': [-1, -1, -1],
            'black': [100, 150, 200],
        })

        result = get_correction_text(df)
        assert "0 (0.00%)" in result  # No filled
        assert "3 (100.00%)" in result or "100.00%" in result  # All untouched

    def test_percentage_formatting(self):
        """Test that percentages are formatted correctly."""
        df = pd.DataFrame({
            'student': list(range(10)),
            'manual': [1, 0, -1, 1, 0, -1, 1, 0, -1, 1],
            'black': [100, 200, 100, 150, 190, 120, 140, 210, 160, 170],
        })

        result = get_correction_text(df)
        # Check that percentages contain decimal places
        assert "%" in result
        # Should have format like "X.XX%"
        import re
        percentages = re.findall(r'\d+\.\d{2}%', result)
        assert len(percentages) == 3  # Three percentages in the text

    def test_large_dataset(self):
        """Test with a large dataset."""
        df = pd.DataFrame({
            'student': list(range(1000)),
            'manual': np.random.choice([1, 0, -1], 1000),
            'black': np.random.randint(0, 255, 1000),
        })

        result = get_correction_text(df)
        assert "1000 boxes in total" in result
        assert "This examination is comprised of" in result

    def test_boundary_darkness_threshold(self):
        """Test boxes exactly at the darkness threshold (180)."""
        df = pd.DataFrame({
            'student': [1, 2, 3],
            'manual': [1, 0, 1],
            'black': [180, 180, 100],
        })

        result = get_correction_text(df)
        # Threshold is 180:
        # - manual=1 & black<180: filled (only index 2)
        # - manual=0 & black>180: emptied (none, 180 is not >180)
        # - Others are untouched
        assert "1 (33.33%)" in result or "33.33%" in result  # 1 filled
