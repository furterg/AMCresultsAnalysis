"""
Pytest tests for AMC Report Generator - ExamData class.

These tests validate the core functionality of loading and analyzing
exam data from AMC SQLite databases.
"""
import os

import pandas as pd
import pytest


def test_exam_data_initialization(exam_data, test_data_path):
    """Test ExamData instance is properly initialized."""
    assert exam_data.path == test_data_path
    assert exam_data.threshold == 99


def test_database_files_exist(exam_data):
    """Test that both required database files exist."""
    assert os.path.exists(exam_data.scoring_db), "Scoring database should exist"
    assert os.path.exists(exam_data.capture_db), "Capture database should exist"


def test_student_code_length(exam_data):
    """Test student code length is correctly retrieved."""
    assert isinstance(exam_data.scl, int), "Student code length should be an integer"
    assert exam_data.scl == 8, "Expected student code length of 8"


def test_marks_dataframe(exam_data, sample_marks_data):
    """Test marks DataFrame is correctly loaded from database."""
    assert isinstance(exam_data.marks, pd.DataFrame), "Marks should be a DataFrame"
    assert exam_data.marks.head(10).to_dict() == sample_marks_data


def test_scores_dataframe(exam_data, sample_scores_data):
    """Test scores DataFrame is correctly loaded from database."""
    assert isinstance(exam_data.scores, pd.DataFrame), "Scores should be a DataFrame"
    assert exam_data.scores.head(10).to_dict() == sample_scores_data


def test_questions_dataframe(exam_data, sample_questions_data):
    """Test questions DataFrame is correctly loaded and calculated."""
    assert isinstance(exam_data.questions, pd.DataFrame), "Questions should be a DataFrame"
    assert exam_data.questions.head(10).to_dict() == sample_questions_data


def test_items_dataframe(exam_data, sample_items_data):
    """Test items DataFrame is correctly loaded and calculated."""
    assert isinstance(exam_data.items, pd.DataFrame), "Items should be a DataFrame"
    assert exam_data.items[3:9].head(10).to_dict() == sample_items_data


def test_general_statistics(exam_data, expected_general_stats):
    """
    Test general statistics are correctly calculated.

    Validates all key statistical measures including:
    - Count statistics (examinees, questions)
    - Score statistics (min, max, mean, median, mode)
    - Dispersion measures (std dev, variance)
    - Distribution shape (skew, kurtosis)
    """
    assert isinstance(exam_data.general_stats, dict), "General stats should be a dictionary"

    # Test each statistic individually for better error messages
    for stat_name, expected_value in expected_general_stats.items():
        actual_value = exam_data.general_stats[stat_name]
        assert actual_value == pytest.approx(expected_value), \
            f"{stat_name}: expected {expected_value}, got {actual_value}"


def test_statistics_table(exam_data):
    """Test that statistics table DataFrame is generated."""
    assert isinstance(exam_data.table, pd.DataFrame), "Statistics table should be a DataFrame"
    assert not exam_data.table.empty, "Statistics table should not be empty"


def test_ticked_values(exam_data, sample_ticked_data):
    """
    Test ticked values in capture data.

    Validates that the 'ticked' column correctly indicates
    which answer boxes were marked by students.
    """
    for index in range(0, len(sample_ticked_data)):
        expected = sample_ticked_data[index]
        actual = exam_data.capture['ticked'][index]
        assert actual == expected, \
            f"Ticked value at index {index}: expected {expected}, got {actual}"


# Additional validation tests

def test_exam_data_has_required_attributes(exam_data):
    """Test that ExamData has all required attributes."""
    required_attrs = [
        'path', 'threshold', 'scoring_db', 'capture_db',
        'marks', 'scores', 'questions', 'items', 'capture',
        'general_stats', 'definitions', 'findings', 'table'
    ]

    for attr in required_attrs:
        assert hasattr(exam_data, attr), f"ExamData should have attribute '{attr}'"


def test_dataframes_not_empty(exam_data):
    """Test that all major DataFrames contain data."""
    dataframes = {
        'marks': exam_data.marks,
        'scores': exam_data.scores,
        'questions': exam_data.questions,
        'items': exam_data.items,
        'capture': exam_data.capture,
    }

    for name, df in dataframes.items():
        assert not df.empty, f"{name} DataFrame should not be empty"
        assert len(df) > 0, f"{name} DataFrame should have rows"


def test_dictionaries_not_empty(exam_data):
    """Test that dictionary attributes are populated."""
    assert len(exam_data.general_stats) > 0, "General stats should not be empty"
    assert len(exam_data.definitions) > 0, "Definitions should not be empty"
    # Note: findings might be empty if no issues found, so we just check it exists
    assert isinstance(exam_data.findings, dict), "Findings should be a dictionary"
