"""
Pytest configuration and shared fixtures for AMC Report Generator tests.

This file provides common fixtures used across multiple test modules.
"""
import os

import pytest


@pytest.fixture(scope="session")
def test_data_path():
    """
    Provide the path to test data directory.

    Returns:
        str: Absolute path to the test data directory
    """
    return os.getcwd()


@pytest.fixture(scope="session")
def sample_marks_data():
    """
    Provide sample marks data for testing.

    Returns:
        dict: Sample marks data dictionary
    """
    return {
        'copy': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        'mark': {0: 18.5, 1: 11.5, 2: 14.0, 3: 18.0, 4: 11.5, 5: 13.5, 6: 12.5, 7: 11.5, 8: 17.5, 9: 17.0},
        'max': {0: 40.0, 1: 40.0, 2: 40.0, 3: 40.0, 4: 40.0, 5: 40.0, 6: 40.0, 7: 40.0, 8: 40.0, 9: 40.0},
        'student': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 12, 9: 13},
        'total': {0: 37.0, 1: 23.0, 2: 28.0, 3: 36.0, 4: 23.0, 5: 27.0, 6: 25.0, 7: 23.0, 8: 35.0, 9: 34.0}
    }


@pytest.fixture(scope="session")
def sample_scores_data():
    """
    Provide sample scores data for testing.

    Returns:
        dict: Sample scores data dictionary
    """
    return {
        'correct': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        'empty': {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False},
        'error': {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False},
        'max': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0},
        'question': {0: 14, 1: 40, 2: 11, 3: 44, 4: 25, 5: 19, 6: 29, 7: 30, 8: 27, 9: 24},
        'replied': {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True},
        'score': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0},
        'student': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        'title': {0: 'Q001', 1: 'Q002', 2: 'Q003', 3: 'Q004', 4: 'Q005', 5: 'Q006', 6: 'Q007', 7: 'Q008', 8: 'Q009', 9: 'Q010'}
    }


@pytest.fixture(scope="session")
def sample_questions_data():
    """
    Provide sample questions data for testing.

    Returns:
        dict: Sample questions data dictionary
    """
    return {
        'question': {0: 9, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16, 8: 17, 9: 18},
        'title': {0: 'Q017', 1: 'Q016', 2: 'Q003', 3: 'Q011', 4: 'Q013', 5: 'Q001', 6: 'Q040', 7: 'Q028', 8: 'Q018', 9: 'Q036'},
        'correct': {0: 281, 1: 213, 2: 350, 3: 182, 4: 345, 5: 298, 6: 398, 7: 318, 8: 433, 9: 347},
        'empty': {0: 5, 1: 10, 2: 4, 3: 3, 4: 5, 5: 2, 6: 2, 7: 1, 8: 3, 9: 7},
        'error': {0: 2, 1: 2, 2: 0, 3: 0, 4: 1, 5: 16, 6: 4, 7: 0, 8: 0, 9: 0},
        'max': {0: 474.0, 1: 474.0, 2: 474.0, 3: 474.0, 4: 474.0, 5: 474.0, 6: 474.0, 7: 474.0, 8: 474.0, 9: 474.0},
        'replied': {0: 467, 1: 462, 2: 470, 3: 471, 4: 468, 5: 456, 6: 468, 7: 473, 8: 471, 9: 467},
        'score': {0: 281.0, 1: 213.0, 2: 350.0, 3: 182.0, 4: 345.0, 5: 298.0, 6: 398.0, 7: 318.0, 8: 433.0, 9: 347.0},
        'presented': {0: 474, 1: 474, 2: 474, 3: 474, 4: 474, 5: 474, 6: 474, 7: 474, 8: 474, 9: 474},
        'difficulty': {0: 0.5928270042194093, 1: 0.44936708860759494, 2: 0.7383966244725738, 3: 0.38396624472573837,
                       4: 0.7278481012658228, 5: 0.6286919831223629, 6: 0.8396624472573839, 7: 0.6708860759493671,
                       8: 0.9135021097046413, 9: 0.7320675105485233},
        'discrimination': {0: 0.3671875, 1: 0.390625, 2: 0.359375, 3: 0.359375, 4: 0.484375, 5: 0.2265625, 6: 0.25,
                           7: 0.5546875, 8: 0.1796875, 9: 0.28125},
        'correlation': {0: 0.3173495638922919, 1: 0.36031896564299354, 2: 0.31463705038664697,
                        3: 0.29833978514380544, 4: 0.37881702536551237, 5: 0.19032832848928796,
                        6: 0.29112653133639765, 7: 0.42286324443180573, 8: 0.22577274248615295,
                        9: 0.23012605242119163}
    }


@pytest.fixture(scope="session")
def sample_items_data():
    """
    Provide sample items data for testing.

    Returns:
        dict: Sample items data dictionary
    """
    return {
        'question': {3: 14, 4: 40, 5: 40, 6: 40, 7: 40, 8: 11},
        'title': {3: 'Q001', 4: 'Q002', 5: 'Q002', 6: 'Q002', 7: 'Q002', 8: 'Q003'},
        'answer': {3: 4, 4: 1, 5: 2, 6: 3, 7: 4, 8: 1},
        'correct': {3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0},
        'ticked': {3: 83.0, 4: 49.0, 5: 178.0, 6: 172.0, 7: 79.0, 8: 30.0},
        'discrimination': {3: -0.0546875, 4: -0.0546875, 5: -0.2109375, 6: 0.3046875, 7: -0.0625, 8: -0.1015625},
        'correlation': {3: -0.04986524106757333, 4: -0.06578468362575698, 5: -0.20712916273334014,
                        6: 0.2715063079185362, 7: -0.05015494837646934, 8: -0.1834758886587201}
    }


@pytest.fixture(scope="session")
def sample_ticked_data():
    """
    Provide sample ticked data for testing.

    Returns:
        list: Sample ticked data
    """
    return [1, 0, 0, 0, 0, 0, 1, 0, 1, 0]


@pytest.fixture(scope="session")
def expected_general_stats():
    """
    Provide expected general statistics for validation.

    Returns:
        dict: Expected general statistics
    """
    return {
        'Number of examinees': 474,
        'Number of questions': 40,
        'Maximum possible mark': 20.0,
        'Minimum achieved mark': 7.0,
        'Maximum achieved mark': 20.0,
        'Mean': 14.079113924050633,
        'Median': 14.0,
        'Mode': 15.0,
        'Standard deviation': 2.5704793556984846,
        'Variance': 6.607364118072096,
        'Standard error of mean': 0.11806602706531438,
        'Standard error of measurement': 0.11806602706531438,
        'Skew': -0.11830246494808731,
        'Kurtosis': -0.31934211231713405
    }


@pytest.fixture
def exam_data(test_data_path):
    """
    Provide an ExamData instance for testing.

    This fixture creates an ExamData object from the test database files.
    It's function-scoped so each test gets a fresh instance.

    Args:
        test_data_path: Path to test data directory (from fixture)

    Returns:
        ExamData: Instance loaded with test data
    """
    from amcreport import ExamData
    return ExamData(test_data_path)
