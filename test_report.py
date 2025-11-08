"""
Pytest tests for report module - get_label function.

These tests validate the label classification logic used in PDF reports
to categorize difficulty, discrimination, and correlation values.
"""
import pytest

from report import get_label


class TestDifficultyLabels:
    """Test difficulty label classification."""

    @pytest.mark.parametrize("value,expected_label,expected_color", [
        (0.0, 'Difficult', 'red'),
        (0.3, 'Difficult', 'red'),
        (0.39, 'Difficult', 'red'),
        (0.4, 'Intermediate', 'yellow'),
        (0.5, 'Intermediate', 'yellow'),
        (0.59, 'Intermediate', 'yellow'),
        (0.6, 'Easy', 'green'),
        (0.8, 'Easy', 'green'),
        (1.0, 'Easy', 'green'),
    ])
    def test_difficulty_labels(self, value, expected_label, expected_color):
        """
        Test difficulty labels for various values.

        Difficulty interpretation:
        - < 0.4: Difficult (red) - Less than 40% answered correctly
        - 0.4-0.59: Intermediate (yellow) - 40-59% answered correctly
        - ≥ 0.6: Easy (green) - 60% or more answered correctly
        """
        assert get_label('difficulty', value) == (expected_label, expected_color)


class TestDiscriminationLabels:
    """Test discrimination label classification."""

    @pytest.mark.parametrize("value,expected_label,expected_color", [
        (-1.0, 'Review!', 'red'),
        (-0.1, 'Review!', 'red'),
        (-0.01, 'Review!', 'red'),
        (0.0, 'Low', 'grey'),
        (0.1, 'Low', 'grey'),
        (0.19, 'Low', 'grey'),
        (0.2, 'Moderate', 'yellow'),
        (0.25, 'Moderate', 'yellow'),
        (0.29, 'Moderate', 'yellow'),
        (0.3, 'High', 'green'),
        (0.4, 'High', 'green'),
        (0.49, 'High', 'green'),
        (0.5, 'Very high', 'blue'),
        (0.6, 'Very high', 'blue'),
        (1.0, 'Very high', 'blue'),
    ])
    def test_discrimination_labels(self, value, expected_label, expected_color):
        """
        Test discrimination labels for various values.

        Discrimination interpretation (top 27% vs bottom 27%):
        - < 0: Review! (red) - Bottom students perform better (wrong key?)
        - 0-0.19: Low (grey) - Poor discrimination
        - 0.2-0.29: Moderate (yellow) - Acceptable discrimination
        - 0.3-0.49: High (green) - Good discrimination
        - ≥ 0.5: Very high (blue) - Excellent discrimination
        """
        assert get_label('discrimination', value) == (expected_label, expected_color)


class TestCorrelationLabels:
    """Test correlation label classification."""

    @pytest.mark.parametrize("value,expected_label,expected_color", [
        (-1.0, 'Review!', 'red'),
        (-0.1, 'Review!', 'red'),
        (-0.01, 'Review!', 'red'),
        (0.0, 'None', 'white'),
        (0.05, 'None', 'white'),
        (0.09, 'None', 'white'),
        (0.1, 'Low', 'grey'),
        (0.15, 'Low', 'grey'),
        (0.19, 'Low', 'grey'),
        (0.2, 'Moderate', 'yellow'),
        (0.25, 'Moderate', 'yellow'),
        (0.29, 'Moderate', 'yellow'),
        (0.3, 'Strong', 'green'),
        (0.4, 'Strong', 'green'),
        (0.49, 'Strong', 'green'),
        (0.5, 'Very strong', 'blue'),
        (0.6, 'Very strong', 'blue'),
        (1.0, 'Very strong', 'blue'),
    ])
    def test_correlation_labels(self, value, expected_label, expected_color):
        """
        Test correlation labels for various values.

        Point-biserial correlation interpretation:
        - < 0: Review! (red) - Negative correlation (wrong key?)
        - 0-0.09: None (white) - No correlation
        - 0.1-0.19: Low (grey) - Weak positive correlation
        - 0.2-0.29: Moderate (yellow) - Moderate correlation
        - 0.3-0.49: Strong (green) - Strong correlation
        - ≥ 0.5: Very strong (blue) - Very strong correlation
        """
        assert get_label('correlation', value) == (expected_label, expected_color)


class TestInvalidColumn:
    """Test error handling for invalid column names."""

    @pytest.mark.parametrize("column,value", [
        ('invalid_column', 0.5),
        ('unknown', 0.0),
        ('typo', 1.0),
        ('', 0.5),
    ])
    def test_invalid_column_names(self, column, value):
        """Test that invalid column names return default values."""
        assert get_label(column, value) == ('-', 'white')


class TestEdgeCases:
    """Test edge cases and boundary values."""

    @pytest.mark.parametrize("column,value", [
        ('difficulty', 0.4),  # Boundary between difficult and intermediate
        ('difficulty', 0.6),  # Boundary between intermediate and easy
        ('discrimination', 0.0),  # Boundary between negative and low
        ('discrimination', 0.2),  # Boundary between low and moderate
        ('discrimination', 0.3),  # Boundary between moderate and high
        ('discrimination', 0.5),  # Boundary between high and very high
        ('correlation', 0.0),  # Boundary between negative and none
        ('correlation', 0.1),  # Boundary between none and low
        ('correlation', 0.2),  # Boundary between low and moderate
        ('correlation', 0.3),  # Boundary between moderate and strong
        ('correlation', 0.5),  # Boundary between strong and very strong
    ])
    def test_boundary_values(self, column, value):
        """
        Test boundary values to ensure correct classification.

        Boundary values are critical as they determine classification transitions.
        """
        result = get_label(column, value)
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (label, color)"
        assert isinstance(result[0], str), "Label should be a string"
        assert isinstance(result[1], str), "Color should be a string"
