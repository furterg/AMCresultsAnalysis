"""
Pytest tests for Pydantic settings validation.

These tests demonstrate that Pydantic correctly validates configuration
settings and catches errors with helpful messages.
"""
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from settings import AMCSettings, get_settings, reload_settings


@pytest.fixture(autouse=True)
def clean_settings_singleton():
    """
    Clean up the settings singleton before and after each test.

    This ensures tests don't interfere with each other by resetting
    the global settings state.
    """
    import settings
    settings._settings = None
    yield
    settings._settings = None


class TestValidSettings:
    """Test valid settings configuration."""

    @patch.dict(os.environ, {'AMC_ENABLE_AI_ANALYSIS': 'false'})
    def test_load_settings_from_env(self):
        """Test that valid settings load correctly from .env file."""
        # Disable AI analysis to avoid API key requirement
        settings = get_settings()

        assert settings.projects_dir.exists(), "Projects directory should exist"
        assert settings.student_threshold >= 10, "Threshold should be >= 10"
        assert isinstance(settings.company_name, str), "Company name should be string"
        assert isinstance(settings.enable_ai_analysis, bool), "AI flag should be boolean"
        assert settings.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], "Log level should be valid"

    @patch.dict(os.environ, {'AMC_ENABLE_AI_ANALYSIS': 'false'})
    def test_colour_palette_method(self):
        """Test the colour palette method returns correct structure."""
        # Disable AI analysis to avoid API key requirement
        settings = get_settings()
        palette = settings.get_colour_palette()

        # Check structure
        assert isinstance(palette, dict), "Palette should be a dictionary"
        assert len(palette) > 0, "Palette should not be empty"

        # Check expected keys
        expected_keys = ['heading_1', 'heading_2', 'heading_3', 'white', 'yellow', 'red', 'green', 'grey', 'blue']
        for key in expected_keys:
            assert key in palette, f"Palette should have key '{key}'"

        # Check color format (RGBA tuples)
        for color_name, rgba in palette.items():
            assert isinstance(rgba, tuple), f"{color_name} should be a tuple"
            assert len(rgba) == 4, f"{color_name} should have 4 values (RGBA)"
            for value in rgba:
                assert 0 <= value <= 255, f"{color_name} values should be 0-255"


class TestInvalidThreshold:
    """Test validation of student threshold field."""

    def test_negative_threshold_rejected(self):
        """Test that negative student threshold raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AMCSettings(
                projects_dir="/tmp",
                student_threshold=-5,
                enable_ai_analysis=False  # Disable to avoid API key check
            )

        errors = exc_info.value.errors()
        assert any('greater than or equal to' in str(e['msg']).lower() for e in errors)

    def test_threshold_below_minimum_rejected(self):
        """Test that threshold below 10 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AMCSettings(
                projects_dir="/tmp",
                student_threshold=5,
                enable_ai_analysis=False
            )

        errors = exc_info.value.errors()
        assert any('10' in str(e['msg']) for e in errors)

    @pytest.mark.parametrize("invalid_threshold", [-1, -10, 0, 5, 9])
    def test_various_invalid_thresholds(self, invalid_threshold):
        """Test multiple invalid threshold values."""
        with pytest.raises(ValidationError):
            AMCSettings(
                projects_dir="/tmp",
                student_threshold=invalid_threshold,
                enable_ai_analysis=False
            )


class TestInvalidLogLevel:
    """Test validation of log level enum field."""

    def test_invalid_log_level_rejected(self):
        """Test that invalid log level raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AMCSettings(
                projects_dir="/tmp",
                log_level="TRACE",  # Not in allowed values
                enable_ai_analysis=False
            )

        errors = exc_info.value.errors()
        # Check that error mentions the allowed values
        error_msg = str(errors[0]['msg']).lower()
        assert 'debug' in error_msg or 'info' in error_msg or 'literal' in error_msg

    @pytest.mark.parametrize("invalid_level", ["TRACE", "VERBOSE", "trace", "invalid", ""])
    def test_various_invalid_log_levels(self, invalid_level):
        """Test multiple invalid log level values."""
        with pytest.raises(ValidationError):
            AMCSettings(
                projects_dir="/tmp",
                log_level=invalid_level,
                enable_ai_analysis=False
            )

    @pytest.mark.parametrize("valid_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_valid_log_levels_accepted(self, valid_level):
        """Test that all valid log levels are accepted."""
        settings = AMCSettings(
            projects_dir="/tmp",
            log_level=valid_level,
            enable_ai_analysis=False
        )
        assert settings.log_level == valid_level


class TestMissingRequiredFields:
    """Test validation of required fields."""

    def test_missing_projects_dir_env_variable(self, monkeypatch):
        """Test that missing projects_dir raises ValidationError."""
        # Remove environment variable
        monkeypatch.delenv('AMC_PROJECTS_DIR', raising=False)
        # Ignore any local .env file, which may also define projects_dir
        monkeypatch.setitem(AMCSettings.model_config, 'env_file', None)

        with pytest.raises(ValidationError) as exc_info:
            reload_settings()

        errors = exc_info.value.errors()
        assert any('projects_dir' in str(e).lower() for e in errors)


class TestAPIKeyOptional:
    """The API key is optional: AI analysis degrades gracefully without it."""

    def test_ai_enabled_without_api_key_does_not_raise(self, monkeypatch):
        for var in ('AMC_CLAUDE_API_KEY', 'CLAUDE_API_KEY', 'ANTHROPIC_API_KEY'):
            monkeypatch.delenv(var, raising=False)
        settings = AMCSettings(projects_dir="/tmp", enable_ai_analysis=True, claude_api_key="")
        assert settings.claude_api_key == ""


class TestFieldDefaults:
    """Test default values for optional fields."""

    def test_default_values(self):
        """Test that default values are applied correctly when not in environment."""
        # Note: Pydantic Settings loads from environment even for new instances
        # The actual default in settings.py is 99, but .env sets it to 90
        # This test documents the actual behavior
        settings = AMCSettings(
            projects_dir="/tmp",
            enable_ai_analysis=False
        )

        # Check values from .env (not Field defaults)
        assert settings.student_threshold == 90, "Threshold from .env should be 90"
        assert settings.company_name == "Print&Scan", "Company name from .env"
        assert settings.company_url == "www.printandscan.fr", "Company URL from .env"
        assert settings.log_level == "INFO", "Default log level should be INFO"
