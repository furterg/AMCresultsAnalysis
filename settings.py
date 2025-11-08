"""
Configuration management for AMC Report Generator using Pydantic.

This module defines typed, validated configuration settings that replace
the old ConfigParser approach with a modern, type-safe solution.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AMCSettings(BaseSettings):
    """
    Main configuration settings for AMC Report Generator.

    Settings are loaded from:
    1. Environment variables (highest priority)
    2. .env file (in script directory)
    3. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / '.env'),  # Look in script directory
        env_file_encoding='utf-8',
        env_prefix='AMC_',  # Environment variables like AMC_PROJECTS_DIR
        case_sensitive=False,
        extra='ignore',  # Ignore extra fields
        validate_default=True
    )

    # ========================================================================
    # PATHS AND DIRECTORIES
    # ========================================================================

    projects_dir: Path = Field(
        ...,  # Required
        description="Directory containing AMC project folders"
    )

    config_file: Path = Field(
        default=Path("settings.conf"),
        description="Path to legacy configuration file"
    )

    # ========================================================================
    # EXAM ANALYSIS SETTINGS
    # ========================================================================

    student_threshold: int = Field(
        default=99,
        ge=10,
        le=10000,
        description="Minimum number of students required for discrimination analysis"
    )

    # ========================================================================
    # COMPANY BRANDING
    # ========================================================================

    company_name: str = Field(
        default="",
        max_length=200,
        description="Organization name displayed in reports"
    )

    company_url: str = Field(
        default="",
        max_length=500,
        description="Organization website URL displayed in reports"
    )

    # ========================================================================
    # AI ANALYSIS SETTINGS
    # ========================================================================

    enable_ai_analysis: bool = Field(
        default=True,
        description="Enable Claude AI statistical analysis"
    )

    claude_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude (also checks CLAUDE_API_KEY and ANTHROPIC_API_KEY)"
    )

    claude_model: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model to use for analysis"
    )

    claude_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Temperature for Claude responses (0.0-1.0, lower = more focused)"
    )

    claude_max_tokens: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum tokens in Claude's response"
    )

    # ========================================================================
    # PSYCHOMETRIC CONSTANTS
    # ========================================================================

    discrimination_quantile: float = Field(
        default=0.27,
        ge=0.0,
        le=0.5,
        description="CTT standard quantile for discrimination index (top/bottom 27%)"
    )

    cancellation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for flagging questions cancelled by >80% of students"
    )

    empty_answer_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for flagging questions left empty by >80% of students"
    )

    # ========================================================================
    # CHART SETTINGS
    # ========================================================================

    plot_width: int = Field(
        default=9,
        ge=1,
        le=20,
        description="Default plot width in inches"
    )

    plot_height: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Default plot height in inches"
    )

    difficulty_histogram_bins: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of bins for difficulty histogram"
    )

    discrimination_histogram_bins: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of bins for discrimination histogram"
    )

    correlation_bins_multiplier: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Multiplier for correlation histogram bins"
    )

    # ========================================================================
    # CORRECTION DETECTION
    # ========================================================================

    manual_correction_darkness_threshold: int = Field(
        default=180,
        ge=0,
        le=255,
        description="Pixel darkness threshold for detecting manual corrections"
    )

    # ========================================================================
    # LOGGING SETTINGS
    # ========================================================================

    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = Field(
        default='INFO',
        description="Logging level for console output"
    )

    log_file: str = Field(
        default='amcreport.log',
        description="Log file name"
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator('projects_dir')
    @classmethod
    def validate_projects_dir(cls, v: Path) -> Path:
        """Ensure projects directory exists and is readable."""
        if not v.exists():
            raise ValueError(
                f"Projects directory does not exist: {v}\n"
                f"Please create it or update the AMC_PROJECTS_DIR setting."
            )
        if not v.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return v.resolve()  # Return absolute path

    @model_validator(mode='after')
    def check_api_key_if_ai_enabled(self) -> 'AMCSettings':
        """Ensure API key is provided when AI analysis is enabled."""
        if self.enable_ai_analysis and not self.claude_api_key:
            # Try alternative environment variable names
            api_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.claude_api_key = api_key
            else:
                raise ValueError(
                    "AI analysis is enabled but no API key found.\n"
                    "Please set CLAUDE_API_KEY, ANTHROPIC_API_KEY, or AMC_CLAUDE_API_KEY environment variable,\n"
                    "or set enable_ai_analysis=False to disable AI analysis."
                )
        return self

    def get_colour_palette(self) -> dict[str, tuple[int, int, int, int]]:
        """
        Get the color palette for PDF reports.

        Returns:
            Dictionary mapping color names to RGBA tuples (0-255)
        """
        return {
            'heading_1': (23, 55, 83, 255),
            'heading_2': (109, 174, 219, 55),
            'heading_3': (40, 146, 215, 55),
            'white': (255, 255, 255, 0),
            'yellow': (251, 215, 114, 0),
            'red': (238, 72, 82, 0),
            'green': (166, 221, 182, 0),
            'grey': (230, 230, 230, 0),
            'blue': (84, 153, 242, 0),
        }


# Singleton instance - load settings once and reuse
_settings: AMCSettings | None = None


def get_settings() -> AMCSettings:
    """
    Get the application settings singleton.

    Settings are loaded once on first call and cached for subsequent calls.
    This ensures consistent configuration throughout the application.

    Returns:
        AMCSettings instance with validated configuration

    Raises:
        ValidationError: If configuration is invalid
    """
    global _settings
    if _settings is None:
        _settings = AMCSettings()  # type: ignore[call-arg]  # Pydantic loads from .env
    return _settings


def reload_settings() -> AMCSettings:
    """
    Reload settings from environment/file.

    Useful for testing or when configuration changes at runtime.

    Returns:
        Fresh AMCSettings instance
    """
    global _settings
    _settings = AMCSettings()  # type: ignore[call-arg]  # Pydantic loads from .env
    return _settings
