"""
Pytest tests for Settings and ExamProject classes.

These tests validate configuration management and project selection functionality.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from amcreport import Settings, ExamProject, ConfigurationError
from settings import AMCSettings


@pytest.fixture
def mock_pydantic_settings(tmp_path):
    """
    Create a mock AMCSettings instance for testing.

    Args:
        tmp_path: pytest temporary directory

    Returns:
        Mock AMCSettings with test configuration
    """
    settings = Mock(spec=AMCSettings)
    settings.projects_dir = tmp_path / "test-projects"
    settings.projects_dir.mkdir(exist_ok=True)
    settings.student_threshold = 99
    settings.company_name = "Test University"
    settings.company_url = "https://test.edu"
    settings.enable_ai_analysis = False
    settings.claude_api_key = ""
    settings.log_level = "INFO"
    return settings


@pytest.fixture
def projects_structure(tmp_path):
    """
    Create a realistic project directory structure for testing.

    Returns:
        dict with paths to various directories
    """
    projects_dir = tmp_path / "Projets-QCM"
    projects_dir.mkdir()

    # Create some test projects
    (projects_dir / "Project1").mkdir()
    (projects_dir / "Project2").mkdir()
    (projects_dir / "Project3").mkdir()
    (projects_dir / "_Archive").mkdir()

    # Create data directories with SQLite files
    for project in ["Project1", "Project2", "Project3"]:
        data_dir = projects_dir / project / "data"
        data_dir.mkdir()
        (data_dir / "scoring.sqlite").touch()
        (data_dir / "capture.sqlite").touch()

    return {
        "projects_dir": projects_dir,
        "project1": projects_dir / "Project1",
        "project2": projects_dir / "Project2",
        "project3": projects_dir / "Project3",
        "archive": projects_dir / "_Archive",
    }


class TestSettingsInitialization:
    """Test Settings class initialization."""

    def test_settings_init_with_pydantic(self, mock_pydantic_settings):
        """Test Settings initialization with AMCSettings instance."""
        settings = Settings(settings=mock_pydantic_settings)

        assert settings.projects == str(mock_pydantic_settings.projects_dir)
        assert settings.threshold == 99
        assert settings.company_name == "Test University"
        assert settings.company_url == "https://test.edu"

    @patch('amcreport.get_settings')
    def test_settings_init_without_args(self, mock_get):
        """Test Settings initialization without arguments (uses singleton)."""
        # Mock the get_settings to return a valid AMCSettings
        mock_settings = Mock(spec=AMCSettings)
        mock_settings.projects_dir = Path("/tmp/test")
        mock_settings.student_threshold = 99
        mock_settings.company_name = "Test"
        mock_settings.company_url = "test.com"
        mock_get.return_value = mock_settings

        # Should load from get_settings()
        settings = Settings()

        assert hasattr(settings, 'projects')
        assert hasattr(settings, 'threshold')
        assert hasattr(settings, 'company_name')
        assert hasattr(settings, 'company_url')

    def test_settings_backward_compatibility_attributes(self, mock_pydantic_settings):
        """Test that Settings exposes backward compatibility attributes."""
        settings = Settings(settings=mock_pydantic_settings)

        # Check all expected attributes exist
        assert hasattr(settings, 'projects')
        assert hasattr(settings, 'threshold')
        assert hasattr(settings, 'company_name')
        assert hasattr(settings, 'company_url')
        assert hasattr(settings, 'config_file')

    def test_settings_config_file_default(self, mock_pydantic_settings):
        """Test that config_file has default value."""
        settings = Settings(settings=mock_pydantic_settings)

        assert settings.config_file == "settings.conf"

    def test_settings_config_file_custom(self, mock_pydantic_settings):
        """Test Settings with custom config_file."""
        settings = Settings(config_file="custom.conf", settings=mock_pydantic_settings)

        assert settings.config_file == "custom.conf"


class TestSettingsErrorHandling:
    """Test Settings error handling."""

    @patch('amcreport.get_settings')
    def test_settings_raises_configuration_error_on_validation(self, mock_get):
        """Test that ValidationError is converted to ConfigurationError."""
        # Make get_settings raise ValidationError
        validation_error = ValidationError.from_exception_data(
            'AMCSettings',
            [{'type': 'missing', 'loc': ('student_threshold',), 'msg': 'Field required', 'input': {}}]
        )
        mock_get.side_effect = validation_error

        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            Settings()

    @patch('amcreport.get_settings')
    @patch('amcreport.Settings._setup_projects_dir')
    def test_settings_calls_setup_on_projects_dir_error(self, mock_setup, mock_get):
        """Test that _setup_projects_dir is called when projects_dir is missing."""
        # First call raises ValidationError about projects_dir
        # Second call (after setup) succeeds
        validation_error = ValidationError.from_exception_data(
            'AMCSettings',
            [{'type': 'missing', 'loc': ('projects_dir',), 'msg': 'Field required', 'input': {}}]
        )

        mock_settings = Mock(spec=AMCSettings)
        mock_settings.projects_dir = Path("/tmp/test")
        mock_settings.student_threshold = 99
        mock_settings.company_name = "Test"
        mock_settings.company_url = "test.com"

        mock_get.side_effect = [validation_error, mock_settings]

        Settings()

        # Verify setup was called
        mock_setup.assert_called_once()


class TestSettingsSetupProjectsDir:
    """Test _setup_projects_dir method."""

    @patch('builtins.input')
    @patch('os.walk')
    @patch('pathlib.Path.write_text')
    def test_setup_finds_projets_qcm(self, mock_write, mock_walk, mock_input):
        """Test that setup finds Projets-QCM directory."""
        mock_input.return_value = 'y'
        mock_walk.return_value = [
            ('/home/user', ['Projets-QCM', 'other'], []),
        ]

        settings = Mock(spec=AMCSettings)
        settings.projects_dir = Path("/home/user/Projets-QCM")
        settings.student_threshold = 99
        settings.company_name = ""
        settings.company_url = ""

        with patch('amcreport.get_settings') as mock_get:
            validation_error = ValidationError.from_exception_data(
                'AMCSettings',
                [{'type': 'missing', 'loc': ('projects_dir',), 'msg': 'Field required', 'input': {}}]
            )
            mock_get.side_effect = [validation_error, settings]

            Settings()

        # Verify .env file was written
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]
        assert 'AMC_PROJECTS_DIR=/home/user/Projets-QCM' in written_content

    @patch('builtins.input')
    def test_setup_user_declines(self, mock_input):
        """Test that ConfigurationError is raised when user declines setup."""
        mock_input.return_value = 'n'

        with patch('amcreport.get_settings') as mock_get:
            validation_error = ValidationError.from_exception_data(
                'AMCSettings',
                [{'type': 'missing', 'loc': ('projects_dir',), 'msg': 'Field required', 'input': {}}]
            )
            mock_get.side_effect = validation_error

            with pytest.raises(ConfigurationError, match="Projects directory must be configured"):
                Settings()

    @patch('builtins.input')
    @patch('os.walk')
    def test_setup_directory_not_found(self, mock_walk, mock_input):
        """Test that ConfigurationError is raised when directory is not found."""
        mock_input.return_value = 'y'
        # No Projets-QCM or MC-Projects found
        mock_walk.return_value = [
            ('/home/user', ['other', 'dirs'], []),
        ]

        with patch('amcreport.get_settings') as mock_get:
            validation_error = ValidationError.from_exception_data(
                'AMCSettings',
                [{'type': 'missing', 'loc': ('projects_dir',), 'msg': 'Field required', 'input': {}}]
            )
            mock_get.side_effect = validation_error

            with pytest.raises(ConfigurationError, match="Could not find"):
                Settings()


class TestExamProjectInitialization:
    """Test ExamProject class initialization."""

    def test_exam_project_init(self, mock_pydantic_settings, projects_structure):
        """Test ExamProject initialization."""
        # Update mock settings to point to test projects
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        with patch('builtins.input', return_value='1'):
            settings = Settings(settings=mock_pydantic_settings)
            project = ExamProject(settings)

            assert project.projects == str(projects_structure["projects_dir"])
            assert project.company_name == "Test University"
            assert project.company_url == "https://test.edu"
            assert project.threshold == 99
            assert project.name == "Project1"

    def test_exam_project_attributes_from_settings(self, mock_pydantic_settings, projects_structure):
        """Test that ExamProject correctly copies attributes from Settings."""
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        with patch('builtins.input', return_value='2'):
            settings = Settings(settings=mock_pydantic_settings)
            project = ExamProject(settings)

            assert project.company_name == settings.company_name
            assert project.company_url == settings.company_url
            assert project.threshold == settings.threshold


class TestExamProjectGetPath:
    """Test _get_path method."""

    def test_get_path_with_command_line_arg(self, mock_pydantic_settings, projects_structure):
        """Test project selection via command-line argument."""
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        with patch.object(sys, 'argv', ['script.py', 'Project2']):
            settings = Settings(settings=mock_pydantic_settings)
            project = ExamProject(settings)

            assert project.name == "Project2"
            assert "Project2" in project.path

    def test_get_path_invalid_command_line_arg(self, mock_pydantic_settings, projects_structure):
        """Test that invalid command-line arg falls back to user input."""
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        with patch.object(sys, 'argv', ['script.py', 'NonexistentProject']):
            with patch('builtins.input', return_value='1'):
                settings = Settings(settings=mock_pydantic_settings)
                project = ExamProject(settings)

                # Should fall back to user input (Project1)
                assert project.name == "Project1"

    def test_get_path_nonexistent_directory(self, mock_pydantic_settings):
        """Test error when projects directory doesn't exist."""
        mock_pydantic_settings.projects_dir = Path("/nonexistent/path")

        settings = Settings(settings=mock_pydantic_settings)

        with pytest.raises(ValueError, match="does not exist"):
            ExamProject(settings)

    def test_get_path_archive_excluded(self, mock_pydantic_settings, projects_structure):
        """Test that _Archive directory is excluded from project list."""
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        with patch('builtins.input', return_value='1') as mock_input:
            settings = Settings(settings=mock_pydantic_settings)
            project = ExamProject(settings)

            # Verify _Archive is not in the name
            assert project.name != "_Archive"


class TestExamProjectUserInput:
    """Test _user_input method."""

    @patch('builtins.input')
    def test_user_input_valid_selection(self, mock_input, mock_pydantic_settings, projects_structure):
        """Test valid user input selection."""
        mock_input.return_value = '2'
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        settings = Settings(settings=mock_pydantic_settings)
        project = ExamProject(settings)

        assert project.name == "Project2"

    @patch('builtins.input')
    def test_user_input_invalid_then_valid(self, mock_input, mock_pydantic_settings, projects_structure):
        """Test that invalid input prompts again."""
        # First invalid, then valid
        mock_input.side_effect = ['invalid', '99', 'abc', '1']
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        settings = Settings(settings=mock_pydantic_settings)
        project = ExamProject(settings)

        assert project.name == "Project1"
        # Should have been called 4 times (3 invalid + 1 valid)
        assert mock_input.call_count == 4

    @patch('builtins.input')
    def test_user_input_zero_shows_list_again(self, mock_input, mock_pydantic_settings, projects_structure):
        """Test that entering 0 shows the list again."""
        # 0 to show list, then valid selection
        mock_input.side_effect = ['0', '1']
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        settings = Settings(settings=mock_pydantic_settings)
        project = ExamProject(settings)

        assert project.name == "Project1"

    @patch('builtins.input')
    def test_user_input_out_of_range(self, mock_input, mock_pydantic_settings, projects_structure):
        """Test handling of out-of-range selections."""
        # Out of range (4, when only 3 projects), then valid
        mock_input.side_effect = ['4', '3']
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        settings = Settings(settings=mock_pydantic_settings)
        project = ExamProject(settings)

        assert project.name == "Project3"


class TestExamProjectName:
    """Test project name extraction."""

    def test_project_name_extracted(self, mock_pydantic_settings, projects_structure):
        """Test that project name is correctly extracted from path."""
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]

        with patch('builtins.input', return_value='1'):
            settings = Settings(settings=mock_pydantic_settings)
            project = ExamProject(settings)

            # Name should be just the directory name, not full path
            assert project.name == "Project1"
            assert "/" not in project.name
            assert "\\" not in project.name


class TestIntegrationSettingsAndProject:
    """Integration tests for Settings and ExamProject working together."""

    def test_settings_to_project_flow(self, projects_structure):
        """Test complete flow from Settings to ExamProject."""
        # Create a temporary .env file
        env_content = f"AMC_PROJECTS_DIR={projects_structure['projects_dir']}\n"
        env_content += "AMC_STUDENT_THRESHOLD=95\n"
        env_content += "AMC_COMPANY_NAME=Integration Test\n"
        env_content += "AMC_ENABLE_AI_ANALYSIS=false\n"

        env_file = Path(".env.test")
        env_file.write_text(env_content)

        try:
            # Patch the env file location
            with patch('settings.AMCSettings.model_config') as mock_config:
                # Create settings and project
                mock_settings = Mock(spec=AMCSettings)
                mock_settings.projects_dir = projects_structure["projects_dir"]
                mock_settings.student_threshold = 95
                mock_settings.company_name = "Integration Test"
                mock_settings.company_url = ""

                with patch('builtins.input', return_value='2'):
                    settings = Settings(settings=mock_settings)
                    project = ExamProject(settings)

                    # Verify data flows correctly
                    assert project.threshold == 95
                    assert project.company_name == "Integration Test"
                    assert project.name == "Project2"

        finally:
            # Clean up
            if env_file.exists():
                env_file.unlink()

    def test_multiple_projects_selection(self, mock_pydantic_settings, projects_structure):
        """Test selecting different projects in sequence."""
        mock_pydantic_settings.projects_dir = projects_structure["projects_dir"]
        settings = Settings(settings=mock_pydantic_settings)

        # Select Project1
        with patch('builtins.input', return_value='1'):
            project1 = ExamProject(settings)
            assert project1.name == "Project1"

        # Select Project3
        with patch('builtins.input', return_value='3'):
            project2 = ExamProject(settings)
            assert project2.name == "Project3"
