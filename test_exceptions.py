"""
Pytest tests for custom exception classes.

These tests verify that custom exceptions are properly defined,
inherit correctly, and maintain exception hierarchy.
"""
import pytest

from amcreport import (
    AMCReportError,
    DatabaseError,
    AIAnalysisError,
    ConfigurationError,
    ExamData
)


class TestDatabaseExceptions:
    """Test database-related exception handling."""

    def test_database_error_raised_for_nonexistent_path(self):
        """Test that DatabaseError is raised when database doesn't exist."""
        with pytest.raises(DatabaseError) as exc_info:
            ExamData._check_db("/nonexistent/path/database.sqlite")

        assert "does not exist" in str(exc_info.value).lower()

    def test_database_error_message(self):
        """Test that DatabaseError stores and displays messages correctly."""
        test_message = "Test database not found"

        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError(test_message)

        assert str(exc_info.value) == test_message


class TestExceptionHierarchy:
    """Test custom exception inheritance and hierarchy."""

    @pytest.mark.parametrize("exception_class,test_message", [
        (DatabaseError, "Test database error"),
        (AIAnalysisError, "Test AI analysis error"),
        (ConfigurationError, "Test configuration error"),
    ])
    def test_exceptions_inherit_from_base(self, exception_class, test_message):
        """
        Test that all custom exceptions inherit from AMCReportError.

        This allows catching all application exceptions with a single except clause.
        """
        with pytest.raises(AMCReportError):
            raise exception_class(test_message)

    @pytest.mark.parametrize("exception_class", [
        DatabaseError,
        AIAnalysisError,
        ConfigurationError,
    ])
    def test_exceptions_are_subclasses(self, exception_class):
        """Test that custom exceptions are proper subclasses of AMCReportError."""
        assert issubclass(exception_class, AMCReportError)
        assert issubclass(exception_class, Exception)

    def test_base_error_is_exception(self):
        """Test that AMCReportError itself inherits from Exception."""
        assert issubclass(AMCReportError, Exception)


class TestExceptionMessages:
    """Test exception message handling."""

    @pytest.mark.parametrize("exception_class,message", [
        (AMCReportError, "Base error message"),
        (DatabaseError, "Database connection failed"),
        (AIAnalysisError, "API timeout occurred"),
        (ConfigurationError, "Invalid configuration value"),
    ])
    def test_exception_message_storage(self, exception_class, message):
        """Test that exceptions properly store and display custom messages."""
        with pytest.raises(exception_class) as exc_info:
            raise exception_class(message)

        assert str(exc_info.value) == message

    @pytest.mark.parametrize("exception_class", [
        AMCReportError,
        DatabaseError,
        AIAnalysisError,
        ConfigurationError,
    ])
    def test_exception_with_empty_message(self, exception_class):
        """Test that exceptions can be raised with empty messages."""
        with pytest.raises(exception_class):
            raise exception_class("")


class TestExceptionCatching:
    """Test exception catching patterns."""

    def test_catch_specific_then_base(self):
        """
        Test catching specific exception before base exception.

        This demonstrates proper exception handling hierarchy.
        """
        # First, catch specific exception
        with pytest.raises(DatabaseError):
            raise DatabaseError("Specific error")

        # Then verify it can also be caught as base exception
        with pytest.raises(AMCReportError):
            raise DatabaseError("Caught as base")

    def test_multiple_exception_types(self):
        """Test that different exception types are distinguishable."""
        with pytest.raises(DatabaseError):
            raise DatabaseError("Database issue")

        with pytest.raises(AIAnalysisError):
            raise AIAnalysisError("AI analysis issue")

        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config issue")


class TestExceptionAttributes:
    """Test exception attributes and properties."""

    def test_exception_has_args_attribute(self):
        """Test that exceptions store args attribute."""
        message = "Test error message"
        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError(message)

        assert exc_info.value.args == (message,)

    @pytest.mark.parametrize("exception_class", [
        AMCReportError,
        DatabaseError,
        AIAnalysisError,
        ConfigurationError,
    ])
    def test_exception_repr(self, exception_class):
        """Test exception representation."""
        message = "Test message"
        exc = exception_class(message)

        # repr should include class name
        assert exception_class.__name__ in repr(exc)
