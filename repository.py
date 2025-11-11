"""
Exam Repository module for AMC Report Generator.

This module provides a unified interface for storing exam metrics in various
backend systems (Airtable, Baserow, etc.) for historical tracking and comparison.
"""

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from settings import AMCSettings

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class ExamMetrics:
    """
    Data class for exam metrics to be stored in the repository.

    This encapsulates all the statistical data we want to track over time.
    """

    def __init__(
        self,
        project_name: str,
        analysis_date: str,
        num_students: int,
        num_questions: int,
        avg_grade: float,
        median_grade: float,
        std_dev_grade: float,
        min_grade: float,
        max_grade: float,
        avg_difficulty: Optional[float] = None,
        avg_discrimination: Optional[float] = None,
        avg_correlation: Optional[float] = None,
        pass_rate: Optional[float] = None,
        cronbach_alpha: Optional[float] = None,
    ):
        self.project_name = project_name
        self.analysis_date = analysis_date
        self.num_students = num_students
        self.num_questions = num_questions
        self.avg_grade = avg_grade
        self.median_grade = median_grade
        self.std_dev_grade = std_dev_grade
        self.min_grade = min_grade
        self.max_grade = max_grade
        self.avg_difficulty = avg_difficulty
        self.avg_discrimination = avg_discrimination
        self.avg_correlation = avg_correlation
        self.pass_rate = pass_rate
        self.cronbach_alpha = cronbach_alpha

        # Parse project name for year, month, and course code
        self._parse_project_name()

    def _parse_project_name(self):
        """Extract year, month, and course code from project name."""
        # Expected format: YYYYMM-CourseCode
        match = re.match(r'^(\d{4})(\d{2})-(.+)$', self.project_name)
        if match:
            self.year = match.group(1)
            self.month = match.group(2)
            self.course_code = match.group(3)
        else:
            self.year = None
            self.month = None
            self.course_code = self.project_name

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for storage."""
        return {
            'project_name': self.project_name,
            'analysis_date': self.analysis_date,
            'year': self.year,
            'month': self.month,
            'course_code': self.course_code,
            'num_students': self.num_students,
            'num_questions': self.num_questions,
            'avg_grade': round(self.avg_grade, 2) if self.avg_grade is not None else None,
            'median_grade': round(self.median_grade, 2) if self.median_grade is not None else None,
            'std_dev_grade': round(self.std_dev_grade, 2) if self.std_dev_grade is not None else None,
            'min_grade': round(self.min_grade, 2) if self.min_grade is not None else None,
            'max_grade': round(self.max_grade, 2) if self.max_grade is not None else None,
            'avg_difficulty': round(self.avg_difficulty, 3) if self.avg_difficulty is not None else None,
            'avg_discrimination': round(self.avg_discrimination, 3) if self.avg_discrimination is not None else None,
            'avg_correlation': round(self.avg_correlation, 3) if self.avg_correlation is not None else None,
            'pass_rate': round(self.pass_rate, 3) if self.pass_rate is not None else None,
            'cronbach_alpha': round(self.cronbach_alpha, 3) if self.cronbach_alpha is not None else None,
        }


class RepositoryBackend(ABC):
    """Abstract base class for repository backends."""

    @abstractmethod
    def save(self, metrics: ExamMetrics) -> bool:
        """
        Save or update exam metrics in the repository.

        Args:
            metrics: ExamMetrics object containing the data to save

        Returns:
            True if save was successful, False otherwise

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection to the repository backend.

        Returns:
            True if connection is successful, False otherwise
        """
        pass


class AirtableBackend(RepositoryBackend):
    """Airtable implementation of the repository backend."""

    def __init__(self, settings: AMCSettings):
        """
        Initialize Airtable backend.

        Args:
            settings: AMCSettings object with Airtable configuration
        """
        self.settings = settings
        self.api = None
        self.table = None
        self._initialize()

    def _initialize(self):
        """Initialize Airtable API connection."""
        try:
            from pyairtable import Api

            self.api = Api(self.settings.airtable_api_key)
            self.table = self.api.table(
                self.settings.airtable_base_id,
                self.settings.airtable_table_name
            )
            logger.info("Airtable backend initialized successfully")
        except ImportError:
            raise RepositoryError(
                "pyairtable package not installed. "
                "Run: pip install pyairtable"
            )
        except Exception as e:
            raise RepositoryError(f"Failed to initialize Airtable backend: {e}")

    def test_connection(self) -> bool:
        """Test connection to Airtable."""
        try:
            # Try to fetch the first record (or all if table is small)
            self.table.first()
            logger.info("Airtable connection test successful")
            return True
        except Exception as e:
            logger.error(f"Airtable connection test failed: {e}")
            return False

    def save(self, metrics: ExamMetrics) -> bool:
        """
        Save or update exam metrics in Airtable.

        Uses project_name as unique identifier (UPSERT logic).
        """
        try:
            # Check if record already exists
            formula = f"{{project_name}}='{metrics.project_name}'"
            existing_records = self.table.all(formula=formula)

            metrics_dict = metrics.to_dict()

            if existing_records:
                # Update existing record
                record_id = existing_records[0]['id']
                self.table.update(record_id, metrics_dict)
                logger.info(f"Updated existing Airtable record for {metrics.project_name}")
            else:
                # Create new record
                self.table.create(metrics_dict)
                logger.info(f"Created new Airtable record for {metrics.project_name}")

            return True

        except Exception as e:
            error_msg = f"Failed to save metrics to Airtable: {e}"
            logger.error(error_msg)
            raise RepositoryError(error_msg)


class BaserowBackend(RepositoryBackend):
    """Baserow implementation of the repository backend."""

    def __init__(self, settings: AMCSettings):
        """
        Initialize Baserow backend.

        Args:
            settings: AMCSettings object with Baserow configuration
        """
        self.settings = settings
        self.client = None
        self._initialize()

    def _initialize(self):
        """Initialize Baserow API connection."""
        try:
            from pybaserow import Baserow

            self.client = Baserow(token=self.settings.baserow_api_key)
            logger.info("Baserow backend initialized successfully")
        except ImportError:
            raise RepositoryError(
                "pybaserow package not installed. "
                "Run: pip install pybaserow"
            )
        except Exception as e:
            raise RepositoryError(f"Failed to initialize Baserow backend: {e}")

    def test_connection(self) -> bool:
        """Test connection to Baserow."""
        try:
            # Try to get table information
            self.client.get_rows(
                table_id=int(self.settings.baserow_table_id),
                size=1
            )
            logger.info("Baserow connection test successful")
            return True
        except Exception as e:
            logger.error(f"Baserow connection test failed: {e}")
            return False

    def save(self, metrics: ExamMetrics) -> bool:
        """
        Save or update exam metrics in Baserow.

        Uses project_name as unique identifier (UPSERT logic).
        """
        try:
            table_id = int(self.settings.baserow_table_id)

            # Search for existing record by project_name
            # Note: Baserow search/filter API may vary - this is a simplified approach
            existing_rows = self.client.get_rows(table_id)
            existing_row = None

            for row in existing_rows:
                if row.get('project_name') == metrics.project_name:
                    existing_row = row
                    break

            metrics_dict = metrics.to_dict()

            if existing_row:
                # Update existing record
                self.client.update_row(
                    table_id=table_id,
                    row_id=existing_row['id'],
                    data=metrics_dict
                )
                logger.info(f"Updated existing Baserow record for {metrics.project_name}")
            else:
                # Create new record
                self.client.add_row(
                    table_id=table_id,
                    data=metrics_dict
                )
                logger.info(f"Created new Baserow record for {metrics.project_name}")

            return True

        except Exception as e:
            error_msg = f"Failed to save metrics to Baserow: {e}"
            logger.error(error_msg)
            raise RepositoryError(error_msg)


class ExamRepository:
    """
    Main repository interface for storing and retrieving exam metrics.

    This class provides a unified interface that delegates to the appropriate
    backend based on configuration.
    """

    def __init__(self, settings: AMCSettings):
        """
        Initialize the exam repository.

        Args:
            settings: AMCSettings object with repository configuration
        """
        self.settings = settings
        self.backend: Optional[RepositoryBackend] = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate backend based on settings."""
        if self.settings.repository_backend == 'none':
            logger.info("Repository backend set to 'none' - metrics will not be stored")
            return

        try:
            if self.settings.repository_backend == 'airtable':
                self.backend = AirtableBackend(self.settings)
            elif self.settings.repository_backend == 'baserow':
                self.backend = BaserowBackend(self.settings)
            else:
                raise RepositoryError(
                    f"Unknown repository backend: {self.settings.repository_backend}"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize repository backend: {e}")
            logger.warning("Continuing without repository functionality")
            self.backend = None

    def is_enabled(self) -> bool:
        """Check if repository is enabled and initialized."""
        return self.backend is not None

    def test_connection(self) -> bool:
        """Test connection to the repository backend."""
        if not self.is_enabled():
            return False
        return self.backend.test_connection()

    def save_exam_metrics(self, metrics: ExamMetrics) -> bool:
        """
        Save exam metrics to the repository.

        Args:
            metrics: ExamMetrics object to save

        Returns:
            True if save was successful, False otherwise
        """
        if not self.is_enabled():
            logger.info("Repository not enabled - skipping save")
            return False

        try:
            return self.backend.save(metrics)
        except RepositoryError as e:
            logger.error(f"Repository error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving to repository: {e}")
            return False


def create_exam_metrics_from_data(
    project_name: str,
    exam_data: Any,
    pass_threshold: float = 10.0
) -> ExamMetrics:
    """
    Create ExamMetrics object from ExamData instance.

    Args:
        project_name: Name of the AMC project
        exam_data: ExamData instance containing the statistics
        pass_threshold: Grade threshold for pass rate calculation (default: 10.0)

    Returns:
        ExamMetrics object ready to be saved
    """
    stats = exam_data.general_statistics

    # Calculate pass rate
    pass_rate = None
    if 'mark' in exam_data.marks.columns:
        passing_students = (exam_data.marks['mark'] >= pass_threshold).sum()
        total_students = len(exam_data.marks)
        pass_rate = passing_students / total_students if total_students > 0 else 0.0

    # Get average psychometric metrics
    avg_difficulty = None
    avg_discrimination = None
    avg_correlation = None

    if 'difficulty' in exam_data.questions.columns:
        avg_difficulty = exam_data.questions['difficulty'].mean()

    if 'discrimination' in exam_data.questions.columns:
        avg_discrimination = exam_data.questions['discrimination'].mean()

    if 'correlation' in exam_data.questions.columns:
        avg_correlation = exam_data.questions['correlation'].mean()

    # Calculate Cronbach's alpha (reliability coefficient)
    cronbach_alpha = None
    try:
        import pingouin as pg
        # Create a matrix of student scores per question
        score_matrix = exam_data.scores.pivot(
            index='student',
            columns='question',
            values='score'
        )
        cronbach_alpha = pg.cronbach_alpha(data=score_matrix)[0]
    except Exception as e:
        logger.debug(f"Could not calculate Cronbach's alpha: {e}")

    return ExamMetrics(
        project_name=project_name,
        analysis_date=datetime.now().strftime('%Y-%m-%d'),
        num_students=stats['examinees'],
        num_questions=stats['questions'],
        avg_grade=stats['average'],
        median_grade=stats['median'],
        std_dev_grade=stats['std_deviation'],
        min_grade=stats['minimum'],
        max_grade=stats['maximum'],
        avg_difficulty=avg_difficulty,
        avg_discrimination=avg_discrimination,
        avg_correlation=avg_correlation,
        pass_rate=pass_rate,
        cronbach_alpha=cronbach_alpha,
    )
