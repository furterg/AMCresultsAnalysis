#!/usr/bin/env python3
import datetime
import glob
import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anthropic import Anthropic
from icecream import install, ic
from pydantic import ValidationError
from scipy import stats

from report import generate_pdf_report
from repository import ExamRepository, create_exam_metrics_from_data, RepositoryError
from settings import AMCSettings, get_settings

matplotlib.use('agg')

# === Deprecated - Now using Pydantic settings ===
# These constants are kept for backward compatibility but values come from settings.py
NBQ: str = 'Number of questions'


# ============================================================================
# CONSTANTS
# ============================================================================

# === Psychometric Analysis Constants ===
DISCRIMINATION_QUANTILE = 0.27  # Top/bottom 27% for discrimination index (CTT standard)
"""Classical Test Theory standard: use top and bottom 27% of students to calculate
discrimination indices. This percentile maximizes the difference between groups."""

CANCELLATION_THRESHOLD = 0.8  # Flag questions cancelled >80% of the time
"""Threshold for identifying problematic questions that were cancelled by most students"""

EMPTY_ANSWER_THRESHOLD = 0.8  # Flag questions left empty >80% of the time
"""Threshold for identifying questions that most students didn't attempt"""

# === Chart/Plot Constants ===
PLOT_WIDTH = 9  # Standard plot width in inches
PLOT_HEIGHT = 4  # Standard plot height in inches
DIFFICULTY_HISTOGRAM_BINS = 30  # Number of bins for difficulty histogram
DISCRIMINATION_HISTOGRAM_BINS = 30  # Number of bins for discrimination histogram
CORRELATION_BINS_MULTIPLIER = 2  # Multiplier for correlation histogram bins

# === Correction Detection Constants ===
MANUAL_CORRECTION_DARKNESS_THRESHOLD = 180  # Pixel darkness threshold for manual corrections
"""Threshold value for detecting manually corrected answer boxes based on pixel darkness"""

# === AI Analysis Constants ===
CLAUDE_MODEL = "claude-sonnet-4-5"  # Claude 4.5 Sonnet for statistical analysis
CLAUDE_TEMPERATURE = 0.4  # Temperature for Claude responses (0.0-1.0)
CLAUDE_MAX_TOKENS = 512  # Maximum tokens in Claude's response

# Enhanced system prompt for Classical Test Theory analysis
CLAUDE_SYSTEM_PROMPT = """You are an expert psychometrician specializing in Classical Test Theory (CTT).
Your role is to analyze exam statistics and provide clear, actionable insights for educators.

Context:
- Difficulty: Proportion of students answering correctly (0-1, higher = easier)
- Discrimination: How well a question differentiates high/low performers (-1 to 1, higher = better)
- Correlation: Point-biserial correlation between item and total score (-1 to 1, higher = better)
- CTT standards: Good discrimination > 0.3, good correlation > 0.2

Your analysis should:
1. Identify patterns in the data (e.g., overall difficulty level, question quality)
2. Highlight specific concerns (e.g., questions with negative discrimination)
3. Provide actionable recommendations for exam improvement
4. Use plain language accessible to educators without deep statistical background
5. Be concise but thorough (2-3 paragraphs maximum)

IMPORTANT FORMATTING RULES:
- use simple markdown formatting ONLY (*, **)
- Do NOT use titles and markdown titles formatting (no #, ## etc.)
- Separate paragraphs with blank lines
- Write in complete sentences and paragraphs"""


# Custom Exceptions
class AMCReportError(Exception):
    """Base exception for AMCresultsAnalysis application"""
    pass


class DatabaseError(AMCReportError):
    """Raised when there are issues with the SQLite databases"""
    pass


class AIAnalysisError(AMCReportError):
    """Raised when there are issues with AI-powered statistical analysis"""
    pass


class ConfigurationError(AMCReportError):
    """Raised when there are configuration issues"""
    pass


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_dir: Optional[str] = None, log_level: str = 'INFO') -> logging.Logger:
    """
    Configure logging for the application.

    Sets up both console and file handlers with appropriate formatting:
    - Console: Clean, user-friendly output (INFO level and above)
    - File: Detailed logs with timestamps (DEBUG level and above)

    Args:
        log_dir: Directory for log files (defaults to project root)
        log_level: Logging level for console output ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('AMCReport')
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler - clean output for users
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - detailed logs with rotation
    if log_dir is None:
        log_dir = os.getcwd()

    log_file = os.path.join(log_dir, 'amcreport.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Initialize logger (will be configured in main)
logger = logging.getLogger('AMCReport')


def sanitize_text_for_pdf(text: str) -> str:
    """
    Sanitize text to remove Unicode characters not supported by FPDF's Latin-1 encoding.

    Converts common Unicode characters to their Latin-1 equivalents.

    Args:
        text: Input text that may contain Unicode characters

    Returns:
        Sanitized text compatible with FPDF Helvetica font
    """
    # Replace em dash and en dash with regular hyphen
    text = text.replace('\u2014', '-')  # Em dash —
    #text = text.replace('\u2013', '-')   # En dash –

    # Replace curly quotes with straight quotes
    #text = text.replace('\u201c', '"')   # Left double quote "
    #text = text.replace('\u201d', '"')   # Right double quote "
    #text = text.replace('\u2018', "'")   # Left single quote '
    #text = text.replace('\u2019', "'")   # Right single quote '

    # Replace ellipsis
    #text = text.replace('\u2026', '...')  # Horizontal ellipsis …

    # Replace bullet point
    #text = text.replace('\u2022', '-')    # Bullet •

    # Replace multiplication sign
    #text = text.replace('\u00d7', 'x')    # Multiplication ×

    # Remove any other non-Latin-1 characters by encoding/decoding
    # This will replace unsupported chars with '?'
    #text = text.encode('latin-1', errors='replace').decode('latin-1')

    return text


class ClaudeAnalyzer:
    """
    Uses Claude AI to analyze exam statistics and provide insights.

    This class replaces the previous OpenAI-based implementation with a simpler,
    more efficient approach using Claude's Messages API.
    """

    def __init__(
        self,
        stats_table: pd.DataFrame,
        model: str = CLAUDE_MODEL,
        temperature: float = CLAUDE_TEMPERATURE,
        max_tokens: int = CLAUDE_MAX_TOKENS
    ) -> None:
        """
        Initialize the Claude analyzer.

        Args:
            stats_table: DataFrame containing exam statistics
            model: Claude model to use for analysis
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response

        Raises:
            AIAnalysisError: If Claude API key is not found or initialization fails
        """
        # Get API key from environment (supports both CLAUDE_API_KEY and ANTHROPIC_API_KEY)
        api_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise AIAnalysisError(
                "No API key found. Please set CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable. "
                "Get your key from: https://console.anthropic.com/"
            )

        self.client: Anthropic = Anthropic(api_key=api_key)
        self.model: str = model
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.stats_table: pd.DataFrame = stats_table
        self.response: str = self._analyze()

    def _format_stats_for_analysis(self) -> str:
        """
        Format the statistics table into a clear, readable format for Claude.

        Returns:
            Formatted string representation of statistics
        """
        # Convert DataFrame to a clean string format
        stats_str = self.stats_table.to_string(index=False, float_format=lambda x: f'{x:.3f}')

        return f"""Here are the exam statistics to analyze:

{stats_str}

Please analyze these results and provide insights about:
1. Overall exam performance and difficulty
2. Question quality (based on discrimination and correlation)
3. Any concerning patterns or outliers
4. Specific recommendations for improvement"""

    def _analyze(self) -> str:
        """
        Send statistics to Claude and get analysis.

        Returns:
            Claude's analysis as a string

        Raises:
            AIAnalysisError: If the API call fails
        """
        logger.info("Analyzing exam statistics with Claude AI...")

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=CLAUDE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": self._format_stats_for_analysis()
                }]
            )

            # Extract text from response
            response_text = message.content[0].text
            ic(response_text)
            # Sanitize text to remove Unicode characters not supported by FPDF
            response_text = sanitize_text_for_pdf(response_text)

            logger.info("✓ Analysis complete")
            return response_text

        except Exception as err:
            raise AIAnalysisError(f"Claude analysis failed: {err}") from err


class Settings:
    """
    Wrapper around Pydantic settings for backward compatibility.

    This class maintains the old interface while using the new Pydantic-based
    configuration system underneath.
    """

    def __init__(self, config_file: Optional[str] = None, settings: Optional[AMCSettings] = None) -> None:
        """
        Initialize settings from Pydantic configuration.

        Args:
            config_file: Deprecated, kept for backward compatibility
            settings: Optional AMCSettings instance (uses singleton if not provided)
        """
        if settings is None:
            try:
                settings = get_settings()
            except ValidationError as e:
                # If validation fails, try to help the user set up configuration
                logger.error("Configuration validation failed:")
                logger.error(str(e))

                # Check if projects_dir is the issue
                if "projects_dir" in str(e):
                    self._setup_projects_dir()
                    # Try loading settings again after setup
                    settings = get_settings()
                else:
                    raise ConfigurationError(
                        f"Invalid configuration: {e}\n"
                        "Please check your .env file or environment variables."
                    ) from e

        self._settings: AMCSettings = settings
        self.config_file: str = config_file or "settings.conf"  # Kept for compatibility

        # Expose settings as instance attributes for backward compatibility
        self.projects: str = str(settings.projects_dir)
        self.threshold: int = settings.student_threshold
        self.company_name: str = settings.company_name
        self.company_url: str = settings.company_url

    def _setup_projects_dir(self) -> None:
        """
        Interactive setup to find and configure projects directory.

        Creates a .env file with the projects directory path.
        """
        create_file = input(
            f"Projects directory not configured. Do you want to search for it? (Y/n): "
        )
        if create_file.lower() != 'y':
            raise ConfigurationError("Projects directory must be configured.")

        home_dir = os.path.expanduser("~")
        projects_dir: str | None = None

        logger.info("Searching for Projets-QCM or MC-Projects directory...")
        for dir_path, dir_names, filenames in os.walk(home_dir):
            if "Projets-QCM" in dir_names:
                projects_dir = os.path.join(dir_path, "Projets-QCM")
                break
            elif "MC-Projects" in dir_names:
                projects_dir = os.path.join(dir_path, "MC-Projects")
                break

        if not projects_dir:
            raise ConfigurationError(
                "Could not find 'Projets-QCM' or 'MC-Projects' directory.\n"
                "Please create a .env file with: AMC_PROJECTS_DIR=/path/to/your/projects"
            )

        logger.info(f"Found projects directory: {projects_dir}")

        # Create .env file
        env_file = Path(".env")
        env_content = f"AMC_PROJECTS_DIR={projects_dir}\n"

        # Add API key placeholder if not set
        if not os.getenv('CLAUDE_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
            env_content += "# AMC_CLAUDE_API_KEY=your-api-key-here\n"

        env_file.write_text(env_content)
        logger.info(f"Created {env_file} with projects directory")
        logger.info("You can edit this file to add more configuration options.")


class ExamProject:
    """
    Represents an Auto Multiple Choice exam project.

    Manages project paths, selection, and configuration.
    """

    def __init__(self, conf: Settings) -> None:
        """
        Initialize exam project from settings.

        Args:
            conf: Settings object containing project configuration
        """
        self.projects: str = conf.projects  # Path to the Projets-QCM directory
        self.company_name: str = conf.company_name  # Name of the company
        self.company_url: str = conf.company_url  # URL of the company
        self.threshold: int = conf.threshold
        self.path: str = self._get_path()  # Path to the selected project to analyse
        self.name: str = glob.glob(self.path, recursive=False)[0].split('/')[-1]  # Project name

    def _get_path(self) -> str:
        """
        - Get the list of project directories
        - Presents the list to the user
        - Get the user's selection
        :return: path to the project selected by the user
        """
        if os.path.exists(self.projects):
            subdirectories: list[str] = next(os.walk(self.projects))[1]
            subdirectories.remove('_Archive')
            subdirectories.sort()
            if len(sys.argv) > 1 and sys.argv[1] in subdirectories:
                return os.path.join(self.projects, sys.argv[1])
            return self._user_input(subdirectories)
        raise ValueError(f"The path {self.projects} does not exist.")

    def _user_input(self, sub: list[str]) -> str:
        """
        Presents the list of projects to the user, prompts them to select a project,
        validates the input and returns the path to the selected project.

        :param sub: list of project subdirectories
        :return: path to the selected project
        """
        while True:
            # display numbered list of subdirectories
            logger.info("Here's a list of current projects:")
            for i, directory in enumerate(sub):
                logger.info(f"{i + 1}. {directory}")

            # prompt user to select a subdirectory
            selection: str = input("Enter the number of the project you'd like to select: ")

            # validate user input
            while not selection.isdigit() \
                    or int(selection) not in range(0, len(sub) + 1):
                selection = input(
                    "Invalid input. Enter the number of the project you'd like to select \
                    (type 0 for list): ")

            # If user input is 0, then print the list again
            if selection == '0':
                continue

            # store the path to the selected project
            return os.path.join(self.projects, sub[int(selection) - 1])


class ExamData:
    """
    Manages exam data from AMC SQLite databases.

    Loads and processes exam statistics, questions, items, and student responses
    from scoring and capture databases.
    """

    def __init__(self, path: str, threshold: int = 99) -> None:
        """
        Initialize exam data from project path.

        Args:
            path: Path to the exam project directory
            threshold: Minimum students for discrimination analysis (default: 99)

        Raises:
            DatabaseError: If database files don't exist
        """
        self.path: str = path  # Project path
        self.threshold: int = threshold  # bottom limit for calculation of discrimination index
        self.scoring_db: str = os.path.join(self.path, 'data/scoring.sqlite')
        self.capture_db: str = os.path.join(self.path, 'data/capture.sqlite')
        self.conn: sqlite3.Connection  # Database connection (set in context managers below)
        for db in [self.scoring_db, self.capture_db]:
            self._check_db(db)
        with sqlite3.connect(self.scoring_db) as self.conn:
            self.scl: int = self._get_student_code_length()
            self.indicative: pd.DataFrame = pd.read_sql_query("""SELECT DISTINCT question 
                                                                      FROM scoring_question 
                                                                      WHERE indicative = 1""", self.conn)
            self.marks: pd.DataFrame = self._get_marks()
            self.scores: pd.DataFrame = self._get_scores()
            self.variables: pd.DataFrame = pd.read_sql_query("SELECT * FROM scoring_variables",
                                                             self.conn, index_col='name')
            self.answers: pd.DataFrame = self._get_answers()
        with sqlite3.connect(self.capture_db) as self.conn:
            self.capture: pd.DataFrame = self._get_capture()
        self.number_of_examinees: int = self.marks['student'].nunique()
        self.standard_deviation: float = self.marks['mark'].std()
        self.questions: pd.DataFrame = self._get_questions()
        self.items: pd.DataFrame = self._get_items()
        self.general_stats: dict[str, Any] = self._general_stats()
        self.table: pd.DataFrame = self._get_stats_table()
        self.definitions: dict[str, str] = self._get_dictionary('definitions')
        self.findings: dict[str, Any] = self._get_dictionary('findings')

    def _general_stats(self) -> dict[str, Any]:
        return {
            'Number of examinees': self.number_of_examinees,
            NBQ: self.questions['title'].nunique(),
            'Maximum possible mark': float(self.variables['value']['mark_max']),
            'Minimum achieved mark': self.marks['mark'].min(),
            'Maximum achieved mark': self.marks['mark'].max(),
            'Mean': self.marks['mark'].mean(),
            'Median': self.marks['mark'].median(),
            'Mode': self.marks['mark'].mode().iloc[0],
            'Standard deviation': self.standard_deviation,
            'Variance': self.marks['mark'].var(),
            'Standard error of mean': stats.sem(self.marks['mark']),
            'Standard error of measurement': self.standard_deviation / (self.number_of_examinees ** 0.5),
            'Skew': self.marks['mark'].skew(),
            'Kurtosis': self.marks['mark'].kurt(),
        }

    def _get_stats_table(self) -> pd.DataFrame:
        table: pd.DataFrame = pd.DataFrame.from_dict(self.general_stats, orient='index', columns=['Value'])
        table = (table.reset_index(names=['Element', 'Value']).iloc[[0, 2, 3, 4, 5, 6, 7, 8, 12, 13]])
        table['Value'] = table['Value'].apply(pd.to_numeric, errors='coerce')
        return table

    def _get_student_code_length(self) -> int:
        """
        Get the number of student code boxes
        :return: Length of the student code as an integer
        """
        # get the number boxes for the student code, so we can only query the real questions
        scl: int = pd.read_sql_query("""SELECT COUNT(*) FROM scoring_title 
                                     WHERE title LIKE '%student.number%';""",
                                     self.conn).iloc[0].iloc[0]
        return int(scl)

    @staticmethod
    def _check_db(db: str) -> None:
        """
        Check if a database file exists.

        Args:
            db: Path to the database file.

        Raises:
            DatabaseError: If the database file does not exist.
        """
        if not os.path.exists(db):
            raise DatabaseError(f"The database {db} does not exist!")

    def _get_marks(self) -> pd.DataFrame:
        """
        Retrieve student marks from the database.

        Returns:
            DataFrame containing student marks.

        Raises:
            DatabaseError: If no marks have been recorded in the database.
        """
        pd_mark = pd.read_sql_query("SELECT * FROM scoring_mark", self.conn)
        if pd_mark.empty:
            raise DatabaseError("No mark has been recorded in the database")
        return pd_mark

    def _get_scores(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"""
        SELECT ss.student, ss.question, st.title, ss.score, ss.why, ss.max
        FROM scoring_score ss
        JOIN scoring_title st ON ss.question = st.question
        WHERE ss.question > {self.scl}""", self.conn)
        # Clean the scores to keep track of Cancelled (C), Floored (P), Empty (V) and Error (E)
        # questions
        why = pd.get_dummies(df['why'])
        df = pd.concat([df, why], axis=1)
        df.drop('why', axis=1, inplace=True)
        df.rename(columns={'': 'replied', 'C': 'cancelled', 'P': 'floored', 'V': 'empty',
                           'E': 'error'}, inplace=True)
        df['correct'] = df.apply(lambda row: 1 if row['score'] == row['max'] else 0, axis=1)
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)
        return df

    def _get_questions(self) -> pd.DataFrame:
        df = self.scores.pivot_table(index=['question', 'title'],
                                     values=self.scores.columns[3:],
                                     aggfunc='sum',
                                     fill_value=0).reset_index()
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)
        # Get the list of columns to calculate the number of times a question has been presented.
        cols_for_pres = [col for col in ['cancelled', 'empty', 'replied', 'error', 'floored']
                         if col in df.columns]
        df['presented'] = df[cols_for_pres].sum(axis=1)
        # Combined 'replied' and 'floored' as 'replied' otherwise the number of presented is wrong
        # Check if there is a column 'floored' in 'pd_question'
        if 'floored' in df.columns:
            # Create a new column 'replied_new' by summing 'replied' and 'floored'
            df['replied_new'] = df['replied'] + df['floored']
            # Drop the original 'replied' column
            df = df.drop('replied', axis=1)
            # Rename 'replied_new' to 'replied'
            df = df.rename(columns={'replied_new': 'replied'})

        # Get the list of columns for calculate the number of times a question has been replied or
        # left empty...
        cols_for_diff = [col for col in ['floored', 'empty', 'replied', 'error']
                         if col in df.columns]
        # Calculate the difficulty of each question
        df['difficulty'] = df['correct'] / df[cols_for_diff].sum(axis=1)
        # Now the columns are: ['question', 'title', 'cancelled', 'correct', 'empty', 'error', 'max',
        # 'replied', 'score', 'presented', 'difficulty'] - some columns are optional
        # Get item and outcome discrimination if the number of examinees is greater than 99
        if self.number_of_examinees > self.threshold:
            # Create two student dataframes based on the quantile values. They should have the same
            # number of students This should probably be done in a smarter way, outside, in order to
            # be used for item discrimination.
            top_27_df = self.marks.sort_values(by=['mark'], ascending=False).head(
                round(len(self.marks) * DISCRIMINATION_QUANTILE))
            bottom_27_df = self.marks.sort_values(by=['mark'], ascending=False).tail(
                round(len(self.marks) * DISCRIMINATION_QUANTILE))

            df['discrimination'] = self._questions_discrimination(bottom_27_df, top_27_df)

        # Get item (question) correlation
        correlation = self._item_correlation()
        df['correlation'] = df['title'].apply(lambda row: correlation.loc[row]['correlation'])

        return df

    def _get_items(self) -> pd.DataFrame:
        df = self.capture.groupby(['question', 'answer'])['ticked'].sum().reset_index().sort_values(
            by=['question', 'answer'])
        df['correct'] = df.apply(lambda row: self.answers.loc[
            (self.answers['question'] == row['question'])
            & (self.answers['answer'] == row['answer']), 'correct'].values[0], axis=1)
        df = df.merge(self.questions[['question', 'title']], left_on='question', right_on='question')
        df = df[['question', 'title', 'answer', 'correct', 'ticked']].sort_values(
            by=['title', 'answer']).reset_index(drop=True)

        if self.number_of_examinees > self.threshold:
            # Create two student dataframes based on the quantile values. They should have the same
            # number of students This should probably be done in a smarter way, outside, in order to
            # be used for item discrimination.
            top_27_df = self.marks.sort_values(by=['mark'], ascending=False).head(
                round(len(self.marks) * DISCRIMINATION_QUANTILE))
            bottom_27_df = self.marks.sort_values(by=['mark'], ascending=False).tail(
                round(len(self.marks) * DISCRIMINATION_QUANTILE))
            discr = self._items_discrimination(bottom_27_df, top_27_df)
            df = df.merge(discr[['question', 'answer', 'discrimination']], on=['question', 'answer'])
        outcome_correlation: pd.DataFrame = self._outcome_correlation()
        return df.merge(outcome_correlation, on=['question', 'answer'])

    def _get_answers(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"""SELECT DISTINCT question, answer, correct, strategy
                                  FROM scoring_answer 
                                  WHERE question > {self.scl}""", self.conn)
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)
        df['correct'] = df.apply(lambda x: 1 if (x['correct'] == 1) else 0, axis=1)
        return df

    def _get_capture(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"""
        SELECT student, id_a AS 'question', id_b AS 'answer', total, black, manual 
        FROM capture_zone 
        WHERE type = 4 AND id_a > {self.scl}""", self.conn)
        # Remove rows in pd_capture where question is present in pd_indicative
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)

        # Apply specific operations to dataframes before returning them
        # pd_capture
        df['ticked'] = df.apply(self._ticked, axis=1)
        return df

    def _ticked(self, row: pd.Series) -> int:
        """
        Define if an answer box has been ticked by looking at the darkness of the box compared to the \
        threshold.
        To be used with the capture dataframe to determine if an answer box has been ticked.
        :param row: Row from the capture dataframe
        :return: 1 (ticked) or 0 (not ticked)
        """
        # Get thresholds to calculate ticked answers and get the item analysis
        darkness_bottom = float(self.variables.loc['darkness_threshold']['value'])
        darkness_top = float(self.variables.loc['darkness_threshold_up']['value'])

        # If the box has been manually (un-)ticked, set 'ticked' to 1 (ticked) or 0 (un-ticked).
        if row['manual'] != -1:
            return row['manual']
        # If the box darkness is within the threshold => 'ticked' = 1
        elif row['total'] * darkness_bottom < row['black'] <= row['total'] * darkness_top:
            return 1
        else:
            return 0

    def _questions_discrimination(self, bottom: pd.DataFrame, top: pd.DataFrame) -> list[float]:
        """
        Calculate the discrimination index for each question.
        Add a column 'discrimination' to the dataframe 'question_df' with the index for each question
        :return: a list of discrimination indices to be added as a column to question_df
        """
        # Merge questions scores and students mark, bottom quantile
        bottom_merged_df = pd.merge(bottom,
                                    (self.scores.select_dtypes(include=['int64', 'float64'])
                                     if 'cancelled' not in self.scores.columns
                                     else self.scores[self.scores['cancelled'] == 0].
                                     select_dtypes(include=['int64', 'float64'])),
                                    on=['student'], how="inner", validate="many_to_many")

        # Merge questions scores and students mark, top quantile
        top_merged_df = pd.merge(top,
                                 (self.scores.select_dtypes(include=['int64', 'float64'])
                                  if 'cancelled' not in self.scores.columns
                                  else self.scores[self.scores['cancelled'] == 0].
                                  select_dtypes(include=['int64', 'float64'])),
                                 on=['student'], how="inner", validate="many_to_many")

        # Group by question and answer, and calculate the mean mark for each group
        top_mean_df = top_merged_df.groupby(['question', 'student']).mean()
        bottom_mean_df = bottom_merged_df.groupby(['question', 'student']).mean()

        # Calculate the discrimination index for each question
        discrimination = []  # Create a list to store the results
        nb_in_groups = round(len(self.marks) * DISCRIMINATION_QUANTILE)
        for question in top_mean_df.index.levels[0]:
            discr_index = (len(top_mean_df.loc[question][
                                   top_mean_df.loc[question]['score'] == top_mean_df.loc[question][
                                       'score'].max()])
                           - len(bottom_mean_df.loc[question][
                                     bottom_mean_df.loc[question]['score'] ==
                                     bottom_mean_df.loc[question]['score'].max()])) / nb_in_groups
            discrimination.append(discr_index)  # Add the result to the list
        return discrimination

    def _items_discrimination(self, bottom: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the discrimination index for each answer.
        Add a column 'discrimination' to the dataframe 'items_df' with the index for each choice
        :return: a list of discrimination indices to be added as a column to items_df
        """
        # Merge questions scores and students mark, bottom quantile
        bottom_merged_df = bottom.merge(self.capture[['student', 'question', 'answer', 'ticked']],
                                        on='student', how='left')

        # Merge questions scores and students mark, top quantile
        top_merged_df = top.merge(self.capture[['student', 'question', 'answer', 'ticked']],
                                  on='student', how='left')

        # Group by question and answer, and calculate the mean mark for each group
        top_sum_df = top_merged_df[['question', 'answer', 'ticked']].groupby(
            ['question', 'answer']).sum()
        bottom_sum_df = bottom_merged_df[['question', 'answer', 'ticked']].groupby(
            ['question', 'answer']).sum()

        # Calculate the discrimination index for each question
        # Create a dictionary to store the results
        discrimination: dict[str, list[Any]] = {'question': [], 'answer': [], 'discrimination': []}
        nb_in_groups = round(len(self.marks) * DISCRIMINATION_QUANTILE)
        for question in top_sum_df.index.levels[0]:
            for answer in top_sum_df.loc[question].index:
                discr_index = (top_sum_df.loc[question, answer]['ticked']
                               - bottom_sum_df.loc[question, answer]['ticked']) \
                              / nb_in_groups
                discrimination['question'].append(question)
                discrimination['answer'].append(answer)
                discrimination['discrimination'].append(discr_index)
        return pd.DataFrame.from_dict(discrimination, orient='columns')

    def _item_correlation(self) -> pd.DataFrame:
        """
        Calculate the item correlation for each question.
        :return: DataFrame of item correlations with questions as index
        """
        if 'cancelled' in self.scores.columns:
            filtered_scores = self.scores[self.scores['cancelled'] is False]
            logger.debug(f"Filtered Scores Columns: {filtered_scores.columns}")
            logger.debug(f"Marks Columns: {self.marks.columns}")

            merged_df = pd.merge(filtered_scores, self.marks, on='student',
                                 how="inner", validate="many_to_many")
        else:
            merged_df = pd.merge(self.scores, self.marks, on='student', how="inner", validate="many_to_many")
        item_corr = {}
        questions = merged_df['title'].unique()
        for question in questions:
            item_scores = merged_df.loc[merged_df['title'] == question, 'correct']
            total_scores = merged_df.loc[merged_df['title'] == question, 'mark']
            correlation = stats.pointbiserialr(item_scores, total_scores)
            item_corr[question] = correlation[0]
        return pd.DataFrame.from_dict(item_corr, orient='index', columns=['correlation'])

    def _outcome_correlation(self) -> pd.DataFrame:
        """
        Calculate the outcome correlation for each outcome of each question.
        :return: DataFrame of outcome correlations
        """
        if 'cancelled' in self.scores.columns:
            merged_df = pd.merge(self.capture, self.scores[['student', 'question', 'cancelled']],
                                 on=['student', 'question'], how="inner", validate="many_to_many")
            merged_df = merged_df[merged_df['cancelled'] is False].merge(self.marks[['student', 'mark']],
                                                                         on='student')
        else:
            merged_df = self.capture.merge(self.marks[['student', 'mark']], on='student')
        outcome_corr: dict[str, list[Any]] = {'question': [], 'answer': [], 'correlation': []}
        questions = merged_df['question'].unique()
        for question in questions:
            answers = merged_df['answer'][merged_df['question'] == question].unique()
            for answer in answers:
                item_scores = merged_df.loc[
                    (merged_df['question'] == question) & (merged_df['answer'] == answer), 'ticked']
                total_scores = merged_df.loc[
                    (merged_df['question'] == question) & (merged_df['answer'] == answer), 'mark']
                correlation = stats.pointbiserialr(item_scores.values, total_scores.values)
                outcome_corr['question'].append(question)
                outcome_corr['answer'].append(answer)
                outcome_corr['correlation'].append(correlation[0])
        return pd.DataFrame.from_dict(outcome_corr)

    @staticmethod
    def _get_dictionary(dictionary: str) -> dict:
        """
        Get the definitions from the definitions.json file
        :return: a dictionary of definitions
        """
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dictionary + ".json")

        try:
            with open(file_path, "r") as json_file:
                data_dict = json.load(json_file)
        except FileNotFoundError:
            logger.error(f"File '{file_path}' not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
        return data_dict


class Charts:
    """
    Generates and saves statistical charts for exam analysis.

    Creates histograms and visualizations for marks, difficulty, discrimination,
    and correlation metrics.
    """

    def __init__(self, exam_project: ExamProject, exam_data: ExamData) -> None:
        """
        Initialize charts generator and create all charts.

        Args:
            exam_project: ExamProject instance with project configuration
            exam_data: ExamData instance with exam statistics

        Side Effects:
            Creates image files in the project's img directory
        """
        self.exam: ExamProject = exam_project
        self.data: ExamData = exam_data
        self.image_path: str = os.path.join(self.exam.path, 'img')
        os.makedirs(self.image_path, exist_ok=True)
        self.question_data_columns: list[str] = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
        self.actual_data_columns: list[str] = list(set(self.question_data_columns).intersection(self.data.questions.columns))
        # Calculate the number of bins based on the maximum and minimum marks
        self.mark_bins: int = int(self.data.marks['mark'].max() - self.data.marks['mark'].min())
        self._create_mark_histogram()
        self._create_difficulty_histogram()
        if self.data.number_of_examinees > self.data.threshold:
            self._create_discrimination_histogram()
            self._create_difficulty_vs_discrimination_histogram()
        self._create_question_correlation_histogram()
        self._create_bar_chart()

    def _create_mark_histogram(self) -> None:
        """
        Create a histogram of the 'marks' column in the dataset and save it as an image file.
        """
        # Create a subplot with specified dimensions
        plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT))

        # Create a histogram of the 'marks' column with KDE and specified bin count
        sns.histplot(self.data.marks['mark'], kde=True, bins=self.mark_bins)

        # Calculate the average value
        average_value: float = self.data.general_stats['Mean']

        # Add a vertical line for the average value
        plt.axvline(average_value, color='red', linestyle='--',
                    label=f'Mean ({round(average_value, 2)})')

        # Set plot labels and legend
        plt.xlabel('Mark')
        plt.ylabel('Number of students')
        plt.legend()

        # Save the plot as an image file
        plt.savefig(os.path.join(self.image_path, 'marks.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_difficulty_histogram(self) -> None:
        """
        Create a histogram of the 'difficulty' column in the dataset and save it as an image file.
        """
        ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT))[1]
        sns.histplot(self.data.questions['difficulty'], bins=DIFFICULTY_HISTOGRAM_BINS, color='blue', ax=ax)
        average_value = self.data.questions['difficulty'].mean()
        ax.axvline(average_value, color='red', linestyle='--', label=f'Average ({round(average_value, 2)})')
        ax.set_xlabel('Difficulty level (higher is easier)')
        ax.set_ylabel(NBQ)
        ax.legend()
        # Set the color of the bars in the histogram
        threshold1 = 13
        threshold2 = 23
        for patch in ax.patches:
            if patch.get_x() < threshold1:  # type: ignore[attr-defined]  # matplotlib Patch has get_x
                patch.set_color('tab:red')
            elif patch.get_x() < threshold2:  # type: ignore[attr-defined]  # matplotlib Patch has get_x
                patch.set_color('tab:blue')
            else:
                patch.set_color('tab:green')
        plt.savefig(os.path.join(self.image_path, 'difficulty.png'), transparent=False, facecolor='white',
                    bbox_inches="tight")

    def _create_discrimination_histogram(self) -> None:
        """
        create a histogram of discrimination if enough students
        """
        # Define constants
        average_line_color = 'red'
        histogram_colors = ['tab:orange', 'tab:blue', 'tab:green']

        _, axis = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        sns.histplot(self.data.questions['discrimination'], bins=DISCRIMINATION_HISTOGRAM_BINS, ax=axis,
                     color=histogram_colors[0], label='Discrimination Index')
        average_value = self.data.questions['discrimination'].mean()
        axis.axvline(average_value, color=average_line_color, linestyle='--',
                     label=f'Average ({round(average_value, 2)})')
        axis.set_xlabel('Discrimination index (the higher the better)')
        axis.set_ylabel(NBQ)
        axis.legend()
        plt.savefig(os.path.join(self.image_path, 'discrimination.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_difficulty_vs_discrimination_histogram(self) -> None:
        """
        Create a scatter plot of discrimination index vs difficulty level of all questions.
        If there are fewer than 90 students, no plot is generated.
        The average difficulty and discrimination index are shown as horizontal and vertical lines respectively.

        :raises ValueError: If there is no data or if the required columns are missing from the data.
        """
        if not self.data.questions.size:
            raise ValueError("No data available")

        required_columns = ['discrimination', 'difficulty']
        if not all(col in self.data.questions.columns for col in required_columns):
            raise ValueError("Missing required columns")

        _, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        sns.scatterplot(x=self.data.questions['discrimination'], y=self.data.questions['difficulty'], ax=ax)

        average_x = np.nanmean(self.data.questions['discrimination'])
        average_y = np.nanmean(self.data.questions['difficulty'])

        ax.axhline(average_y, color='red', linestyle='--',
                   label=f'Average Difficulty ({average_y:.2f})')
        ax.axvline(average_x, color='blue', linestyle='--',
                   label=f'Average Discrimination ({average_x:.2f})')

        ax.set_xlabel('Discrimination index (the higher the better)')
        ax.set_ylabel('Difficulty level (higher is easier)')
        ax.legend()
        plt.savefig(os.path.join(self.image_path, 'discrimination_vs_difficulty.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_question_correlation_histogram(self) -> None:
        """
        Create a histogram of question correlation
        """
        _, ax3 = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        sns.histplot(self.data.questions['correlation'], kde=True, bins=self.mark_bins * CORRELATION_BINS_MULTIPLIER)
        average_value = self.data.questions['correlation'].mean()
        ax3.axvline(average_value, color='red', linestyle='--',
                    label=f'Average ({average_value:.2f})')
        ax3.set(xlabel='Item correlation', ylabel=NBQ)
        ax3.legend()
        plt.savefig(os.path.join(self.image_path, 'item_correlation.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_bar_chart(self) -> None:
        """
        Create a bar chart for questions data columns
        """
        columns = self.actual_data_columns
        values = self.data.questions[columns].mean().round(2)
        sorted_values = values.sort_values(ascending=False)
        _, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT))

        sns.barplot(x=sorted_values, y=sorted_values.index, ax=ax)

        ax.set_xlabel('Average Number of Students')
        ax.set_ylabel('Question Status')

        for i, v in enumerate(sorted_values):
            ax.text(v + 3, i, str(v), color='black', ha='center', va='center')

        plt.savefig(os.path.join(self.image_path, 'question_columns.png'),
                    transparent=False, facecolor='white', bbox_inches="tight")


def get_list_questions(qlist: list[str]) -> str:
    """
    Format a list of question titles into a grammatical string.

    Args:
        qlist: List of question titles

    Returns:
        Formatted string like "Q1, Q2 and Q3"
    """
    return ', '.join(qlist[:-1]) + ' and ' + qlist[-1]


def add_blurb_conditionally(df: pd.DataFrame, column: str, threshold: float, message_template: str) -> str:
    """
    Add a blurb to the report if the condition is met.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    column : str
        The column to check.
    threshold : float
        The threshold value.
    message_template : str
        The template for the blurb message.

    Returns
    -------
    str
        The blurb message if the condition is met, empty string otherwise.
    """
    if column in df.columns:
        filtered_titles = df[df[column] > threshold].sort_values('title')['title'].values
        if filtered_titles.size > 0:
            qlist = get_list_questions(filtered_titles)
            if len(filtered_titles) > 1:
                blb_message = message_template.format(qlist=qlist)
            else:
                blb_message = message_template.format(qlist=filtered_titles[0])
            return blb_message
    return ''


def get_blurb(exam: ExamData) -> str:
    """
    Generate a first level of analysis on the performance of the questions. This text can either
    be used as is in the report or passed to ChatGPT for a better wording.
    :return: a string of text describing the data and how to improve the exam questions.
    """
    intro: str = "According to the data collected, the following questions should probably be reviewed:\n"
    blb: str = ''

    # Conditions for cancelled questions
    blb += add_blurb_conditionally(
        exam.questions,
        'cancelled',
        exam.questions['presented'] * CANCELLATION_THRESHOLD,
        "- Questions {qlist} have been cancelled more than 80% of the time.\n"
    )

    # Conditions for empty questions
    blb += add_blurb_conditionally(
        exam.questions,
        'empty',
        exam.questions['presented'] * EMPTY_ANSWER_THRESHOLD,
        "- Questions {qlist} have been empty more than 80% of the time.\n"
    )

    # Conditions for negative discrimination
    if 'discrimination' in exam.questions.columns:
        negative_discrimination = exam.questions[exam.questions['discrimination'] < 0].sort_values('title')[
            'title'].values
        if negative_discrimination.size > 0:
            qlist = get_list_questions(negative_discrimination)
            blb += f"- Questions {qlist} have a negative discrimination, meaning that there is a " \
                   f"possibility of an error in the questions (incorrect outcome indicated as correct).\n"

    # Check for not ticked questions
    if exam.items[exam.items['ticked'] == 0]['title'].values.size > 0:
        not_ticked = exam.items[exam.items['ticked'] == 0]['title'].unique()
        not_ticked.sort()
        if len(not_ticked) > 1:
            qlist = get_list_questions(not_ticked)
            blb += f"- Questions {qlist} have distractors that have never been chosen.\n"
        else:
            blb += f"- Question {not_ticked[0]} has distractors that have never been chosen.\n"

    # Final message construction
    if blb:
        return intro + blb
    else:
        return ("According to the data collected, there are no questions to review based on their "
                "performance.")


def print_dataframes() -> None:
    """
    Print the dataframes for debugging purposes.

    Logs all major dataframes at DEBUG level for troubleshooting.
    """
    logger.debug(f"\nGeneral Statistics:\n{data.table}")
    logger.debug(f"\nList of questions (question_df):\n{data.questions.head()}")
    logger.debug(f"\nList of answers (answer_df):\n{data.answers.head()}")
    logger.debug(f"\nList of items (items_df):\n{data.items.sort_values(by='title').head()}")
    logger.debug(f"\nList of variables (variables_df):\n{data.variables}")
    logger.debug(f"\nList of capture (capture_df):\n{data.capture.head()}")
    logger.debug(f"\nList of mark (mark_df):\n{data.marks.head()}")
    logger.debug(f"\nList of score (score_df):\n{data.scores.head()}")


def get_correction_text(df: pd.DataFrame) -> str:
    """
    Generate a paragraphe of text from the capture data to explain the number of boxes ticked or
    un-ticked during the marking process.

    :param df: Dataframe with the capture data
    :type df: pd.Dataframe
    :return: a paragraph of text to display in the report
    :rtype: str
    """
    tres: int = MANUAL_CORRECTION_DARKNESS_THRESHOLD
    nb_box_filled = df[(df['manual'] == 1) & (df['black'] < tres)]['student'].count()
    nb_box_emptied = df[(df['manual'] == 0) & (df['black'] > tres)]['student'].count()
    nb_box_untouched = df[(df['manual'] == -1)]['student'].count()
    nb_box_total = df['student'].count()
    nb_box_untouched += df[(df['manual'] == 1) & (df['black'] > tres)]['student'].count()
    nb_box_untouched += df[(df['manual'] == 0) & (df['black'] < tres)]['student'].count()
    pc_filled: str = f"{nb_box_filled * 100 / nb_box_total:.2f}%"
    pc_emptied: str = f"{nb_box_emptied * 100 / nb_box_total:.2f}%"
    pc_untouched: str = f"{nb_box_untouched * 100 / nb_box_total:.2f}%"
    txt: str = (f"This examination is comprised of {nb_box_total} boxes in total. During the "
                f"marking process, {nb_box_filled} ({pc_filled}) have been manually filled "
                f"(ticked), {nb_box_emptied} ({pc_emptied}) have been manually emptied (un-ticked) "
                f"and {nb_box_untouched} ({pc_untouched}) have not been changed.")
    return txt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        # Initialize settings first (to get log level)
        try:
            initial_settings = get_settings()
            log_level = initial_settings.log_level
        except ValidationError:
            # If settings fail, use default log level
            log_level = 'INFO'

        # Configure logging
        logger = setup_logging(log_level=log_level)
        logger.info("AMC Report Generator started")

        install()
        ic.disable()
        sns.set_theme()
        sns.set_style('darkgrid')
        sns.set_style()
        sns.color_palette("tab10")
        # colour palette (red, green, blue, text color)
        # Load application settings
        app_settings = get_settings()

        # Get color palette from settings
        colour_palette: dict = app_settings.get_colour_palette()

        # Get some directory information
        # Get the dir name to check if it matches 'Projets-QCM'
        current_dir_name: str = os.path.basename(os.getcwd())
        # Get the full path in case it matches
        current_full_path: str = os.path.dirname(os.getcwd()) + '/' + current_dir_name
        today = datetime.datetime.now().strftime('%d/%m/%Y')

        config: Settings = Settings(settings=app_settings)
        # Get the Project directory and questions file paths
        project: ExamProject = ExamProject(config)
        data: ExamData = ExamData(project.path)

        blurb: str = ''
        if app_settings.enable_ai_analysis:
            try:
                analyzer = ClaudeAnalyzer(data.table)
                ic(analyzer.response)
                blurb = analyzer.response + '\n\n'
            except AIAnalysisError as e:
                logger.warning(f"AI analysis failed: {e}")
                logger.warning("Continuing without AI-generated summary...")
                # Continue without AI summary

        # Generate the report
        blurb += get_blurb(data)
        correction_text: str = get_correction_text(data.capture)
        ic(blurb)
        ic(correction_text)
        report_params = {
            'project_name': project.name,
            'project_path': project.path,
            'questions': data.questions,
            'items': data.items,
            'stats': data.general_stats,
            'threshold': project.threshold,
            'marks': data.marks,
            'definitions': data.definitions,
            'findings': data.findings,
            'palette': colour_palette,
            'blurb': blurb,
            'company_name': project.company_name,
            'company_url': project.company_url,
            'correction': correction_text,
        }
        # plot_charts(report_params)
        charts: Charts = Charts(project, data)
        report_url: str = generate_pdf_report(report_params)

        # Save exam metrics to repository if enabled
        try:
            repository = ExamRepository(app_settings)
            if repository.is_enabled():
                logger.info(f"Repository backend: {app_settings.repository_backend}")

                # Prompt user to save metrics
                print("\n" + "="*60)
                print("Exam Repository: Save Metrics")
                print("="*60)
                save_choice = input("Save exam metrics to repository? [Y/n]: ").strip().lower()

                if save_choice != 'n':
                    # Create metrics from exam data
                    metrics = create_exam_metrics_from_data(
                        project_name=project.name,
                        exam_data=data
                    )

                    # Save to repository
                    if repository.save_exam_metrics(metrics):
                        print(f"✓ Exam metrics saved successfully to {app_settings.repository_backend}")
                        logger.info(f"Exam metrics saved for {project.name}")
                    else:
                        print(f"✗ Failed to save exam metrics")
                        logger.warning(f"Failed to save exam metrics for {project.name}")
                else:
                    print("Skipped saving exam metrics")
                print("="*60 + "\n")
        except RepositoryError as e:
            logger.warning(f"Repository error: {e}")
            logger.warning("Continuing without saving exam metrics...")
        except Exception as e:
            logger.warning(f"Unexpected error with repository: {e}")
            logger.warning("Continuing without saving exam metrics...")

        # Open the report
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', report_url))
        elif platform.system() == 'Windows':  # Windows
            os.startfile(report_url)  # type: ignore[attr-defined]  # Windows-only attribute
        else:  # linux variants
            if shutil.which("zathura") is not None:
                subprocess.Popen(['zathura', report_url], start_new_session=True, stderr=subprocess.DEVNULL)
            else:
                subprocess.call(('xdg-open', report_url))

    except DatabaseError as e:
        logger.error(f"Database Error: {e}")
        logger.error("Please ensure the AMC project has been properly processed and the database files exist.")
        sys.exit(1)

    except ConfigurationError as e:
        logger.error(f"Configuration Error: {e}")
        logger.error("Please check your settings.conf file and ensure all paths are correct.")
        sys.exit(1)

    except AMCReportError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error("Please report this issue with the full error message.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
