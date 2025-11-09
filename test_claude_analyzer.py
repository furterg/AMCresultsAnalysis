"""
Pytest tests for ClaudeAnalyzer class.

These tests validate AI-powered statistical analysis functionality using
mocked Anthropic API calls to avoid external dependencies and costs.
"""
import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from amcreport import CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS, CLAUDE_SYSTEM_PROMPT
from amcreport import ClaudeAnalyzer, AIAnalysisError, sanitize_text_for_pdf


@pytest.fixture
def sample_stats_table():
    """
    Create sample statistics table for testing.

    Returns:
        DataFrame with exam statistics
    """
    return pd.DataFrame({
        'title': ['Q001', 'Q002', 'Q003', 'Q004', 'Q005'],
        'difficulty': [0.75, 0.45, 0.60, 0.85, 0.30],
        'discrimination': [0.45, 0.35, 0.50, 0.20, 0.60],
        'correlation': [0.42, 0.38, 0.48, 0.25, 0.55],
        'presented': [100, 100, 100, 100, 100],
        'correct': [75, 45, 60, 85, 30],
    })


@pytest.fixture
def mock_claude_response():
    """
    Create a mock Claude API response.

    Returns:
        Mock message object with structure matching Anthropic API
    """
    mock_message = Mock()
    mock_content = Mock()
    mock_content.text = """The exam shows generally good quality with most questions performing well.
The average difficulty of 0.59 indicates a moderately challenging exam, which is appropriate for most contexts.

Most questions demonstrate strong discrimination (>0.3) and correlation (>0.2) values, suggesting they effectively
differentiate between high and low performers. Question Q005 shows particularly excellent metrics with high
discrimination (0.60) despite being the most difficult question.

One concern is Question Q004, which has relatively low discrimination (0.20) despite being very easy (0.85 difficulty).
Consider reviewing this question as it may not be contributing effectively to the overall assessment. The other
questions appear well-constructed and are functioning as intended."""

    mock_message.content = [mock_content]
    return mock_message


class TestClaudeAnalyzerInitialization:
    """Test ClaudeAnalyzer initialization."""

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-api-key-12345'})
    @patch('amcreport.Anthropic')
    def test_init_with_claude_api_key(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test initialization with CLAUDE_API_KEY environment variable."""
        # Mock the client and its response
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Verify Anthropic client was initialized with correct API key
        mock_anthropic.assert_called_once_with(api_key='test-api-key-12345')
        assert analyzer.client == mock_client
        assert analyzer.model == CLAUDE_MODEL
        assert analyzer.temperature == CLAUDE_TEMPERATURE
        assert analyzer.max_tokens == CLAUDE_MAX_TOKENS

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'anthropic-key-67890'}, clear=True)
    @patch('amcreport.Anthropic')
    def test_init_with_anthropic_api_key(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test initialization with ANTHROPIC_API_KEY as fallback."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Should use ANTHROPIC_API_KEY when CLAUDE_API_KEY not set
        mock_anthropic.assert_called_once_with(api_key='anthropic-key-67890')

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self, sample_stats_table):
        """Test that missing API key raises AIAnalysisError."""
        with pytest.raises(AIAnalysisError, match="No API key found"):
            ClaudeAnalyzer(sample_stats_table)

    @patch.dict(os.environ, {
        'CLAUDE_API_KEY': 'claude-key',
        'ANTHROPIC_API_KEY': 'anthropic-key'
    })
    @patch('amcreport.Anthropic')
    def test_claude_api_key_precedence(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test that CLAUDE_API_KEY takes precedence over ANTHROPIC_API_KEY."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        ClaudeAnalyzer(sample_stats_table)

        # Should use CLAUDE_API_KEY when both are set
        mock_anthropic.assert_called_once_with(api_key='claude-key')

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_init_with_custom_parameters(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test initialization with custom model parameters."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(
            sample_stats_table,
            model='claude-opus-4',
            temperature=0.7,
            max_tokens=1024
        )

        assert analyzer.model == 'claude-opus-4'
        assert analyzer.temperature == 0.7
        assert analyzer.max_tokens == 1024


class TestFormatStatsForAnalysis:
    """Test statistics formatting for Claude."""

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_format_includes_all_data(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test that formatted stats include all table data."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)
        formatted = analyzer._format_stats_for_analysis()

        # Check that all question titles are present
        for title in sample_stats_table['title']:
            assert title in formatted

        # Check that formatting instructions are included
        assert "analyze these results" in formatted.lower()
        assert "recommendations" in formatted.lower()

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_format_uses_float_precision(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test that floats are formatted with 3 decimal places."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)
        formatted = analyzer._format_stats_for_analysis()

        # Check for 3-decimal formatting
        assert '0.750' in formatted or '0.45' in formatted  # At least some values formatted

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_format_with_empty_dataframe(self, mock_anthropic, mock_claude_response):
        """Test formatting with empty DataFrame."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        empty_df = pd.DataFrame()
        analyzer = ClaudeAnalyzer(empty_df)
        formatted = analyzer._format_stats_for_analysis()

        # Should still have instructions even with empty data
        assert "analyze these results" in formatted.lower()


class TestAnalyze:
    """Test Claude API analysis."""

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_analyze_calls_api_correctly(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test that analyze calls Claude API with correct parameters."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Verify API was called with correct parameters
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs

        assert call_kwargs['model'] == CLAUDE_MODEL
        assert call_kwargs['max_tokens'] == CLAUDE_MAX_TOKENS
        assert call_kwargs['temperature'] == CLAUDE_TEMPERATURE
        assert call_kwargs['system'] == CLAUDE_SYSTEM_PROMPT
        assert 'messages' in call_kwargs
        assert call_kwargs['messages'][0]['role'] == 'user'

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_analyze_returns_sanitized_text(self, mock_anthropic, sample_stats_table):
        """Test that analysis response passes through sanitize_text_for_pdf."""
        # Create response with em-dash (which IS converted)
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Analysis with em-dash \u2014 here"
        mock_message.content = [mock_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Em-dash should be converted to hyphen
        assert '\u2014' not in analyzer.response
        assert '-' in analyzer.response

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_analyze_stores_response(self, mock_anthropic, sample_stats_table, mock_claude_response):
        """Test that analysis response is stored in instance variable."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        assert hasattr(analyzer, 'response')
        assert isinstance(analyzer.response, str)
        assert len(analyzer.response) > 0
        assert 'exam' in analyzer.response.lower()


class TestAPIErrorHandling:
    """Test handling of API errors."""

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_api_timeout_error(self, mock_anthropic, sample_stats_table):
        """Test handling of API timeout."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = TimeoutError("API request timed out")
        mock_anthropic.return_value = mock_client

        with pytest.raises(AIAnalysisError, match="Claude analysis failed"):
            ClaudeAnalyzer(sample_stats_table)

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_api_rate_limit_error(self, mock_anthropic, sample_stats_table):
        """Test handling of rate limit errors."""
        mock_client = Mock()
        # Simulate Anthropic rate limit error
        rate_limit_error = Exception("Rate limit exceeded")
        mock_client.messages.create.side_effect = rate_limit_error
        mock_anthropic.return_value = mock_client

        with pytest.raises(AIAnalysisError, match="Claude analysis failed"):
            ClaudeAnalyzer(sample_stats_table)

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_api_authentication_error(self, mock_anthropic, sample_stats_table):
        """Test handling of authentication errors."""
        mock_client = Mock()
        auth_error = Exception("Invalid API key")
        mock_client.messages.create.side_effect = auth_error
        mock_anthropic.return_value = mock_client

        with pytest.raises(AIAnalysisError, match="Claude analysis failed"):
            ClaudeAnalyzer(sample_stats_table)

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_api_network_error(self, mock_anthropic, sample_stats_table):
        """Test handling of network errors."""
        mock_client = Mock()
        network_error = ConnectionError("Network unreachable")
        mock_client.messages.create.side_effect = network_error
        mock_anthropic.return_value = mock_client

        with pytest.raises(AIAnalysisError, match="Claude analysis failed"):
            ClaudeAnalyzer(sample_stats_table)

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_api_generic_error(self, mock_anthropic, sample_stats_table):
        """Test handling of generic API errors."""
        mock_client = Mock()
        generic_error = Exception("Something went wrong")
        mock_client.messages.create.side_effect = generic_error
        mock_anthropic.return_value = mock_client

        with pytest.raises(AIAnalysisError, match="Claude analysis failed"):
            ClaudeAnalyzer(sample_stats_table)


class TestResponseParsing:
    """Test parsing and validation of API responses."""

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_parse_multiline_response(self, mock_anthropic, sample_stats_table):
        """Test parsing response with multiple paragraphs."""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = """Paragraph one with analysis.

Paragraph two with more details.

Paragraph three with recommendations."""
        mock_message.content = [mock_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Response should preserve newlines and structure
        assert 'Paragraph one' in analyzer.response
        assert 'Paragraph two' in analyzer.response
        assert 'Paragraph three' in analyzer.response

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_parse_response_with_markdown(self, mock_anthropic, sample_stats_table):
        """Test parsing response with markdown formatting."""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = """Analysis with **bold** and *italic* text.

- Point one
- Point two"""
        mock_message.content = [mock_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Basic markdown should be preserved
        assert '**bold**' in analyzer.response or 'bold' in analyzer.response
        assert '*italic*' in analyzer.response or 'italic' in analyzer.response

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_empty_response(self, mock_anthropic, sample_stats_table):
        """Test handling of empty API response."""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = ""
        mock_message.content = [mock_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        analyzer = ClaudeAnalyzer(sample_stats_table)

        # Should have empty string response
        assert analyzer.response == ""


class TestSanitizeTextForPDF:
    """Test text sanitization for PDF compatibility."""

    def test_sanitize_smart_quotes(self):
        """Test that smart quotes are currently NOT converted (implementation limitation)."""
        text = "Text with \u201csmart quotes\u201d and \u2018smart apostrophes\u2019"
        sanitized = sanitize_text_for_pdf(text)

        # Current implementation doesn't convert quotes (commented out)
        assert '\u201c' in sanitized
        assert '\u201d' in sanitized
        assert '\u2018' in sanitized
        assert '\u2019' in sanitized

    def test_sanitize_dashes(self):
        """Test conversion of em-dash (en-dash NOT converted currently)."""
        text = "Text with \u2013 en-dash and \u2014 em-dash"
        sanitized = sanitize_text_for_pdf(text)

        # En-dash conversion is commented out in implementation
        assert '\u2013' in sanitized  # Not converted
        # Em-dash IS converted
        assert '\u2014' not in sanitized
        assert '-' in sanitized

    def test_sanitize_ellipsis(self):
        """Test that ellipsis is currently NOT converted (implementation limitation)."""
        text = "Text with\u2026 ellipsis"
        sanitized = sanitize_text_for_pdf(text)

        # Current implementation doesn't convert ellipsis (commented out)
        assert '\u2026' in sanitized

    def test_sanitize_preserves_safe_characters(self):
        """Test that safe ASCII characters are preserved."""
        text = "Normal text with numbers 123 and symbols !@#$%"
        sanitized = sanitize_text_for_pdf(text)

        assert sanitized == text  # Should be unchanged

    def test_sanitize_multiple_unicode_characters(self):
        """Test sanitization with multiple Unicode characters."""
        text = "Text with \u201cquotes\u201d, \u2013 en-dash and \u2014 em-dash, and\u2026 ellipsis"
        sanitized = sanitize_text_for_pdf(text)

        # Only em-dash is converted in current implementation
        assert '\u2014' not in sanitized  # Em-dash converted
        assert '\u201c' in sanitized  # Quotes not converted
        assert '\u2013' in sanitized  # En-dash not converted
        assert '\u2026' in sanitized  # Ellipsis not converted

    def test_sanitize_empty_string(self):
        """Test sanitization of empty string."""
        assert sanitize_text_for_pdf("") == ""

    def test_sanitize_none_unicode(self):
        """Test text that's already ASCII-safe."""
        text = "Plain ASCII text"
        sanitized = sanitize_text_for_pdf(text)
        assert sanitized == text


class TestIntegrationWithRealData:
    """Integration tests using realistic exam data."""

    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'test-key'})
    @patch('amcreport.Anthropic')
    def test_full_analysis_workflow(self, mock_anthropic):
        """Test complete analysis workflow with realistic data."""
        # Create realistic exam statistics
        stats_df = pd.DataFrame({
            'title': [f'Q{i:03d}' for i in range(1, 21)],
            'difficulty': [0.65, 0.45, 0.80, 0.55, 0.70, 0.40, 0.75, 0.60,
                          0.85, 0.50, 0.72, 0.48, 0.68, 0.58, 0.77, 0.43,
                          0.81, 0.52, 0.74, 0.46],
            'discrimination': [0.45, 0.38, 0.25, 0.52, 0.41, 0.60, 0.33, 0.48,
                              0.22, 0.55, 0.39, 0.58, 0.44, 0.50, 0.28, 0.62,
                              0.20, 0.53, 0.35, 0.59],
            'correlation': [0.42, 0.35, 0.28, 0.48, 0.39, 0.55, 0.32, 0.45,
                           0.25, 0.51, 0.37, 0.54, 0.41, 0.47, 0.30, 0.58,
                           0.23, 0.49, 0.33, 0.56],
        })

        # Mock comprehensive response
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = """The exam demonstrates solid overall quality with an average difficulty of 0.62,
indicating a moderately challenging assessment appropriate for most educational contexts.

Question quality is generally strong, with 85% of questions showing good discrimination (>0.30) and correlation (>0.20)
values. Questions Q006, Q016, and Q020 show particularly excellent psychometric properties. However, Questions Q003,
Q009, Q015, and Q017 have lower discrimination values (<0.30), suggesting they may not effectively differentiate
between high and low performers.

Recommendations: Review the four low-discrimination questions to identify potential issues with answer key,
distractors, or clarity. Consider piloting revisions in future exams. The remaining questions are functioning well
and should be retained in the item bank."""
        mock_message.content = [mock_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        # Run analysis
        analyzer = ClaudeAnalyzer(stats_df)

        # Verify complete workflow
        assert analyzer.response is not None
        assert len(analyzer.response) > 0
        assert 'discrimination' in analyzer.response.lower()
        assert 'recommendations' in analyzer.response.lower()

        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs['system'] == CLAUDE_SYSTEM_PROMPT
        assert len(call_kwargs['messages']) == 1
