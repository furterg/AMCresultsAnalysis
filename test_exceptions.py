#!/usr/bin/env python3
"""
Simple test to verify exception handling works correctly.
This test doesn't require the full environment to be set up.
"""
import os
import sys

# Add the current directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import just the exception classes
from amcreport import (
    AMCReportError,
    DatabaseError,
    LLMError,
    ConfigurationError,
    ExamData
)


def test_database_error():
    """Test that DatabaseError is raised when database doesn't exist"""
    print("Testing DatabaseError...")
    try:
        ExamData._check_db("/nonexistent/path/database.sqlite")
        print("❌ FAILED: Should have raised DatabaseError")
        return False
    except DatabaseError as e:
        print(f"✓ PASSED: DatabaseError correctly raised: {e}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Wrong exception type: {type(e)}")
        return False


def test_exception_hierarchy():
    """Test that custom exceptions inherit from AMCReportError"""
    print("\nTesting exception hierarchy...")
    try:
        raise DatabaseError("Test database error")
    except AMCReportError:
        print("✓ PASSED: DatabaseError is a subclass of AMCReportError")
        passed = True
    except Exception:
        print("❌ FAILED: DatabaseError should inherit from AMCReportError")
        passed = False

    try:
        raise LLMError("Test LLM error")
    except AMCReportError:
        print("✓ PASSED: LLMError is a subclass of AMCReportError")
    except Exception:
        print("❌ FAILED: LLMError should inherit from AMCReportError")
        passed = False

    try:
        raise ConfigurationError("Test config error")
    except AMCReportError:
        print("✓ PASSED: ConfigurationError is a subclass of AMCReportError")
    except Exception:
        print("❌ FAILED: ConfigurationError should inherit from AMCReportError")
        passed = False

    return passed


def test_exception_messages():
    """Test that exceptions properly store and display messages"""
    print("\nTesting exception messages...")
    test_message = "This is a test error message"

    try:
        raise DatabaseError(test_message)
    except DatabaseError as e:
        if str(e) == test_message:
            print(f"✓ PASSED: Exception message correctly stored: '{e}'")
            return True
        else:
            print(f"❌ FAILED: Expected '{test_message}', got '{e}'")
            return False


if __name__ == '__main__':
    print("=" * 60)
    print("Exception Handling Verification Tests")
    print("=" * 60)

    all_passed = True
    all_passed &= test_database_error()
    all_passed &= test_exception_hierarchy()
    all_passed &= test_exception_messages()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
