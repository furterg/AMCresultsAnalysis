#!/usr/bin/env python3
"""
Test script to demonstrate Pydantic settings validation.

This shows how Pydantic catches configuration errors with helpful messages.
"""

import os

from pydantic import ValidationError

from settings import AMCSettings, get_settings


def test_valid_settings():
    """Test that valid settings load correctly"""
    print("=" * 60)
    print("Test 1: Loading valid settings from .env")
    print("=" * 60)
    try:
        settings = get_settings()
        print("✓ Settings loaded successfully!")
        print(f"  Projects dir: {settings.projects_dir}")
        print(f"  Student threshold: {settings.student_threshold}")
        print(f"  Company: {settings.company_name}")
        print(f"  AI enabled: {settings.enable_ai_analysis}")
        print(f"  Log level: {settings.log_level}")
        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return False


def test_invalid_threshold():
    """Test validation of student threshold"""
    print("=" * 60)
    print("Test 2: Invalid student threshold (negative value)")
    print("=" * 60)
    try:
        # Temporarily set invalid environment variable
        os.environ['AMC_STUDENT_THRESHOLD'] = '-5'
        settings = AMCSettings(
            projects_dir="/tmp",  # Dummy path for testing
            student_threshold=-5
        )
        print("✗ Should have raised ValidationError!\n")
        return False
    except ValidationError as e:
        print("✓ Validation error caught correctly:")
        print(f"  {e.errors()[0]['msg']}")
        print()
        return True
    finally:
        # Clean up
        os.environ.pop('AMC_STUDENT_THRESHOLD', None)


def test_invalid_temperature():
    """Test validation of Claude temperature"""
    print("=" * 60)
    print("Test 3: Invalid Claude temperature (out of range)")
    print("=" * 60)
    try:
        settings = AMCSettings(
            projects_dir="/tmp",
            claude_temperature=2.0  # Must be 0.0-1.0
        )
        print("✗ Should have raised ValidationError!\n")
        return False
    except ValidationError as e:
        print("✓ Validation error caught correctly:")
        print(f"  {e.errors()[0]['msg']}")
        print()
        return True


def test_invalid_log_level():
    """Test validation of log level enum"""
    print("=" * 60)
    print("Test 4: Invalid log level (not in allowed values)")
    print("=" * 60)
    try:
        settings = AMCSettings(
            projects_dir="/tmp",
            log_level="TRACE"  # Not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        )
        print("✗ Should have raised ValidationError!\n")
        return False
    except ValidationError as e:
        print("✓ Validation error caught correctly:")
        print(f"  {e.errors()[0]['msg']}")
        print()
        return True


def test_missing_projects_dir():
    """Test validation of required fields"""
    print("=" * 60)
    print("Test 5: Missing required projects_dir")
    print("=" * 60)

    # Backup and remove environment variable
    backup = os.environ.pop('AMC_PROJECTS_DIR', None)

    try:
        settings = AMCSettings()
        print("✗ Should have raised ValidationError!\n")
        return False
    except ValidationError as e:
        print("✓ Validation error caught correctly:")
        print(f"  {e.errors()[0]['msg']}")
        print()
        return True
    finally:
        # Restore
        if backup:
            os.environ['AMC_PROJECTS_DIR'] = backup


def test_colour_palette():
    """Test the colour palette method"""
    print("=" * 60)
    print("Test 6: Colour palette method")
    print("=" * 60)
    try:
        settings = get_settings()
        palette = settings.get_colour_palette()
        print("✓ Colour palette retrieved:")
        print(f"  Keys: {', '.join(palette.keys())}")
        print(f"  Heading 1 color: {palette['heading_1']}")
        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PYDANTIC SETTINGS VALIDATION TESTS")
    print("=" * 60 + "\n")

    results = []
    results.append(("Valid settings", test_valid_settings()))
    results.append(("Invalid threshold", test_invalid_threshold()))
    results.append(("Invalid temperature", test_invalid_temperature()))
    results.append(("Invalid log level", test_invalid_log_level()))
    results.append(("Missing projects_dir", test_missing_projects_dir()))
    results.append(("Colour palette", test_colour_palette()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    return passed == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
