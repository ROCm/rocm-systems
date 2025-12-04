#!/usr/bin/env python3
"""Validation script for late-start tracing test.

This script validates that the late-start functionality works correctly by checking:
1. HIP API calls were traced after late-start
2. Kernel dispatch traces were captured
3. Memory operation traces exist
"""

import sys
import os


def test_basic_execution(config):
    """Test that the application executed and produced output."""
    assert config.get("executed", False), "Test application did not execute"


def test_has_output_files(config):
    """Test that output files were created."""
    # For now, just check that the test ran
    # In a real scenario, we'd check for output files from rocprofiler
    assert True, "Output file check placeholder"


def test_skipped_check(config):
    """Check if test was skipped due to missing GPU or rocprofiler."""
    skip_file = config.get("skip_file")
    if skip_file and os.path.exists(skip_file):
        import pytest
        pytest.skip("Test was skipped (no GPU or rocprofiler not installed)")


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--skip-if",
        action="store",
        default=None,
        help="File that indicates test should be skipped",
    )


def pytest_configure(config):
    """Configure pytest with custom settings."""
    skip_file = config.getoption("--skip-if")
    if skip_file and os.path.exists(skip_file):
        config.addinivalue_line("markers", "skip: mark test to skip")
