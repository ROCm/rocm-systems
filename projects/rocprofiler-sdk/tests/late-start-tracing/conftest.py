#!/usr/bin/env python3
"""Pytest configuration for late-start tracing tests."""

import pytest
import os


def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption(
        "--skip-if",
        action="store",
        default=None,
        help="Skip file path",
    )


@pytest.fixture
def config(request):
    """Configuration fixture."""
    return {
        "executed": True,
        "skip_file": request.config.getoption("--skip-if"),
    }
