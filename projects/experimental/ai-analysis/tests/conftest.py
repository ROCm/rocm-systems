#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Shared pytest configuration and fixtures for rocm-ai-analysis tests.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require a real database"
    )
    config.addinivalue_line(
        "markers", "requires_anthropic: marks tests that require an Anthropic API key"
    )
    config.addinivalue_line(
        "markers", "requires_openai: marks tests that require an OpenAI API key"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn():
    """Return a MagicMock that mimics an ``AnalysisConnection``.

    The mock exposes a real in-memory ``sqlite3.Connection`` as its
    ``.connection`` attribute, so SQL statements can be executed against
    it when needed.  For pure-mock usage the ``execute`` and ``cursor``
    methods are also available as mocks.
    """
    conn = MagicMock()
    conn.connection = sqlite3.connect(":memory:")
    conn._paths = ["/tmp/mock_trace.db"]
    conn.table_info = {}

    # Allow execute() to fall through to the real connection
    conn.execute = conn.connection.execute
    conn.executescript = conn.connection.executescript

    yield conn

    conn.connection.close()


@pytest.fixture
def mock_cursor():
    """Return a MagicMock that mimics a ``sqlite3.Cursor``.

    Useful for testing code that consumes query results without needing
    a real database.
    """
    cursor = MagicMock(spec=sqlite3.Cursor)
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    cursor.description = []
    return cursor
