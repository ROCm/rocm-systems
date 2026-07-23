#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""conftest.py for rocpd backward-compatibility tests.

Registers the command-line options that CMakeLists.txt passes when invoking
pytest so that schema version sets and output directory paths remain defined
in CMake rather than hard-coded in validate.py.

Options
-------
--output-root
    Root directory under which CMake's execute tests wrote conversion outputs.
    Layout: <root>/<version>/{csv,pftrace,otf2}/

--tested-schemas
    Space-separated list of schema versions with explicit test coverage (used
    to enforce that no version ships without a backward-compat test).

--old-schemas
    Schema versions before the latest one (they lack graph-launch tables).
    Used to parametrize the ``old_schema`` fixture.

--latest-schema
    The newest supported schema version (has graph-launch tables).
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Command-line option registration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--output-root",
        action="store",
        default=None,
        help="Root directory containing pre-generated conversion outputs per schema version",
    )
    parser.addoption(
        "--tested-schemas",
        nargs="+",
        default=None,
        help="Schema versions with explicit backward-compat test coverage",
    )
    parser.addoption(
        "--old-schemas",
        nargs="+",
        default=None,
        help="Schema versions before the latest (lack graph-launch tables)",
    )
    parser.addoption(
        "--latest-schema",
        action="store",
        default=None,
        help="Latest (newest) supported schema version",
    )


# ---------------------------------------------------------------------------
# Dynamic parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Parametrize the ``old_schema`` fixture from the --old-schemas option."""
    if "old_schema" in metafunc.fixturenames:
        schemas = metafunc.config.getoption("--old-schemas", default=None) or [
            "3.0.0",
            "3.0.1",
        ]
        metafunc.parametrize("old_schema", schemas)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_root(request) -> Path:
    """Path to the root of pre-generated conversion outputs."""
    root = request.config.getoption("--output-root", default=None)
    if root is None:
        pytest.skip("--output-root not provided; run tests via CTest")
    return Path(root)


@pytest.fixture
def tested_schemas(request) -> list:
    """Schema versions with explicit backward-compat test coverage."""
    schemas = request.config.getoption("--tested-schemas", default=None)
    return schemas or ["3.0.0", "3.0.1", "3.0.2"]


@pytest.fixture
def latest_schema(request) -> str:
    """Latest (newest) supported schema version."""
    schema = request.config.getoption("--latest-schema", default=None)
    return schema or "3.0.2"
