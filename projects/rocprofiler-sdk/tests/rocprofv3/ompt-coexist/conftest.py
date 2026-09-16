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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import os
import sqlite3

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--rocpd-input",
        action="store",
        default="out_results.db",
        help="Input rocpd SQLite database written by rocprofv3",
    )
    parser.addoption(
        "--mock-input",
        action="store",
        default="coexist-mock.json",
        help="JSON summary the mock OMPT tool writes at teardown",
    )


@pytest.fixture
def rocpd_conn(request):
    """Open the rocpd SQLite database. OMPT is a rocpd-only trace, so all OMPT
    validators read from here. Skipped when the database is unavailable (the
    execute test that produces it was skipped or failed)."""
    filename = request.config.getoption("--rocpd-input")
    if not os.path.isfile(filename):
        return pytest.skip("rocpd output unavailable")
    return sqlite3.connect(filename)


@pytest.fixture
def mock_summary(request):
    """Parsed JSON summary from the mock OMPT tool, or None when the file is
    absent. Absence is itself an assertable outcome: the mock only writes the
    summary once the OpenMP runtime has loaded it, so no file means the runtime
    never went looking for a second tool."""
    filename = request.config.getoption("--mock-input")
    if not os.path.isfile(filename):
        return None
    with open(filename, "r") as inp:
        return json.load(inp)["coexist-mock-tool"]
