#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices,
# Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from pathlib import Path

import pytest

from validate import load_csv_data


def pytest_addoption(parser):
    """Register command-line options for this test module."""
    parser.addoption(
        "--csv-input",
        action="store",
        help="Path to kernel trace CSV file to validate.",
    )


@pytest.fixture
def csv_data(request):
    """
    Return CSV data as a list of dictionaries.
    
    This fixture follows the existing format used in other tests.
    All validation logic has been moved to validate.py.
    """
    filename = request.config.getoption("--csv-input")
    if not filename:
        raise RuntimeError("--csv-input option is required for this test")
    
    path = Path(filename)
    return load_csv_data(path)
