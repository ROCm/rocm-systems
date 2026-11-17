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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--output-root", action="store", help="Repeated ATT output directory"
    )
    parser.addoption(
        "--expected-dispatches",
        action="store",
        default=3,
        type=int,
        help="Expected captured dispatches in each attachment",
    )
    parser.addoption("--skip-if", action="store", help="Skip test if file exists")


def _skip_if(request):
    skip_file = request.config.getoption("--skip-if")
    if skip_file and Path(skip_file).exists():
        pytest.skip("Attach tests unavailable due to insufficient ptrace permissions")


@pytest.fixture(scope="session")
def output_root(request):
    _skip_if(request)
    value = request.config.getoption("--output-root")
    if not value:
        pytest.skip("--output-root not provided")
    return Path(value)


@pytest.fixture(scope="session")
def expected_dispatches(request):
    _skip_if(request)
    return request.config.getoption("--expected-dispatches")
