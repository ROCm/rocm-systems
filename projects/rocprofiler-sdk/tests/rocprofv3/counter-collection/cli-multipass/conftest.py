#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir",
        action="store",
        help="Path to output directory.",
    )
    parser.addoption(
        "--expected-counters",
        action="store",
        nargs="+",
        default=[],
        help="Ordered list of counters, one per pass (pass_1, pass_2, ...). "
        "The number of passes is derived from the length of this list.",
    )
    parser.addoption(
        "--binary-dir",
        action="store",
        help="Path to the directory holding the per-test output directories.",
    )
    parser.addoption(
        "--absent-globs",
        action="store",
        nargs="+",
        default=[],
        help="Glob patterns, relative to --binary-dir, that must not match any "
        "directory because the runs that would have created them were rejected.",
    )
    parser.addoption(
        "--present-globs",
        action="store",
        nargs="+",
        default=[],
        help="Glob patterns, relative to --binary-dir, that must match. Confirms "
        "--binary-dir points where the absent patterns are meant to be checked.",
    )


@pytest.fixture
def output_dir(request):
    return request.config.getoption("--output-dir")


@pytest.fixture
def expected_counters(request):
    return request.config.getoption("--expected-counters")


@pytest.fixture
def binary_dir(request):
    return request.config.getoption("--binary-dir")


@pytest.fixture
def absent_globs(request):
    return request.config.getoption("--absent-globs")


@pytest.fixture
def present_globs(request):
    return request.config.getoption("--present-globs")
