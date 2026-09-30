#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import csv
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--kernel-input",
        action="store",
        help="Path to kernel tracing CSV file.",
    )
    parser.addoption(
        "--expected-dispatch-count",
        action="store",
        type=int,
        help="Expected number of kernel dispatch records.",
    )
    parser.addoption(
        "--expected-kernels",
        action="store",
        type=int,
        help="Expected kernels per graph launch.",
    )
    parser.addoption(
        "--expected-iterations",
        action="store",
        type=int,
        help="Expected number of graph launches.",
    )


@pytest.fixture
def kernel_input_data(request):
    filename = request.config.getoption("--kernel-input")
    if filename is None:
        pytest.fail("--kernel-input argument is required but was not provided")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    assert (
        len(data) > 0
    ), f"CSV file '{filename}' contained no data rows. The profiler may have failed to produce output."
    return data


@pytest.fixture
def expected_dispatch_count(request):
    value = request.config.getoption("--expected-dispatch-count")
    if value is None:
        pytest.fail("--expected-dispatch-count argument is required but was not provided")
    return value


@pytest.fixture
def expected_kernels(request):
    value = request.config.getoption("--expected-kernels")
    if value is None:
        pytest.fail("--expected-kernels argument is required but was not provided")
    return value


@pytest.fixture
def expected_iterations(request):
    value = request.config.getoption("--expected-iterations")
    if value is None:
        pytest.fail("--expected-iterations argument is required but was not provided")
    return value
