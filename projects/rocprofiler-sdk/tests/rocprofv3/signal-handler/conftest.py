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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

# Recognized --expect outcomes. An unknown value must fail loudly rather than silently skip
# every check (which would let a scenario pass with zero assertions run).
VALID_EXPECT = ("clean-exit", "flushed-markers")


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir", action="store", help="Output directory from rocprofv3"
    )
    parser.addoption(
        "--expect",
        action="store",
        help="Expected outcome to assert: clean-exit or flushed-markers",
    )
    parser.addoption("--process-type", action="store", help="Process type tested")


@pytest.fixture
def output_dir(request):
    val = request.config.getoption("--output-dir")
    if val is None:
        pytest.fail("--output-dir not provided")
    return val


@pytest.fixture
def expect(request):
    val = request.config.getoption("--expect")
    if val is None:
        pytest.fail("--expect not provided")
    if val not in VALID_EXPECT:
        pytest.fail(f"invalid --expect '{val}'; expected one of {list(VALID_EXPECT)}")
    return val


@pytest.fixture
def process_type(request):
    val = request.config.getoption("--process-type")
    if val is None:
        pytest.fail("--process-type not provided")
    return val
