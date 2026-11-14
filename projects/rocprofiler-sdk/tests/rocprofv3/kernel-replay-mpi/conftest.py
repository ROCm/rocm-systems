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

import pytest


def pytest_addoption(parser):
    parser.addoption("--output-dir", action="store", required=True)
    parser.addoption("--mpi-ranks", action="store", type=int, required=True)
    parser.addoption(
        "--profiled-ranks",
        action="store",
        type=int,
        nargs="+",
        required=True,
        help="ranks that run under the profiler; the others run the application unprofiled",
    )
    parser.addoption("--iters", action="store", type=int, required=True)
    parser.addoption(
        "--groups",
        action="store",
        type=int,
        required=True,
        help="number of --pmc groups == replay passes per replayed dispatch",
    )
    parser.addoption("--range-first", action="store", type=int, required=True)
    parser.addoption("--range-last", action="store", type=int, required=True)


@pytest.fixture
def output_dir(request):
    return request.config.getoption("--output-dir")


@pytest.fixture
def mpi_ranks(request):
    return request.config.getoption("--mpi-ranks")


@pytest.fixture
def profiled_ranks(request):
    return list(request.config.getoption("--profiled-ranks"))


@pytest.fixture
def iters(request):
    return request.config.getoption("--iters")


@pytest.fixture
def groups(request):
    return request.config.getoption("--groups")


@pytest.fixture
def range_first(request):
    return request.config.getoption("--range-first")


@pytest.fixture
def range_last(request):
    return request.config.getoption("--range-last")
