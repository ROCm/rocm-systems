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
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader


def pytest_addoption(parser):

    parser.addoption(
        "--agent-index",
        choices=("absolute", "relative", "type-relative"),
        help="...",
    )
    parser.addoption(
        "--csv-kernel-input",
        action="store",
        help="Path to kernel tracing CSV file",
    )
    parser.addoption(
        "--csv-memory-allocation-input",
        action="store",
        help="Path to memory allocation tracing CSV file",
    )
    parser.addoption(
        "--csv-memory-copy-input",
        action="store",
        help="Path to memory allocation tracing CSV file",
    )
    parser.addoption(
        "--json-input",
        action="store",
        help="Path to JSON file.",
    )


def read_csv(filename):
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def csv_kernel_input(request):
    filename = request.config.getoption("--csv-kernel-input")
    return read_csv(filename)


@pytest.fixture
def csv_memory_copy_input(request):
    filename = request.config.getoption("--csv-memory-copy-input")
    return read_csv(filename)


@pytest.fixture
def csv_memory_allocation_input(request):
    filename = request.config.getoption("--csv-memory-allocation-input")
    return read_csv(filename)


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def agent_index(request):
    return request.config.getoption("--agent-index")
