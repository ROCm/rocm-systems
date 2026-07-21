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
import json
import sqlite3
import pytest

from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption("--kernel-input", action="store", default=None)
    parser.addoption("--memory-copy-input", action="store", default=None)
    parser.addoption("--graph-launch-input", action="store", default=None)
    parser.addoption("--hip-api-input", action="store", default=None)
    parser.addoption("--json-input", action="store", default=None)
    parser.addoption("--rocpd-input", action="store", default=None)
    parser.addoption("--expected-iterations", action="store", type=int, default=None)
    parser.addoption("--expected-execs", action="store", type=int, default=None)
    parser.addoption(
        "--expected-nodes-per-launch", action="store", type=int, default=None
    )
    parser.addoption(
        "--expected-kernel-nodes-per-launch", action="store", type=int, default=None
    )
    parser.addoption(
        "--expected-memcpy-nodes-per-launch", action="store", type=int, default=None
    )
    parser.addoption(
        "--expected-distinct-kernels", action="store", type=int, default=None
    )


def _read_csv(filename):
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def kernel_input_data(request):
    filename = request.config.getoption("--kernel-input")
    if filename is None:
        pytest.fail("--kernel-input argument is required")
    data = _read_csv(filename)
    assert len(data) > 0, f"CSV file '{filename}' contained no data rows"
    return data


@pytest.fixture
def memory_copy_input_data(request):
    """Memory copy CSV may be absent on AMD HIP when memcpys are implemented
    as blit kernels and surface in KERNEL_DISPATCH instead of MEMORY_COPY.
    Tests skip rather than fail in that case."""
    import os

    filename = request.config.getoption("--memory-copy-input")
    if filename is None:
        pytest.fail("--memory-copy-input argument is required")
    if not os.path.exists(filename):
        pytest.skip(f"memory_copy CSV '{filename}' was not produced")
    data = _read_csv(filename)
    if len(data) == 0:
        pytest.skip(f"memory_copy CSV '{filename}' contained no data rows")
    return data


@pytest.fixture
def graph_launch_input_data(request):
    """HIP_GRAPH CSV output is deprecated; fixture returns an empty list when
    the file is absent so the legacy CSV tests can skip cleanly."""
    import os

    filename = request.config.getoption("--graph-launch-input")
    if filename is None or not os.path.exists(filename):
        return []
    return _read_csv(filename)


@pytest.fixture
def hip_api_input_data(request):
    filename = request.config.getoption("--hip-api-input")
    if filename is None:
        pytest.fail("--hip-api-input argument is required")
    return _read_csv(filename)


@pytest.fixture
def json_input_data(request):
    filename = request.config.getoption("--json-input")
    if filename is None:
        pytest.fail("--json-input argument is required")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def rocpd_connection(request):
    filename = request.config.getoption("--rocpd-input")
    if filename is None:
        pytest.fail("--rocpd-input argument is required")
    conn = sqlite3.connect(filename)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def expected_iterations(request):
    return request.config.getoption("--expected-iterations")


@pytest.fixture
def expected_execs(request):
    return request.config.getoption("--expected-execs")


@pytest.fixture
def expected_nodes_per_launch(request):
    return request.config.getoption("--expected-nodes-per-launch")


@pytest.fixture
def expected_kernel_nodes_per_launch(request):
    return request.config.getoption("--expected-kernel-nodes-per-launch")


@pytest.fixture
def expected_memcpy_nodes_per_launch(request):
    return request.config.getoption("--expected-memcpy-nodes-per-launch")


@pytest.fixture
def expected_distinct_kernels(request):
    return request.config.getoption("--expected-distinct-kernels")
