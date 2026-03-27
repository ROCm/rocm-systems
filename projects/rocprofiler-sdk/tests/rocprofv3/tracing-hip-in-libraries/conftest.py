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

import os
import csv
import glob
import pytest
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader
from rocprofiler_sdk.pytest_utils.otf2_reader import OTF2Reader


def pytest_addoption(parser):
    parser.addoption(
        "--agent-input",
        action="store",
        help="Path to agent info CSV file.",
    )
    parser.addoption(
        "--hsa-input",
        action="store",
        help="Path to HSA API tracing CSV file.",
    )
    parser.addoption(
        "--kernel-input",
        action="store",
        help="Path to kernel tracing CSV file.",
    )
    parser.addoption(
        "--memory-copy-input",
        action="store",
        help="Path to memory-copy tracing CSV file.",
    )
    parser.addoption(
        "--marker-input",
        action="store",
        help="Path to marker API tracing CSV file.",
    )
    parser.addoption(
        "--hip-input",
        action="store",
        help="Path to HIP runtime and compiler API tracing CSV file.",
    )
    parser.addoption(
        "--hip-stats",
        action="store",
        help="Path to HIP stats CSV file.",
    )
    parser.addoption(
        "--hsa-stats",
        action="store",
        help="Path to HSA stats CSV file.",
    )
    parser.addoption(
        "--kernel-stats",
        action="store",
        help="Path to kernel stats CSV file.",
    )
    parser.addoption(
        "--memory-copy-stats",
        action="store",
        help="Path to memory copy stats CSV file.",
    )
    parser.addoption(
        "--json-input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        help="Path to Perfetto trace file.",
    )
    parser.addoption(
        "--otf2-input",
        action="store",
        help="Path to OTF2 trace file.",
    )


@pytest.fixture
def agent_info_input_data(request):
    filename_pattern = request.config.getoption("--agent-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def hsa_input_data(request):
    filename_pattern = request.config.getoption("--hsa-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def kernel_input_data(request):
    filename_pattern = request.config.getoption("--kernel-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def memory_copy_input_data(request):
    filename_pattern = request.config.getoption("--memory-copy-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def marker_input_data(request):
    filename_pattern = request.config.getoption("--marker-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def hip_input_data(request):
    filename_pattern = request.config.getoption("--hip-input")
    matches = glob.glob(filename_pattern)
    data = []
    if matches:
        assert (
            len(matches) == 1
        ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
        with open(matches[0], "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def hip_stats_data(request):
    filename_pattern = request.config.getoption("--hip-stats")
    matches = glob.glob(filename_pattern)
    data = []
    if matches:
        assert (
            len(matches) == 1
        ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
        with open(matches[0], "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def hsa_stats_data(request):
    filename_pattern = request.config.getoption("--hsa-stats")
    matches = glob.glob(filename_pattern)
    data = []
    if matches:
        assert (
            len(matches) == 1
        ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
        with open(matches[0], "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def kernel_stats_data(request):
    filename_pattern = request.config.getoption("--kernel-stats")
    matches = glob.glob(filename_pattern)
    data = []
    if matches:
        assert (
            len(matches) == 1
        ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
        with open(matches[0], "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def memory_copy_stats_data(request):
    filename_pattern = request.config.getoption("--memory-copy-stats")
    matches = glob.glob(filename_pattern)
    data = []
    if matches:
        assert (
            len(matches) == 1
        ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
        with open(matches[0], "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def json_data(request):
    filename_pattern = request.config.getoption("--json-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    with open(matches[0], "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def pftrace_data(request):
    filename_pattern = request.config.getoption("--pftrace-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    return PerfettoReader(matches[0]).read()[0]


@pytest.fixture
def otf2_data(request):
    filename_pattern = request.config.getoption("--otf2-input")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    return OTF2Reader(matches[0]).read()[0]
