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
import pytest
import json
import pandas as pd
import shutil
import subprocess

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader


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
        "--counter-input",
        action="store",
        help="Path to counter collection CSV file.",
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
        "--json-input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        help="Path to Perfetto trace file.",
    )


@pytest.fixture
def agent_info_input_data(request):
    filename = request.config.getoption("--agent-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def hsa_input_data(request):
    filename = request.config.getoption("--hsa-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def kernel_input_data(request):
    filename = request.config.getoption("--kernel-input")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)

    return None


@pytest.fixture
def counter_input_data(request):
    filename = request.config.getoption("--counter-input")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)

    return None


@pytest.fixture
def memory_copy_input_data(request):
    filename = request.config.getoption("--memory-copy-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def marker_input_data(request):
    filename = request.config.getoption("--marker-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def hip_input_data(request):
    filename = request.config.getoption("--hip-input")
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def hip_stats_data(request):
    filename = request.config.getoption("--hip-stats")
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def hsa_stats_data(request):
    filename = request.config.getoption("--hsa-stats")
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def kernel_stats_data(request):
    filename = request.config.getoption("--kernel-stats")
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def memory_copy_stats_data(request):
    filename = request.config.getoption("--memory-copy-stats")
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

    return data


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def pftrace_reader(request):
    filename = request.config.getoption("--pftrace-input")
    return PerfettoReader(filename)


@pytest.fixture
def pftrace_data(pftrace_reader):
    return pftrace_reader.read()[0]


def _gpu_uses_auto_perf_level(agent):
    if not agent.get("name", "").startswith(("gfx11", "gfx12")):
        return False

    amd_smi = shutil.which("amd-smi") or os.path.join(
        os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "amd-smi"
    )
    try:
        result = subprocess.run(
            [amd_smi, "metric", "--gpu", "0", "--perf-level", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        metrics = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        # An unavailable performance level must not hide a counter regression.
        return False
    if isinstance(metrics, dict):
        metrics = metrics.get("gpu_data", [])
    return (
        isinstance(metrics, list)
        and len(metrics) == 1
        and isinstance(metrics[0], dict)
        and metrics[0].get("perf_level") == "AMDSMI_DEV_PERF_LEVEL_AUTO"
    )


@pytest.fixture
def zero_counters_in_auto_mode(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    entries = data["callback_records"]["counter_collection"]
    values = [record["value"] for entry in entries for record in entry["records"]]
    if not values or any(value != 0 for value in values):
        return False
    first_gpu = next((agent for agent in data["agents"] if agent["type"] == 2), None)
    return first_gpu is not None and _gpu_uses_auto_perf_level(first_gpu)
