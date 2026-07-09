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
import json
import pytest
import csv
import yaml
import importlib.util
from importlib.machinery import SourceFileLoader

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--agent-input",
        action="store",
        help="Path to agent info CSV file.",
    )
    parser.addoption(
        "--counter-input",
        action="store",
        help="Path to counter collection CSV file.",
    )
    parser.addoption(
        "--multiplex-input",
        action="store",
        help="Path to the JSON/YAML multiplex layout input used for the run.",
    )
    parser.addoption(
        "--rocprofv3",
        action="store",
        help="Path to the rocprofv3 script (imported for GPU-free parse unit tests).",
    )
    parser.addoption(
        "--max-value-ratio",
        action="store",
        type=float,
        default=None,
        help="If set, assert per-counter values are stable across identical "
        "dispatches (max <= ratio * min).",
    )
    parser.addoption(
        "--allow-zero-counter-values",
        action="store_true",
        default=False,
        help="Relax the per-row Counter_Value check from '> 0' to '>= 0' (for the "
        "oversubscription layout, whose counters the workload may never exercise).",
    )
    parser.addoption(
        "--counter-input-b",
        action="store",
        default=None,
        help="Path to a second counter CSV (same layout, other input format) for "
        "the run-level JSON/YAML equivalence test.",
    )
    parser.addoption(
        "--present-counters",
        action="store",
        default=None,
        help="Comma-separated exact set of counters expected to survive, for the "
        "graceful-degradation test.",
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
def counter_input_data(request):
    filename = request.config.getoption("--counter-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def multiplex_layout(request):
    # Layout is read from the input file so the validator stays in sync with the
    # run. Returns None when --multiplex-input is absent so layout-specific tests
    # can skip cleanly.
    filename = request.config.getoption("--multiplex-input")
    if not filename:
        return None
    with open(filename, "r") as inp:
        if filename.endswith((".yml", ".yaml")):
            config = yaml.safe_load(inp)
        else:
            config = json.load(inp)

    job = config["jobs"][0]
    pmc_groups = [list(group) for group in job["pmc_groups"]]

    # interval defaults to 1; an invalid value must fail here rather than as a
    # ZeroDivisionError/TypeError in the group derivation.
    pmc_group_interval = job.get("pmc_group_interval", 1)
    assert (
        isinstance(pmc_group_interval, int) and pmc_group_interval > 0
    ), "pmc_group_interval must be a positive integer"

    return pmc_groups, pmc_group_interval


@pytest.fixture
def max_value_ratio(request):
    return request.config.getoption("--max-value-ratio")


@pytest.fixture
def allow_zero_counter_values(request):
    return request.config.getoption("--allow-zero-counter-values")


@pytest.fixture
def counter_input_b_data(request):
    filename = request.config.getoption("--counter-input-b")
    if not filename:
        return None
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def present_counters(request):
    value = request.config.getoption("--present-counters")
    if not value:
        return None
    return {name.strip() for name in value.split(",") if name.strip()}


@pytest.fixture
def rocprofv3_path(request):
    path = request.config.getoption("--rocprofv3")
    assert path, "--rocprofv3 must point to the rocprofv3 script"
    return path


@pytest.fixture
def rocprofv3_module(request):
    # Import the rocprofv3 script directly (no .py extension) for GPU-free parse tests.
    path = request.config.getoption("--rocprofv3")
    assert path, "--rocprofv3 must point to the rocprofv3 script"

    loader = SourceFileLoader("rocprofv3_under_test", os.path.realpath(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
