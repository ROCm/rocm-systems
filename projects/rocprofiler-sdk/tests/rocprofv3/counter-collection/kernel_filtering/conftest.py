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

import json
import pytest
import pandas as pd

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):

    parser.addoption(
        "--input-json-pass1",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass1",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-pass2",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass2",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-pass3",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass3",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-pass4",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass4",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-csv-pmc1",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-csv-iteration-range",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-iteration-range",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--kernel-iteration-range",
        action="store",
        help="Kernel iteration range passed on the command line.",
    )

    parser.addoption(
        "--iteration-config",
        action="store",
        help="Path to the input config (JSON/YAML) whose per-pass "
        "kernel_iteration_range is used to check captured dispatch counts.",
    )

    parser.addoption(
        "--input-csv-file",
        action="store",
        help="Path to a generic CSV file (used by the CLI filter tests).",
    )

    parser.addoption(
        "--input-json-file",
        action="store",
        help="Path to a generic JSON file (used by the CLI filter tests).",
    )


def tokenize(kernel_iteration_range):
    range_str = kernel_iteration_range.replace("[", "").replace("]", "")
    split_list = range_str.split(",")
    _range = []
    for split_string in split_list:
        if "-" in split_string:
            interval = split_string.split("-")
            [
                _range.append(i)
                for i in list(range((int)(interval[0]), (int)(interval[1]) + 1))
            ]
        else:
            _range.append(int(split_string))
    return _range


@pytest.fixture
def input_csv_pass1(request):
    filename = request.config.getoption("--input-csv-pass1")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pass2(request):
    filename = request.config.getoption("--input-csv-pass2")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pass3(request):
    filename = request.config.getoption("--input-csv-pass3")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pass4(request):
    filename = request.config.getoption("--input-csv-pass4")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pmc1(request):
    filename = request.config.getoption("--input-csv-pmc1")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_iteration_range(request):
    filename = request.config.getoption("--input-csv-iteration-range")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_file(request):
    filename = request.config.getoption("--input-csv-file")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_json_file(request):
    filename = request.config.getoption("--input-json-file")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_iteration_range(request):
    filename = request.config.getoption("--input-json-iteration-range")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def iteration_range(request):
    kernel_iteration_range = request.config.getoption("--kernel-iteration-range")
    return tokenize(kernel_iteration_range.strip())


def _load_config_jobs(config_path):
    if config_path.endswith((".yml", ".yaml")):
        import yaml

        with open(config_path, "r") as inp:
            return yaml.safe_load(inp)["jobs"]
    with open(config_path, "r") as inp:
        return json.load(inp)["jobs"]


def _iteration_range_for_pass(config_path, pass_index):
    # match rocprofv3: a YAML list of ranges joins with ", " like the CLI
    kernel_iteration_range = _load_config_jobs(config_path)[pass_index].get(
        "kernel_iteration_range", ""
    )
    if isinstance(kernel_iteration_range, list):
        kernel_iteration_range = ", ".join(kernel_iteration_range)
    kernel_iteration_range = kernel_iteration_range.strip()
    if not kernel_iteration_range:
        return None
    return tokenize(kernel_iteration_range)


@pytest.fixture
def iteration_range_pass1(request):
    return _iteration_range_for_pass(request.config.getoption("--iteration-config"), 0)


@pytest.fixture
def iteration_range_pass2(request):
    return _iteration_range_for_pass(request.config.getoption("--iteration-config"), 1)


@pytest.fixture
def input_json_pass1(request):
    filename = request.config.getoption("--input-json-pass1")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_pass2(request):
    filename = request.config.getoption("--input-json-pass2")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_pass3(request):
    filename = request.config.getoption("--input-json-pass3")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_pass4(request):
    filename = request.config.getoption("--input-json-pass4")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
