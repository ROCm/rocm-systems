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

import json
import pandas as pd
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption("--pmc-json", action="store", help="Path to PMC JSON file.")
    parser.addoption("--spm-json", action="store", help="Path to SPM JSON file.")
    parser.addoption(
        "--counter-csv", action="store", help="Path to rocpd counter CSV file."
    )
    parser.addoption(
        "--spm-derived-json",
        action="store",
        help="Path to SPM derived JSON file.",
    )
    parser.addoption(
        "--spm-derived-only-json",
        action="store",
        help="Path to SPM derived-only JSON file.",
    )
    parser.addoption(
        "--spm-derived-csv",
        action="store",
        help="Path to SPM derived counter CSV file.",
    )


@pytest.fixture
def pmc_json_data(request):
    filename = request.config.getoption("--pmc-json")
    if filename is None:
        pytest.skip("--pmc-json not provided")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def spm_json_data(request):
    filename = request.config.getoption("--spm-json")
    if filename is None:
        pytest.skip("--spm-json not provided")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def counter_csv(request):
    filename = request.config.getoption("--counter-csv")
    if filename is None:
        pytest.skip("--counter-csv not provided")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def spm_derived_json_data(request):
    filename = request.config.getoption("--spm-derived-json")
    if filename is None:
        pytest.skip("--spm-derived-json not provided")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def spm_derived_only_json_data(request):
    filename = request.config.getoption("--spm-derived-only-json")
    if filename is None:
        pytest.skip("--spm-derived-only-json not provided")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def spm_derived_csv(request):
    filename = request.config.getoption("--spm-derived-csv")
    if filename is None:
        pytest.skip("--spm-derived-csv not provided")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)
