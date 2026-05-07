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
    for pmc_id in (1, 2, 3, 4, 5):
        parser.addoption(
            f"--input-csv-pmc{pmc_id}",
            action="store",
            help=f"Path to CSV file produced by SQG counter pass {pmc_id}.",
        )
        parser.addoption(
            f"--input-json-pmc{pmc_id}",
            action="store",
            help=f"Path to JSON file produced by SQG counter pass {pmc_id}.",
        )


def _read_csv(request, option):
    filename = request.config.getoption(option)
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


def _read_json(request, option):
    filename = request.config.getoption(option)
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_csv_pmc1(request):
    return _read_csv(request, "--input-csv-pmc1")


@pytest.fixture
def input_json_pmc1(request):
    return _read_json(request, "--input-json-pmc1")


@pytest.fixture
def input_csv_pmc2(request):
    return _read_csv(request, "--input-csv-pmc2")


@pytest.fixture
def input_json_pmc2(request):
    return _read_json(request, "--input-json-pmc2")


@pytest.fixture
def input_csv_pmc3(request):
    return _read_csv(request, "--input-csv-pmc3")


@pytest.fixture
def input_json_pmc3(request):
    return _read_json(request, "--input-json-pmc3")


@pytest.fixture
def input_csv_pmc4(request):
    return _read_csv(request, "--input-csv-pmc4")


@pytest.fixture
def input_json_pmc4(request):
    return _read_json(request, "--input-json-pmc4")


@pytest.fixture
def input_csv_pmc5(request):
    return _read_csv(request, "--input-csv-pmc5")


@pytest.fixture
def input_json_pmc5(request):
    return _read_json(request, "--input-json-pmc5")
