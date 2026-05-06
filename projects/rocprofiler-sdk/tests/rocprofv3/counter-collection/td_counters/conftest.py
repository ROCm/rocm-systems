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
import pytest
import pandas as pd

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list

# Each --input-csv-pmcN/--input-json-pmcN pair maps to one `pmc:` line in
# input.txt (rocprofv3 emits a separate pmc_N/ output directory per pass).
NUM_PASSES = 5


def pytest_addoption(parser):
    for n in range(1, NUM_PASSES + 1):
        parser.addoption(
            f"--input-csv-pmc{n}",
            action="store",
            help=f"Path to pmc_{n} counter_collection.csv file.",
        )
        parser.addoption(
            f"--input-json-pmc{n}",
            action="store",
            help=f"Path to pmc_{n} results.json file.",
        )


def _read_csv(filename):
    if not filename:
        return None
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


def _read_json(filename):
    if not filename:
        return None
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_csv_pmc1(request):
    return _read_csv(request.config.getoption("--input-csv-pmc1"))


@pytest.fixture
def input_csv_pmc2(request):
    return _read_csv(request.config.getoption("--input-csv-pmc2"))


@pytest.fixture
def input_csv_pmc3(request):
    return _read_csv(request.config.getoption("--input-csv-pmc3"))


@pytest.fixture
def input_csv_pmc4(request):
    return _read_csv(request.config.getoption("--input-csv-pmc4"))


@pytest.fixture
def input_csv_pmc5(request):
    return _read_csv(request.config.getoption("--input-csv-pmc5"))


@pytest.fixture
def input_json_pmc1(request):
    return _read_json(request.config.getoption("--input-json-pmc1"))


@pytest.fixture
def input_json_pmc2(request):
    return _read_json(request.config.getoption("--input-json-pmc2"))


@pytest.fixture
def input_json_pmc3(request):
    return _read_json(request.config.getoption("--input-json-pmc3"))


@pytest.fixture
def input_json_pmc4(request):
    return _read_json(request.config.getoption("--input-json-pmc4"))


@pytest.fixture
def input_json_pmc5(request):
    return _read_json(request.config.getoption("--input-json-pmc5"))
