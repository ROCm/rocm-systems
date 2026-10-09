#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.rocpd_reader import RocpdReader
from rocprofiler_sdk.pytest_utils.otf2_reader import OTF2Reader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        help="Path to JSON results file.",
    )
    parser.addoption(
        "--rocpd-input",
        action="store",
        help="Path to rocpd SQLite3 database file.",
    )

    parser.addoption(
        "--otf2-input",
        action="store",
        default="hip-event-trace/hip-events-trace/out_results.otf2",
        help="Input OTF2",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    if filename is None:
        pytest.fail("--json-input argument is required but was not provided")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def rocpd_data(request):
    filename = request.config.getoption("--rocpd-input")
    if filename is None:
        pytest.fail("--rocpd-input argument is required but was not provided")
    return RocpdReader(filename).read()[0]


@pytest.fixture
def otf2_data(request):
    filename = request.config.getoption("--otf2-input")
    if not os.path.isfile(filename):
        return pytest.skip("hip event tracing unavailable")
    return OTF2Reader(filename).read()[0]
