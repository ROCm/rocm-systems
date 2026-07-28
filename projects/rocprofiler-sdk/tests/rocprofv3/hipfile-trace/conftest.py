#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import json
import os
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.rocpd_reader import RocpdReader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        default="hipfile-trace-client-trace/out_results.json",
        help="Input JSON",
    )
    parser.addoption(
        "--rocpd-input",
        action="store",
        default="hipfile-trace-client-trace/out_results.db",
        help="Input rocpd SQLite3 database",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    if not os.path.isfile(filename):
        return pytest.skip("hipFILE tracing unavailable")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def rocpd_data(request):
    filename = request.config.getoption("--rocpd-input")
    if not os.path.isfile(filename):
        return pytest.skip("hipFILE tracing unavailable")
    return RocpdReader(filename).read()[0]
