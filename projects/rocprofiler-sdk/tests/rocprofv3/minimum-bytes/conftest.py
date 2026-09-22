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
import csv

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):

    parser.addoption(
        "--trace-input-csv",
        action="store",
        help="Path to kernel trace CSV file.",
    )

    parser.addoption(
        "--trace-input-json",
        action="store",
        help="Path to kernel trace JSON file.",
    )

    parser.addoption(
        "--trace-input-pftrace",
        action="store",
        help="Path to kernel trace perfetto file.",
    )


@pytest.fixture
def trace_data_csv(request):
    filename = request.config.getoption("--trace-input-csv")
    return filename


@pytest.fixture
def trace_data_json(request):
    filename = request.config.getoption("--trace-input-json")
    return filename


@pytest.fixture
def trace_data_pftrace(request):
    filename = request.config.getoption("--trace-input-pftrace")
    return filename
