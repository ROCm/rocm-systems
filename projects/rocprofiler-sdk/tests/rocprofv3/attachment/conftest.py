#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import csv
import json
import pytest


def pytest_addoption(parser):
    parser.addoption("--kernel-input", action="store", help="Kernel trace input")
    parser.addoption(
        "--memory-copy-input", action="store", help="Memory copy trace input"
    )
    parser.addoption("--hsa-input", action="store", help="HSA API trace input")
    parser.addoption("--agent-input", action="store", help="Agent info input")


def get_csv_data(request, field):
    inp_data = request.config.getoption(field)
    if not inp_data:
        return []
    with open(inp_data, "r") as inp:
        csv_reader = csv.DictReader(inp)
        return [row for row in csv_reader]


@pytest.fixture
def kernel_input_data(request):
    return get_csv_data(request, "--kernel-input")


@pytest.fixture
def memory_copy_input_data(request):
    return get_csv_data(request, "--memory-copy-input")


@pytest.fixture
def hsa_input_data(request):
    return get_csv_data(request, "--hsa-input")


@pytest.fixture
def agent_info_input_data(request):
    return get_csv_data(request, "--agent-input")
