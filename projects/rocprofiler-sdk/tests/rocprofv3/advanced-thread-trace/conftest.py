#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import csv
import pytest
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader

import re
import os


def pytest_addoption(parser):

    parser.addoption(
        "--input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--code-object-input",
        action="store",
        help="Path to code object file.",
    )
    parser.addoption(
        "--output-path",
        action="store",
        help="Output Path.",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def output_path(request):
    return request.config.getoption("--output-path")


@pytest.fixture
def code_object_file_path(request):
    file_path = request.config.getoption("--code-object-input")
    # hsa_file_load = re.compile(".*copy.hsaco$")
    code_object_files = {}
    code_object_memory = []
    hsa_memory_load_pattern = "gfx[a-z0-9]+_copy_memory.hsaco"
    for root, dirs, files in os.walk(file_path, topdown=True):
        for file in files:
            filename = os.path.join(root, file)
            if re.search(hsa_memory_load_pattern, filename):
                code_object_memory.append(filename)
    code_object_files["hsa_memory_load"] = code_object_memory
    return code_object_files
