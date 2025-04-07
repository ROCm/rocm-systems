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
import re

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):

    parser.addoption(
        "--input-json-pass1",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-json-pass2",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-json-pass3",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--json-config",
        action="store",
        help="Path to input JSON file.",
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


def extract_iteration_list(jobs, pass_):

    kernel_iteration_range = jobs[pass_]["kernel_iteration_range"].strip()
    return tokenize(kernel_iteration_range)


def process_config(out_file, input_config, pass_):

    ret_dict = {}

    with open(out_file, "r") as inp:
        ret_dict["json_data"] = dotdict(collapse_dict_list(json.load(inp)))

    with open(input_config, "r") as inp:
        jobs = dotdict(collapse_dict_list(json.load(inp)))["jobs"]
        ret_dict["iteration_range"] = extract_iteration_list(jobs, pass_)

    return ret_dict


@pytest.fixture
def input_json_pass1(request):
    out_file = request.config.getoption("--input-json-pass1")
    input_config = request.config.getoption("--json-config")
    return process_config(out_file, input_config, 0)


@pytest.fixture
def input_json_pass2(request):
    out_file = request.config.getoption("--input-json-pass2")
    input_config = request.config.getoption("--json-config")
    return process_config(out_file, input_config, 1)


@pytest.fixture
def input_json_pass3(request):
    out_file = request.config.getoption("--input-json-pass3")
    input_config = request.config.getoption("--json-config")
    return process_config(out_file, input_config, 2)
