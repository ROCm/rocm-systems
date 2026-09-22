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

import pandas as pd
import pytest
import json
import os
import io

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader
from rocprofiler_sdk.pytest_utils.otf2_reader import OTF2Reader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        help="Path to Perfetto trace file.",
    )
    parser.addoption(
        "--otf2-input",
        action="store",
        help="Path to OTF2 trace file.",
    )
    parser.addoption(
        "--summary-input",
        action="store",
        help="Path to summary markdown file.",
    )

    pd.set_option("display.width", 2000)
    # increase debug display of pandas dataframes
    for itr in ["rows", "columns", "colwidth"]:
        pd.set_option(f"display.max_{itr}", None)


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def pftrace_data(request):
    filename = request.config.getoption("--pftrace-input")
    return PerfettoReader(filename).read()[0]


@pytest.fixture
def otf2_data(request):
    filename = request.config.getoption("--otf2-input")
    if not os.path.exists(filename):
        raise FileExistsError(f"{filename} does not exist")
    return OTF2Reader(filename).read()[0]


@pytest.fixture
def summary_data(request):
    filename = request.config.getoption("--summary-input")
    if not os.path.exists(filename):
        raise FileExistsError(f"{filename} does not exist")

    domains = {}
    with open(filename, "r") as inp:
        lines = [itr.strip() for itr in inp.readlines()]
        lines = [itr for itr in lines if itr and not itr.startswith("|--")]

    def rework(x):
        tmp = [itr.strip(" ") for itr in x.split("|")]
        tmp = [itr.strip() for itr in tmp if len(itr.strip()) > 0]
        if tmp[0] == "NAME":
            tmp = [f'"{itr}"' for itr in tmp]
        else:
            tmp[0] = f'"{tmp[0]}"'
            tmp[1] = f'"{tmp[1]}"'
        return ",".join(tmp)

    def process_current_domain(_name, _list):
        if _name and _list:
            _list = [rework(itr) for itr in _list]
            _contents = "{}\n".format("\n".join(_list))
            ifs = io.StringIO(_contents)
            df = pd.read_csv(ifs)
            domains[_name] = df
            _name = None
            _list = []
        return (None, [])

    current_name = None
    current_list = []

    for itr in lines:
        if not itr.startswith("|") or itr.startswith("ROCPROFV3 "):
            current_name, current_list = process_current_domain(
                current_name, current_list
            )
            current_name = itr.strip().strip(":").replace("ROCPROFV3 ", "", 1)
            rpos = current_name.rfind(" SUMMARY")
            if rpos >= 0:
                current_name = current_name[:rpos]
        else:
            current_list += [itr]

    process_current_domain(current_name, current_list)

    return domains
