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
        "--csv-directory-input",
        action="store",
        help="Path to stats csv file.",
    )


@pytest.fixture
def csv_data(request):
    import glob

    directory = request.config.getoption("--csv-directory-input")

    data = []
    if not os.path.isdir(directory):
        raise FileExistsError(f"{directory} does not exist")

    stats_glob = os.path.join(directory, "stats*.csv")
    for filename in glob.glob(stats_glob):
        file_data = []
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                file_data.append(row)
        data.append(file_data)
    return data
