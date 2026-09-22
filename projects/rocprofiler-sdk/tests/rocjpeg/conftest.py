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

import json
import os
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="rocjpeg-tracing-test.json",
        help="Input JSON",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    if not os.path.isfile(filename):
        return pytest.skip("rocjpeg tracing unavailable")
    with open(filename, "r") as inp:
        return dotdict(json.load(inp))
