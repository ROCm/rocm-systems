#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
import re


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="kernel-tracing-test.json",
        help="Input JSON",
    )
    parser.addoption(
        "--signal-less-log",
        action="store",
        default=None,
        help="stderr of a run that requested KFD dispatch-log signal-less completion",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return json.load(inp)


@pytest.fixture
def signal_less_summary(request):
    """Counters from the signal-less summary line, or None when signal-less completion
    was not requested"""
    filename = request.config.getoption("--signal-less-log")
    if filename is None:
        return None
    with open(filename, "r") as inp:
        log = inp.read()
    found = re.search(r"KFD dispatch-log signal-less summary: ([^;]*);", log)
    assert found is not None, f"no signal-less summary in {filename}:\n{log}"
    return {
        key: int(val) for key, val in (itr.split("=") for itr in found.group(1).split())
    }
