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


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="hip-graph-forkjoin-deadlock-test.json",
        help="Input JSON",
    )
    parser.addoption(
        "--width",
        action="store",
        type=int,
        required=True,
        help="Branch kernels per layer",
    )
    parser.addoption(
        "--depth",
        action="store",
        type=int,
        required=True,
        help="Diamond layers per replay",
    )
    parser.addoption(
        "--replays", action="store", type=int, required=True, help="Graph launches"
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return json.load(inp)


@pytest.fixture
def expected_dispatches(request):
    """Kernel nodes the workload launches, which is the floor on dispatch records rather
    than the exact total. Per replay: one root node, then per layer `width` branch kernels
    and one join. The geometry is passed in by the test definition so it cannot drift from
    the arguments the workload actually ran with."""
    width = request.config.getoption("--width")
    depth = request.config.getoption("--depth")
    replays = request.config.getoption("--replays")
    return replays * (1 + depth * (width + 1))
