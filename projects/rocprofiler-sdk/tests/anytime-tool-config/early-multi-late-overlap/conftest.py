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


def pytest_addoption(parser):
    parser.addoption(
        "--tool-a-input",
        action="store",
        default="early-multi-late-overlap-tool-a.json",
        help="JSON output from the early tool A (HIP,HSA,dispatch,memcpy,marker)",
    )
    parser.addoption(
        "--tool-b-input",
        action="store",
        default="early-multi-late-overlap-tool-b.json",
        help="JSON output from the late tool B (HSA,dispatch,memcpy,marker,alloc)",
    )
    parser.addoption(
        "--phase2-dispatches",
        action="store",
        type=int,
        default=20,
        help="Kernel dispatches issued after Tool B registered",
    )
    parser.addoption(
        "--phase2-copies",
        action="store",
        type=int,
        default=40,
        help="Memory copies issued after Tool B registered",
    )
    parser.addoption(
        "--phase2-allocates",
        action="store",
        type=int,
        default=5,
        help="Explicit hipMalloc allocations issued after Tool B registered",
    )


def _load(path):
    with open(path, "r") as inp:
        return json.load(inp)


@pytest.fixture
def tool_a_data(request):
    return _load(request.config.getoption("--tool-a-input"))


@pytest.fixture
def tool_b_data(request):
    return _load(request.config.getoption("--tool-b-input"))


@pytest.fixture
def phase2_dispatches(request):
    return request.config.getoption("--phase2-dispatches")


@pytest.fixture
def phase2_copies(request):
    return request.config.getoption("--phase2-copies")


@pytest.fixture
def phase2_allocates(request):
    return request.config.getoption("--phase2-allocates")
