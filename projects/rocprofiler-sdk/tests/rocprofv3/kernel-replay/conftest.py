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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        default=None,
        help="rocprofv3 results JSON produced by a --kernel-replay-beta-enabled run",
    )
    parser.addoption(
        "--passes",
        action="store",
        type=int,
        default=None,
        help="expected number of replay passes per dispatch (number of --pmc groups)",
    )
    parser.addoption(
        "--common-counters",
        action="store",
        nargs="+",
        default=["SQ_WAVES", "SQ_INSTS_VALU"],
        help="counters shared by every --pmc group; must be constant across a kernel's passes",
    )
    parser.addoption(
        "--pass-groups",
        action="store",
        nargs="+",
        default=None,
        help="counter unique to each --pmc group, in group order; pass i must collect entry i",
    )


@pytest.fixture
def json_data(request):
    path = request.config.getoption("--json-input")
    assert path, "--json-input is required by this test"
    assert os.path.isfile(path), f"missing JSON input: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def expected_passes(request):
    passes = request.config.getoption("--passes")
    assert passes is not None, "--passes is required by this test"
    return passes


@pytest.fixture
def common_counters(request):
    return list(request.config.getoption("--common-counters"))


@pytest.fixture
def pass_groups(request):
    return list(request.config.getoption("--pass-groups") or [])
