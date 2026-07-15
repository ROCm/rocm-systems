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

import os
import json

import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption("--input-csv", action="store", default=None)
    parser.addoption("--input-json", action="store", default=None)
    parser.addoption("--agent-info-csv", action="store", default=None)
    parser.addoption("--skip-if", action="store", default=None)


def _check_skip(request):
    skip_path = request.config.getoption("--skip-if")
    if skip_path and os.path.exists(skip_path):
        pytest.skip("attach tests unavailable (insufficient ptrace permissions)")


@pytest.fixture
def input_csv(request):
    _check_skip(request)
    fname = request.config.getoption("--input-csv")
    if not fname or not os.path.isfile(fname):
        pytest.skip("PC sampling unavailable")
    return pd.read_csv(
        fname, na_filter=False, keep_default_na=False, dtype={"Exec_Mask": "uint64"}
    )


@pytest.fixture
def input_json(request):
    _check_skip(request)
    fname = request.config.getoption("--input-json")
    if not fname or not os.path.isfile(fname):
        pytest.skip("PC sampling unavailable")
    with open(fname, "r") as inp:
        return json.load(inp)


@pytest.fixture
def wave_size(request):
    fname = request.config.getoption("--agent-info-csv")
    if not fname or not os.path.isfile(fname):
        pytest.skip("agent info unavailable")
    df = pd.read_csv(fname)
    gpu = df[df["Wave_Front_Size"].astype(int) > 0]
    assert not gpu.empty, "no GPU agent with a wave front size in agent info"
    return int(gpu["Wave_Front_Size"].astype(int).iloc[0])
