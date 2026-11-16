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

import glob
import json
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--output-root",
        required=True,
        help="directory holding the -d directories of --att-remap runs; the newest is checked",
    )
    parser.addoption(
        "--layout-script",
        default="",
        help="the remapper's att_merge/layout.py; its checks are skipped when empty",
    )


@pytest.fixture
def output_dir(request):
    root = request.config.getoption("--output-root")
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    assert runs, f"no --att-remap run under {root}"
    return max(runs, key=os.path.getmtime)


@pytest.fixture
def dispatch_dir(output_dir):
    dirs = glob.glob(os.path.join(output_dir, "att_remap", "*", "dispatch-*"))
    assert len(dirs) == 1, f"expected one remapped dispatch under {output_dir}: {dirs}"
    return dirs[0]


@pytest.fixture
def report(dispatch_dir):
    with open(os.path.join(dispatch_dir, "report.json")) as ifs:
        return json.load(ifs)


@pytest.fixture
def passes(dispatch_dir):
    with open(os.path.join(dispatch_dir, "passes.json")) as ifs:
        return json.load(ifs)


@pytest.fixture
def layout_script(request):
    path = request.config.getoption("--layout-script")
    if not path or not os.path.isfile(path):
        pytest.skip("the remapper's layout.py is not installed")
    return path
