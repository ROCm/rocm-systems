#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import importlib.util
import os

from importlib.machinery import SourceFileLoader

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--rocprofv3-path",
        action="store",
        default="",
        help="Path to the rocprofv3 launcher script to import and unit-test.",
    )


@pytest.fixture(scope="session")
def rocprofv3(request):
    """Import the rocprofv3 launcher as a module (no GPU required)."""
    path = request.config.getoption("--rocprofv3-path")
    assert path, "--rocprofv3-path must point at rocprofv3"
    assert os.path.isfile(path), f"rocprofv3 script not found: {path}"

    # installed launcher is named "rocprofv3" (no .py), so load via explicit loader
    loader = SourceFileLoader("rocprofv3_under_test", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
