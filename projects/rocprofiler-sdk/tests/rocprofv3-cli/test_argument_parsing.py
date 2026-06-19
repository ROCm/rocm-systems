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

"""Unit tests for the rocprofv3 command-line argument parser.

These tests load the (CMake-templated) ``rocprofv3.py`` script directly and
exercise its pure-Python argument parsing. They require no GPU or ROCm runtime.
"""

import importlib.util
import os

import pytest

_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "source",
        "bin",
        "rocprofv3.py",
    )
)


def load_rocprofv3():
    spec = importlib.util.spec_from_file_location("rocprofv3_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_boolean_flag_value_exits_with_argparse_error(capsys):
    """A non-boolean value for a boolean flag (e.g. ``-s ./a.out``) should
    produce a clean argparse usage error and exit code 2, not an uncaught
    ValueError traceback."""
    rocprofv3 = load_rocprofv3()

    with pytest.raises(SystemExit) as excinfo:
        rocprofv3.parse_arguments(["-s", "./a.out"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "invalid" in captured.err.lower()
    assert "./a.out" in captured.err


def test_valid_boolean_flag_values_parse(capsys):
    """Explicit truthy/falsy values for a boolean flag still parse correctly."""
    rocprofv3 = load_rocprofv3()

    truthy, _ = rocprofv3.parse_arguments(["-s", "true"])
    assert truthy.sys_trace is True

    falsy, _ = rocprofv3.parse_arguments(["-s", "false"])
    assert falsy.sys_trace is False
