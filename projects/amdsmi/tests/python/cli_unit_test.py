#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
CLI unit test runner — discovers and runs all tests under cli/.

Usage (installed):
    /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -v
    /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -b -v
    /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -k "gpu" -v

Usage (source):
    tests/python/cli_unit_test.py -v

Options:
    -v / --verbose    Verbose output (show per-test names)
    -q / --quiet      Quiet output
    -b / --buffer     Buffer stdout/stderr during tests
    -k "pattern"      Only run tests matching the substring
    --list / -l       List all available tests without running them
"""

import os
import sys

# Resolve the package root (the directory containing common/) regardless of
# where this script is installed or invoked from.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import common.common as common  # noqa: E402  (sys.path bootstrapped above)

common.run_test_dir("cli", "AMD SMI CLI Tests", _here)
