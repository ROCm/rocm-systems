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
Integration (functional) test runner — discovers and runs all tests under functional/.
These tests require live hardware and may require elevated privileges (sudo/root).

Usage (installed):
    /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v
    /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -b -v
    /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "power" -v

Usage (source):
    tests/python/integration_test.py -v

Options:
    -v / --verbose    Verbose output (show per-test names)
    -q / --quiet      Quiet output
    -b / --buffer     Buffer stdout/stderr during tests
    -k "pattern"      Only run tests matching the substring
    --list / -l       List all available tests without running them
"""

import os
import sys
import unittest

# Resolve the package root (the directory containing common/) regardless of
# where this script is installed or invoked from.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import common.helpers as common

verbose = common.verbose

if "-h" in sys.argv or "--help" in sys.argv:
    common.print_unittest_help()
    common.print_amdsmi_path_help()
    sys.exit(0)

loader = unittest.TestLoader()

# Support -k pattern filtering via unittest's built-in keyword matching
_k_pattern = None
for i, arg in enumerate(sys.argv):
    if arg == "-k" and i + 1 < len(sys.argv):
        _k_pattern = sys.argv[i + 1]
        break
    if arg.startswith("-k"):
        _k_pattern = arg[2:]
        break

if _k_pattern:
    loader.testNamePatterns = [f"*{_k_pattern}*"]

functional_dir = os.path.join(_here, "functional")
suite = loader.discover(start_dir=functional_dir, pattern="test_*.py", top_level_dir=_here)

if "--list" in sys.argv or "-l" in sys.argv:
    common.print_test_ids(suite)
    sys.exit(0)

if verbose < common.VERBOSITY_VERBOSE:
    common.print_legend()

if verbose > common.VERBOSITY_QUIET:
    print("AMD SMI Integration Tests\n")
    print("Running tests...\n")

runner = common.GTestSummaryRunner(
    stream=sys.stderr,
    verbosity=common.make_runner_verbosity(verbose),
    buffer="-b" in sys.argv or "--buffer" in sys.argv,
)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
