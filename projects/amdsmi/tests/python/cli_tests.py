#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
CLI unit test runner — discovers and runs all tests under cli/.

Usage (installed):
    /opt/rocm/share/amd_smi/tests/python_unittest/cli_tests.py -v
    /opt/rocm/share/amd_smi/tests/python_unittest/cli_tests.py -b -v
    /opt/rocm/share/amd_smi/tests/python_unittest/cli_tests.py -k "gpu" -v

Usage (source):
    tests/python/cli_tests.py -v

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
