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

"""Hardware-free unit tests for the `amd-smi metric -X/--partition` flag.

These tests build the CLI argument parser directly and assert that the
partition flag is wired up correctly. The driver-state helpers are patched so
the test runs without a GPU; only the parser wiring is exercised, no library
calls are made.
"""

import os
import sys
import unittest
from unittest.mock import patch


def _resolve_amdsmi_path():
    return os.environ.get("AMDSMI_PATH") or os.path.join(
        os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm", "share/amd_smi"
    )


def _resolve_cli_source_dir():
    # The CLI sources live in the project tree, not the installed share dir.
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "amdsmi_cli"))
    return candidate if os.path.isdir(candidate) else None


class TestAmdSmiPartitionFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli_dir = _resolve_cli_source_dir()
        if cli_dir is None:
            raise unittest.SkipTest("amdsmi_cli source directory not found")
        amdsmi_path = _resolve_amdsmi_path()
        if not os.path.isdir(amdsmi_path):
            raise unittest.SkipTest(f"amdsmi path '{amdsmi_path}' does not exist")

        sys.path.insert(0, cli_dir)
        sys.path.insert(0, amdsmi_path)
        try:
            from amdsmi_helpers import AMDSMIHelpers
            from amdsmi_parser import AMDSMIParser
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import the CLI parser: {e}")

        cls.AMDSMIHelpers = AMDSMIHelpers
        cls.AMDSMIParser = AMDSMIParser

    def _build_metric_parser(self):
        """Build a parser with the metric subcommand on a patched baremetal Linux
        host so the partition flag is registered without a real GPU."""

        def noop(args=None):
            return None

        callbacks = [noop] * 20
        with (
            patch.object(self.AMDSMIHelpers, "is_amdgpu_initialized", return_value=True),
            patch.object(self.AMDSMIHelpers, "is_baremetal", return_value=True),
            patch.object(self.AMDSMIHelpers, "is_linux", return_value=True),
            patch.object(self.AMDSMIHelpers, "is_hypervisor", return_value=False),
            patch.object(self.AMDSMIHelpers, "is_windows", return_value=False),
            patch.object(self.AMDSMIHelpers, "get_gpu_choices", return_value=({}, "")),
        ):
            helpers = self.AMDSMIHelpers()
            return self.AMDSMIParser(*callbacks, sys_argv=["amd-smi", "metric"], helpers=helpers)

    def test_partition_short_flag(self):
        parser = self._build_metric_parser()
        args = parser.parse_args(["metric", "-X"])
        self.assertTrue(args.partition)

    def test_partition_long_flag(self):
        parser = self._build_metric_parser()
        args = parser.parse_args(["metric", "--partition"])
        self.assertTrue(args.partition)

    def test_partition_defaults_false(self):
        parser = self._build_metric_parser()
        args = parser.parse_args(["metric", "-m"])
        self.assertFalse(args.partition)


if __name__ == "__main__":
    unittest.main()
