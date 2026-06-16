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
"""CPU DIMM: power consumption, temperature range and refresh rate, thermal sensor."""

import os
import sys
import unittest

# Allow direct execution: walk up to find the 'common/' package root
import os as _os

_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, "common")):
    _d = _os.path.dirname(_d)
sys.path.insert(0, _d)
del _d, _os

import common.helpers as common

# common.helpers owns path resolution, sys.path setup, and amdsmi loading — borrow the
# reference so AMDSMI_PATH/ROCM_HOME/ROCM_PATH resolution and the stale-package check
# (see ROCM-1552 / PR #6359) are not duplicated or bypassed here.
from common.helpers import amdsmi

verbose = common.VERBOSITY_NORMAL
if "-q" in sys.argv or "--quiet" in sys.argv:
    verbose = common.VERBOSITY_QUIET
elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
    verbose = common.VERBOSITY_VERBOSE


class TestCpuDimm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_dimm_power_consumption(self):
        self.common.print_func_name("")

        # TODO Find better way to get dimm_addr
        dimm_addr = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_dimm_power_consumption=amdsmi.amdsmi_get_cpu_dimm_power_consumption,
            dimm_addr=dimm_addr,
        )
        return

    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_dimm_temp_range_and_refresh_rate as it fails."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO Find better way to get dimm_addr
        dimm_addr = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_dimm_temp_range_and_refresh_rate=amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate,
            dimm_addr=dimm_addr,
        )
        return

    def test_get_cpu_dimm_thermal_sensor(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_dimm_thermal_sensor as it fails."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO Find better way to get dimm_addr
        dimm_addr = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_dimm_thermal_sensor=amdsmi.amdsmi_get_cpu_dimm_thermal_sensor,
            dimm_addr=dimm_addr,
        )
        return


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Warning: Some tests may require elevated privileges (sudo/root).\n", file=sys.stderr)
    verbose = common.VERBOSITY_NORMAL
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbose = common.VERBOSITY_QUIET
    elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
        verbose = common.VERBOSITY_VERBOSE
    if not ("-k" in sys.argv or "--keyword" in sys.argv):
        if verbose > common.VERBOSITY_QUIET:
            common.print_tests(__name__)
    if "-h" in sys.argv or "--help" in sys.argv:
        unittest.main()
    if verbose < common.VERBOSITY_VERBOSE:
        common.print_legend()
    runner = unittest.TextTestRunner(
        stream=sys.stderr, verbosity=common.make_runner_verbosity(verbose)
    )
    common.expand_glob_k_arg(globals())
    unittest.main(testRunner=runner)
