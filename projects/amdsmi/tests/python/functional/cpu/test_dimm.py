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

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuDimm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)

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
