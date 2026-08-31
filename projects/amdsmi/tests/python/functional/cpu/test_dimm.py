#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
