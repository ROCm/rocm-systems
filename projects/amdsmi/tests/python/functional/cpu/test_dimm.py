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
        self.common.amdsmi_smart_init()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_dimm_power_consumption(self):
        self.common.print_func_name("")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_dimm_power_consumption,
            "amdsmi_get_cpu_dimm_power_consumption",
            arg_sets=[(f"dimm_addr={dimm_addr:#x}", (dimm_addr,))],
        )
        return

    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self.common.print_func_name("")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0x87
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate,
            "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
            arg_sets=[(f"dimm_addr={dimm_addr:#x}", (dimm_addr,))],
        )
        return

    def test_get_cpu_dimm_thermal_sensor(self):
        self.common.print_func_name("")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0x87
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_dimm_thermal_sensor,
            "amdsmi_get_cpu_dimm_thermal_sensor",
            arg_sets=[(f"dimm_addr={dimm_addr:#x}", (dimm_addr,))],
        )
        return
