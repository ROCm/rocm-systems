#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU DIMM APIs."""

import unittest

import common.api_test as api


class TestCpuDimm(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self.both(
            "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
            self.handle,
            api.integer("dimm_addr", 0),
        )

    def test_get_cpu_dimm_power_consumption(self):
        self.both("amdsmi_get_cpu_dimm_power_consumption", self.handle, api.integer("dimm_addr", 0))

    def test_get_cpu_dimm_thermal_sensor(self):
        self.both("amdsmi_get_cpu_dimm_thermal_sensor", self.handle, api.integer("dimm_addr", 0))

    def test_get_cpu_dimm_sb_reg(self):
        self.both(
            "amdsmi_get_cpu_dimm_sb_reg",
            self.handle,
            api.integer("dimm_addr", 0),
            api.integer("lid", 0),
            api.integer("reg_offset", 0),
            api.integer("reg_space", 0),
        )

    def test_set_cpu_dimm_sb_reg(self):
        self.reject_only(
            "amdsmi_set_cpu_dimm_sb_reg",
            self.handle,
            api.integer("dimm_addr", 0),
            api.integer("lid", 0),
            api.integer("reg_offset", 0),
            api.integer("reg_space", 0),
            api.integer("write_data", 0),
        )


if __name__ == "__main__":
    unittest.main()
