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
