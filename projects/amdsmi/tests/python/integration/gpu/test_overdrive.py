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
"""GPU overdrive, performance level and voltage APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuOverdrive(api.ApiTestCase):
    def test_get_gpu_overdrive_level(self):
        self.both("amdsmi_get_gpu_overdrive_level", self.handle)

    def test_get_gpu_mem_overdrive_level(self):
        self.both("amdsmi_get_gpu_mem_overdrive_level", self.handle)

    def test_get_gpu_od_volt_info(self):
        self.both("amdsmi_get_gpu_od_volt_info", self.handle)

    def test_get_gpu_od_volt_curve_regions(self):
        self.both(
            "amdsmi_get_gpu_od_volt_curve_regions", self.handle, api.integer("num_regions", 1)
        )

    def test_get_gpu_perf_level(self):
        self.both("amdsmi_get_gpu_perf_level", self.handle)

    def test_get_gpu_volt_metric(self):
        self.both(
            "amdsmi_get_gpu_volt_metric",
            self.handle,
            api.enum("sensor_type", common.VOLTAGE_TYPES),
            api.enum("metric", common.VOLTAGE_METRICS),
        )

    def test_get_gpu_reg_table_info(self):
        self.both(
            "amdsmi_get_gpu_reg_table_info", self.handle, api.enum("reg_type", common.REG_TYPES)
        )

    def test_set_gpu_overdrive_level(self):
        self.reject_only(
            "amdsmi_set_gpu_overdrive_level", self.handle, api.integer("overdrive_value", 0)
        )

    def test_set_gpu_od_volt_info(self):
        self.reject_only(
            "amdsmi_set_gpu_od_volt_info",
            self.handle,
            api.integer("vpoint", 0),
            api.integer("clk_value", 0),
            api.integer("volt_value", 0),
        )

    def test_set_gpu_od_clk_info(self):
        self.reject_only(
            "amdsmi_set_gpu_od_clk_info",
            self.handle,
            api.enum("level", common.FREQ_INDS),
            api.integer("value", 0),
            api.enum("clk_type", common.CLK_TYPES),
        )

    def test_set_gpu_perf_level(self):
        self.reject_only(
            "amdsmi_set_gpu_perf_level", self.handle, api.enum("perf_level", common.DEV_PERF_LEVELS)
        )

    def test_set_gpu_perf_determinism_mode(self):
        self.reject_only(
            "amdsmi_set_gpu_perf_determinism_mode", self.handle, api.integer("clkvalue", 0)
        )


if __name__ == "__main__":
    unittest.main()
