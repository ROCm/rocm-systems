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

import common.common as common


class TestGpuOverdrive(common.ApiTestCase):
    def test_get_gpu_overdrive_level(self):
        self.both("amdsmi_get_gpu_overdrive_level", self.handle)

    def test_get_gpu_mem_overdrive_level(self):
        self.both("amdsmi_get_gpu_mem_overdrive_level", self.handle)

    def test_get_gpu_od_volt_info(self):
        self.both("amdsmi_get_gpu_od_volt_info", self.handle)

    def test_get_gpu_od_volt_curve_regions(self):
        self.both(
            "amdsmi_get_gpu_od_volt_curve_regions", self.handle, common.integer("num_regions", 1)
        )

    def test_get_gpu_perf_level(self):
        self.both("amdsmi_get_gpu_perf_level", self.handle)

    def test_get_gpu_volt_metric(self):
        self.both(
            "amdsmi_get_gpu_volt_metric",
            self.handle,
            common.enum("sensor_type", common.VOLTAGE_TYPES),
            common.enum("metric", common.VOLTAGE_METRICS),
        )

    def test_get_gpu_reg_table_info(self):
        self.both(
            "amdsmi_get_gpu_reg_table_info", self.handle, common.enum("reg_type", common.REG_TYPES)
        )

    def test_set_gpu_overdrive_level(self):
        self.reject_only(
            "amdsmi_set_gpu_overdrive_level", self.handle, common.integer("overdrive_value", 0)
        )

    def test_set_gpu_od_volt_info(self):
        self.reject_only(
            "amdsmi_set_gpu_od_volt_info",
            self.handle,
            common.integer("vpoint", 0),
            common.integer("clk_value", 0),
            common.integer("volt_value", 0),
        )

    def test_set_gpu_od_clk_info(self):
        self.reject_only(
            "amdsmi_set_gpu_od_clk_info",
            self.handle,
            common.enum("level", common.FREQ_INDS),
            common.integer("value", 0),
            common.enum("clk_type", common.CLK_TYPES),
        )

    def test_set_gpu_perf_level(self):
        self.reject_only(
            "amdsmi_set_gpu_perf_level",
            self.handle,
            common.enum("perf_level", common.DEV_PERF_LEVELS),
        )

    def test_set_gpu_perf_determinism_mode(self):
        self.reject_only(
            "amdsmi_set_gpu_perf_determinism_mode", self.handle, common.integer("clkvalue", 0)
        )


if __name__ == "__main__":
    unittest.main()
