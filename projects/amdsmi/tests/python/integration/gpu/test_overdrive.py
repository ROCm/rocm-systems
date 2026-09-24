#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
