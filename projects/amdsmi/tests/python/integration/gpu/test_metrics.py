#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU metrics and activity APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuMetrics(api.ApiTestCase):
    def test_get_gpu_activity(self):
        self.both("amdsmi_get_gpu_activity", self.handle)

    def test_get_gpu_busy_percent(self):
        self.both("amdsmi_get_gpu_busy_percent", self.handle)

    def test_get_vcn_busy_percent(self):
        self.both("amdsmi_get_vcn_busy_percent", self.handle)

    def test_get_gpu_cache_info(self):
        self.both("amdsmi_get_gpu_cache_info", self.handle)

    def test_get_gpu_metrics_header_info(self):
        self.both("amdsmi_get_gpu_metrics_header_info", self.handle)

    def test_get_gpu_metrics_info(self):
        self.both("amdsmi_get_gpu_metrics_info", self.handle)

    def test_get_gpu_partition_metrics_info(self):
        self.both("amdsmi_get_gpu_partition_metrics_info", self.handle)

    def test_get_gpu_pm_metrics_info(self):
        self.both("amdsmi_get_gpu_pm_metrics_info", self.handle)

    def test_get_utilization_count(self):
        counters = api.Param(
            "counter_types",
            (
                "COARSE_GRAIN_GFX_ACTIVITY",
                [common.amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_GFX_ACTIVITY],
            ),
            [("bad-type", api.BAD_SEQUENCE), ("empty", [])],
        )
        self.both("amdsmi_get_utilization_count", self.handle, counters)


if __name__ == "__main__":
    unittest.main()
