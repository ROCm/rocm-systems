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
"""GPU metrics and activity APIs."""

import unittest

import common.common as common


class TestGpuMetrics(common.ApiTestCase):
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
        counters = common.Param(
            "counter_types",
            (
                "COARSE_GRAIN_GFX_ACTIVITY",
                [common.amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_GFX_ACTIVITY],
            ),
            [("bad-type", common.BAD_SEQUENCE), ("empty", [])],
        )
        self.both("amdsmi_get_utilization_count", self.handle, counters)


if __name__ == "__main__":
    unittest.main()
