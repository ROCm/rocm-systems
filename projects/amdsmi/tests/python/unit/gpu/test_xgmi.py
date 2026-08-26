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
"""GPU XGMI and fabric APIs."""

import unittest

import common.common as common


class TestGpuXgmi(common.ApiTestCase):
    def test_get_xgmi_info(self):
        self.both("amdsmi_get_xgmi_info", self.handle)

    def test_get_gpu_xgmi_link_status(self):
        self.both("amdsmi_get_gpu_xgmi_link_status", self.handle)

    def test_gpu_xgmi_error_status(self):
        self.both("amdsmi_gpu_xgmi_error_status", self.handle)

    def test_get_xgmi_plpd(self):
        self.both("amdsmi_get_xgmi_plpd", self.handle)

    def test_get_gpu_fabric_info(self):
        self.both("amdsmi_get_gpu_fabric_info", self.handle)

    def test_get_fabric_telemetry_data(self):
        self.both(
            "amdsmi_get_fabric_telemetry_data", self.handle, common.integer("category_mask", 1)
        )

    def test_set_xgmi_plpd(self):
        self.reject_only("amdsmi_set_xgmi_plpd", self.handle, common.integer("policy_id", 0))

    def test_reset_gpu_xgmi_error(self):
        self.reject_only("amdsmi_reset_gpu_xgmi_error", self.handle)

    def test_alloc_fabric_telemetry(self):
        self.reject_only(
            "amdsmi_alloc_fabric_telemetry", self.handle, common.integer("category_mask", 0)
        )

    def test_free_fabric_telemetry(self):
        self.reject_only("amdsmi_free_fabric_telemetry", self.handle, common.opaque("telemetry"))

    def test_fabric_telem_id_to_string(self):
        self.reject_only("amdsmi_fabric_telem_id_to_string", common.integer("telem_id", 0))


if __name__ == "__main__":
    unittest.main()
