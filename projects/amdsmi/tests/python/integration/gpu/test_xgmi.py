#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU XGMI and fabric APIs."""

import unittest

import common.api_test as api


class TestGpuXgmi(api.ApiTestCase):
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
        self.both("amdsmi_get_fabric_telemetry_data", self.handle, api.integer("category_mask", 1))

    def test_set_xgmi_plpd(self):
        self.reject_only("amdsmi_set_xgmi_plpd", self.handle, api.integer("policy_id", 0))

    def test_reset_gpu_xgmi_error(self):
        self.reject_only("amdsmi_reset_gpu_xgmi_error", self.handle)

    def test_alloc_fabric_telemetry(self):
        self.reject_only(
            "amdsmi_alloc_fabric_telemetry", self.handle, api.integer("category_mask", 0)
        )

    def test_free_fabric_telemetry(self):
        self.reject_only("amdsmi_free_fabric_telemetry", self.handle, api.opaque("telemetry"))

    def test_fabric_telem_id_to_string(self):
        self.reject_only("amdsmi_fabric_telem_id_to_string", api.integer("telem_id", 0))


if __name__ == "__main__":
    unittest.main()
