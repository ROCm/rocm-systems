#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU device identity APIs: bad arguments are rejected, good reads are valid."""

import unittest

import common.api_test as api


class TestGpuIdentity(api.ApiTestCase):
    def test_get_fw_info(self):
        self.assertTrue(
            self.both("amdsmi_get_fw_info", self.handle, require_success=True), "fw_info is empty"
        )

    def test_get_gpu_asic_info(self):
        self.both(
            "amdsmi_get_gpu_asic_info",
            self.handle,
            require_success=True,
            require_populated=("market_name", "vendor_id"),
        )

    def test_get_gpu_bdf_id(self):
        self.both("amdsmi_get_gpu_bdf_id", self.handle, require_success=True)

    def test_get_gpu_board_info(self):
        self.both("amdsmi_get_gpu_board_info", self.handle)

    def test_get_gpu_device_bdf(self):
        self.both(
            "amdsmi_get_gpu_device_bdf", self.handle, require_success=True, require_populated=True
        )

    def test_get_gpu_device_bdf_bdf(self):
        self.both("amdsmi_get_gpu_device_bdf_bdf", self.handle)

    def test_get_gpu_device_cuid(self):
        self.both("amdsmi_get_gpu_device_cuid", self.handle)

    def test_get_gpu_device_uuid(self):
        self.both(
            "amdsmi_get_gpu_device_uuid", self.handle, require_success=True, require_populated=True
        )

    def test_get_gpu_driver_info(self):
        self.both(
            "amdsmi_get_gpu_driver_info", self.handle, require_success=True, require_populated=True
        )

    def test_get_gpu_enumeration_info(self):
        self.both("amdsmi_get_gpu_enumeration_info", self.handle)

    def test_get_gpu_id(self):
        self.both("amdsmi_get_gpu_id", self.handle, require_success=True)

    def test_get_gpu_kfd_info(self):
        self.both("amdsmi_get_gpu_kfd_info", self.handle)

    def test_get_gpu_revision(self):
        self.both("amdsmi_get_gpu_revision", self.handle)

    def test_get_gpu_subsystem_id(self):
        self.both("amdsmi_get_gpu_subsystem_id", self.handle)

    def test_get_gpu_subsystem_name(self):
        self.both("amdsmi_get_gpu_subsystem_name", self.handle)

    def test_get_gpu_vbios_info(self):
        self.both("amdsmi_get_gpu_vbios_info", self.handle)

    def test_get_gpu_vendor_name(self):
        self.both("amdsmi_get_gpu_vendor_name", self.handle)

    def test_get_gpu_virtualization_mode(self):
        self.both("amdsmi_get_gpu_virtualization_mode", self.handle)

    def test_get_gpu_vram_info(self):
        self.both("amdsmi_get_gpu_vram_info", self.handle)

    def test_get_gpu_vram_vendor(self):
        self.both("amdsmi_get_gpu_vram_vendor", self.handle)

    def test_get_gpu_xcd_counter(self):
        self.both("amdsmi_get_gpu_xcd_counter", self.handle)


if __name__ == "__main__":
    unittest.main()
