#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU power APIs: bad arguments are rejected, good reads are valid."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuPower(api.ApiTestCase):
    def test_get_power_info(self):
        self.both("amdsmi_get_power_info", self.handle)

    def test_get_power_cap_info(self):
        self.both(
            "amdsmi_get_power_cap_info", self.handle, api.integer("sensor_ind", 0, bounds=True)
        )

    def test_get_supported_power_cap(self):
        self.both("amdsmi_get_supported_power_cap", self.handle)

    def test_get_gpu_power_profile_presets(self):
        self.both("amdsmi_get_gpu_power_profile_presets", self.handle, api.integer("sensor_idx", 0))

    def test_is_gpu_power_management_enabled(self):
        self.both("amdsmi_is_gpu_power_management_enabled", self.handle)

    def test_get_energy_count(self):
        self.both("amdsmi_get_energy_count", self.handle)

    def test_set_power_cap(self):
        self.reject_only(
            "amdsmi_set_power_cap", self.handle, api.integer("sensor_ind", 0), api.integer("cap", 0)
        )

    def test_set_gpu_power_profile(self):
        self.reject_only(
            "amdsmi_set_gpu_power_profile",
            self.handle,
            api.integer("reserved", 0),
            api.enum("profile", common.POWER_PROFILE_PRESET_MASKS),
        )


if __name__ == "__main__":
    unittest.main()
