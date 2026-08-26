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
"""GPU power APIs: bad arguments are rejected, good reads are valid."""

import unittest

import common.common as common


class TestGpuPower(common.ApiTestCase):
    def test_get_power_info(self):
        self.both("amdsmi_get_power_info", self.handle)

    def test_get_power_cap_info(self):
        self.both(
            "amdsmi_get_power_cap_info", self.handle, common.integer("sensor_ind", 0, bounds=True)
        )

    def test_get_supported_power_cap(self):
        self.both("amdsmi_get_supported_power_cap", self.handle)

    def test_get_gpu_power_profile_presets(self):
        self.both(
            "amdsmi_get_gpu_power_profile_presets", self.handle, common.integer("sensor_idx", 0)
        )

    def test_is_gpu_power_management_enabled(self):
        self.both("amdsmi_is_gpu_power_management_enabled", self.handle)

    def test_get_energy_count(self):
        self.both("amdsmi_get_energy_count", self.handle)

    def test_set_power_cap(self):
        self.reject_only(
            "amdsmi_set_power_cap",
            self.handle,
            common.integer("sensor_ind", 0),
            common.integer("cap", 0),
        )

    def test_set_gpu_power_profile(self):
        self.reject_only(
            "amdsmi_set_gpu_power_profile",
            self.handle,
            common.integer("reserved", 0),
            common.enum("profile", common.POWER_PROFILE_PRESET_MASKS),
        )


if __name__ == "__main__":
    unittest.main()
