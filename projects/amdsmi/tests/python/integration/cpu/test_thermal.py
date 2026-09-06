#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU thermal: socket temperature, PROCHOT status, C0 residency."""

import unittest

import common.api_test as api


class TestCpuThermal(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_socket_temperature(self):
        self.both("amdsmi_get_cpu_socket_temperature", self.handle)

    def test_get_cpu_prochot_status(self):
        self.both("amdsmi_get_cpu_prochot_status", self.handle)

    def test_get_cpu_tdelta(self):
        self.both("amdsmi_get_cpu_tdelta", self.handle)

    def test_get_cpu_svi3_vr_controller_temp(self):
        self.both(
            "amdsmi_get_cpu_svi3_vr_controller_temp",
            self.handle,
            api.integer("rail_selection", 0),
            api.integer("rail_index", 0),
        )


if __name__ == "__main__":
    unittest.main()
