#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU energy APIs."""

import unittest

import common.api_test as api


class TestCpuEnergy(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_core_energy(self):
        self.both("amdsmi_get_cpu_core_energy", self.handle)

    def test_get_cpu_socket_energy(self):
        self.both("amdsmi_get_cpu_socket_energy", self.handle)

    def test_get_cpu_core_ccd_power(self):
        self.both("amdsmi_get_cpu_core_ccd_power", self.handle)


if __name__ == "__main__":
    unittest.main()
