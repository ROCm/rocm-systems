#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU power, boost limit and C-state APIs."""

import unittest

import common.api_test as api


class TestCpuPower(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_socket_power(self):
        self.both("amdsmi_get_cpu_socket_power", self.handle)

    def test_get_cpu_socket_power_cap(self):
        self.both("amdsmi_get_cpu_socket_power_cap", self.handle)

    def test_get_cpu_socket_power_cap_max(self):
        self.both("amdsmi_get_cpu_socket_power_cap_max", self.handle)

    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        self.both("amdsmi_get_cpu_pwr_svi_telemetry_all_rails", self.handle)

    def test_get_cpu_pwr_efficiency_mode(self):
        self.both("amdsmi_get_cpu_pwr_efficiency_mode", self.handle)

    def test_get_cpu_core_boostlimit(self):
        self.both("amdsmi_get_cpu_core_boostlimit", self.handle)

    def test_get_cpu_socket_c0_residency(self):
        self.both("amdsmi_get_cpu_socket_c0_residency", self.handle)

    def test_get_cpu_sdps_limit(self):
        self.both("amdsmi_get_cpu_sdps_limit", self.handle)

    def test_get_cpu_rail_isofreq_policy(self):
        self.both("amdsmi_get_cpu_rail_isofreq_policy", self.handle)

    def test_get_cpu_dfc_ctrl(self):
        self.both("amdsmi_get_cpu_dfc_ctrl", self.handle)

    def test_get_cpu_pc6_enable(self):
        self.both("amdsmi_get_cpu_pc6_enable", self.handle)

    def test_get_cpu_cc6_enable(self):
        self.both("amdsmi_get_cpu_cc6_enable", self.handle)

    def test_set_cpu_socket_power_cap(self):
        self.reject_only(
            "amdsmi_set_cpu_socket_power_cap", self.handle, api.integer("power_cap", 0)
        )

    def test_set_cpu_pwr_efficiency_mode(self):
        self.reject_only(
            "amdsmi_set_cpu_pwr_efficiency_mode",
            self.handle,
            api.integer("mode", 0),
            api.integer("util", 0),
            api.integer("ppt_limit", 0),
        )

    def test_set_cpu_core_boostlimit(self):
        self.reject_only(
            "amdsmi_set_cpu_core_boostlimit", self.handle, api.integer("boostlimit", 0)
        )

    def test_set_cpu_socket_boostlimit(self):
        self.reject_only(
            "amdsmi_set_cpu_socket_boostlimit", self.handle, api.integer("boostlimit", 0)
        )

    def test_set_cpu_sdps_limit(self):
        self.reject_only("amdsmi_set_cpu_sdps_limit", self.handle, api.integer("sdps_limit", 0))

    def test_set_cpu_rail_isofreq_policy(self):
        self.reject_only("amdsmi_set_cpu_rail_isofreq_policy", self.handle, api.integer("value", 0))

    def test_set_cpu_dfc_ctrl(self):
        self.reject_only("amdsmi_set_cpu_dfc_ctrl", self.handle, api.integer("value", 0))

    def test_set_cpu_pc6_enable(self):
        self.reject_only("amdsmi_set_cpu_pc6_enable", self.handle, api.integer("value", 0))

    def test_set_cpu_cc6_enable(self):
        self.reject_only("amdsmi_set_cpu_cc6_enable", self.handle, api.integer("value", 0))


if __name__ == "__main__":
    unittest.main()
