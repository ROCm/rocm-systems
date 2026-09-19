#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU identity APIs."""

import unittest

import common.api_test as api
import common.common as common

_AFFINITY_SCOPES = [
    (member.name, member, common.PASS) for member in common.amdsmi.AmdSmiAffinityScope
]


class TestCpuIdentity(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_handles(self):
        self.assertGreaterEqual(self.expect_only("amdsmi_get_cpu_handles")["cpu_count"], 1)

    def test_get_cpucore_handles(self):
        self.expect_only("amdsmi_get_cpucore_handles")

    def test_get_cpu_socket_count(self):
        self.expect_only("amdsmi_get_cpu_socket_count")

    def test_get_cpu_cores_per_socket(self):
        self.both("amdsmi_get_cpu_cores_per_socket", api.integer("sock_count", 0))

    def test_get_cpu_model_name(self):
        self.both("amdsmi_get_cpu_model_name", self.handle)

    def test_get_cpu_smu_fw_version(self):
        self.both("amdsmi_get_cpu_smu_fw_version", self.handle)

    def test_get_cpu_enabled_commands(self):
        self.both("amdsmi_get_cpu_enabled_commands", self.handle)

    def test_first_online_core_on_cpu_socket(self):
        self.both("amdsmi_first_online_core_on_cpu_socket", self.handle)

    def test_get_cpu_affinity_with_scope(self):
        # Resolves the CPU behind a device, so it takes a GPU handle but needs
        # CPU enumeration to answer.
        self.both(
            "amdsmi_get_cpu_affinity_with_scope",
            api.Handle("gpu", self.common.processors),
            api.enum("scope", _AFFINITY_SCOPES),
        )

    def test_get_cpu_family(self):
        self.expect_only("amdsmi_get_cpu_family")

    def test_get_cpu_model(self):
        self.expect_only("amdsmi_get_cpu_model")

    def test_get_threads_per_core(self):
        self.expect_only("amdsmi_get_threads_per_core")


if __name__ == "__main__":
    unittest.main()
