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
"""CPU identity APIs."""

import unittest

import common.common as common

_AFFINITY_SCOPES = [
    (member.name, member, common.PASS) for member in common.amdsmi.AmdSmiAffinityScope
]


class TestCpuIdentity(common.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_handles(self):
        self.assertGreaterEqual(self.expect_only("amdsmi_get_cpu_handles")["cpu_count"], 1)

    def test_get_cpucore_handles(self):
        self.expect_only("amdsmi_get_cpucore_handles")

    def test_get_cpu_socket_count(self):
        self.expect_only("amdsmi_get_cpu_socket_count")

    def test_get_cpu_cores_per_socket(self):
        self.both("amdsmi_get_cpu_cores_per_socket", common.integer("sock_count", 0))

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
            common.Handle("gpu", self.common.processors),
            common.enum("scope", _AFFINITY_SCOPES),
        )

    def test_get_cpu_family(self):
        self.expect_only("amdsmi_get_cpu_family")

    def test_get_cpu_model(self):
        self.expect_only("amdsmi_get_cpu_model")

    def test_get_threads_per_core(self):
        self.expect_only("amdsmi_get_threads_per_core")


if __name__ == "__main__":
    unittest.main()
