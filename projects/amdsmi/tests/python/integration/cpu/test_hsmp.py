#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU HSMP transport and ESMI error APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestCpuHsmp(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_hsmp_proto_ver(self):
        self.both("amdsmi_get_cpu_hsmp_proto_ver", self.handle)

    def test_get_cpu_hsmp_driver_version(self):
        self.both("amdsmi_get_cpu_hsmp_driver_version", self.handle)

    def test_get_hsmp_metrics_table(self):
        self.both("amdsmi_get_hsmp_metrics_table", self.handle)

    def test_get_hsmp_metrics_table_version(self):
        self.both("amdsmi_get_hsmp_metrics_table_version", self.handle)

    def test_get_cpu_ddr_bw(self):
        self.both("amdsmi_get_cpu_ddr_bw", self.handle)

    def test_get_esmi_err_msg(self):
        self.both("amdsmi_get_esmi_err_msg", api.enum("status", common.STATUS_TYPES))


if __name__ == "__main__":
    unittest.main()
