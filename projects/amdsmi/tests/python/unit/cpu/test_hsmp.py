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
"""CPU HSMP transport and ESMI error APIs."""

import unittest

import common.common as common


class TestCpuHsmp(common.ApiTestCase):
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
        self.both("amdsmi_get_esmi_err_msg", common.enum("status", common.STATUS_TYPES))


if __name__ == "__main__":
    unittest.main()
