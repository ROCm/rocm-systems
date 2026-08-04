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
"""CPU HSMP: driver version, protocol version, DDR bandwidth, ESMI error messages, metrics table."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuHsmp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_ddr_bw=amdsmi.amdsmi_get_cpu_ddr_bw)
        return

    def test_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_hsmp_driver_version as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_hsmp_driver_version=amdsmi.amdsmi_get_cpu_hsmp_driver_version
        )
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_hsmp_proto_ver as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_hsmp_proto_ver=amdsmi.amdsmi_get_cpu_hsmp_proto_ver
        )
        return

    def test_get_esmi_err_msg(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_esmi_err_msg as it fails (Unknown Error)."
            self.common.print(msg)
            self.skipTest(msg)

        for _, status_type, status_cond in common.STATUS_TYPES:
            msg = f"\t### amdsmi_get_esmi_err_msg(status_type={status_type}):"
            try:
                ret = amdsmi.amdsmi_get_esmi_err_msg(status_type)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg, e, status_cond):
                    self.raise_exception = e
            self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_hsmp_metrics_table(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_hsmp_metrics_table=amdsmi.amdsmi_get_hsmp_metrics_table
        )
        return

    def test_get_hsmp_metrics_table_version(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_hsmp_metrics_table_version=amdsmi.amdsmi_get_hsmp_metrics_table_version
        )
        return
