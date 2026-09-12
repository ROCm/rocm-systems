#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
