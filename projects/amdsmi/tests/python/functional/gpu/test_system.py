#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU system operations: init, shutdown, reset, status codes, socket handles."""

import ctypes
import unittest

import common.common as common
from common.common import amdsmi


class TestGpuSystem(unittest.TestCase):
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

    def test_get_socket_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_socket_handles():"
        try:
            ret = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in ret])
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_init(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_init=amdsmi.amdsmi_init)
        return

    def test_reset_gpu(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu as it fails (MI350X, Hang)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu=amdsmi.amdsmi_reset_gpu)
        return

    def test_reset_gpu_fan(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_fan=amdsmi.amdsmi_reset_gpu_fan, index=0)
        return

    def test_shut_down(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_shut_down=amdsmi.amdsmi_shut_down)
        return

    def test_status_code_to_string(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_status_code_to_string as it fails (Unhashable type)."
            self.common.print(msg)
            self.skipTest(msg)

        for error_num, _ in self.common.error_map.items():
            msg = f"\t### amdsmi_status_code_to_string(error_num={error_num}):"
            try:
                ret = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(int(error_num, 0)))
                self.common.print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
            self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return
