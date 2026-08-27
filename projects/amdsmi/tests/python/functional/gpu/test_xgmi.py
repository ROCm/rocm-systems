#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU XGMI: XGMI info, link status, error status, topology, NUMA affinity, P2P."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuXgmi(unittest.TestCase):
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

    def test_gpu_xgmi_error_status(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_xgmi_error_status as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_gpu_xgmi_error_status=amdsmi.amdsmi_gpu_xgmi_error_status
        )
        return

    def test_reset_gpu_xgmi_error(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu_xgmi_error as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_xgmi_error=amdsmi.amdsmi_reset_gpu_xgmi_error)
        return
