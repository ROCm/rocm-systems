#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU RAS: ECC count/status/enabled, RAS block features, total ECC count, EEPROM validation, counters."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuRas(unittest.TestCase):
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

    def test_get_gpu_ecc_status(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_ecc_status as it fails."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_ecc_status=amdsmi.amdsmi_get_gpu_ecc_status, gpu_block=common.GPU_BLOCKS
        )
        return

    def test_gpu_validate_ras_eeprom(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_validate_ras_eepromas it fails (File Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_gpu_validate_ras_eeprom=amdsmi.amdsmi_gpu_validate_ras_eeprom
        )
        return
