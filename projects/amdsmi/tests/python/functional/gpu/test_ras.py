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

    def test_get_gpu_available_counters(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_available_counters=amdsmi.amdsmi_get_gpu_available_counters,
            event_group=common.EVENT_GROUPS,
        )
        return

    def test_get_gpu_ecc_count(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_ecc_count=amdsmi.amdsmi_get_gpu_ecc_count, gpu_block=common.GPU_BLOCKS
        )
        return

    def test_get_gpu_ecc_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_ecc_enabled=amdsmi.amdsmi_get_gpu_ecc_enabled)
        return

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

    def test_get_gpu_ras_block_features_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_ras_block_features_enabled=amdsmi.amdsmi_get_gpu_ras_block_features_enabled
        )
        return

    def test_get_gpu_ras_feature_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_ras_feature_info=amdsmi.amdsmi_get_gpu_ras_feature_info
        )
        return

    def test_get_gpu_total_ecc_count(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_total_ecc_count=amdsmi.amdsmi_get_gpu_total_ecc_count
        )
        return

    def test_gpu_counter_group_supported(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_gpu_counter_group_supported=amdsmi.amdsmi_gpu_counter_group_supported,
            event_group=common.EVENT_GROUPS,
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
