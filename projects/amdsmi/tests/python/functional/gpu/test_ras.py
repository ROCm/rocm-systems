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
        # amdsmi_get_gpu_total_ecc_count() reads ras/features once per call and returns:
        #   SUCCESS       - mask read and parsed
        #   NOT_SUPPORTED - RAS unsupported on this ASIC (ras/features absent)
        # AMDSMI_STATUS_API_FAILED (a malformed/unreadable mask) is deliberately not
        # accepted: on present, real hardware that would mean the mask-read fix this
        # suite guards has regressed, not an outcome to tolerate.
        accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                self.common.print_device_header(i)
                msg = f"\t### amdsmi_get_gpu_total_ecc_count(gpu={i}):"
                data = "N/A"
                with self.common.expect_status(msg, accept):
                    data = amdsmi.amdsmi_get_gpu_total_ecc_count(gpu)
                self.common.print(f"\t\tamdsmi_get_gpu_total_ecc_count(gpu={i}): {data}")
                self.common.print("")
        return

    def test_get_gpu_total_ecc_count_rejects_non_handle(self):
        self.common.print_func_name("")
        # A non-handle fails the Python-side isinstance check before any library
        # call happens, so it is not a status expect_status can judge.
        with self.assertRaises(amdsmi.AmdSmiParameterException) as cm:
            amdsmi.amdsmi_get_gpu_total_ecc_count("not_a_handle")
        # Confirms it is *this* isinstance check that fired, not some other
        # AmdSmiParameterException raised earlier/elsewhere for a different reason.
        self.assertEqual(cm.exception.expectedType, amdsmi.amdsmi_wrapper.amdsmi_processor_handle)
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
