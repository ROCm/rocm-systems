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

    def test_cpu_apis_reject_gpu_handle(self) -> None:
        if not self.common._check_amd_hsmp_driver():
            self.skipTest("AMD HSMP driver is unavailable")

        cpu_handles = amdsmi.amdsmi_get_cpu_handles()["processor_handles"]
        if not cpu_handles:
            self.skipTest("No AMD CPU socket handles available")

        gpu_handles = [
            processor
            for processor in self.common.processors
            if amdsmi.amdsmi_get_processor_type(processor)["processor_type"]
            == amdsmi.AmdSmiProcessorType.AMD_GPU.name
        ]
        if not gpu_handles:
            self.skipTest("No AMD GPU handles available")

        amdsmi.amdsmi_get_cpu_hsmp_proto_ver(cpu_handles[0])
        for processor in gpu_handles:
            for cpu_api in (
                amdsmi.amdsmi_get_cpu_ddr_bw,
                amdsmi.amdsmi_get_cpu_core_energy,
                amdsmi.amdsmi_get_cpu_hsmp_proto_ver,
            ):
                with self.subTest(api=cpu_api.__name__):
                    with self.assertRaises(amdsmi.AmdSmiLibraryException) as error:
                        cpu_api(processor)
                    self.assertEqual(
                        error.exception.get_error_code(), amdsmi.AmdSmiStatus.NOT_SUPPORTED
                    )

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
