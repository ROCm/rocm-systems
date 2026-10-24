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
        self.common.amdsmi_smart_init()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(amdsmi.amdsmi_get_cpu_ddr_bw, "amdsmi_get_cpu_ddr_bw")
        return

    def test_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")
        # Returns the global HSMP driver version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_hsmp_driver_version,
            "amdsmi_get_cpu_hsmp_driver_version",
            check_wrong_handle=False,
        )
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")
        # Returns the global HSMP protocol version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_hsmp_proto_ver,
            "amdsmi_get_cpu_hsmp_proto_ver",
            check_wrong_handle=False,
        )
        return

    def test_get_esmi_err_msg(self):
        self.common.print_func_name("")
        # Pure status-code -> message-string mapper, no processor handle. Its
        # valid input domain is a real ESMI status: SUCCESS maps to its string,
        # while the native mapper rejects everything else (the old test only
        # appeared to cover all AmdSmiStatus members because it skipped itself on
        # CPU-less hosts). So: POSITIVE a known-good status, NEGATIVE a None.
        failures = []
        # POSITIVE: a valid status maps to a non-empty message, status SUCCESS.
        try:
            msg = "\t### amdsmi_get_esmi_err_msg(status_type=SUCCESS):"
            val = "N/A"
            with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.SUCCESS]):
                val = amdsmi.amdsmi_get_esmi_err_msg(amdsmi.AmdSmiStatus.SUCCESS)
            self.common.print(f"\t    data: {val}")
            self.assertIsInstance(val, str)
            self.assertTrue(val)
        except Exception as e:  # noqa: BLE001 - collected and re-raised together
            failures.append(e)
        # NEGATIVE: None is rejected by the wrapper with AmdSmiParameterException.
        try:
            msg = "\t### amdsmi_get_esmi_err_msg(status_type=None):"
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_get_esmi_err_msg(None)
            self.common.print(f"{msg}\n\tTEST SUCCESS, raised AmdSmiParameterException as expected")
        except Exception as e:  # noqa: BLE001 - collected and re-raised together
            failures.append(e)
        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return

    def test_get_hsmp_metrics_table(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_hsmp_metrics_table, "amdsmi_get_hsmp_metrics_table"
        )
        return

    def test_get_hsmp_metrics_table_version(self):
        self.common.print_func_name("")
        # Returns the global metrics-table version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_hsmp_metrics_table_version,
            "amdsmi_get_hsmp_metrics_table_version",
            check_wrong_handle=False,
        )
        return
