#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU HSMP: driver version, protocol version, DDR bandwidth, ESMI error messages, metrics table."""

import unittest

import common.common as common
from common.common import amdsmi

# amdsmi_get_esmi_err_msg() reads its argument as an esmi_status_t code and returns the
# amdsmi status that code maps to (esmi_status_map in amd_smi_common.h), so an input
# inside the esmi enum range comes back as that mapped status rather than SUCCESS.
# Keyed by esmi_status_t ordinal (e_smi.h).
_ESMI_CODE_TO_STATUS = {
    0: amdsmi.AmdSmiStatus.SUCCESS,  # ESMI_SUCCESS / ESMI_INITIALIZED
    1: amdsmi.AmdSmiStatus.NO_ENERGY_DRV,
    2: amdsmi.AmdSmiStatus.NO_MSR_DRV,
    3: amdsmi.AmdSmiStatus.NO_HSMP_DRV,
    4: amdsmi.AmdSmiStatus.NO_HSMP_SUP,
    5: amdsmi.AmdSmiStatus.NO_DRV,
    6: amdsmi.AmdSmiStatus.FILE_NOT_FOUND,
    7: amdsmi.AmdSmiStatus.BUSY,
    8: amdsmi.AmdSmiStatus.NO_PERM,
    9: amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    10: amdsmi.AmdSmiStatus.FILE_ERROR,
    11: amdsmi.AmdSmiStatus.INTERRUPT,
    12: amdsmi.AmdSmiStatus.IO,
    13: amdsmi.AmdSmiStatus.UNEXPECTED_SIZE,
    14: amdsmi.AmdSmiStatus.UNKNOWN_ERROR,
    15: amdsmi.AmdSmiStatus.ARG_PTR_NULL,
    16: amdsmi.AmdSmiStatus.OUT_OF_RESOURCES,
    17: amdsmi.AmdSmiStatus.NOT_INIT,
    18: amdsmi.AmdSmiStatus.INVAL,
    19: amdsmi.AmdSmiStatus.HSMP_TIMEOUT,
    20: amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
}


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
        self.common.TODO_SKIP_FAIL = not self.common._check_amd_hsmp_driver()
        self.common.TODO_SKIP_NOT_COMPLETE = False

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(amdsmi_get_cpu_ddr_bw=amdsmi.amdsmi_get_cpu_ddr_bw)
        return

    def test_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_hsmp_driver_version as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_hsmp_driver_version=amdsmi.amdsmi_get_cpu_hsmp_driver_version
        )
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_hsmp_proto_ver as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_hsmp_proto_ver=amdsmi.amdsmi_get_cpu_hsmp_proto_ver
        )
        return

    def test_get_esmi_err_msg(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_esmi_err_msg as it fails (Unknown Error)."
            self.common.print(msg)
            self.skipTest(msg)

        for _, status_type, _ in common.STATUS_TYPES:
            # Inputs outside the esmi enum range match no entry and return SUCCESS.
            expected = _ESMI_CODE_TO_STATUS.get(status_type.value, amdsmi.AmdSmiStatus.SUCCESS)
            status_cond = f"AMDSMI_STATUS_{expected.name}"
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
        self.common.Test_API_Per_CPU(
            amdsmi_get_hsmp_metrics_table=amdsmi.amdsmi_get_hsmp_metrics_table
        )
        return

    def test_get_hsmp_metrics_table_version(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(
            amdsmi_get_hsmp_metrics_table_version=amdsmi.amdsmi_get_hsmp_metrics_table_version
        )
        return
