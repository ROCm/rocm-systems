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

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(amdsmi_get_cpu_ddr_bw=amdsmi.amdsmi_get_cpu_ddr_bw)
        return

    def test_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_hsmp_driver_version=amdsmi.amdsmi_get_cpu_hsmp_driver_version
        )
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_hsmp_proto_ver=amdsmi.amdsmi_get_cpu_hsmp_proto_ver
        )
        return

    # Statuses that amdsmi_get_esmi_err_msg reports back instead of SUCCESS: it
    # returns the value mapped from esmi_status_map rather than a success code
    # whenever the requested status carries an ESMI message, so the binding raises
    # for exactly the inputs that resolve to a message.
    ESMI_MAPPED_STATUSES = [
        amdsmi.AmdSmiStatus.INVAL,
        amdsmi.AmdSmiStatus.NOT_SUPPORTED,
        amdsmi.AmdSmiStatus.NO_PERM,
        amdsmi.AmdSmiStatus.INTERRUPT,
        amdsmi.AmdSmiStatus.IO,
        amdsmi.AmdSmiStatus.FILE_ERROR,
        amdsmi.AmdSmiStatus.OUT_OF_RESOURCES,
        amdsmi.AmdSmiStatus.BUSY,
        amdsmi.AmdSmiStatus.NOT_INIT,
        amdsmi.AmdSmiStatus.UNEXPECTED_SIZE,
        amdsmi.AmdSmiStatus.UNKNOWN_ERROR,
        amdsmi.AmdSmiStatus.NO_ENERGY_DRV,
        amdsmi.AmdSmiStatus.NO_MSR_DRV,
        amdsmi.AmdSmiStatus.NO_HSMP_DRV,
        amdsmi.AmdSmiStatus.NO_HSMP_SUP,
        amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
        amdsmi.AmdSmiStatus.HSMP_TIMEOUT,
        amdsmi.AmdSmiStatus.NO_DRV,
        amdsmi.AmdSmiStatus.FILE_NOT_FOUND,
        amdsmi.AmdSmiStatus.ARG_PTR_NULL,
    ]

    def test_get_esmi_err_msg(self):
        self.common.print_func_name("")

        accept = [self.common.PASS, *self.ESMI_MAPPED_STATUSES]
        with self.common.status_sweep():
            for status_name, status_type, _ in common.STATUS_TYPES:
                msg = f"\t### amdsmi_get_esmi_err_msg(status_type={status_name}):"
                with self.common.expect_status(msg, accept):
                    amdsmi.amdsmi_get_esmi_err_msg(status_type)
                self.common.print("")
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
