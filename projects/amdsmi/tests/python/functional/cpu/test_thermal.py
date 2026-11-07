#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU thermal: socket temperature, PROCHOT status, C0 residency."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuThermal(unittest.TestCase):
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

    def test_get_cpu_prochot_status(self):
        self.common.print_func_name("")
        # NOT_SUPPORTED / NO_HSMP_MSG_SUP: platform lacks this HSMP message.
        # NOT_SUPPORTED / NO_HSMP_MSG_SUP / NOT_FOUND / INVAL: a GPU or bogus handle
        # is rejected.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_prochot_status,
            "amdsmi_get_cpu_prochot_status",
            wrong_accept=[
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
                amdsmi.AmdSmiStatus.NOT_FOUND,
                amdsmi.AmdSmiStatus.INVAL,
            ],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[amdsmi.AmdSmiStatus.INVAL],
        )
        return

    def test_get_cpu_socket_c0_residency(self):
        self.common.print_func_name("")
        # Same status sets as test_get_cpu_prochot_status (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_c0_residency,
            "amdsmi_get_cpu_socket_c0_residency",
            wrong_accept=[
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
                amdsmi.AmdSmiStatus.NOT_FOUND,
                amdsmi.AmdSmiStatus.INVAL,
            ],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[amdsmi.AmdSmiStatus.INVAL],
        )
        return

    def test_get_cpu_socket_temperature(self):
        self.common.print_func_name("")
        # Same status sets as test_get_cpu_prochot_status (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_temperature,
            "amdsmi_get_cpu_socket_temperature",
            wrong_accept=[
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
                amdsmi.AmdSmiStatus.NOT_FOUND,
                amdsmi.AmdSmiStatus.INVAL,
            ],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[amdsmi.AmdSmiStatus.INVAL],
        )
        return
