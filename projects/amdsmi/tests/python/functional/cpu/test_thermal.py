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

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_prochot_status(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_prochot_status, "amdsmi_get_cpu_prochot_status"
        )
        return

    def test_get_cpu_socket_c0_residency(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_c0_residency, "amdsmi_get_cpu_socket_c0_residency"
        )
        return

    def test_get_cpu_socket_temperature(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_temperature, "amdsmi_get_cpu_socket_temperature"
        )
        return
