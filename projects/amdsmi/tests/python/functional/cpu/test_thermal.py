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
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_prochot_status(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_prochot_status=amdsmi.amdsmi_get_cpu_prochot_status
        )
        return

    def test_get_cpu_socket_c0_residency(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_c0_residency=amdsmi.amdsmi_get_cpu_socket_c0_residency
        )
        return

    def test_get_cpu_socket_temperature(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_temperature=amdsmi.amdsmi_get_cpu_socket_temperature
        )
        return
