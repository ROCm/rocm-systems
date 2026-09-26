#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU energy: core energy, socket energy."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuEnergy(unittest.TestCase):
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

    def test_get_cpu_core_energy(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU_Core(
            amdsmi_get_cpu_core_energy=amdsmi.amdsmi_get_cpu_core_energy
        )
        return

    def test_get_cpu_socket_energy(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_socket_energy=amdsmi.amdsmi_get_cpu_socket_energy
        )
        return
