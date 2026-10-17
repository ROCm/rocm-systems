#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU energy: core energy, socket energy."""

import unittest

import common.common as common
from common.common import amdsmi

# On a discrete-GPU host the energy APIs reach the ESMI IO path before the HSMP
# message-support check, so a wrong (GPU) handle is rejected with AMDSMI_STATUS_IO
# rather than NO_HSMP_MSG_SUP; on other hosts/APUs it is rejected with INVAL. All
# are valid rejections of a bogus handle - the negative portion only has to reject
# SUCCESS.
ENERGY_WRONG_ACCEPT = [
    amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
    amdsmi.AmdSmiStatus.NOT_FOUND,
    amdsmi.AmdSmiStatus.IO,
    amdsmi.AmdSmiStatus.INVAL,
]


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
        self.common.amdsmi_smart_init()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_cpu_core_energy(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_core_energy,
            "amdsmi_get_cpu_core_energy",
            handle_kind="core",
            wrong_accept=ENERGY_WRONG_ACCEPT,
        )
        return

    def test_get_cpu_socket_energy(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_energy,
            "amdsmi_get_cpu_socket_energy",
            handle_kind="socket",
            wrong_accept=ENERGY_WRONG_ACCEPT,
        )
        return
