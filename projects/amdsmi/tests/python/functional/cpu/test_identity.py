#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU identity: family, model, handles, SMU firmware version, threads per core."""

import unittest

import common.common as common
from common.common import amdsmi

# amdsmi_first_online_core_on_cpu_socket reaches the ESMI IO path before the HSMP
# message-support check, so a wrong (GPU) handle is rejected with AMDSMI_STATUS_IO
# on a discrete-GPU host rather than NO_HSMP_MSG_SUP; on other hosts/APUs it is
# rejected with INVAL. All are valid rejections of a bogus handle - the negative
# portion only has to reject SUCCESS.
FIRST_ONLINE_WRONG_ACCEPT = [
    amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
    amdsmi.AmdSmiStatus.NOT_FOUND,
    amdsmi.AmdSmiStatus.IO,
    amdsmi.AmdSmiStatus.INVAL,
]

# amdsmi_get_cpu_family / _model / _threads_per_core take no processor handle and
# read CPU identity directly. On a host with no CPU/HSMP target they fail with
# AMDSMI_STATUS_IO; on a real CPU they return SUCCESS. Accept both (plus the
# platform "absent" status) so the positive portion runs and validates without
# skipping, while still rejecting INVAL and friends.
NO_CPU_INFO_ACCEPT = [
    amdsmi.AmdSmiStatus.SUCCESS,
    amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    amdsmi.AmdSmiStatus.IO,
]


class TestCpuIdentity(unittest.TestCase):
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

    def test_first_online_core_on_cpu_socket(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_first_online_core_on_cpu_socket,
            "amdsmi_first_online_core_on_cpu_socket",
            wrong_accept=FIRST_ONLINE_WRONG_ACCEPT,
        )
        return

    def test_get_cpu_family(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_family,
            "amdsmi_get_cpu_family",
            handle_kind="none",
            correct_accept=NO_CPU_INFO_ACCEPT,
        )
        return

    def test_get_cpu_handles(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_handles, "amdsmi_get_cpu_handles", handle_kind="none"
        )
        return

    def test_get_cpu_model(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_model,
            "amdsmi_get_cpu_model",
            handle_kind="none",
            correct_accept=NO_CPU_INFO_ACCEPT,
        )
        return

    def test_get_cpu_smu_fw_version(self):
        self.common.print_func_name("")
        # Returns the global SMU firmware version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_smu_fw_version,
            "amdsmi_get_cpu_smu_fw_version",
            check_wrong_handle=False,
        )
        return

    def test_get_threads_per_core(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_threads_per_core,
            "amdsmi_get_threads_per_core",
            handle_kind="none",
            correct_accept=NO_CPU_INFO_ACCEPT,
        )
        return
