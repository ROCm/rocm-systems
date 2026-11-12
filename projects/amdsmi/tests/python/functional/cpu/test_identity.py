#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU identity: family, model, handles, SMU firmware version, threads per core."""

import unittest

import common.common as common
from common.common import amdsmi


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

    def test_first_online_core_on_cpu_socket(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_first_online_core_on_cpu_socket,
            "amdsmi_first_online_core_on_cpu_socket",
            wrong_accept=[
                # NOT_SUPPORTED: a GPU handle reaches the CPU path, which reports the feature
                # absent. Incidental; there is no processor-type check yet.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: a GPU handle reaches the HSMP message check, which reports the
                # message absent. Incidental; there is no processor-type check yet.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
                # NOT_FOUND: a GPU handle is not a CPU socket, so the lookup fails. The CPU path has
                # no processor-type check yet, so this rejection is incidental.
                amdsmi.AmdSmiStatus.NOT_FOUND,
                # IO: on a discrete-GPU host a GPU handle reaches the ESMI IO path first; on other
                # hosts it gets INVAL.
                amdsmi.AmdSmiStatus.IO,
                # INVAL: a GPU handle is rejected as invalid input. The CPU path has no processor-
                # type check yet, so this rejection is incidental.
                amdsmi.AmdSmiStatus.INVAL,
            ],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return

    def test_get_cpu_family(self):
        self.common.print_func_name("")
        # Takes no processor handle, so only the positive portion runs.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_family,
            "amdsmi_get_cpu_family",
            handle_kind="none",
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # IO: no CPU or HSMP target on the host, so the identity read fails.
                amdsmi.AmdSmiStatus.IO,
            ],
            null_accept=[],
        )
        return

    def test_get_cpu_handles(self):
        self.common.print_func_name("")
        # Takes no processor handle, so only the positive portion runs.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_handles,
            "amdsmi_get_cpu_handles",
            handle_kind="none",
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[],
        )
        return

    def test_get_cpu_model(self):
        self.common.print_func_name("")
        # Takes no processor handle, so only the positive portion runs. Same
        # status set as test_get_cpu_family (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_model,
            "amdsmi_get_cpu_model",
            handle_kind="none",
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # IO: no CPU or HSMP target on the host, so the identity read fails.
                amdsmi.AmdSmiStatus.IO,
            ],
            null_accept=[],
        )
        return

    def test_get_cpu_smu_fw_version(self):
        self.common.print_func_name("")
        # Returns the global SMU firmware version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. The wrong-handle
        # negative does not apply (check_wrong_handle=False), so wrong_accept is empty.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_smu_fw_version,
            "amdsmi_get_cpu_smu_fw_version",
            check_wrong_handle=False,
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return

    def test_get_threads_per_core(self):
        self.common.print_func_name("")
        # Takes no processor handle, so only the positive portion runs. Same
        # status set as test_get_cpu_family (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_threads_per_core,
            "amdsmi_get_threads_per_core",
            handle_kind="none",
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # IO: no CPU or HSMP target on the host, so the identity read fails.
                amdsmi.AmdSmiStatus.IO,
            ],
            null_accept=[],
        )
        return
