#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU DIMM: power consumption, temperature range and refresh rate, thermal sensor."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuDimm(unittest.TestCase):
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

    def test_get_cpu_dimm_power_consumption(self):
        self.common.print_func_name("")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_dimm_power_consumption,
            "amdsmi_get_cpu_dimm_power_consumption",
            arg_sets=[(f"dimm_addr={dimm_addr:#x}", (dimm_addr,))],
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

    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self.common.print_func_name("")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0x87
        # Same status set as test_get_cpu_dimm_power_consumption (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate,
            "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
            arg_sets=[(f"dimm_addr={dimm_addr:#x}", (dimm_addr,))],
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

    def test_get_cpu_dimm_thermal_sensor(self):
        self.common.print_func_name("")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0x87
        # Same status set as test_get_cpu_dimm_power_consumption (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_dimm_thermal_sensor,
            "amdsmi_get_cpu_dimm_thermal_sensor",
            arg_sets=[(f"dimm_addr={dimm_addr:#x}", (dimm_addr,))],
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
