#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU power: boostlimit set, socket power cap set."""

import unittest

import common.common as common
from common.common import amdsmi

# Writing a 0 boostlimit / power cap is the invalid-input probe the original
# tests used. Firmware may clamp-and-succeed or reject with INVAL, and a platform
# without HSMP answers NOT_SUPPORTED/NO_HSMP_MSG_SUP; accept that spread (same
# tolerance the old check_ret path had) so the write is meaningful without
# widening to anything.
INVALID_SET_ACCEPT = [
    amdsmi.AmdSmiStatus.SUCCESS,
    amdsmi.AmdSmiStatus.INVAL,
    amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
]


class TestCpuPower(unittest.TestCase):
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

    def test_cpu_core_boostlimit(self):
        self.common.print_func_name("")

        def positive(handles, failures):
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    # Write an invalid boostlimit (0); firmware may reject or clamp.
                    inval_msg = f"\t### amdsmi_set_cpu_core_boostlimit(cpu={i}, boost_limit=0):"
                    with self.common.expect_status(inval_msg, INVALID_SET_ACCEPT):
                        amdsmi.amdsmi_set_cpu_core_boostlimit(cpu, 0)
                    # Read the current boostlimit so it can be restored.
                    orig = {}
                    get_msg = f"\t### amdsmi_get_cpu_core_boostlimit(cpu={i}):"
                    with self.common.expect_status(get_msg, self.common.CPU_CORRECT_ACCEPT):
                        orig["val"] = amdsmi.amdsmi_get_cpu_core_boostlimit(cpu)
                    if "val" not in orig:
                        continue  # getter did not succeed; nothing to restore
                    set_msg = (
                        f"\t### amdsmi_set_cpu_core_boostlimit(cpu={i}, boost_limit={orig['val']}):"
                    )
                    with self.common.expect_status(set_msg, self.common.CPU_CORRECT_ACCEPT):
                        amdsmi.amdsmi_set_cpu_core_boostlimit(cpu, orig["val"])

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_core_boostlimit,
            "amdsmi_set_cpu_core_boostlimit",
            handle_kind="core",
            arg_sets=[("boost_limit=0", (0,))],
            positive=positive,
        )
        return

    # integration

    def test_cpu_socket_power_cap(self):
        self.common.print_func_name("")

        def positive(handles, failures):
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    # Write an invalid power cap (0); firmware may reject or clamp.
                    inval_msg = f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap=0):"
                    with self.common.expect_status(inval_msg, INVALID_SET_ACCEPT):
                        amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, 0)
                    # Read the current cap and the max so the cap can be exercised
                    # and then restored.
                    vals = {}
                    get_msg = f"\t### amdsmi_get_cpu_socket_power_cap(cpu={i}):"
                    with self.common.expect_status(get_msg, self.common.CPU_CORRECT_ACCEPT):
                        vals["orig"] = amdsmi.amdsmi_get_cpu_socket_power_cap(cpu)
                    max_msg = f"\t### amdsmi_get_cpu_socket_power_cap_max(cpu={i}):"
                    with self.common.expect_status(max_msg, self.common.CPU_CORRECT_ACCEPT):
                        vals["max"] = amdsmi.amdsmi_get_cpu_socket_power_cap_max(cpu)
                    if "orig" not in vals or "max" not in vals:
                        continue  # a getter did not succeed; nothing to set/restore
                    set_max_msg = (
                        f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={vals['max']}):"
                    )
                    with self.common.expect_status(set_max_msg, self.common.CPU_CORRECT_ACCEPT):
                        amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, vals["max"])
                    restore_msg = (
                        f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={vals['orig']}):"
                    )
                    with self.common.expect_status(restore_msg, self.common.CPU_CORRECT_ACCEPT):
                        amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, vals["orig"])

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_socket_power_cap,
            "amdsmi_set_cpu_socket_power_cap",
            arg_sets=[("power_cap=0", (0,))],
            positive=positive,
        )
        return

    # integration

    def test_cpu_socket_boostlimit(self):
        self.common.print_func_name("")
        # TODO boost_limit = 0: the original test only writes a 0 boostlimit; it
        # is preserved here as the positive portion (one set per socket).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_socket_boostlimit,
            "amdsmi_set_cpu_socket_boostlimit",
            arg_sets=[("boost_limit=0", (0,))],
            correct_accept=INVALID_SET_ACCEPT,
        )
        return

    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails,
            "amdsmi_get_cpu_pwr_svi_telemetry_all_rails",
        )
        return

    def test_get_cpu_socket_power(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_power, "amdsmi_get_cpu_socket_power"
        )
        return

    def test_get_cpu_socket_power_cap(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_power_cap, "amdsmi_get_cpu_socket_power_cap"
        )
        return

    def test_get_cpu_socket_power_cap_max(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_power_cap_max, "amdsmi_get_cpu_socket_power_cap_max"
        )
        return

    def test_set_cpu_pwr_efficiency_mode(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_pwr_efficiency_mode,
            "amdsmi_set_cpu_pwr_efficiency_mode",
            arg_sets=[(f"mode={mode}", (mode,)) for mode in (0, 1, 2)],
        )
        return
