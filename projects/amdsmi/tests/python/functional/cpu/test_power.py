#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU power: boostlimit set, socket power cap set."""

import unittest

import common.common as common
from common.common import amdsmi


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

    def test_cpu_core_boostlimit(self):
        self.common.print_func_name("")
        # Writing a 0 boostlimit is the invalid-input probe. Firmware clamps the zero value
        # and succeeds.
        invalid_set_accept = [
            amdsmi.AmdSmiStatus.SUCCESS,
            # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU sockets
            # are present or the CPU family is unsupported.
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
            # message.
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
        ]
        correct_accept = [
            amdsmi.AmdSmiStatus.SUCCESS,
            # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU sockets
            # are present or the CPU family is unsupported.
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
            # message.
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
        ]

        def positive(handles, failures):
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    # Write an invalid boostlimit (0); firmware may reject or clamp.
                    inval_msg = f"\t### amdsmi_set_cpu_core_boostlimit(cpu={i}, boost_limit=0):"
                    val = "N/A"
                    with self.common.expect_status(inval_msg, invalid_set_accept):
                        val = amdsmi.amdsmi_set_cpu_core_boostlimit(cpu, 0)
                    self.common.print(f"\t    data: {val}")
                    # Read the current boostlimit so it can be restored.
                    orig = {}
                    get_msg = f"\t### amdsmi_get_cpu_core_boostlimit(cpu={i}):"
                    with self.common.expect_status(get_msg, correct_accept):
                        orig["val"] = amdsmi.amdsmi_get_cpu_core_boostlimit(cpu)
                    self.common.print(f"\t    data: {orig.get('val', 'N/A')}")
                    if "val" not in orig:
                        continue  # getter did not succeed; nothing to restore
                    set_msg = (
                        f"\t### amdsmi_set_cpu_core_boostlimit(cpu={i}, boost_limit={orig['val']}):"
                    )
                    val = "N/A"
                    with self.common.expect_status(set_msg, correct_accept):
                        val = amdsmi.amdsmi_set_cpu_core_boostlimit(cpu, orig["val"])
                    self.common.print(f"\t    data: {val}")

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_core_boostlimit,
            "amdsmi_set_cpu_core_boostlimit",
            handle_kind="core",
            arg_sets=[("boost_limit=0", (0,))],
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
            correct_accept=correct_accept,
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
            positive=positive,
        )
        return

    def test_cpu_socket_power_cap(self):
        self.common.print_func_name("")
        # Writing a 0 power cap is the invalid-input probe. Firmware clamps the zero value
        # and succeeds. Same status sets as test_cpu_core_boostlimit (see there).
        invalid_set_accept = [
            amdsmi.AmdSmiStatus.SUCCESS,
            # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU sockets
            # are present or the CPU family is unsupported.
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
            # message.
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
        ]
        correct_accept = [
            amdsmi.AmdSmiStatus.SUCCESS,
            # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU sockets
            # are present or the CPU family is unsupported.
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
            # message.
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
        ]

        def positive(handles, failures):
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    # Write an invalid power cap (0); firmware may reject or clamp.
                    inval_msg = f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap=0):"
                    val = "N/A"
                    with self.common.expect_status(inval_msg, invalid_set_accept):
                        val = amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, 0)
                    self.common.print(f"\t    data: {val}")
                    # Read the current cap and the max so the cap can be exercised
                    # and then restored.
                    vals = {}
                    get_msg = f"\t### amdsmi_get_cpu_socket_power_cap(cpu={i}):"
                    with self.common.expect_status(get_msg, correct_accept):
                        vals["orig"] = amdsmi.amdsmi_get_cpu_socket_power_cap(cpu)
                    self.common.print(f"\t    data: {vals.get('orig', 'N/A')}")
                    max_msg = f"\t### amdsmi_get_cpu_socket_power_cap_max(cpu={i}):"
                    with self.common.expect_status(max_msg, correct_accept):
                        vals["max"] = amdsmi.amdsmi_get_cpu_socket_power_cap_max(cpu)
                    self.common.print(f"\t    data: {vals.get('max', 'N/A')}")
                    if "orig" not in vals or "max" not in vals:
                        continue  # a getter did not succeed; nothing to set/restore
                    set_max_msg = (
                        f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={vals['max']}):"
                    )
                    val = "N/A"
                    with self.common.expect_status(set_max_msg, correct_accept):
                        val = amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, vals["max"])
                    self.common.print(f"\t    data: {val}")
                    restore_msg = (
                        f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={vals['orig']}):"
                    )
                    val = "N/A"
                    with self.common.expect_status(restore_msg, correct_accept):
                        val = amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, vals["orig"])
                    self.common.print(f"\t    data: {val}")

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_socket_power_cap,
            "amdsmi_set_cpu_socket_power_cap",
            arg_sets=[("power_cap=0", (0,))],
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
            correct_accept=correct_accept,
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
            positive=positive,
        )
        return

    def test_cpu_socket_boostlimit(self):
        self.common.print_func_name("")
        # TODO boost_limit = 0: the original test only writes a 0 boostlimit; it
        # is preserved here as the positive portion (one set per socket).
        # SUCCESS: firmware clamps the 0 boostlimit and accepts it.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_socket_boostlimit,
            "amdsmi_set_cpu_socket_boostlimit",
            arg_sets=[("boost_limit=0", (0,))],
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

    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails,
            "amdsmi_get_cpu_pwr_svi_telemetry_all_rails",
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

    def test_get_cpu_socket_power(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_power,
            "amdsmi_get_cpu_socket_power",
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

    def test_get_cpu_socket_power_cap(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_power_cap,
            "amdsmi_get_cpu_socket_power_cap",
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

    def test_get_cpu_socket_power_cap_max(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_power_cap_max,
            "amdsmi_get_cpu_socket_power_cap_max",
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

    def test_set_cpu_pwr_efficiency_mode(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_pwr_efficiency_mode,
            "amdsmi_set_cpu_pwr_efficiency_mode",
            arg_sets=[(f"mode={mode}", (mode,)) for mode in (0, 1, 2)],
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
