#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU clock: set clock frequency."""

import unittest

import common.common as common
from common.common import amdsmi

# Limits the firmware enforces, from esmi_ib_library: DF_PSTATE_MAX_LIMIT and
# GMI3_LINK_WIDTH_LIMIT in e_smi_plat.c, the 2/8/16 lane note on
# esmi_xgmi_width_set, and the 2-bit prev_mode mask on esmi_pcie_link_rate_set.
DF_PSTATE_MAX = 2
GMI3_LINK_WIDTH_MAX = 2
XGMI_WIDTH_MIN = 2
XGMI_WIDTH_MAX = 16
PCIE_RATE_CTRLS = (0, 1, 2)
NBIO_IDS = (0, 1, 2, 3)


class TestCpuClock(unittest.TestCase):
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

    def test_cpu_apb_disable(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_cpu_apb_disable,
            "amdsmi_cpu_apb_disable",
            arg_sets=[("pstate=0", (0,))],
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

    def test_cpu_apb_enable(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_cpu_apb_enable,
            "amdsmi_cpu_apb_enable",
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

    def test_get_cpu_cclk_limit(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_cclk_limit,
            "amdsmi_get_cpu_cclk_limit",
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

    def test_get_cpu_core_current_freq_limit(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there), per core handle.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_core_current_freq_limit,
            "amdsmi_get_cpu_core_current_freq_limit",
            handle_kind="core",
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

    def test_get_cpu_current_io_bandwidth(self):
        self.common.print_func_name("")

        def positive(handles, failures):
            # Link names are legal only for the platform's HSMP protocol version, so
            # resolve the table per socket. A version with no table is skipped for
            # that socket; a legal link returning INVAL is a real failure.
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    try:
                        proto_ver = amdsmi.amdsmi_get_cpu_hsmp_proto_ver(cpu)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        self.common.print(
                            f"\t### amdsmi_get_cpu_hsmp_proto_ver(cpu={i}): unavailable ({e}), skipping"
                        )
                        continue
                    links = common.io_bw_encodings_for_proto_ver(proto_ver)
                    if not links:
                        self.common.print(
                            f"\t### HSMP protocol version {proto_ver} defines no IO link encodings, skipping cpu={i}"
                        )
                        continue
                    for link_name, encoding, _ in links:
                        msg = (
                            f"\t### amdsmi_get_cpu_current_io_bandwidth(cpu={i}, "
                            f"encoding={encoding}, link_name={link_name}):"
                        )
                        val = "N/A"
                        with self.common.expect_status(msg, correct_accept):
                            val = amdsmi.amdsmi_get_cpu_current_io_bandwidth(
                                cpu, encoding, link_name
                            )
                        self.common.print(f"\t    data: {val}")

        # Positive accepts only feature-absent statuses; INVAL is not accepted for a
        # legal link ID (see positive above).
        correct_accept = [
            amdsmi.AmdSmiStatus.SUCCESS,
            # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU sockets
            # are present or the CPU family is unsupported.
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
            # message.
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
        ]
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_current_io_bandwidth,
            "amdsmi_get_cpu_current_io_bandwidth",
            arg_sets=[
                (f"encoding={encoding} link_name={link_name}", (encoding, link_name))
                for link_name, encoding, _ in common.IO_BW_ENCODINGS
            ],
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

    def test_get_cpu_fclk_mclk(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_fclk_mclk,
            "amdsmi_get_cpu_fclk_mclk",
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

    def test_get_cpu_socket_current_active_freq_limit(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit,
            "amdsmi_get_cpu_socket_current_active_freq_limit",
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

    def test_get_cpu_socket_freq_range(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_freq_range,
            "amdsmi_get_cpu_socket_freq_range",
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

    def test_get_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there), per NBIO id.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level,
            "amdsmi_get_cpu_socket_lclk_dpm_level",
            arg_sets=[(f"nbio_id={nbio_id}", (nbio_id,)) for nbio_id in NBIO_IDS],
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

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
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
            # Narrow the range, then hand P-state selection back to the firmware.
            # Firmware requires max_pstate <= min_pstate, with 0 the highest P-state
            # and 2 the lowest. There is no getter for the DF P-state range, so
            # enabling APB is the documented restore path.
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    set_msg = (
                        f"\t### amdsmi_set_cpu_df_pstate_range(cpu={i}, "
                        f"min_pstate={DF_PSTATE_MAX}, max_pstate=1):"
                    )
                    val = "N/A"
                    with self.common.expect_status(set_msg, correct_accept):
                        val = amdsmi.amdsmi_set_cpu_df_pstate_range(cpu, DF_PSTATE_MAX, 1)
                    self.common.print(f"\t    data: {val}")
                    restore_msg = f"\t### amdsmi_cpu_apb_enable(cpu={i}):"
                    val = "N/A"
                    with self.common.expect_status(restore_msg, correct_accept):
                        val = amdsmi.amdsmi_cpu_apb_enable(cpu)
                    self.common.print(f"\t    data: {val}")

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_df_pstate_range,
            "amdsmi_set_cpu_df_pstate_range",
            arg_sets=[(f"min_pstate={DF_PSTATE_MAX}, max_pstate=1", (DF_PSTATE_MAX, 1))],
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

    def test_set_cpu_gmi3_link_width_range(self):
        self.common.print_func_name("")
        # Narrow, then widen back. No getter exists, so the restore target is the
        # full range; the positive portion issues both sets per handle.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_gmi3_link_width_range,
            "amdsmi_set_cpu_gmi3_link_width_range",
            arg_sets=[
                (
                    f"min_link_width=1, max_link_width={GMI3_LINK_WIDTH_MAX}",
                    (1, GMI3_LINK_WIDTH_MAX),
                ),
                (
                    f"min_link_width=0, max_link_width={GMI3_LINK_WIDTH_MAX}",
                    (0, GMI3_LINK_WIDTH_MAX),
                ),
            ],
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

    def test_set_cpu_pcie_link_rate(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
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
            # The setter reports the mode it replaced, which is the only way to
            # read the current rate control value; keep the first one to restore.
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    prev_mode = None
                    for rate_ctrl in PCIE_RATE_CTRLS:
                        msg = (
                            f"\t### amdsmi_set_cpu_pcie_link_rate(cpu={i}, rate_ctrl={rate_ctrl}):"
                        )
                        ret = "N/A"
                        with self.common.expect_status(msg, correct_accept):
                            ret = amdsmi.amdsmi_set_cpu_pcie_link_rate(cpu, rate_ctrl)
                            if prev_mode is None:
                                prev_mode = int(ret)
                        self.common.print(f"\t    data: {ret}")
                    if prev_mode is not None:
                        msg = (
                            f"\t### amdsmi_set_cpu_pcie_link_rate(cpu={i}, rate_ctrl={prev_mode}):"
                        )
                        ret = "N/A"
                        with self.common.expect_status(msg, correct_accept):
                            ret = amdsmi.amdsmi_set_cpu_pcie_link_rate(cpu, prev_mode)
                        self.common.print(f"\t    data: {ret}")

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_pcie_link_rate,
            "amdsmi_set_cpu_pcie_link_rate",
            arg_sets=[("rate_ctrl=0", (0,))],
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

    def test_set_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")
        # Same status sets as test_cpu_apb_disable (see there).
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
            # Read the current bounds so they can be put back afterwards.
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    for nbio_id in NBIO_IDS:
                        bounds = {}
                        get_msg = (
                            f"\t### amdsmi_get_cpu_socket_lclk_dpm_level(cpu={i}, "
                            f"nbio_id={nbio_id}):"
                        )
                        dpm = "N/A"
                        with self.common.expect_status(get_msg, correct_accept):
                            dpm = amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(cpu, nbio_id)
                            bounds["min"] = dpm["nbio_min_dpm_level"]
                            bounds["max"] = dpm["nbio_max_dpm_level"]
                        self.common.print(f"\t    data: {dpm}")
                        if "max" not in bounds:
                            continue  # getter did not succeed; nothing to set/restore
                        min_orig, max_orig = bounds["min"], bounds["max"]
                        # Pin to the top level, then restore the original bounds.
                        for min_val, max_val in ((max_orig, max_orig), (min_orig, max_orig)):
                            set_msg = (
                                f"\t### amdsmi_set_cpu_socket_lclk_dpm_level(cpu={i}, "
                                f"nbio_id={nbio_id}, min_val={min_val}, max_val={max_val}):"
                            )
                            val = "N/A"
                            with self.common.expect_status(set_msg, correct_accept):
                                val = amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level(
                                    cpu, nbio_id, min_val, max_val
                                )
                            self.common.print(f"\t    data: {val}")

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
            "amdsmi_set_cpu_socket_lclk_dpm_level",
            arg_sets=[("nbio_id=0, min_val=0, max_val=1", (0, 0, 1))],
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

    def test_set_cpu_xgmi_width(self):
        self.common.print_func_name("")
        # Narrow to the top lane count only, then reopen the full range. As with
        # GMI3 there is no getter, so full range is the restore target.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_xgmi_width,
            "amdsmi_set_cpu_xgmi_width",
            arg_sets=[
                (
                    f"min_width={XGMI_WIDTH_MAX}, max_width={XGMI_WIDTH_MAX}",
                    (XGMI_WIDTH_MAX, XGMI_WIDTH_MAX),
                ),
                (
                    f"min_width={XGMI_WIDTH_MIN}, max_width={XGMI_WIDTH_MAX}",
                    (XGMI_WIDTH_MIN, XGMI_WIDTH_MAX),
                ),
            ],
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
