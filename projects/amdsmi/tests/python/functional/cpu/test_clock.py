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

# amdsmi_set_cpu_xgmi_width reaches the ESMI IO path before the HSMP
# message-support check, so a wrong (GPU) handle is rejected with
# AMDSMI_STATUS_IO on a discrete-GPU host rather than NO_HSMP_MSG_SUP; on other
# hosts/APUs it is rejected with INVAL. All are valid rejections of a bogus
# handle - the negative portion only has to reject SUCCESS.
XGMI_WRONG_ACCEPT = [
    amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
    amdsmi.AmdSmiStatus.NOT_FOUND,
    amdsmi.AmdSmiStatus.IO,
    amdsmi.AmdSmiStatus.INVAL,
]

# amdsmi_set_cpu_xgmi_width and amdsmi_get_cpu_current_io_bandwidth answer a real
# CPU handle with INVAL where the target is absent on the platform: no xGMI width
# control (single-socket host / APU) for the former, and a non-existent IO link
# (several entries in IO_BW_ENCODINGS are not wired on every platform) for the
# latter. Treat INVAL as "target not present here" alongside the documented
# feature-absent statuses so the positive portion still runs everywhere without
# false failures.
CORRECT_ACCEPT_WITH_INVAL = [
    amdsmi.AmdSmiStatus.SUCCESS,
    amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
    amdsmi.AmdSmiStatus.INVAL,
]


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
        self.raise_exception = None
        self.common.amdsmi_smart_init()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_cpu_apb_disable(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_cpu_apb_disable, "amdsmi_cpu_apb_disable", arg_sets=[("pstate=0", (0,))]
        )
        return

    def test_cpu_apb_enable(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(amdsmi.amdsmi_cpu_apb_enable, "amdsmi_cpu_apb_enable")
        return

    def test_get_cpu_cclk_limit(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_cclk_limit, "amdsmi_get_cpu_cclk_limit"
        )
        return

    def test_get_cpu_core_current_freq_limit(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_core_current_freq_limit,
            "amdsmi_get_cpu_core_current_freq_limit",
            handle_kind="core",
        )
        return

    def test_get_cpu_current_io_bandwidth(self):
        self.common.print_func_name("")
        for i, cpu in enumerate(self.common.skip_without_cpu()):
            # Link names are only legal if the platform's encoding table lists them,
            # and that table is selected by HSMP protocol version.
            try:
                proto_ver = amdsmi.amdsmi_get_cpu_hsmp_proto_ver(cpu)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.skipTest(f"HSMP protocol version unavailable: {e}")
            links = common.io_bw_encodings_for_proto_ver(proto_ver)
            if not links:
                self.skipTest(f"HSMP protocol version {proto_ver} defines no IO link encodings")

            for encoding_name, encoding, encoding_cond in links:
                msg = f"\t### amdsmi_get_cpu_current_io_bandwidth(cpu={i}, encoding={encoding} encoding_name={encoding_name}):"
                try:
                    ret = amdsmi.amdsmi_get_cpu_current_io_bandwidth(cpu, encoding, encoding_name)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, encoding_cond):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_fclk_mclk(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_fclk_mclk, "amdsmi_get_cpu_fclk_mclk"
        )
        return

    def test_get_cpu_socket_current_active_freq_limit(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit,
            "amdsmi_get_cpu_socket_current_active_freq_limit",
        )
        return

    def test_get_cpu_socket_freq_range(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_freq_range, "amdsmi_get_cpu_socket_freq_range"
        )
        return

    def test_get_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level,
            "amdsmi_get_cpu_socket_lclk_dpm_level",
            arg_sets=[(f"nbio_id={nbio_id}", (nbio_id,)) for nbio_id in NBIO_IDS],
        )
        return

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        def positive(handles, failures):
            # Narrow the range, then hand P-state selection back to the firmware.
            # Note the parameter order is (min, max). There is no getter for the
            # DF P-state range, so enabling APB is the documented restore path.
            with self.common.status_sweep():
                for i, cpu in enumerate(handles):
                    set_msg = (
                        f"\t### amdsmi_set_cpu_df_pstate_range(cpu={i}, "
                        f"min_pstate=1, max_pstate={DF_PSTATE_MAX}):"
                    )
                    with self.common.expect_status(set_msg, self.common.CPU_CORRECT_ACCEPT):
                        amdsmi.amdsmi_set_cpu_df_pstate_range(cpu, 1, DF_PSTATE_MAX)
                    restore_msg = f"\t### amdsmi_cpu_apb_enable(cpu={i}):"
                    with self.common.expect_status(restore_msg, self.common.CPU_CORRECT_ACCEPT):
                        amdsmi.amdsmi_cpu_apb_enable(cpu)

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_df_pstate_range,
            "amdsmi_set_cpu_df_pstate_range",
            arg_sets=[(f"min_pstate=1, max_pstate={DF_PSTATE_MAX}", (1, DF_PSTATE_MAX))],
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
        )
        return

    def test_set_cpu_pcie_link_rate(self):
        self.common.print_func_name("")

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
                        with self.common.expect_status(msg, self.common.CPU_CORRECT_ACCEPT):
                            ret = amdsmi.amdsmi_set_cpu_pcie_link_rate(cpu, rate_ctrl)
                            if prev_mode is None:
                                prev_mode = int(ret)
                    if prev_mode is not None:
                        msg = (
                            f"\t### amdsmi_set_cpu_pcie_link_rate(cpu={i}, rate_ctrl={prev_mode}):"
                        )
                        with self.common.expect_status(msg, self.common.CPU_CORRECT_ACCEPT):
                            amdsmi.amdsmi_set_cpu_pcie_link_rate(cpu, prev_mode)

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_pcie_link_rate,
            "amdsmi_set_cpu_pcie_link_rate",
            arg_sets=[("rate_ctrl=0", (0,))],
            positive=positive,
        )
        return

    def test_set_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

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
                        with self.common.expect_status(get_msg, self.common.CPU_CORRECT_ACCEPT):
                            dpm = amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(cpu, nbio_id)
                            bounds["min"] = dpm["nbio_min_dpm_level"]
                            bounds["max"] = dpm["nbio_max_dpm_level"]
                        if "max" not in bounds:
                            continue  # getter did not succeed; nothing to set/restore
                        min_orig, max_orig = bounds["min"], bounds["max"]
                        # Pin to the top level, then restore the original bounds.
                        for min_val, max_val in ((max_orig, max_orig), (min_orig, max_orig)):
                            set_msg = (
                                f"\t### amdsmi_set_cpu_socket_lclk_dpm_level(cpu={i}, "
                                f"nbio_id={nbio_id}, min_val={min_val}, max_val={max_val}):"
                            )
                            with self.common.expect_status(set_msg, self.common.CPU_CORRECT_ACCEPT):
                                amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level(
                                    cpu, nbio_id, min_val, max_val
                                )

        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
            "amdsmi_set_cpu_socket_lclk_dpm_level",
            arg_sets=[("nbio_id=0, min_val=0, max_val=1", (0, 0, 1))],
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
            wrong_accept=XGMI_WRONG_ACCEPT,
            correct_accept=CORRECT_ACCEPT_WITH_INVAL,
        )
        return
