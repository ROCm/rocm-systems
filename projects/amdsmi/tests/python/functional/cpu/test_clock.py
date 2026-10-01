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
        self.raise_exception = None
        self.common.amdsmi_smart_init()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_cpu_apb_disable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(amdsmi_cpu_apb_disable=amdsmi.amdsmi_cpu_apb_disable, pstate=0)
        return

    def test_cpu_apb_enable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(amdsmi_cpu_apb_enable=amdsmi.amdsmi_cpu_apb_enable)
        return

    def test_get_cpu_cclk_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(amdsmi_get_cpu_cclk_limit=amdsmi.amdsmi_get_cpu_cclk_limit)
        return

    def test_get_cpu_core_current_freq_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU_Core(
            amdsmi_get_cpu_core_current_freq_limit=amdsmi.amdsmi_get_cpu_core_current_freq_limit
        )
        return

    def test_get_cpu_current_io_bandwidth(self):
        self.common.print_func_name("")
        for i, cpu in enumerate(self.common.skip_without_cpu()):
            for encoding_name, encoding, encoding_cond in common.IO_BW_ENCODINGS:
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
        self.common.Test_API_Per_CPU(amdsmi_get_cpu_fclk_mclk=amdsmi.amdsmi_get_cpu_fclk_mclk)
        return

    def test_get_cpu_socket_current_active_freq_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_socket_current_active_freq_limit=amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit
        )
        return

    def test_get_cpu_socket_freq_range(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_CPU(
            amdsmi_get_cpu_socket_freq_range=amdsmi.amdsmi_get_cpu_socket_freq_range
        )
        return

    def test_get_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        for i, cpu in enumerate(self.common.skip_without_cpu()):
            for nbio_id in NBIO_IDS:
                msg = f"\t### amdsmi_get_cpu_socket_lclk_dpm_level(cpu={i}, nbio_id={nbio_id}):"
                try:
                    ret = amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(cpu, nbio_id)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
                self.common.print("")

        if self.raise_exception:
            raise self.raise_exception
        return

    def _call(self, msg, func, *args):
        """Run one set call, record an unexpected status, and report success."""
        try:
            ret = func(*args)
            self.common.print(msg, ret)
            self.common.check_ret("", "", self.common.PASS)
            return True
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
            return False

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        for i, cpu in enumerate(self.common.skip_without_cpu()):
            # Narrow the range, then hand P-state selection back to the firmware.
            # Note the parameter order is (min, max) - the previous placeholder
            # version passed max first, which programs the inverse range.
            self._call(
                f"\t### amdsmi_set_cpu_df_pstate_range(cpu={i}, "
                f"min_pstate=1, max_pstate={DF_PSTATE_MAX}):",
                amdsmi.amdsmi_set_cpu_df_pstate_range,
                cpu,
                1,
                DF_PSTATE_MAX,
            )

            # There is no getter for the DF P-state range, so the original bounds
            # cannot be read back. Enabling APB is the documented way to return
            # P-state selection to automatic.
            self._call(f"\t### amdsmi_cpu_apb_enable(cpu={i}):", amdsmi.amdsmi_cpu_apb_enable, cpu)
            self.common.print("")

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_gmi3_link_width_range(self):
        self.common.print_func_name("")

        for i, cpu in enumerate(self.common.skip_without_cpu()):
            # Narrow, then widen back. No getter exists, so the restore target is
            # the full range rather than the range found on entry.
            for min_link_width, max_link_width in (
                (1, GMI3_LINK_WIDTH_MAX),
                (0, GMI3_LINK_WIDTH_MAX),
            ):
                self._call(
                    f"\t### amdsmi_set_cpu_gmi3_link_width_range(cpu={i}, "
                    f"min_link_width={min_link_width}, max_link_width={max_link_width}):",
                    amdsmi.amdsmi_set_cpu_gmi3_link_width_range,
                    cpu,
                    min_link_width,
                    max_link_width,
                )
            self.common.print("")

        if self.raise_exception:
            raise self.raise_exception
        return

    # param modes

    def test_set_cpu_pcie_link_rate(self):
        self.common.print_func_name("")

        for i, cpu in enumerate(self.common.skip_without_cpu()):
            # The setter reports the mode it replaced, which is the only way to
            # read the current rate control value; keep the first one to restore.
            prev_mode = None
            for rate_ctrl in PCIE_RATE_CTRLS:
                msg = f"\t### amdsmi_set_cpu_pcie_link_rate(cpu={i}, rate_ctrl={rate_ctrl}):"
                try:
                    ret = amdsmi.amdsmi_set_cpu_pcie_link_rate(cpu, rate_ctrl)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                    if prev_mode is None:
                        prev_mode = int(ret)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e

            if prev_mode is not None:
                self._call(
                    f"\t### amdsmi_set_cpu_pcie_link_rate(cpu={i}, rate_ctrl={prev_mode}):",
                    amdsmi.amdsmi_set_cpu_pcie_link_rate,
                    cpu,
                    prev_mode,
                )
            self.common.print("")

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        for i, cpu in enumerate(self.common.skip_without_cpu()):
            for nbio_id in NBIO_IDS:
                # Read the current bounds so they can be put back afterwards.
                msg = f"\t### amdsmi_get_cpu_socket_lclk_dpm_level(cpu={i}, nbio_id={nbio_id}):"
                try:
                    dpm = amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(cpu, nbio_id)
                    self.common.print(msg, dpm)
                    self.common.check_ret("", "", self.common.PASS)
                    min_orig = dpm["nbio_min_dpm_level"]
                    max_orig = dpm["nbio_max_dpm_level"]
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
                    continue

                # Pin to the top level, then restore the original bounds.
                for min_val, max_val in ((max_orig, max_orig), (min_orig, max_orig)):
                    self._call(
                        f"\t### amdsmi_set_cpu_socket_lclk_dpm_level(cpu={i}, "
                        f"nbio_id={nbio_id}, min_val={min_val}, max_val={max_val}):",
                        amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
                        cpu,
                        nbio_id,
                        min_val,
                        max_val,
                    )
                self.common.print("")

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_xgmi_width(self):
        self.common.print_func_name("")

        for i, cpu in enumerate(self.common.skip_without_cpu()):
            # Narrow to the top lane count only, then reopen the full range. As
            # with GMI3 there is no getter, so full range is the restore target.
            for min_width, max_width in (
                (XGMI_WIDTH_MAX, XGMI_WIDTH_MAX),
                (XGMI_WIDTH_MIN, XGMI_WIDTH_MAX),
            ):
                self._call(
                    f"\t### amdsmi_set_cpu_xgmi_width(cpu={i}, "
                    f"min_width={min_width}, max_width={max_width}):",
                    amdsmi.amdsmi_set_cpu_xgmi_width,
                    cpu,
                    min_width,
                    max_width,
                )
            self.common.print("")

        if self.raise_exception:
            raise self.raise_exception
        return
