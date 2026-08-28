#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""CPU clock: set clock frequency, SOC P-state."""

import unittest

import common.common as common
from common.common import amdsmi


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
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_soc_pstate(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # Get current policy info
            msg = f"\t### amdsmi_get_soc_pstate(gpu={i}):"
            try:
                policy_info = amdsmi.amdsmi_get_soc_pstate(gpu)
                self.common.print(msg, "")
                self.common.check_ret("", "", self.common.PASS)

                num_supported = policy_info["num_supported"]
                if not isinstance(num_supported, int):
                    self.common.print(f"Cannot determine num_supported={num_supported}", "")
                    continue
                policy_id_current = policy_info["current_id"]
                if not isinstance(policy_id_current, int):
                    self.common.print(f"Cannot determine policy_id_current={policy_id_current}", "")
                    continue
                policy_id_orig = policy_info["policies"][policy_id_current]["policy_id"]
                if not isinstance(policy_id_orig, int):
                    self.common.print(f"Cannot determine orig policy_id={policy_id_orig}", "")
                    continue

                index = 0
                if num_supported >= 2:
                    if policy_id_current != 0:
                        index = 1
                policy_id = policy_info["policies"][index]["policy_id"]
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set SOC Pstate policy
            msg = f"\t### amdsmi_set_soc_pstate(gpu={i}):"
            try:
                ret = amdsmi.amdsmi_set_soc_pstate(gpu, policy_id)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set back to original policy
            msg = f"\t### amdsmi_set_soc_pstate(gpu={i}, policy_id={policy_id_orig}):"
            try:
                ret = amdsmi.amdsmi_set_soc_pstate(gpu, policy_id_orig)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_cpu_apb_disable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_cpu_apb_disable=amdsmi.amdsmi_cpu_apb_disable, pstate=0)
        return

    def test_cpu_apb_enable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_cpu_apb_enable=amdsmi.amdsmi_cpu_apb_enable)
        return

    def test_get_cpu_cclk_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_cclk_limit=amdsmi.amdsmi_get_cpu_cclk_limit)
        return

    def test_get_cpu_core_current_freq_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_core_current_freq_limit=amdsmi.amdsmi_get_cpu_core_current_freq_limit
        )
        return

    def test_get_cpu_current_io_bandwidth(self):
        self.common.print_func_name("")
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for encoding_name, encoding, encoding_cond in common.IO_BW_ENCODINGS:
                msg = f"\t### amdsmi_get_cpu_current_io_bandwidth(gpu={i}, encoding={encoding} encoding_name={encoding_name}):"
                try:
                    ret = amdsmi.amdsmi_get_cpu_current_io_bandwidth(gpu, encoding, encoding_name)
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
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_fclk_mclk=amdsmi.amdsmi_get_cpu_fclk_mclk)
        return

    def test_get_cpu_socket_current_active_freq_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_current_active_freq_limit=amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit
        )
        return

    def test_get_cpu_socket_freq_range(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_freq_range=amdsmi.amdsmi_get_cpu_socket_freq_range
        )
        return

    def test_get_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        # TODO nbio_id = 0
        nbio_id = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_lclk_dpm_level=amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level,
            nbio_id=nbio_id,
        )
        return

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_df_pstate_range as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO max_pstate = 0, min_pstate = 0
        max_pstate = 0
        min_pstate = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_df_pstate_range=amdsmi.amdsmi_set_cpu_df_pstate_range,
            max_pstate=max_pstate,
            min_pstate=min_pstate,
        )
        return

    def test_set_cpu_gmi3_link_width_range(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_gmi3_link_width_range as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO min_link_width = 0, max_link_width = 0
        min_link_width = 0
        max_link_width = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_gmi3_link_width_range=amdsmi.amdsmi_set_cpu_gmi3_link_width_range,
            min_link_width=min_link_width,
            max_link_width=max_link_width,
        )
        return

    # param modes

    def test_set_cpu_pcie_link_rate(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_pcie_link_rate as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO rate_ctrl = 0
        rate_ctrl = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_pcie_link_rate=amdsmi.amdsmi_set_cpu_pcie_link_rate, rate_ctrl=rate_ctrl
        )
        return

    def test_set_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_socket_lclk_dpm_level as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO nbio_id = 0, min_val = 0, max_val = 0
        nbio_id = 0
        min_val = 0
        max_val = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_socket_lclk_dpm_level=amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
            nbio_id=nbio_id,
            min_val=min_val,
            max_val=max_val,
        )
        return

    def test_set_cpu_xgmi_width(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_xgmi_width as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO min_width = 0, max_width = 0
        min_width = 0
        max_width = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_xgmi_width=amdsmi.amdsmi_set_cpu_xgmi_width,
            min_width=min_width,
            max_width=max_width,
        )
        return
