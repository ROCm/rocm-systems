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
"""GPU clock: set clock frequency, set PCI bandwidth."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuClock(unittest.TestCase):
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

    def test_set_clk_freq(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_clk_freq as it fails (Perm failure)."
            self.common.print(msg)
            self.skipTest(msg)

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # amdsmi_set_clk_freq() accepts only these specific string names.
            # AmdSmiClkType enum names (e.g. SYS/MEM/DF/SOC) are not accepted;
            # map only the supported ones and skip the rest.
            clk_type_str_map = {"SYS": "sclk", "MEM": "mclk", "DF": "fclk", "SOC": "socclk"}
            for clk_type_name, clk_type, clk_cond in common.CLK_TYPES:
                clk_type_str = clk_type_str_map.get(clk_type_name)
                if clk_type_str is None:
                    # No string mapping for this clock type; amdsmi_set_clk_freq
                    # would raise AmdSmiParameterException before reaching the API.
                    self.common.print(
                        f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_name}): skipped (no string mapping)"
                    )
                    continue
                # Set invalid clock frequency
                try:
                    freq_bitmask = 0x1234
                    msg = f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_str}, freq_bitmask={freq_bitmask}):"
                    ret = amdsmi.amdsmi_set_clk_freq(gpu, clk_type_str, freq_bitmask)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.FAIL)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.FAIL):
                        self.raise_exception = e

                # Get clock frequency info
                msg = f"\t### amdsmi_get_clk_freq(gpu={i}, clk_type={clk_type_name}):"
                try:
                    clk_freq_info = amdsmi.amdsmi_get_clk_freq(gpu, clk_type)
                    self.common.print(msg, clk_freq_info)
                    self.common.check_ret("", "", self.common.PASS)

                    current = clk_freq_info["current"]
                    num_supported = clk_freq_info["num_supported"]
                    frequencies = clk_freq_info["frequency"]
                    if num_supported == 0:
                        self.common.print(f"No supported frequencies for clk_type={clk_type_name}")
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, clk_cond):
                        self.raise_exception = e
                        continue

                for index in range(0, num_supported):
                    # Set clock frequency for each frequency supported
                    try:
                        freq_bitmask = frequencies[index]
                        msg = f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_str}, freq_bitmask={freq_bitmask}):"
                        ret = amdsmi.amdsmi_set_clk_freq(gpu, clk_type_str, freq_bitmask)
                        self.common.print(msg, ret)
                        self.common.check_ret("", "", self.common.PASS)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, clk_cond):
                            self.raise_exception = e
                            continue

                # Set clock frequency back
                try:
                    freq_bitmask = frequencies[current]
                    msg = f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_str}, freq_bitmask={freq_bitmask}):"
                    ret = amdsmi.amdsmi_set_clk_freq(gpu, clk_type_str, freq_bitmask)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, clk_cond):
                        self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_set_gpu_pci_bandwidth(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # Get current PCI bandwidth info
            msg = f"\t### amdsmi_get_gpu_pci_bandwidth(gpu={i}):"
            try:
                bandwidth_info = amdsmi.amdsmi_get_gpu_pci_bandwidth(gpu)
                self.common.print(msg, bandwidth_info)
                self.common.check_ret("", "", self.common.PASS)

                current_bandwidth_index = bandwidth_info["transfer_rate"]["current"]
                if current_bandwidth_index > 0:
                    bitmask = 1 << (current_bandwidth_index - 1)
                else:
                    bitmask = 1 << (current_bandwidth_index)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set PCI bandwidth
            msg = f"\t### amdsmi_set_gpu_pci_bandwidth(gpu={i}, bitmask={bitmask}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_pci_bandwidth(gpu, bitmask)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set back to original PCI bandwidth
            msg = f"\t### amdsmi_set_gpu_pci_bandwidth(gpu={i}, bitmask={bitmask}):"
            try:
                bitmask = 1 << (current_bandwidth_index)
                ret = amdsmi.amdsmi_set_gpu_pci_bandwidth(gpu, bitmask)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_get_clk_freq(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = (
                "\tSkipping test_get_clk_freq as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            )
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_clk_freq=amdsmi.amdsmi_get_clk_freq)
        return

    def test_get_clock_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_clock_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_clock_info=amdsmi.amdsmi_get_clock_info)
        return
