#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU PTL (PCIe telemetry logging): enable/disable set + read-back round-trip."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuPtl(unittest.TestCase):
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

    def test_set_gpu_ptl_state(self):
        """Setting PTL state must actually change the reported state.

        Regression guard: the set path used to write "1"/"0" to the
        ptl/ptl_enable sysfs node, which only accepts "enabled"/"disabled".
        The driver silently ignored the numeric write, so the state never
        changed even though the API reported success. GPUs that do not
        support PTL return a not-supported status and are skipped.
        """
        self.common.print_func_name("")

        mismatches = []
        tested = 0
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)

            # Read the original state; GPUs without PTL support are skipped
            # (check_ret treats not-supported codes as a non-failure).
            msg = f"\t### amdsmi_get_gpu_ptl_state(gpu={i}):"
            try:
                original = amdsmi.amdsmi_get_gpu_ptl_state(gpu)
                self.common.print(msg, original)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            tested += 1

            # Drive each state and confirm the read-back matches what was set.
            for desired in (True, False):
                set_msg = f"\t### amdsmi_set_gpu_ptl_state(gpu={i}, enable={desired}):"
                try:
                    amdsmi.amdsmi_set_gpu_ptl_state(gpu, int(desired))
                    self.common.print(set_msg, "")
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(set_msg, e, self.common.PASS):
                        self.raise_exception = e
                    continue

                read_back = amdsmi.amdsmi_get_gpu_ptl_state(gpu)
                self.common.print(f"\t### amdsmi_get_gpu_ptl_state(gpu={i}) after set:", read_back)
                self.common.check_ret("", "", self.common.PASS)
                if read_back != desired:
                    mismatches.append(
                        f"gpu({i}): set PTL state to {desired} but read back {read_back}"
                    )

            # Restore the original state; a genuine failure (not merely
            # not-supported) fails the test instead of being swallowed.
            restore_msg = f"\t### restore amdsmi_set_gpu_ptl_state(gpu={i}):"
            try:
                amdsmi.amdsmi_set_gpu_ptl_state(gpu, int(original))
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(restore_msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        if mismatches:
            self.fail("PTL set did not take effect:\n  " + "\n  ".join(mismatches))
        if tested == 0:
            self.skipTest("No PTL-capable GPUs found")
