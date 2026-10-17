#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU energy count."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuEnergy(unittest.TestCase):
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

    def test_get_energy_count(self):
        self.common.print_func_name("")

        # Every ASIC is expected to report an energy count, so accept SUCCESS
        # only: a missing or malformed reading (eg. NOT_SUPPORTED or
        # UNEXPECTED_DATA) is a real failure, never a silently tolerated status.
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                msg = f"\t### amdsmi_get_energy_count(gpu={i}):"
                val = "N/A"
                with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.SUCCESS]):
                    val = amdsmi.amdsmi_get_energy_count(gpu)
                self.common.print(f"\t    data: {val}")
        return
