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
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_energy_count(self):
        self.common.print_func_name("")

        # Run every check even if an earlier one fails, then report all failures,
        # each labelled with the check that produced it.
        checks = [
            ("None input", self._check_none_is_rejected),
            ("null handle", self._check_null_handle_is_rejected),
            ("unregistered handle", self._check_unregistered_handle_is_rejected),
            ("real GPU handles", self._check_every_gpu_reports_energy),
        ]
        failures = []
        for name, check in checks:
            try:
                check()
            except Exception as exc:
                failures.append(f"[{name}] {common.failure_text(exc)}")

        if failures:
            raise AssertionError("\n\n".join(failures))

    def _check_none_is_rejected(self):
        # The Python wrapper rejects a non-handle before any native call.
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_energy_count(None)
        self.common.print(
            "\t### amdsmi_get_energy_count(gpu=None):\n"
            "\tTEST SUCCESS, raised AmdSmiParameterException as expected"
        )

    def _check_null_handle_is_rejected(self):
        null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        msg = "\t### amdsmi_get_energy_count(gpu=null handle):"
        with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.INVAL]):
            amdsmi.amdsmi_get_energy_count(null_handle)

    def _check_unregistered_handle_is_rejected(self):
        # A non-null pointer the library never issued.
        unregistered_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle(0xDEADBEEF)
        msg = "\t### amdsmi_get_energy_count(gpu=unregistered handle):"
        with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.NOT_FOUND]):
            amdsmi.amdsmi_get_energy_count(unregistered_handle)

    def _check_every_gpu_reports_energy(self):
        # Every ASIC is expected to report an energy count, so accept SUCCESS
        # only: a missing or malformed reading (eg. NOT_SUPPORTED or
        # UNEXPECTED_DATA) is a real failure, never a silently tolerated status.
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                msg = f"\t### amdsmi_get_energy_count(gpu={i}):"
                energy = "N/A"
                with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.SUCCESS]):
                    energy = amdsmi.amdsmi_get_energy_count(gpu)
                self.common.print(f"\t    data: {energy}")
