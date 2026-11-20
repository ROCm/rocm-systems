#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU getter handle-validation unit tests (hardware-free)."""

import unittest

from common.common import amdsmi

_GETTERS = (
    "amdsmi_get_cpu_apb_status",
    "amdsmi_get_cpu_xgmi_width",
    "amdsmi_get_cpu_df_pstate_range",
)


class TestCpuGetterHandleValidation(unittest.TestCase):
    """CPU getters reject non-handle arguments before any C call."""

    @classmethod
    def setUpClass(cls):
        # The default /opt/rocm install can predate these getters; the check this
        # file makes is only meaningful against a build that has them (point
        # AMDSMI_PATH at a staged build of this branch).
        missing = [name for name in _GETTERS if not hasattr(amdsmi, name)]
        if missing:
            raise unittest.SkipTest(
                f"installed amdsmi lacks {', '.join(missing)}; "
                "set AMDSMI_PATH to a build of this branch to run"
            )

    def test_apb_status_rejects_non_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_cpu_apb_status(object())

    def test_xgmi_width_rejects_non_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_cpu_xgmi_width(object())

    def test_df_pstate_range_rejects_non_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_cpu_df_pstate_range(object())
