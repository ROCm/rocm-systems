#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_gpu_activity N/A handling (hardware-free)."""

import unittest
from unittest import mock

from common.common import amdsmi

_KEYS = ("gfx_activity", "umc_activity", "mm_activity")


def _activity_with(value: int) -> dict:
    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

    def _stub(_handle, info_ptr):
        for key in _KEYS:
            setattr(info_ptr._obj, key, value)
        return 0

    with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_activity", _stub):
        return amdsmi.amdsmi_interface.amdsmi_get_gpu_activity(handle)


class TestGpuActivityNa(unittest.TestCase):
    def test_uint32_max_is_na(self):
        self.assertEqual(_activity_with(0xFFFFFFFF), dict.fromkeys(_KEYS, "N/A"))

    def test_uint16_max_is_na_for_older_libraries(self):
        self.assertEqual(_activity_with(0xFFFF), dict.fromkeys(_KEYS, "N/A"))

    def test_non_na_values_pass_through(self):
        for value in (0, 50, 100, 0xFFFE, 0xFFFFFFFE):
            self.assertEqual(_activity_with(value), dict.fromkeys(_KEYS, value))


if __name__ == "__main__":
    unittest.main()
