#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the "current_node_power" key in amdsmi_get_npm_info()'s dict.

Uses the real compiled amdsmi package (via `from common.common import amdsmi`)
but mocks the ctypes call (`amdsmi_wrapper.amdsmi_get_npm_info`), so no
hardware or root is needed.

`current_node_power` is a new uint32 field in `amdsmi_npm_info_t` (after
`max_node_power_limit`) and follows the same UINT32_MAX -> "N/A" convention as
`"ubb_power_threshold"`.
"""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi


def _make_node_handle() -> "amdsmi.amdsmi_wrapper.amdsmi_node_handle":
    return amdsmi.amdsmi_wrapper.amdsmi_node_handle()


def _make_npm_info(
    current_node_power: int,
    limit: int = 0,
    status: int = 0,
    ubb_power_threshold: int = 0,
    max_node_power_limit: int = 0,
) -> "amdsmi.amdsmi_wrapper.amdsmi_npm_info_t":
    info = amdsmi.amdsmi_wrapper.amdsmi_npm_info_t()
    info.status = status
    info.limit = limit
    info.ubb_power_threshold = ubb_power_threshold
    info.max_node_power_limit = max_node_power_limit
    info.current_node_power = current_node_power
    return info


class TestNpmInfoCurrentNodePowerKey(unittest.TestCase):
    def _get_npm_info_with_mocked_c_call(
        self, current_node_power_raw: int, max_node_power_limit_raw: int = 0
    ):
        node_handle = _make_node_handle()

        def _fake_amdsmi_get_npm_info(_node_handle, info_ptr):
            info_ptr._obj.__init__()  # zero-init as the real ctypes.byref target would be
            populated = _make_npm_info(
                current_node_power=current_node_power_raw,
                max_node_power_limit=max_node_power_limit_raw,
            )
            ctypes.memmove(
                ctypes.addressof(info_ptr._obj),
                ctypes.addressof(populated),
                ctypes.sizeof(populated),
            )
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_npm_info", side_effect=_fake_amdsmi_get_npm_info
        ):
            return amdsmi.amdsmi_get_npm_info(node_handle)

    def test_current_node_power_key_present_with_concrete_value(self):
        result = self._get_npm_info_with_mocked_c_call(current_node_power_raw=5800)
        self.assertIn("current_node_power", result)
        self.assertEqual(result["current_node_power"], 5800)

    def test_current_node_power_sentinel_becomes_na(self):
        uint32_max = amdsmi.amdsmi_interface.MaxUIntegerTypes.UINT32_T
        result = self._get_npm_info_with_mocked_c_call(current_node_power_raw=uint32_max)
        self.assertEqual(result["current_node_power"], "N/A")

    def test_current_node_power_independent_of_max_node_power_limit(self):
        # current_node_power and max_node_power_limit are adjacent fields in
        # the struct; make sure the sentinel conversion is applied per-field,
        # not as an all-or-nothing pass keyed off one of them.
        uint64_max = amdsmi.amdsmi_interface.MaxUIntegerTypes.UINT64_T
        result = self._get_npm_info_with_mocked_c_call(
            current_node_power_raw=5800, max_node_power_limit_raw=uint64_max
        )
        self.assertEqual(result["current_node_power"], 5800)
        self.assertEqual(result["max_node_power_limit"], "N/A")

    def test_current_node_power_zero_is_not_treated_as_na(self):
        # 0 W is a legitimate (if unusual) reading and must not be conflated
        # with the sentinel/absent case.
        result = self._get_npm_info_with_mocked_c_call(current_node_power_raw=0)
        self.assertEqual(result["current_node_power"], 0)
