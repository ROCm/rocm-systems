#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the amdsmi_get/set_npm_balancing_mode() Python wrappers.

Uses the real compiled amdsmi package (resolved the same way test_check_res.py
and test_apu_metrics.py do, via `from common.common import amdsmi`), but mocks
the underlying ctypes calls (`amdsmi_wrapper.amdsmi_get_npm_balancing_mode` /
`amdsmi_wrapper.amdsmi_set_npm_balancing_mode`) so no real hardware/root access
is required -- these tests exercise only the Python-level parameter
validation, PB/FB/N/A string mapping, and status-to-exception mapping added
around the new C API, per the frozen contract:

    def amdsmi_get_npm_balancing_mode(node_handle: processor_handle_t) -> str

which validates node_handle, calls amdsmi_wrapper.amdsmi_get_npm_balancing_mode
(always AMDSMI_STATUS_SUCCESS per the C contract), and maps the returned
amdsmi_npm_balancing_mode_t value to "PB" / "FB" / "N/A" (INVALID, e.g. when
NPM is disabled on the node); and

    def amdsmi_set_npm_balancing_mode(node_handle: processor_handle_t, mode: str) -> None

which validates node_handle and that mode is one of "PB" / "FB", then calls
amdsmi_wrapper.amdsmi_set_npm_balancing_mode(node_handle,
_STR_TO_NPM_BALANCING_MODE[mode]) and funnels the returned status code through
the standard _check_res() mapping (AmdSmiLibraryException with the propagated
error code on any non-SUCCESS status, e.g. AMDSMI_STATUS_NO_PERM or
AMDSMI_STATUS_NOT_SUPPORTED; no exception on AMDSMI_STATUS_SUCCESS).
"""

from __future__ import annotations

import unittest
from unittest import mock

from common.common import amdsmi


def _make_node_handle() -> "amdsmi.amdsmi_wrapper.amdsmi_node_handle":
    """A node handle instance satisfying the isinstance() check, independent
    of any real device/board path -- both wrappers' Python-level validation
    only checks the ctypes type, not the pointee's validity."""
    return amdsmi.amdsmi_wrapper.amdsmi_node_handle()


class TestAmdSmiGetNpmBalancingModeParameterValidation(unittest.TestCase):
    def test_rejects_non_node_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_npm_balancing_mode("not-a-node-handle")

    def test_rejects_none_node_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_npm_balancing_mode(None)


class TestAmdSmiGetNpmBalancingModeStringMapping(unittest.TestCase):
    def _get_with_mocked_c_call(self, mode_value: int):
        node_handle = _make_node_handle()

        def _fake_amdsmi_get_npm_balancing_mode(_node_handle, mode_ptr):
            mode_ptr._obj.value = mode_value
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_get_npm_balancing_mode",
            side_effect=_fake_amdsmi_get_npm_balancing_mode,
        ) as mocked_call:
            result = amdsmi.amdsmi_get_npm_balancing_mode(node_handle)
        return result, mocked_call

    def test_power_balancing_maps_to_pb(self):
        result, mocked_call = self._get_with_mocked_c_call(
            amdsmi.AmdSmiNpmBalancingMode.POWER_BALANCING
        )
        self.assertEqual(result, "PB")
        mocked_call.assert_called_once()

    def test_frequency_balancing_maps_to_fb(self):
        result, _ = self._get_with_mocked_c_call(amdsmi.AmdSmiNpmBalancingMode.FREQUENCY_BALANCING)
        self.assertEqual(result, "FB")

    def test_invalid_maps_to_na(self):
        # AMDSMI_NPM_BALANCING_MODE_INVALID -- e.g. NPM disabled on the node --
        # is always AMDSMI_STATUS_SUCCESS per the C contract, never an error.
        result, _ = self._get_with_mocked_c_call(amdsmi.AmdSmiNpmBalancingMode.INVALID)
        self.assertEqual(result, "N/A")


class TestAmdSmiSetNpmBalancingModeParameterValidation(unittest.TestCase):
    def test_rejects_non_node_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_balancing_mode("not-a-node-handle", "PB")

    def test_rejects_none_node_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_balancing_mode(None, "PB")

    def test_rejects_unknown_mode_string(self):
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_balancing_mode(node_handle, "BOGUS")

    def test_rejects_lowercase_mode_string(self):
        # Case matters -- "pb"/"fb" are not accepted, only the exact "PB"/"FB".
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_balancing_mode(node_handle, "pb")

    def test_rejects_none_mode(self):
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_balancing_mode(node_handle, None)


class TestAmdSmiSetNpmBalancingModeStatusMapping(unittest.TestCase):
    def test_success_returns_none_and_forwards_args(self):
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_balancing_mode",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS,
        ) as mocked_call:
            result = amdsmi.amdsmi_set_npm_balancing_mode(node_handle, "FB")

        self.assertIsNone(result)
        mocked_call.assert_called_once()
        called_node_handle, called_mode = mocked_call.call_args[0]
        self.assertIs(called_node_handle, node_handle)
        self.assertEqual(called_mode, amdsmi.AmdSmiNpmBalancingMode.FREQUENCY_BALANCING)

    def test_no_perm_status_raises_library_exception_with_code(self):
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_balancing_mode",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM,
        ):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_set_npm_balancing_mode(node_handle, "PB")
        self.assertEqual(
            ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
        )

    def test_not_supported_status_raises_library_exception_with_code(self):
        # The balancing-mode-specific divergence from amdsmi_set_npm_limit():
        # NPM disabled on the node must surface as AMDSMI_STATUS_NOT_SUPPORTED,
        # not AMDSMI_STATUS_INVAL.
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_balancing_mode",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
        ):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_set_npm_balancing_mode(node_handle, "PB")
        self.assertEqual(
            ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        )
