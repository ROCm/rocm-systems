#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the amdsmi_set_npm_limit() Python wrapper.

Uses the real compiled amdsmi package (via `from common.common import amdsmi`)
but mocks the ctypes call (`amdsmi_wrapper.amdsmi_set_npm_limit`), so no
hardware or root is needed. Covers the local parameter checks (node handle
type; limit an int, not bool, in 1..2**64-1) and the _check_res() status mapping.
"""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi


def _make_node_handle() -> "amdsmi.amdsmi_wrapper.amdsmi_node_handle":
    """A node handle instance satisfying the isinstance() check, independent
    of any real device/board path -- amdsmi_set_npm_limit()'s Python-level
    validation only checks the ctypes type, not the pointee's validity."""
    return amdsmi.amdsmi_wrapper.amdsmi_node_handle()


class TestAmdSmiSetNpmLimitParameterValidation(unittest.TestCase):
    def test_rejects_non_node_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit("not-a-node-handle", 100)

    def test_rejects_non_int_limit(self):
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit(node_handle, "100")

    def test_rejects_none_node_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit(None, 100)

    def test_rejects_float_limit(self):
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit(node_handle, 100.5)

    def test_rejects_bool_limit(self):
        # bool is an int subclass; True must not be written as a 1 W limit.
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit(node_handle, True)

    def test_rejects_negative_limit(self):
        # A Python int can represent negative values a raw ctypes.c_uint64()
        # cannot (it would silently wrap modulo 2**64); the explicit range
        # check must reject this before ever reaching ctypes.
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit(node_handle, -1)

    def test_rejects_limit_over_uint64_max(self):
        node_handle = _make_node_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_npm_limit(node_handle, 0xFFFFFFFFFFFFFFFF + 1)


class TestAmdSmiSetNpmLimitStatusMapping(unittest.TestCase):
    def test_success_returns_none_and_forwards_args(self):
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_limit",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS,
        ) as mocked_call:
            result = amdsmi.amdsmi_set_npm_limit(node_handle, 250)

        self.assertIsNone(result)
        mocked_call.assert_called_once()
        called_node_handle, called_limit = mocked_call.call_args[0]
        self.assertIs(called_node_handle, node_handle)
        self.assertIsInstance(called_limit, ctypes.c_uint64)
        self.assertEqual(called_limit.value, 250)

    def test_no_perm_status_raises_library_exception_with_code(self):
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_limit",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM,
        ):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_set_npm_limit(node_handle, 250)
        self.assertEqual(
            ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
        )

    def test_not_supported_status_raises_library_exception_with_code(self):
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_limit",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
        ):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_set_npm_limit(node_handle, 250)
        self.assertEqual(
            ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        )

    def test_inval_status_raises_library_exception_with_code(self):
        node_handle = _make_node_handle()
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_npm_limit",
            return_value=amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL,
        ):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_set_npm_limit(node_handle, 250)
        self.assertEqual(ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL)
