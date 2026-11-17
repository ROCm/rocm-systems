#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for amdsmi_interface.amdsmi_get_ainic_cper_entries()

Tests the Python wrapper for AI-NIC CPER retrieval without real hardware.
Mocks the ctypes layer to verify parameter handling, exception raising, and
data parsing.
"""

import ctypes
import importlib
import os
import struct
import sys
import types
import unittest
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_PY_INTERFACE = os.path.join(_REPO_ROOT, "py-interface")

_PKG = "amdsmi_source_under_test"


def _load_source_interface():
    if os.path.isfile(os.path.join(_PY_INTERFACE, "amdsmi_interface.py")):
        if _PKG not in sys.modules:
            pkg = types.ModuleType(_PKG)
            pkg.__path__ = [_PY_INTERFACE]
            sys.modules[_PKG] = pkg
        return (
            importlib.import_module(f"{_PKG}.amdsmi_interface"),
            importlib.import_module(f"{_PKG}.amdsmi_exception"),
            importlib.import_module(f"{_PKG}.amdsmi_wrapper"),
            False,
        )
    try:
        from amdsmi import (
            amdsmi_interface as installed_interface,
            amdsmi_exception as installed_exception,
            amdsmi_wrapper as installed_wrapper,
        )
    except (ModuleNotFoundError, ImportError):
        return None, None, None, False
    return (installed_interface, installed_exception, installed_wrapper, True)


(amdsmi_interface, amdsmi_exception, amdsmi_wrapper, _loaded_from_install) = (
    _load_source_interface()
)

_nic_symbol_missing = (
    _loaded_from_install
    and amdsmi_wrapper is not None
    and not hasattr(amdsmi_wrapper, "amdsmi_get_nic_cper_entries")
)


def _patch_nic_symbol():
    return patch.object(amdsmi_wrapper, "amdsmi_get_nic_cper_entries", create=True)


def _set_out(ref, value):
    ref._obj.value = value


@unittest.skipIf(amdsmi_interface is None, "amdsmi_interface not available")
@unittest.skipIf(_nic_symbol_missing, "installed amdsmi build predates the NIC CPER API")
class TestAinicCperInterface(unittest.TestCase):
    """Test amdsmi_interface.amdsmi_get_ainic_cper_entries()"""

    def setUp(self):
        self.processor_handle = amdsmi_wrapper.amdsmi_processor_handle(0xABCD)

    def test_invalid_handle_type(self):
        with self.assertRaises(amdsmi_exception.AmdSmiParameterException):
            amdsmi_interface.amdsmi_get_ainic_cper_entries("not_a_handle", 0xFFFF)

    @_patch_nic_symbol()
    def test_not_supported_error(self, mock_nic_cper):
        mock_nic_cper.return_value = amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        with self.assertRaises(amdsmi_exception.AmdSmiLibraryException) as ctx:
            amdsmi_interface.amdsmi_get_ainic_cper_entries(self.processor_handle, 0xFFFF)

        self.assertEqual(ctx.exception.get_error_code(), amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED)

    @_patch_nic_symbol()
    def test_file_not_found_error(self, mock_nic_cper):
        """FILE_NOT_FOUND is raised when pds_core debugfs path is absent."""
        mock_nic_cper.return_value = amdsmi_wrapper.AMDSMI_STATUS_FILE_NOT_FOUND

        with self.assertRaises(amdsmi_exception.AmdSmiLibraryException) as ctx:
            amdsmi_interface.amdsmi_get_ainic_cper_entries(self.processor_handle, 0xFFFF)

        self.assertEqual(
            ctx.exception.get_error_code(), amdsmi_wrapper.AMDSMI_STATUS_FILE_NOT_FOUND
        )

    @_patch_nic_symbol()
    def test_no_perm_error(self, mock_nic_cper):
        mock_nic_cper.return_value = amdsmi_wrapper.AMDSMI_STATUS_NO_PERM

        with self.assertRaises(amdsmi_exception.AmdSmiLibraryException) as ctx:
            amdsmi_interface.amdsmi_get_ainic_cper_entries(self.processor_handle, 0xFFFF)

        self.assertEqual(ctx.exception.get_error_code(), amdsmi_wrapper.AMDSMI_STATUS_NO_PERM)

    @_patch_nic_symbol()
    def test_success_no_entries(self, mock_nic_cper):
        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            _set_out(entry_count, 0)
            _set_out(buf_size, 0)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_nic_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_ainic_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(len(entries), 0)
        self.assertEqual(new_cursor, 0)
        self.assertEqual(len(cper_data), 0)

    @_patch_nic_symbol()
    def test_success_with_entry(self, mock_nic_cper):
        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            cper_header = bytearray(128)
            cper_header[0:4] = b"CPER"
            struct.pack_into("<H", cper_header, 4, 0x0100)  # revision
            struct.pack_into("<I", cper_header, 6, 0xFFFFFFFF)  # signature_end
            struct.pack_into("<H", cper_header, 10, 2)  # sec_cnt
            struct.pack_into("<I", cper_header, 12, 2)  # AMDSMI_CPER_SEV_NON_FATAL_CORRECTED
            struct.pack_into("<I", cper_header, 20, 128)  # record_length
            # Timestamp fields use raw integer values (not BCD) in this code path.
            cper_header[24] = 0  # seconds = 0
            cper_header[25] = 30  # minutes = 30
            cper_header[26] = 12  # hours = 12
            cper_header[27] = 1  # flag
            cper_header[28] = 17  # day = 17
            cper_header[29] = 11  # month = 11
            cper_header[30] = 26  # year = 26 → 2026
            cper_header[31] = 20  # century (unused in this path)

            ctypes.memmove(buf, bytes(cper_header), 128)
            hdrs[0] = ctypes.cast(buf, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))

            _set_out(entry_count, 1)
            _set_out(buf_size, 128)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_nic_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_ainic_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["error_severity"], "non_fatal_corrected")
        self.assertEqual(entry["source"], "ainic")
        self.assertEqual(entry["serial_number"], "")
        self.assertEqual(entry["timestamp"], "2026/11/17 12:30:00")

    @_patch_nic_symbol()
    def test_entries_marked_as_ainic(self, mock_nic_cper):
        """All entries carry source='ainic'."""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            cper_header = bytearray(128)
            cper_header[0:4] = b"CPER"
            struct.pack_into("<I", cper_header, 6, 0xFFFFFFFF)
            struct.pack_into("<I", cper_header, 20, 128)
            ctypes.memmove(buf, bytes(cper_header), 128)
            hdrs[0] = ctypes.cast(buf, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))
            _set_out(entry_count, 1)
            _set_out(buf_size, 128)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_nic_cper.side_effect = mock_impl

        entries, _, _, _ = amdsmi_interface.amdsmi_get_ainic_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )
        self.assertEqual(entries[0]["source"], "ainic")

    @_patch_nic_symbol()
    def test_more_data_pagination(self, mock_nic_cper):
        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor_ref):
            _set_out(entry_count, 0)
            _set_out(buf_size, 0)
            _set_out(cursor_ref, 42)
            return amdsmi_wrapper.AMDSMI_STATUS_MORE_DATA

        mock_nic_cper.side_effect = mock_impl

        entries, new_cursor, _, status = amdsmi_interface.amdsmi_get_ainic_cper_entries(
            self.processor_handle, severity_mask=0xFFFF, cursor=0
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_MORE_DATA)
        self.assertEqual(new_cursor, 42)

    @_patch_nic_symbol()
    def test_custom_buffer_size(self, mock_nic_cper):
        seen = {}

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            seen["buf_len"] = len(buf)
            seen["buf_size_in"] = buf_size._obj.value
            _set_out(entry_count, 0)
            _set_out(buf_size, 0)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_nic_cper.side_effect = mock_impl

        custom = 2 * 1048576
        amdsmi_interface.amdsmi_get_ainic_cper_entries(
            self.processor_handle, severity_mask=0xFFFF, buffer_size=custom
        )

        self.assertEqual(seen["buf_len"], custom)
        self.assertEqual(seen["buf_size_in"], custom)
