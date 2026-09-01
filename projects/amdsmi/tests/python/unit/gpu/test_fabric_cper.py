#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for amdsmi_interface.amdsmi_get_fabric_cper_entries()

Tests the Python wrapper for fabric CPER retrieval without requiring real UALoE hardware.
Mocks the ctypes layer to verify parameter handling, exception raising, and data parsing.
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


# Synthetic package name for the in-tree sources. The ``py-interface`` directory
# is not a valid identifier, and its ``__init__.py`` pulls in the build-generated
# ``_version`` module, so the submodules are imported directly under this name.
# It is deliberately not ``amdsmi``: shadowing that would break sibling suites.
_PKG = "amdsmi_source_under_test"


def _load_source_interface():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [_PY_INTERFACE]
        sys.modules[_PKG] = pkg
    try:
        return (
            importlib.import_module(f"{_PKG}.amdsmi_interface"),
            importlib.import_module(f"{_PKG}.amdsmi_exception"),
            importlib.import_module(f"{_PKG}.amdsmi_wrapper"),
            importlib.import_module(f"{_PKG}.amdsmi_interface_utils"),
        )
    except Exception:
        for name in [n for n in list(sys.modules) if n == _PKG or n.startswith(_PKG + ".")]:
            del sys.modules[name]
        return None, None, None, None


amdsmi_interface, amdsmi_exception, amdsmi_wrapper, amdsmi_interface_utils = (
    _load_source_interface()
)


def _patch_fabric_symbol():
    """Patch the C entry point the interface resolves through the wrapper module.

    ``create=True`` because the symbol is only bound when the loaded
    libamd_smi.so exports it, which is not the case in a source-only checkout.
    """
    return patch.object(amdsmi_wrapper, "amdsmi_get_fabric_cper_entries", create=True)


def _set_out(ref, value):
    """Assign through a ``ctypes.byref()`` out-parameter as the C library would."""
    ref._obj.value = value


@unittest.skipIf(amdsmi_interface is None, "amdsmi_interface not available")
class TestFabricCperInterface(unittest.TestCase):
    """Test amdsmi_interface.amdsmi_get_fabric_cper_entries()"""

    def setUp(self):
        self.processor_handle = amdsmi_wrapper.amdsmi_processor_handle(0x1234)

    def test_invalid_handle_type(self):
        """Test exception on invalid processor_handle type"""
        with self.assertRaises(amdsmi_exception.AmdSmiParameterException):
            amdsmi_interface.amdsmi_get_fabric_cper_entries("not_a_handle", 0xFFFF)

    @_patch_fabric_symbol()
    def test_not_supported_error(self, mock_fabric_cper):
        """Test NOT_SUPPORTED error handling (UALoE unavailable)"""
        mock_fabric_cper.return_value = amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        with self.assertRaises(amdsmi_exception.AmdSmiLibraryException) as context:
            amdsmi_interface.amdsmi_get_fabric_cper_entries(
                self.processor_handle, severity_mask=0xFFFF
            )

        self.assertEqual(
            context.exception.get_error_code(), amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        )

    @_patch_fabric_symbol()
    def test_success_no_entries(self, mock_fabric_cper):
        """Test successful call with no entries returned"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            _set_out(entry_count, 0)
            _set_out(buf_size, 0)
            _set_out(cursor, 0)  # 0 means no more data
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(len(entries), 0)
        self.assertEqual(new_cursor, 0)
        self.assertEqual(len(cper_data), 0)

    @_patch_fabric_symbol()
    def test_success_with_entries(self, mock_fabric_cper):
        """Test successful call with fabric CPER entries"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            # One fabric CPER entry, laid out by byte offset into the 128-byte
            # amdsmi_cper_hdr_t.
            cper_header = bytearray(128)
            cper_header[0:4] = b"CPER"  # signature
            struct.pack_into("<H", cper_header, 4, 0x0100)  # revision
            struct.pack_into("<I", cper_header, 6, 0xFFFFFFFF)  # signature_end
            struct.pack_into("<H", cper_header, 10, 1)  # sec_cnt
            struct.pack_into("<I", cper_header, 12, 1)  # AMDSMI_CPER_SEV_FATAL
            struct.pack_into("<I", cper_header, 16, 0x01)  # valid_bits: timestamp
            struct.pack_into("<I", cper_header, 20, 128)  # record_length, no payload
            # Timestamp: 2026-08-24 12:00:00, as the Python layer reads it.
            cper_header[24] = 0  # seconds
            cper_header[25] = 0  # minutes
            cper_header[26] = 12  # hours
            cper_header[27] = 1  # flag
            cper_header[28] = 24  # day
            cper_header[29] = 8  # month
            cper_header[30] = 26  # year
            cper_header[31] = 20  # century
            cper_header[96:100] = b"IFoE"  # creator_id
            # notify_type at 72:88 stays all zeros: the IFoE placeholder GUID.

            ctypes.memmove(buf, bytes(cper_header), 128)
            hdrs[0] = ctypes.cast(buf, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))

            _set_out(entry_count, 1)
            _set_out(buf_size, 128)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(len(entries), 1)
        self.assertEqual(new_cursor, 0)

        entry = entries[0]
        self.assertEqual(entry["error_severity"], "fatal")
        self.assertEqual(entry["timestamp"], "2026/08/24 12:00:00")
        self.assertEqual(entry["signature"], b"CPER")
        self.assertEqual(entry["revision"], 0x0100)
        self.assertEqual(entry["signature_end"], "0xffffffff")
        self.assertEqual(entry["sec_cnt"], 1)
        self.assertEqual(entry["record_length"], 128)

    @_patch_fabric_symbol()
    def test_more_data_pagination(self, mock_fabric_cper):
        """Test MORE_DATA status and cursor pagination"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor_ref):
            _set_out(entry_count, 0)
            _set_out(buf_size, 0)
            _set_out(cursor_ref, 100)
            return amdsmi_wrapper.AMDSMI_STATUS_MORE_DATA

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF, cursor=0
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_MORE_DATA)
        self.assertEqual(new_cursor, 100)

    @_patch_fabric_symbol()
    def test_custom_buffer_size(self, mock_fabric_cper):
        """buffer_size sizes both the data buffer and the declared buf_size"""
        seen = {}

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            seen["buf_len"] = len(buf)
            seen["buf_size_in"] = buf_size._obj.value
            _set_out(entry_count, 0)
            _set_out(buf_size, 0)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        custom_buffer_size = 2 * 1048576
        _entries, _new_cursor, _cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF, buffer_size=custom_buffer_size
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(seen["buf_len"], custom_buffer_size)
        self.assertEqual(seen["buf_size_in"], custom_buffer_size)

    @_patch_fabric_symbol()
    def test_fabric_entries_are_marked_as_fabric(self, mock_fabric_cper):
        """Fabric entries carry an explicit source marker, not a sniffable GUID string"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            cper_header = bytearray(128)
            cper_header[0:4] = b"CPER"
            struct.pack_into("<H", cper_header, 4, 0x0100)
            struct.pack_into("<I", cper_header, 6, 0xFFFFFFFF)
            struct.pack_into("<H", cper_header, 10, 1)
            struct.pack_into("<I", cper_header, 12, 1)  # FATAL
            struct.pack_into("<I", cper_header, 20, 128)
            cper_header[72:88] = bytes(16)  # notify_type: IFoE placeholder GUID
            cper_header[96:100] = b"IFoE"

            ctypes.memmove(buf, bytes(cper_header), 128)
            hdrs[0] = ctypes.cast(buf, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))

            _set_out(entry_count, 1)
            _set_out(buf_size, 128)
            _set_out(cursor, 0)
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        entries, _new_cursor, _cper_data, _status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source"], "fabric")

        # The IFoE placeholder GUID maps to no known notify type, so the string
        # form is unusable for classification - hence the source marker above.
        self.assertEqual(
            entries[0]["notify_type"], amdsmi_interface_utils._notifyTypeToString(bytes(16))
        )
        self.assertEqual(entries[0]["notify_type"], "Unknown")


if __name__ == "__main__":
    unittest.main()
