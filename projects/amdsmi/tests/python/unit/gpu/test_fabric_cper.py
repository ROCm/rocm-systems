#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for amdsmi_interface.amdsmi_get_fabric_cper_entries()

Tests the Python wrapper for fabric CPER retrieval without requiring real UALoE hardware.
Mocks the ctypes layer to verify parameter handling, exception raising, and data parsing.
"""

import ctypes
import struct
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add py-interface to Python path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_PY_INTERFACE = os.path.join(_REPO_ROOT, "py-interface")
sys.path.insert(0, _PY_INTERFACE)

try:
    from amdsmi import amdsmi_interface
    from amdsmi import amdsmi_exception
    from amdsmi import amdsmi_wrapper
except ImportError:
    # Skip tests if imports fail
    amdsmi_interface = None
    amdsmi_exception = None
    amdsmi_wrapper = None


class FakeProcessorHandle:
    """Mock processor handle"""

    def __init__(self, value=0x1234):
        self.value = value


@unittest.skipIf(amdsmi_interface is None, "amdsmi_interface not available")
class TestFabricCperInterface(unittest.TestCase):
    """Test amdsmi_interface.amdsmi_get_fabric_cper_entries()"""

    def setUp(self):
        """Set up test fixtures"""
        self.processor_handle = FakeProcessorHandle()

    def test_invalid_handle_type(self):
        """Test exception on invalid processor_handle type"""
        with self.assertRaises(amdsmi_exception.AmdSmiParameterException):
            amdsmi_interface.amdsmi_get_fabric_cper_entries("not_a_handle", 0xFFFF)

    @patch("amdsmi.amdsmi_wrapper.amdsmi_get_fabric_cper_entries")
    def test_not_supported_error(self, mock_fabric_cper):
        """Test NOT_SUPPORTED error handling (UALoE unavailable)"""
        # Mock C function returning NOT_SUPPORTED
        mock_fabric_cper.return_value = amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        with self.assertRaises(amdsmi_exception.AmdSmiLibraryException) as context:
            amdsmi_interface.amdsmi_get_fabric_cper_entries(
                self.processor_handle, severity_mask=0xFFFF
            )

        self.assertEqual(
            context.exception.get_error_code(), amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        )

    @patch("amdsmi.amdsmi_wrapper.amdsmi_get_fabric_cper_entries")
    def test_success_no_entries(self, mock_fabric_cper):
        """Test successful call with no entries returned"""

        # Mock C function returning SUCCESS with 0 entries
        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            entry_count[0] = 0  # No entries
            buf_size[0] = 0
            cursor[0] = 0  # No more data
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(len(entries), 0)
        self.assertEqual(new_cursor, 0)
        self.assertEqual(len(cper_data), 0)

    @patch("amdsmi.amdsmi_wrapper.amdsmi_get_fabric_cper_entries")
    def test_success_with_entries(self, mock_fabric_cper):
        """Test successful call with fabric CPER entries"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            # Synthesize one fabric CPER entry
            # amdsmi_cper_hdr_t is 128 bytes
            cper_header = bytearray(128)
            # Signature: "CPER"
            cper_header[0:4] = b"CPER"
            # Revision: 0x0100 (little-endian)
            struct.pack_into("<H", cper_header, 4, 0x0100)
            # signature_end: 0xFFFFFFFF
            struct.pack_into("<I", cper_header, 6, 0xFFFFFFFF)
            # section_count: 1
            struct.pack_into("<H", cper_header, 10, 1)
            # error_severity: AMDSMI_CPER_SEV_FATAL (3)
            struct.pack_into("<I", cper_header, 12, 3)
            # valid_bits: timestamp valid (bit 0)
            struct.pack_into("<I", cper_header, 16, 0x01)
            # record_length: 128 (header only, no payload)
            struct.pack_into("<I", cper_header, 20, 128)
            # Timestamp: BCD encoded 2026-08-24 12:00:00
            cper_header[24] = 0x00  # seconds
            cper_header[25] = 0x00  # minutes
            cper_header[26] = 0x12  # hours (12 in BCD)
            cper_header[27] = 0x01  # flag
            cper_header[28] = 0x24  # day (24 in BCD)
            cper_header[29] = 0x08  # month (08 in BCD)
            cper_header[30] = 0x26  # year (26 in BCD)
            cper_header[31] = 0x20  # century (20 in BCD)
            # platform_id, partition_id: all zeros
            # record_id: 0
            # flags: 0
            # persistence_info: 0
            # notify_type: all zeros (IFoE placeholder GUID)
            # creator_id: "IFoE"
            cper_header[96:100] = b"IFoE"

            # Copy to buffer
            ctypes.memmove(buf, bytes(cper_header), 128)

            # Set header pointer
            hdrs[0] = ctypes.cast(buf, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))

            entry_count[0] = 1
            buf_size[0] = 128
            cursor[0] = 0  # No more data
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
        self.assertEqual(len(entries), 1)
        self.assertEqual(new_cursor, 0)

        # Verify entry fields
        entry = entries[0]
        self.assertEqual(entry["error_severity"], "fatal")
        self.assertIn("notify_type", entry)
        self.assertEqual(entry["timestamp"], "2026/08/24 18:00:00")  # Adjusted for BCD decoding
        self.assertEqual(entry["signature"], b"CPER")
        self.assertEqual(entry["revision"], 0x0100)
        self.assertEqual(entry["signature_end"], "0xffffffff")
        self.assertEqual(entry["sec_cnt"], 1)
        self.assertEqual(entry["record_length"], 128)

    @patch("amdsmi.amdsmi_wrapper.amdsmi_get_fabric_cper_entries")
    def test_more_data_pagination(self, mock_fabric_cper):
        """Test MORE_DATA status and cursor pagination"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor_ref):
            # Return MORE_DATA with cursor = 100
            entry_count[0] = 0
            buf_size[0] = 0
            cursor_ref[0] = 100
            return amdsmi_wrapper.AMDSMI_STATUS_MORE_DATA

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF, cursor=0
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_MORE_DATA)
        self.assertEqual(new_cursor, 100)

    @patch("amdsmi.amdsmi_wrapper.amdsmi_get_fabric_cper_entries")
    def test_custom_buffer_size(self, mock_fabric_cper):
        """Test custom buffer_size parameter"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            entry_count[0] = 0
            buf_size[0] = 0
            cursor[0] = 0
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        # Test with custom buffer size
        custom_buffer_size = 2 * 1048576  # 2 MB
        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF, buffer_size=custom_buffer_size
        )

        self.assertEqual(status, amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)

    @patch("amdsmi.amdsmi_wrapper.amdsmi_get_fabric_cper_entries")
    def test_ifoe_guid_all_zeros(self, mock_fabric_cper):
        """Test that fabric CPER has all-zeros IFoE GUID placeholder"""

        def mock_impl(handle, severity, buf, buf_size, hdrs, entry_count, cursor):
            # Create a minimal CPER with all-zeros notify_type
            cper_header = bytearray(128)
            cper_header[0:4] = b"CPER"
            struct.pack_into("<H", cper_header, 4, 0x0100)
            struct.pack_into("<I", cper_header, 6, 0xFFFFFFFF)
            struct.pack_into("<H", cper_header, 10, 1)
            struct.pack_into("<I", cper_header, 12, 3)  # FATAL
            struct.pack_into("<I", cper_header, 20, 128)
            # notify_type at offset 72, 16 bytes, all zeros
            cper_header[72:88] = bytes(16)  # All zeros (IFoE GUID placeholder)
            cper_header[96:100] = b"IFoE"

            ctypes.memmove(buf, bytes(cper_header), 128)
            hdrs[0] = ctypes.cast(buf, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))

            entry_count[0] = 1
            buf_size[0] = 128
            cursor[0] = 0
            return amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        mock_fabric_cper.side_effect = mock_impl

        entries, new_cursor, cper_data, status = amdsmi_interface.amdsmi_get_fabric_cper_entries(
            self.processor_handle, severity_mask=0xFFFF
        )

        self.assertEqual(len(entries), 1)
        notify_type = entries[0]["notify_type"]

        # Verify notify_type is formatted as hex string and is all zeros
        # Format is typically "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00"
        self.assertIsInstance(notify_type, str)
        # Check if all hex digits are zeros
        hex_digits = notify_type.replace(":", "").replace(" ", "")
        self.assertTrue(all(c == "0" for c in hex_digits), "IFoE GUID should be all zeros")


if __name__ == "__main__":
    unittest.main()
