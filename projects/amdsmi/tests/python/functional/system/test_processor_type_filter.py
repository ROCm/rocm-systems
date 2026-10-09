#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed processor enumeration: count and list agreement, truncation, invalid types."""

import ctypes
import unittest

from common.common import amdsmi

# First value past AMD_BRCM_SWITCH, still inside the enum's representable range.
_OUT_OF_RANGE_TYPE = 10
_GUARD = 1
_NEVER_ENUMERATED = ("NON_AMD_GPU", "NON_AMD_CPU", "AMD_APU")


class TestProcessorTypeFilter(unittest.TestCase):
    def setUp(self):
        amdsmi.amdsmi_init()
        self.addCleanup(amdsmi.amdsmi_shut_down)
        self.wrapper = amdsmi.amdsmi_wrapper
        self.sockets = amdsmi.amdsmi_get_socket_handles()
        if not self.sockets:
            self.skipTest("No sockets available")

    def _query(self, socket, processor_type, capacity=None):
        """Return (available, reported, handles), asserting nothing is written past capacity."""
        available = ctypes.c_uint32()
        self.assertEqual(
            self.wrapper.amdsmi_get_processor_handles_by_type(
                socket, processor_type, None, ctypes.byref(available)
            ),
            self.wrapper.AMDSMI_STATUS_SUCCESS,
        )
        if capacity is None:
            capacity = available.value
        handles = (self.wrapper.amdsmi_processor_handle * (capacity + 1))()
        handles[capacity] = _GUARD
        reported = ctypes.c_uint32(capacity)
        self.assertEqual(
            self.wrapper.amdsmi_get_processor_handles_by_type(
                socket, processor_type, handles, ctypes.byref(reported)
            ),
            self.wrapper.AMDSMI_STATUS_SUCCESS,
        )
        self.assertEqual(handles[capacity], _GUARD)
        return available.value, reported.value, handles

    def test_each_type_returns_only_its_own_processors(self):
        for socket in self.sockets:
            for processor_type in amdsmi.AmdSmiProcessorType:
                if processor_type == amdsmi.AmdSmiProcessorType.UNKNOWN:
                    continue
                with self.subTest(socket=socket.value, processor_type=processor_type):
                    available, reported, handles = self._query(socket, processor_type)
                    self.assertEqual(reported, available)
                    if processor_type.name in _NEVER_ENUMERATED:
                        self.assertEqual(available, 0)
                    for raw_handle in handles[:reported]:
                        actual = amdsmi.amdsmi_get_processor_type(
                            self.wrapper.amdsmi_processor_handle(raw_handle)
                        )
                        self.assertEqual(actual["processor_type"], processor_type.name)

    def test_small_buffer_truncates_and_succeeds(self):
        for socket in self.sockets:
            available, _, _ = self._query(socket, amdsmi.AmdSmiProcessorType.AMD_GPU)
            if available < 2:
                continue
            _, reported, _ = self._query(
                socket, amdsmi.AmdSmiProcessorType.AMD_GPU, capacity=available - 1
            )
            self.assertEqual(reported, available - 1)
            return
        self.skipTest("No socket has at least two GPU processors for a partial fill")

    def test_invalid_types_preserve_count_and_buffer(self):
        for socket in self.sockets:
            for processor_type in (amdsmi.AmdSmiProcessorType.UNKNOWN, _OUT_OF_RANGE_TYPE):
                for capacity in (0, 1, 3):
                    with self.subTest(
                        socket=socket.value, processor_type=processor_type, capacity=capacity
                    ):
                        count = ctypes.c_uint32(capacity)
                        handles = (self.wrapper.amdsmi_processor_handle * 4)(*([_GUARD] * 4))
                        for buffer in (None, handles):
                            self.assertEqual(
                                self.wrapper.amdsmi_get_processor_handles_by_type(
                                    socket, processor_type, buffer, ctypes.byref(count)
                                ),
                                self.wrapper.AMDSMI_STATUS_INVAL,
                            )
                            self.assertEqual(count.value, capacity)
                            self.assertEqual(list(handles), [_GUARD] * 4)

    def test_unknown_type_raises_in_python(self):
        for socket in self.sockets:
            with self.subTest(socket=socket.value):
                with self.assertRaises(amdsmi.AmdSmiLibraryException) as error:
                    amdsmi.amdsmi_get_processor_handles_by_type(
                        socket, amdsmi.AmdSmiProcessorType.UNKNOWN
                    )
                self.assertEqual(error.exception.get_error_code(), self.wrapper.AMDSMI_STATUS_INVAL)
