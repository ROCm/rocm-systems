#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ABI decoding tests for the internal name-value buffer parser.

``_get_name_value`` turns the native ``amdsmi_name_value_t`` array returned by
``amdsmi_get_gpu_pm_metrics_info`` / ``amdsmi_get_gpu_reg_table_info`` into a
list of ``{"name", "value"}`` dicts. These tests feed it a synthetic buffer
built from the generated structure, so they need no GPU and catch a stride or
member-offset that stops matching the C ABI (a 256-byte name char array
followed by a uint64 value).
"""

from __future__ import annotations

import ctypes
import unittest

from common.common import amdsmi


def _build_buffer(records):
    """Pack ``(name, value)`` pairs into a native ``amdsmi_name_value_t`` array.

    Returns the array (which the caller must keep alive), an element count as a
    ``c_uint32``, and a pointer to the first element -- the exact argument shape
    the public APIs hand to ``_get_name_value``.
    """
    name_value_t = amdsmi.amdsmi_wrapper.amdsmi_name_value_t
    name_size = name_value_t.name.size

    array = (name_value_t * len(records))()
    for i, (name, value) in enumerate(records):
        encoded = name.encode("utf-8")
        # Copy raw bytes so a name that fills the whole field stays terminator
        # free, mirroring what the C library can write.
        ctypes.memmove(
            ctypes.addressof(array[i]) + name_value_t.name.offset,
            encoded,
            min(len(encoded), name_size),
        )
        array[i].value = value

    num = ctypes.c_uint32(len(records))
    data = ctypes.cast(array, ctypes.POINTER(name_value_t))
    return array, num, data


class TestNameValueAbi(unittest.TestCase):
    def test_structure_matches_c_abi(self):
        # The parser derives its stride and offsets from this structure, so
        # pin the layout the C ABI documents: name[256] then a uint64 value.
        name_value_t = amdsmi.amdsmi_wrapper.amdsmi_name_value_t
        self.assertEqual(name_value_t.name.offset, 0)
        self.assertEqual(name_value_t.name.size, 256)
        self.assertEqual(name_value_t.value.offset, 256)
        self.assertEqual(ctypes.sizeof(name_value_t), 264)

    def test_two_records_decode_independently(self):
        # The minimal case: a wrong stride reads the second record from inside
        # the first, and a wrong value offset reads zero for the first.
        records = [("alpha", 123), ("beta", 456)]
        array, num, data = _build_buffer(records)
        self.assertIsNotNone(array)

        result = amdsmi.amdsmi_interface_utils._get_name_value(num, data)

        self.assertEqual(result, [{"name": "alpha", "value": 123}, {"name": "beta", "value": 456}])

    def test_long_names_do_not_overlap_neighbours(self):
        # Names longer than the old 72-byte stride would bleed into the next
        # record and shift every value if the stride were wrong.
        records = [("n" * 100, 0xDEADBEEF), ("m" * 200, 0x1234567890AB), ("x" * 50, 42)]
        array, num, data = _build_buffer(records)
        self.assertIsNotNone(array)

        result = amdsmi.amdsmi_interface_utils._get_name_value(num, data)

        self.assertEqual(
            result,
            [
                {"name": "n" * 100, "value": 0xDEADBEEF},
                {"name": "m" * 200, "value": 0x1234567890AB},
                {"name": "x" * 50, "value": 42},
            ],
        )

    def test_full_width_name_is_bounded(self):
        # A 256-byte name has no terminator; the read must stop at the field
        # edge and still find the value in the following record.
        records = [("z" * 256, 0xFFFFFFFFFFFFFFFF), ("tail", 7)]
        array, num, data = _build_buffer(records)
        self.assertIsNotNone(array)

        result = amdsmi.amdsmi_interface_utils._get_name_value(num, data)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"name": "z" * 256, "value": 0xFFFFFFFFFFFFFFFF})
        self.assertEqual(result[1], {"name": "tail", "value": 7})

    def test_zero_records_returns_empty_list(self):
        _array, num, data = _build_buffer([("ignored", 1)])
        num = ctypes.c_uint32(0)
        self.assertEqual(amdsmi.amdsmi_interface_utils._get_name_value(num, data), [])


if __name__ == "__main__":
    unittest.main()
