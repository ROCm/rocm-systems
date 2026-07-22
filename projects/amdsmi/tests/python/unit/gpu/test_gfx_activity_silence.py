#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Hardware-free unit tests for gfx-activity silencing sentinel rendering.

When gfx activity is silenced, the C layer emits the uint-max N/A sentinel and
the Python interface must render it as "N/A". amdsmi_get_gpu_busy_percent routes
its uint32 value and amdsmi_get_utilization_count routes its uint64 counter
values through _validate_if_max_uint; this pins that mapping so a regression in
either getter is caught without a GPU.
"""

import unittest

from common.common import amdsmi


class TestGfxActivitySilence(unittest.TestCase):
    """The uint-max sentinel must render as N/A, real values must pass through."""

    def setUp(self):
        self._validate = amdsmi.amdsmi_interface._validate_if_max_uint
        self._max = amdsmi.amdsmi_interface.MaxUIntegerTypes

    def test_busy_percent_sentinel_is_na(self):
        # amdsmi_get_gpu_busy_percent: uint32 sentinel -> "N/A".
        self.assertEqual(self._validate(self._max.UINT32_T, self._max.UINT32_T), "N/A")

    def test_busy_percent_real_value_passthrough(self):
        # A live percentage is returned unchanged.
        self.assertEqual(self._validate(42, self._max.UINT32_T), 42)

    def test_busy_percent_zero_is_valid(self):
        # 0% is a real reading, not a sentinel.
        self.assertEqual(self._validate(0, self._max.UINT32_T), 0)

    def test_utilization_counter_sentinel_is_na(self):
        # amdsmi_get_utilization_count: uint64 accumulator sentinel -> "N/A".
        self.assertEqual(self._validate(self._max.UINT64_T, self._max.UINT64_T), "N/A")

    def test_utilization_counter_real_value_passthrough(self):
        self.assertEqual(self._validate(123456789, self._max.UINT64_T), 123456789)

    def test_utilization_counter_uint32_sentinel_not_silenced_as_uint64(self):
        # A uint32-max value is a legitimate uint64 counter, not the sentinel.
        self.assertEqual(
            self._validate(self._max.UINT32_T, self._max.UINT64_T), int(self._max.UINT32_T)
        )


if __name__ == "__main__":
    unittest.main()
