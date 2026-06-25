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
"""BDF string parsing and formatting unit tests."""

from __future__ import annotations

import unittest

from common.common import amdsmi

# BDF strings that must parse to their [domain, bus, device, function] components.
VALID_BDFS: dict[str, list[int]] = {
    "00:00.0": [0, 0, 0, 0],
    "01:01.1": [0, 1, 1, 1],
    "FF:1F.7": [0, 255, 31, 7],
    "FF:00.7": [0, 255, 0, 7],
    "11:01.2": [0, 17, 1, 2],
    "11:0a.2": [0, 17, 10, 2],
    "0000:FF:1F.7": [0, 255, 31, 7],
    "0001:ff:1F.7": [1, 255, 31, 7],
    "ffff:FF:1f.7": [65535, 255, 31, 7],
}

# Malformed inputs that _parse_bdf must reject (return None).
INVALID_BDFS: dict[str | None, None] = {
    None: None,
    "": None,
    "00:00:0": None,
    "00.00:0": None,
    "00:00.Z": None,
    "00:0Z.0": None,
    "0Z:00.0": None,
    "Z00:00.0": None,
    "A00:00.0": None,
    "0A00:00.0": None,
    "00:00.07": None,
    "00:00.8": None,
    "00:00.10": None,
    "00:00.11": None,
    "00:00.-1": None,
    "00:00.*-1": None,
    "00:00.123": None,
    "00:20.0": None,
    "00:45.0": None,
    "00:200.0": None,
    "00:002.0": None,
    "100:00.0": None,
    "0100:00.0": None,
    "00100:00.0": None,
    "0101:00.0": None,
    "00001:00.0": None,
    "10001:00.0": None,
    "45:0.0": None,
    ".00:00.0": None,
    "00.00.0": None,
    "00.0.0": None,
    "0.00.0": None,
    "000.00.0": None,
    "00 00 0": None,
    " 00:00.0": None,
    "00:00.0 ": None,
    "0000:00.00.0": None,
    "000:00:00.0": None,
    "00:00:00.1": None,
    "0:00:00.1": None,
    "0000 00 00 0": None,
    "-1-1:00:00.0": None,
    "AAAA:00:AA.0": None,
    "*1*1:00:00.0": None,
    "0000:00:00.07": None,
    "0000:00:00.8": None,
    "0000:00:00.10": None,
    "0000:00:00.11": None,
    "0000:00:00.-1": None,
    "0000:00:00.*-1": None,
    "0000:00:00.123": None,
    "0000:00:20.0": None,
    "0000:00:45.0": None,
    "0000:00:200.0": None,
    "0000:00:002.0": None,
    "0000:100:00.0": None,
    "0000:0100:00.0": None,
    "0000:00100:00.0": None,
    "0000:0101:00.0": None,
    "0000:00001:00.0": None,
    "0000:10001:00.0": None,
    "0000:45:0.0": None,
    ".0000.00:00.0": None,
    "0000.00.0.0": None,
    " 0000:00:00.0": None,
    "0000:00:00.0 ": None,
}


class TestAmdSmiPythonBDF(unittest.TestCase):
    @classmethod
    def _convert_bdf_to_long(cls, bdf):
        if len(bdf) == 12:
            return bdf
        if len(bdf) == 7:
            return "0000:" + bdf
        return None

    def test_parse_bdf(self):
        # Valid bdfs parse to their component list; invalid bdfs parse to None.
        for bdf, expected in {**VALID_BDFS, **INVALID_BDFS}.items():
            result = amdsmi.amdsmi_interface._parse_bdf(bdf)
            self.assertEqual(
                result, expected, f"Expected {expected} for bdf {bdf}, but got {result}"
            )
        return

    def test_format_bdf(self):
        # Only valid bdfs can be formatted back to a canonical string.
        for bdf_string, bdf_list in VALID_BDFS.items():
            smi_bdf = amdsmi.amdsmi_interface._make_amdsmi_bdf_from_list(bdf_list)
            expected = self._convert_bdf_to_long(bdf_string)
            if expected:
                expected = expected.lower()
            if smi_bdf:
                result = amdsmi.amdsmi_interface._format_bdf(smi_bdf)
            else:
                result = "None"
            self.assertEqual(
                result, expected, f"Expected {expected} for bdf {bdf_string}, but got {result}"
            )
        return
