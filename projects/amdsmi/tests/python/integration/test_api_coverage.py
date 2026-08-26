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
"""Guards that every public API is exercised by a test."""

import pathlib
import re
import unittest

import common.common as common

_TIER_DIR = pathlib.Path(__file__).resolve().parent

# Naming an API in a comment or a message must not count as covering it, so
# match only the drivers that actually call one.
_DRIVEN = re.compile(
    r"(?:both|reject_only|expect_only|reject|expect|prerequisite)"
    r'\s*\(\s*"(amdsmi_\w+)"'
)


class TestApiCoverage(unittest.TestCase):
    """Hardware-free: introspects the binding and greps the suites."""

    def test_every_public_api_is_driven(self):
        public = {
            name
            for name in dir(common.amdsmi)
            if name.startswith("amdsmi_") and callable(getattr(common.amdsmi, name))
        }
        driven = set()
        for path in _TIER_DIR.rglob("test_*.py"):
            driven |= set(_DRIVEN.findall(path.read_text()))

        missing = sorted(public - driven)
        self.assertEqual(
            missing,
            [],
            "public APIs with no test under tests/python/integration; add one per API "
            f"or the coverage silently rots: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
