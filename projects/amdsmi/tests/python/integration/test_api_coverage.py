#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards that every public API is exercised by a test."""

import pathlib
import re
import unittest

import common.common as common

_TIER_DIR = pathlib.Path(__file__).resolve().parent

# Module-level helpers that format an already-fetched struct rather than calling
# the library. Their only caller is amdsmi_get_ainic_info(), which is driven.
_FORMATTERS = frozenset({"amdsmi_get_ainic_info_detail", "amdsmi_get_ainic_info_summary"})

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
            if name.startswith("amdsmi_")
            and callable(getattr(common.amdsmi, name))
            and name not in _FORMATTERS
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
