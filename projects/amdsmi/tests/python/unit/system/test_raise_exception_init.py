#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test module that reads ``self.raise_exception`` must also reset it.

The functional tests collect a failure in ``self.raise_exception`` and re-raise
it at the end of the test. unittest.TestCase has no such attribute, so a module
that reads it without ``self.raise_exception = None`` (normally in ``setUp``)
errors with AttributeError on every run where nothing failed.
"""

import os
import re
import unittest

_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_READ = re.compile(r"\bif self\.raise_exception\b|\braise self\.raise_exception\b")
_RESET = re.compile(r"\bself\.raise_exception = None\b")


class RaiseExceptionInitTest(unittest.TestCase):
    def test_modules_reading_raise_exception_reset_it(self):
        offenders = []
        for dirpath, _, filenames in os.walk(_TESTS_ROOT):
            for filename in filenames:
                if not (filename.startswith("test_") and filename.endswith(".py")):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as f:
                    source = f.read()
                if _READ.search(source) and not _RESET.search(source):
                    offenders.append(os.path.relpath(path, _TESTS_ROOT))
        self.assertEqual(sorted(offenders), [], "set self.raise_exception = None in setUp()")
