#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards the unit suite relies on: no real ``amdsmi_init``, and clean failure reports.

Without these tests the guards could rot unnoticed: a broken guard lets a driver-dependent
test pass on a host that has the driver, which is exactly what it exists to prevent.
"""

import importlib
import io
import sys
import unittest
from unittest import mock

from common.common import GTestSummaryRunner, RealAmdsmiInitImportError, fake_module, stub_modules

_PROBE_MODULE = "amdsmi_runner_guard_probe"


def _run_probe_class(set_up_class):
    """Run a throwaway class whose setUpClass is *set_up_class*; return the runner's result."""

    class Probe(unittest.TestCase):
        setUpClass = classmethod(set_up_class)

        def test_noop(self):
            pass

    runner = GTestSummaryRunner(stream=io.StringIO(), verbosity=0)
    return runner.run(unittest.TestSuite([Probe("test_noop")]))


class TestUnitRunnerGuards(unittest.TestCase):
    def test_unit_package_forbids_the_real_amdsmi_init(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("amdsmi_init", None)
            with self.assertRaises(RealAmdsmiInitImportError):
                importlib.import_module("amdsmi_init")

    def test_failed_set_up_class_restores_its_stubs(self):
        def set_up_class(cls):
            stub_modules(cls, {_PROBE_MODULE: fake_module(_PROBE_MODULE)})
            raise RuntimeError("setup failed")

        result = _run_probe_class(set_up_class)

        self.assertNotIn(_PROBE_MODULE, sys.modules)
        self.assertEqual(len(result.errors), 1)
        # A stub left behind would be reported here as a sys.modules leak.
        self.assertEqual(result.failures, [])

    def test_real_amdsmi_init_import_is_reported_without_a_traceback(self):
        def set_up_class(cls):
            raise RealAmdsmiInitImportError("imported the real amdsmi_init")

        ((_test, text),) = _run_probe_class(set_up_class).errors

        self.assertEqual(text, "RealAmdsmiInitImportError: imported the real amdsmi_init\n")
