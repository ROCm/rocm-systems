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
"""Guards the module-isolation contract of the mock tier.

A suite that leaves a stub in ``sys.modules`` shadows the real module for every
test that runs after it, and the resulting ImportError surfaces in an unrelated
file. Pin the contract here so a leak is reported at its source.
"""

import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOCK_DIR = os.path.join(_THIS_DIR, "mock")
_TOP_LEVEL_DIR = os.path.dirname(_THIS_DIR)

# typing.io and friends are hand-built module objects too, so shape alone cannot
# separate a test stub from a standard-library one.
_STDLIB = getattr(sys, "stdlib_module_names", frozenset())


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _declared_isolated_modules(suite):
    names = set()
    for test in _iter_tests(suite):
        names.update(getattr(type(test), "ISOLATED_MODULES", ()))
    return names


def _looks_like_stub(name):
    module = sys.modules.get(name)
    return (
        name.split(".")[0] not in _STDLIB
        and module is not None
        and getattr(module, "__spec__", None) is None
        and not hasattr(module, "__file__")
    )


class TestModuleIsolation(unittest.TestCase):
    def test_mock_tier_restores_sys_modules_and_sys_path(self):
        suite = unittest.TestLoader().discover(
            start_dir=_MOCK_DIR, pattern="test_*.py", top_level_dir=_TOP_LEVEL_DIR
        )
        self.assertGreater(suite.countTestCases(), 0, "no mock tests discovered")

        declared = _declared_isolated_modules(suite)
        self.assertTrue(declared, "no suite declares ISOLATED_MODULES")
        before = {name: sys.modules.get(name) for name in declared}
        modules_before = set(sys.modules)
        path_before = list(sys.path)

        with open(os.devnull, "w") as devnull:
            result = unittest.TextTestRunner(stream=devnull, verbosity=0).run(suite)
        self.assertTrue(
            result.wasSuccessful(), "mock tier must pass for this guard to mean anything"
        )

        not_restored = sorted(n for n in declared if sys.modules.get(n) is not before[n])
        self.assertEqual(not_restored, [], f"isolated modules not restored: {not_restored}")

        undeclared = sorted(n for n in set(sys.modules) - modules_before if _looks_like_stub(n))
        self.assertEqual(undeclared, [], f"undeclared stubs outlived their suite: {undeclared}")

        leaked_paths = [entry for entry in sys.path if entry not in path_before]
        self.assertEqual(leaked_paths, [], f"sys.path entries outlived their suite: {leaked_paths}")
