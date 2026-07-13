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
"""Structural tests for the Python test suite."""

import ast
import contextlib
import io
import re
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import common.common as test_common

PYTHON_TEST_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATHS = (
    PYTHON_TEST_ROOT / "unit_tests.py",
    PYTHON_TEST_ROOT / "integration_test.py",
    PYTHON_TEST_ROOT / "cli_unit_test.py",
)


class TestPythonTestInfrastructure(unittest.TestCase):
    def test_common_module_has_no_ticket_references(self):
        source = (PYTHON_TEST_ROOT / "common" / "common.py").read_text(encoding="utf-8")
        ticket_pattern = re.compile(r"\b(?:ROCM|SWDEV|AILITOOLS)-\d+\b|\bPR #\d+\b")
        self.assertIsNone(ticket_pattern.search(source))

    def test_unit_modules_avoid_future_annotations(self):
        offenders = []
        for path in sorted((PYTHON_TEST_ROOT / "unit").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in tree.body
            ):
                offenders.append(str(path.relative_to(PYTHON_TEST_ROOT)))

        self.assertEqual(offenders, [])

    def test_functional_modules_define_tests(self):
        empty_modules = []
        for path in sorted((PYTHON_TEST_ROOT / "functional").rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            has_test = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                for node in ast.walk(tree)
            )
            if not has_test:
                empty_modules.append(str(path.relative_to(PYTHON_TEST_ROOT)))

        self.assertEqual(empty_modules, [])

    def test_runners_are_import_safe(self):
        for runner_path in RUNNER_PATHS:
            with self.subTest(runner=runner_path.name):
                run_test_dir = mock.Mock()
                common_module = types.ModuleType("common.common")
                common_module.run_test_dir = run_test_dir
                common_package = types.ModuleType("common")
                common_package.__path__ = []
                common_package.common = common_module

                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.dict(
                            sys.modules, {"common": common_package, "common.common": common_module}
                        )
                    )
                    stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
                    runpy.run_path(str(runner_path), run_name="runner_import_probe")

                run_test_dir.assert_not_called()

    def test_runner_docstrings_document_exclude(self):
        missing = []
        for runner_path in RUNNER_PATHS:
            tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
            docstring = ast.get_docstring(tree) or ""
            if "-x" not in docstring or "--exclude" not in docstring:
                missing.append(runner_path.name)

        self.assertEqual(missing, [])

    def test_unit_runner_disables_root_requirement(self):
        tree = ast.parse(RUNNER_PATHS[0].read_text(encoding="utf-8"), filename=str(RUNNER_PATHS[0]))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_test_dir"
        ]
        self.assertEqual(len(calls), 1)
        requires_root = next(
            (keyword.value for keyword in calls[0].keywords if keyword.arg == "requires_root"), None
        )
        self.assertIsNotNone(requires_root)
        self.assertIs(getattr(requires_root, "value", None), False)

    def test_hardware_free_runner_allows_non_root_user(self):
        loader = mock.Mock()
        loader.discover.return_value = unittest.TestSuite()
        result = mock.Mock()
        result.wasSuccessful.return_value = True
        runner = mock.Mock()
        runner.run.return_value = result

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(test_common, "GTestSummaryRunner", return_value=runner)
            )
            stack.enter_context(mock.patch.object(sys, "argv", ["unit_tests.py"]))
            stack.enter_context(mock.patch.object(test_common.os, "geteuid", return_value=1000))
            stack.enter_context(
                mock.patch.object(test_common.unittest, "TestLoader", return_value=loader)
            )
            stack.enter_context(mock.patch.object(test_common.sys, "stdout", io.StringIO()))
            stack.enter_context(mock.patch.object(test_common.sys, "stderr", io.StringIO()))
            context = stack.enter_context(self.assertRaises(SystemExit))
            test_common.run_test_dir(
                "unit", "AMD SMI Unit Tests", str(PYTHON_TEST_ROOT), requires_root=False
            )

        self.assertEqual(context.exception.code, 0)
        loader.discover.assert_called_once_with(
            start_dir=str(PYTHON_TEST_ROOT / "unit"),
            pattern="test_*.py",
            top_level_dir=str(PYTHON_TEST_ROOT),
        )
        runner.run.assert_called_once()

    def test_privileged_runner_rejects_non_root_user(self):
        loader = mock.Mock()
        loader.discover.return_value = unittest.TestSuite()

        with contextlib.ExitStack() as stack:
            runner = stack.enter_context(mock.patch.object(test_common, "GTestSummaryRunner"))
            stack.enter_context(mock.patch.object(sys, "argv", ["integration_test.py"]))
            stack.enter_context(mock.patch.object(test_common.os, "geteuid", return_value=1000))
            stack.enter_context(
                mock.patch.object(test_common.unittest, "TestLoader", return_value=loader)
            )
            stack.enter_context(mock.patch.object(test_common.sys, "stdout", io.StringIO()))
            stack.enter_context(mock.patch.object(test_common.sys, "stderr", io.StringIO()))
            context = stack.enter_context(self.assertRaises(SystemExit))
            test_common.run_test_dir(
                "functional", "AMD SMI Integration Tests", str(PYTHON_TEST_ROOT)
            )

        self.assertEqual(context.exception.code, 1)
        loader.discover.assert_called_once_with(
            start_dir=str(PYTHON_TEST_ROOT / "functional"),
            pattern="test_*.py",
            top_level_dir=str(PYTHON_TEST_ROOT),
        )
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
