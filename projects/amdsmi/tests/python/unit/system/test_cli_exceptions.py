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
"""Hardware-independent checks for the CLI exception framework."""

import ast
import glob
import os
import re
import shutil
import sys
import unittest


class TestAmdSmiCliExceptions(unittest.TestCase):
    """Hardware-independent checks for the CLI exception framework.

    Guards two classes of regression:
      1. A CLI module raising an ``AmdSmi*Exception`` it never imported, which
         only surfaces as a ``NameError`` on an error path (e.g. the missing
         imports in ``subcommands/reset.py`` and ``subcommands/set_value.py``).
      2. The empty-string ``hint`` ``IndexError`` in
         ``AmdSmiInvalidParameterValueException``.
    """

    EXCEPTION_NAME = re.compile(r"^AmdSmi.*Exception$")

    @staticmethod
    def _cli_dir():
        # Prefer the in-repo CLI source tree by walking up from this test file
        # looking for amdsmi_cli/amdsmi_cli_exceptions.py; fall back to the
        # installed CLI the `amd-smi` launcher points at.
        here = os.path.dirname(os.path.abspath(__file__))
        cur = here
        for _ in range(8):
            candidate = os.path.join(cur, "amdsmi_cli")
            if os.path.isfile(os.path.join(candidate, "amdsmi_cli_exceptions.py")):
                return candidate
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        amd_smi = shutil.which("amd-smi")
        if amd_smi:
            cli_dir = os.path.dirname(os.path.realpath(amd_smi))
            if os.path.isfile(os.path.join(cli_dir, "amdsmi_cli_exceptions.py")):
                return cli_dir
        return None

    @staticmethod
    def _resolvable_names(tree):
        names = set(dir(__builtins__))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        return names

    def test_cli_modules_import_every_exception_they_raise(self):
        cli_dir = self._cli_dir()
        if cli_dir is None:
            self.skipTest("Could not locate amdsmi_cli source directory")

        py_files = glob.glob(os.path.join(cli_dir, "*.py"))
        py_files += glob.glob(os.path.join(cli_dir, "subcommands", "*.py"))
        self.assertTrue(py_files, f"No CLI source files found under {cli_dir}")

        offenders = {}
        for path in py_files:
            with open(path, "r", encoding="utf-8") as fin:
                tree = ast.parse(fin.read(), filename=path)
            resolvable = self._resolvable_names(tree)
            used = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and self.EXCEPTION_NAME.match(node.id)
            }
            missing = sorted(name for name in used if name not in resolvable)
            if missing:
                offenders[os.path.relpath(path, cli_dir)] = missing

        self.assertEqual(
            offenders,
            {},
            "CLI modules reference exception classes they never import "
            f"(NameError on the error path): {offenders}",
        )

    def test_invalid_parameter_value_exception_accepts_empty_hint(self):
        cli_dir = self._cli_dir()
        if cli_dir is None:
            self.skipTest("Could not locate amdsmi_cli source directory")
        if cli_dir not in sys.path:
            sys.path.insert(0, cli_dir)
        import amdsmi_cli_exceptions

        try:
            exc = amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                "set", None, "json", hint=""
            )
        except IndexError as e:
            self.fail(f"Empty hint raised IndexError: {e}")

        # The guard is that an empty hint does not raise IndexError; the
        # specific code number is intentionally not asserted here.
        self.assertIsInstance(exc.value, int)
        self.assertIsInstance(str(exc), str)


if __name__ == "__main__":
    unittest.main()
