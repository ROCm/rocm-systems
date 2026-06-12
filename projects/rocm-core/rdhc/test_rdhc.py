#!/usr/bin/env python3
"""Unit tests for rdhc.py"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


def _ensure_optional_deps():
    """Stub optional third-party deps so importing rdhc never sys.exit()s.

    rdhc.py calls check_required_packages() at import time and exits if
    prettytable/yaml are missing. CI installs them via requirements.txt, but
    stubbing keeps the tests runnable in a bare environment too.
    """
    if "prettytable" not in sys.modules:
        try:
            import prettytable  # noqa: F401
        except ImportError:
            stub = types.ModuleType("prettytable")
            stub.PrettyTable = type("PrettyTable", (), {})
            sys.modules["prettytable"] = stub
    if "yaml" not in sys.modules:
        try:
            import yaml  # noqa: F401
        except ImportError:
            sys.modules["yaml"] = types.ModuleType("yaml")


_ensure_optional_deps()
rdhc = importlib.import_module("rdhc")


class LibDepthValidationTest(unittest.TestCase):
    """Covers LIBDIR_MAX_DEPTH validation and safe find/ldd invocation."""

    def setUp(self):
        self.hc = rdhc.ROCMHealthCheck(rocm_path="/opt/rocm")

    def _run(self, env_value, find_stdout="", find_ret=0):
        """Invoke test_check_lib_dependencies with a patched environment.

        Returns (result, calls) where calls is the list of every
        (command, shell) pair passed to run_command. The find invocation is
        always the first entry. find is stubbed so no real filesystem/command
        is touched; os.path.exists returns True so we reach the find call.
        """
        calls = []

        def fake_run_command(command, shell=False):
            calls.append((command, shell))
            # First call is find; any later call (ldd) returns "not dynamic".
            if len(calls) == 1:
                return find_stdout, "", find_ret
            return "", "not a dynamic executable", 0

        env = {} if env_value is None else {"LIBDIR_MAX_DEPTH": env_value}
        with mock.patch.dict(os.environ, env, clear=False):
            if env_value is None:
                os.environ.pop("LIBDIR_MAX_DEPTH", None)
            with mock.patch.object(
                rdhc, "run_command", side_effect=fake_run_command
            ), mock.patch.object(rdhc.os.path, "exists", return_value=True):
                result = self.hc.test_check_lib_dependencies()
        return result, calls

    def test_invalid_non_numeric_fails(self):
        (status, msg), calls = self._run("3; rm -rf /")
        self.assertEqual(status, rdhc.TestStatus.FAIL.value)
        self.assertIn("non-negative integer", msg)
        # Rejected before any command was run.
        self.assertEqual(calls, [])

    def test_negative_value_fails(self):
        (status, _msg), calls = self._run("-1")
        self.assertEqual(status, rdhc.TestStatus.FAIL.value)
        self.assertEqual(calls, [])

    def test_non_ascii_digits_normalized_to_ascii(self):
        # Arabic-Indic digit three: int() parses it and str() normalizes to
        # ASCII "3", so find receives a safe base-10 integer string.
        _result, calls = self._run("٣", find_stdout="")
        find_cmd, shell = calls[0]
        self.assertEqual(shell, False)
        self.assertEqual(find_cmd[find_cmd.index("-maxdepth") + 1], "3")

    def test_whitespace_value_normalized(self):
        # Whitespace should be accepted and normalized, not rejected.
        _result, calls = self._run(" 3 ", find_stdout="")
        find_cmd, shell = calls[0]
        self.assertEqual(shell, False)
        self.assertIsInstance(find_cmd, list)
        self.assertIn("-maxdepth", find_cmd)
        self.assertEqual(find_cmd[find_cmd.index("-maxdepth") + 1], "3")

    def test_find_uses_argv_list_without_shell(self):
        _result, calls = self._run(None, find_stdout="")
        find_cmd, shell = calls[0]
        self.assertIsInstance(find_cmd, list)
        self.assertEqual(find_cmd[0], "find")
        self.assertEqual(shell, False)
        # No -maxdepth when the env var is unset.
        self.assertNotIn("-maxdepth", find_cmd)


class LddInvocationTest(unittest.TestCase):
    """The ldd call must use an argv list with shell=False and a -- guard."""

    def setUp(self):
        self.hc = rdhc.ROCMHealthCheck(rocm_path="/opt/rocm")

    def test_ldd_called_with_argv_and_dashdash(self):
        malicious = "/opt/rocm/lib/evil; touch pwned.so"
        captured = {}

        def fake_run_command(command, shell=False):
            captured["command"] = command
            captured["shell"] = shell
            return "\tnot a dynamic executable", "not a dynamic executable", 0

        with mock.patch.object(
            rdhc, "run_command", side_effect=fake_run_command
        ), mock.patch.object(
            rdhc.os.path, "exists", return_value=True
        ), mock.patch.object(
            rdhc.os.path, "islink", return_value=False
        ):
            self.hc._check_rocm_libs_dependency([malicious], "/opt/rocm/lib")

        self.assertEqual(captured["shell"], False)
        self.assertEqual(captured["command"], ["ldd", "--", malicious])


if __name__ == "__main__":
    unittest.main()
