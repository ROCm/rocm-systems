#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``--file`` overwrite/append prompt and its non-TTY guard.

Loads the installed ``amdsmi_parser`` with its heavy dependencies (``amdsmi``,
``_version``, ``amdsmi_helpers``) stubbed, so the ``_check_output_file_path``
argparse action is exercised without GPU hardware or the compiled ``amdsmi``
package. Pins the fix that makes ``amd-smi <cmd> --file <existing>`` fail fast
instead of blocking forever on the interactive overwrite/append prompt when
stdin is not a terminal (for example when launched by an automation framework
with stdin piped or closed), while preserving the interactive prompt for a TTY
and the ``--overwrite`` / ``--append`` fast paths.
"""

import argparse
import builtins
import importlib.util
import os
import sys
import tempfile
import types
import unittest

from common.common import amdsmi_path, find_cli_dir

# Locate the CLI dir (amdsmi_path first so an AMDSMI_PATH override selects the
# matching install; see common.find_cli_dir). None -> setUpClass skips.
_CLI_DIR = find_cli_dir(amdsmi_path, os.path.dirname(os.path.abspath(__file__)))
PARSER_PATH = os.path.join(_CLI_DIR, "amdsmi_parser.py") if _CLI_DIR else None


class _FakeStdin:
    """Minimal stand-in for sys.stdin exposing only isatty()."""

    def __init__(self, is_tty):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def _install_stubs():
    """Register lightweight stubs for the parser's non-stdlib imports.

    ``setdefault`` leaves any already-imported real module in place (the CLI is
    installed, so ``amdsmi``/``amdsmi_helpers`` may already be loaded by the
    shared test harness); only the missing ones fall back to stubs. The parser
    references these names at import only to bind them, so stubs are sufficient.
    ``amdsmi_cli_exceptions`` is intentionally NOT stubbed -- the test asserts on
    the real exception type raised by the action.
    """
    amdsmi_mod = types.ModuleType("amdsmi")
    amdsmi_mod.amdsmi_interface = types.SimpleNamespace()
    sys.modules.setdefault("amdsmi", amdsmi_mod)

    version_mod = types.ModuleType("_version")
    version_mod.__version__ = "0.0.0-test"
    sys.modules.setdefault("_version", version_mod)

    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    sys.modules.setdefault("amdsmi_helpers", helpers_mod)


def _load_parser_module():
    # Put the installed CLI dir on sys.path so the parser's own
    # ``import amdsmi_cli_exceptions`` resolves to the real module.
    if _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    spec = importlib.util.spec_from_file_location("amdsmi_parser_under_test", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOutputFileStdinGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARSER_PATH or not os.path.isfile(PARSER_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI amdsmi_parser.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        _install_stubs()
        cls.parser_mod = _load_parser_module()
        import amdsmi_cli_exceptions  # resolved from _CLI_DIR above

        cls.exceptions = amdsmi_cli_exceptions

    def _make_action(self):
        """Build the CheckOutputFilePath action from a minimal fake parser."""
        fake_parser = types.SimpleNamespace(
            helpers=types.SimpleNamespace(get_output_format=lambda: "human")
        )
        action_cls = self.parser_mod.AMDSMIParser._check_output_file_path(fake_parser)
        return action_cls(option_strings=["--file"], dest="file")

    def _run_action(self, path, *, is_tty, overwrite=False, append=False, input_response=None):
        """Invoke the action for an existing ``path`` under a controlled stdin.

        Returns (args, input_prompts). ``input_prompts`` records every prompt
        string passed to input(); when ``input_response`` is None a call to
        input() fails the test (proving the prompt was never reached).
        """
        action = self._make_action()
        args = argparse.Namespace(json=False, csv=False, overwrite=overwrite, append=append)
        input_prompts = []

        def _fake_input(prompt=""):
            input_prompts.append(prompt)
            if input_response is None:
                raise AssertionError("input() was called but no prompt was expected")
            return input_response

        saved = (sys.stdin, list(sys.argv), builtins.input)
        try:
            sys.stdin = _FakeStdin(is_tty)
            sys.argv = ["amd-smi"]  # so "--overwrite"/"--append" in sys.argv is False
            builtins.input = _fake_input
            action(None, args, str(path))
        finally:
            sys.stdin, sys.argv[:], builtins.input = saved[0], saved[1], saved[2]
        return args, input_prompts

    def test_non_tty_existing_file_fails_fast(self):
        # Existing file, no --overwrite/--append, stdin not a TTY: the action
        # must raise instead of calling input() (which would block forever on a
        # held-open pipe).
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            with self.assertRaises(self.exceptions.AmdSmiInvalidFilePathException) as ctx:
                self._run_action(tmp, is_tty=False)
            self.assertIn("not a TTY", str(ctx.exception))
        finally:
            os.unlink(tmp)

    def test_overwrite_flag_bypasses_prompt_when_non_tty(self):
        # --overwrite must truncate the file and never prompt, even without a TTY.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"stale-contents")
            tmp = f.name
        try:
            args, prompts = self._run_action(tmp, is_tty=False, overwrite=True)
            self.assertEqual(str(args.file), tmp)
            self.assertEqual(prompts, [])
            self.assertEqual(os.path.getsize(tmp), 0)
        finally:
            os.unlink(tmp)

    def test_append_flag_bypasses_prompt_when_non_tty(self):
        # --append must keep the file and never prompt, even without a TTY.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"keep")
            tmp = f.name
        try:
            args, prompts = self._run_action(tmp, is_tty=False, append=True)
            self.assertEqual(str(args.file), tmp)
            self.assertEqual(prompts, [])
            self.assertEqual(os.path.getsize(tmp), 4)
        finally:
            os.unlink(tmp)

    def test_tty_still_prompts_and_overwrites(self):
        # A real TTY must still get the interactive prompt; answering "o" truncates.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"stale-contents")
            tmp = f.name
        try:
            args, prompts = self._run_action(tmp, is_tty=True, input_response="o")
            self.assertEqual(str(args.file), tmp)
            self.assertEqual(len(prompts), 1)
            self.assertIn("Overwrite", prompts[0])
            self.assertEqual(os.path.getsize(tmp), 0)
        finally:
            os.unlink(tmp)
