#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free unit tests for ``amdsmi_cli.normalize_argv()``.

This is the CLI's argv pre-processing pass (run once, before argparse, in
``amdsmi_cli.py``): it lowercases subcommands/flags while preserving the
original case of path-like and profile-like values -- ``--folder``,
``--file``, ``--gpu``, ``--ampp-configure PROFILE KEY=VALUE ...``, and the
``--power-stabilization-configure @<path>`` replay form (both the
space-separated and ``--flag=value`` spellings).

A prior regression silently lowercased the
``--power-stabilization-configure`` ``@<path>`` value, turning a
case-sensitive filesystem path into a file-not-found error. The production
fix is two list entries in ``amdsmi_cli.py``
(``MULTI_VALUE_CASE_SENSITIVE_ARGS`` and ``CASE_SENSITIVE_PREFIXES``).

``tests/python/cli/test_set.py``'s
``test_power_stabilization_configure_case_sensitive_path`` exercises the same
defect end-to-end through the real CLI binary, but it self-skips on any GPU
that reports Power Stabilization as unsupported -- i.e. on every
hardware-free CI runner. ``normalize_argv()`` is pure list/string
manipulation with zero GPU dependency, so this suite imports it directly and
needs no GPU, mock sysfs, or compiled ``amdsmi`` package. This is the guard
that is actually CI-reachable and will catch a reintroduced regression
regardless of what hardware runs the suite.
"""

import os
import unittest
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
_AMDSMI_CLI_PATH = os.path.join(_CLI_DIR, "amdsmi_cli.py") if _CLI_DIR else ""


class TestArgvCaseNormalization(unittest.TestCase):
    """``normalize_argv()``'s case-preservation rules, independent of GPU hardware."""

    @classmethod
    def setUpClass(cls):
        # amdsmi_cli.py unconditionally imports these five siblings at module
        # scope -- one of them, amdsmi_init, in turn requires the compiled
        # amdsmi package to be installed. Stub all of them so the module loads
        # on a machine with no amd-smi install; normalize_argv() itself never
        # touches any of them (it is pure sys.argv list/string manipulation).
        stub_modules(
            cls,
            {
                "amdsmi_init": fake_module("amdsmi_init"),
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=mock.Mock),
                "amdsmi_commands": fake_module("amdsmi_commands", AMDSMICommands=mock.Mock),
                "amdsmi_parser": fake_module("amdsmi_parser", AMDSMIParser=mock.Mock),
                "amdsmi_logger": fake_module("amdsmi_logger", AMDSMILogger=mock.Mock),
                "amdsmi_cli_exceptions": fake_module("amdsmi_cli_exceptions"),
                "argv_case_normalization_under_test": None,
            },
        )
        cls.module = load_cli_module(
            "argv_case_normalization_under_test", _AMDSMI_CLI_PATH, _CLI_DIR
        )

    def test_power_stabilization_configure_space_form_preserves_path_case(self):
        """Regression guard: the space-separated ``@<path>`` form.

        Without the fix, ``--power-stabilization-configure`` falls through to
        the generic "lowercase everything else" branch, and so does its
        ``@<path>`` value (anything not starting with ``-``) -- this would
        lowercase the mixed-case path below.
        """
        argv = ["amd-smi", "SET", "--power-stabilization-configure", "@/Tmp/PS_Config/Dump.JSON"]
        self.assertEqual(
            self.module.normalize_argv(argv),
            ["amd-smi", "set", "--power-stabilization-configure", "@/Tmp/PS_Config/Dump.JSON"],
        )

    def test_power_stabilization_configure_equals_form_preserves_path_case(self):
        """Regression guard: the ``--flag=@<path>`` form.

        Without the fix, the whole token (flag and value together) falls
        through to the generic lowercasing branch, because it does not match
        any entry in CASE_SENSITIVE_PREFIXES.
        """
        argv = ["--power-stabilization-configure=@/Tmp/PS_Config/Dump.JSON"]
        self.assertEqual(
            self.module.normalize_argv(argv),
            ["--power-stabilization-configure=@/Tmp/PS_Config/Dump.JSON"],
        )

    def test_ampp_configure_preserves_case_until_next_flag(self):
        """Multi-value case preservation covers every token up to the next flag.

        Mirrors --power-stabilization-configure's contract for its sibling
        flag, and gives --ampp-configure its first unit-test coverage.
        """
        argv = [
            "--ampp-configure",
            "MyProfile",
            "PPT0_Limit=100",
            "MaxGfxclkFreq=2000",
            "--gpu",
            "0",
        ]
        self.assertEqual(
            self.module.normalize_argv(argv),
            ["--ampp-configure", "MyProfile", "PPT0_Limit=100", "MaxGfxclkFreq=2000", "--gpu", "0"],
        )

    def test_unrelated_flags_and_subcommands_still_lowercase(self):
        """Baseline: normalization still lowercases ordinary tokens."""
        argv = ["STATIC", "--POWER-STABILIZATION"]
        self.assertEqual(self.module.normalize_argv(argv), ["static", "--power-stabilization"])
