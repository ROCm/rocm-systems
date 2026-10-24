#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for hiding ``--cpu``/``--core`` from GPU-only subcommand help.

``bad-pages``, ``fabric``, ``firmware``, ``xgmi``, ``partition``,
``topology``, ``process``, ``monitor``, and ``list`` reject ``--cpu``/``--core``
at runtime with ``COMMAND_NOT_SUPPORTED`` (see
``test_cli_cpu_unsupported_subcommands.py`` and
``test_cli_list_cpu_unsupported.py``). Registering those flags with their normal
help text made ``<command> --help`` still advertise ``-U``/``--cpu`` and
``-O``/``--core`` under "Device Arguments", contradicting the rejection.

``AMDSMIParser._add_device_arguments`` now takes ``cpu_core_supported``. When
``False`` the flags are registered with ``argparse.SUPPRESS`` help: they are
hidden from ``--help`` (usage line and Device Arguments) yet still parse, so the
runtime guard can raise the informative ``COMMAND_NOT_SUPPORTED`` error. When
``True`` (the default, used by CPU-capable commands like ``static``/``metric``)
the flags keep their normal help.

The parser module is resolved through ``common.find_cli_dir`` so the suite
exercises the install or the checkout as ``cli_search_order`` dictates, and
skips cleanly when no CLI is present.
"""

import argparse
import os
import sys
import types
import unittest
from typing import Any

from common.common import (
    cli_search_order,
    fake_module,
    find_cli_dir,
    generated_version_stub,
    stub_modules_at_import,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
if _CLI_DIR and _CLI_DIR not in sys.path:
    sys.path.append(_CLI_DIR)

# amdsmi_parser imports amdsmi_helpers (``from amdsmi_init import *``) and the
# CMake-generated _version at load. Stub those at import; tearDownModule
# restores them.
_stubs = {}
if _CLI_DIR and "amdsmi_init" not in sys.modules:
    from amdsmi import amdsmi_exception as _amdsmi_exception
    from amdsmi import amdsmi_interface as _amdsmi_interface

    _stubs["amdsmi_init"] = fake_module(
        "amdsmi_init",
        AMDSMI_INIT_FLAG=0,
        AMDSMI_INITIALIZED=True,
        amdsmi_interface=_amdsmi_interface,
        amdsmi_exception=_amdsmi_exception,
    )
if "_version" not in sys.modules:
    _stubs.update(generated_version_stub())
_restore_stubs = stub_modules_at_import(_stubs) if _stubs else None

AMDSMIParser: Any = None
_CLI_IMPORT_ERROR: Any = None
if _CLI_DIR is None:
    _CLI_IMPORT_ERROR = "amd-smi CLI dir not found"
else:
    try:
        from amdsmi_parser import AMDSMIParser as _AMDSMIParser  # noqa: E402

        AMDSMIParser = _AMDSMIParser
    except ImportError as _err:
        _CLI_IMPORT_ERROR = _err


def setUpModule():
    if _CLI_IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"amd-smi CLI unusable ({_CLI_DIR}): {_CLI_IMPORT_ERROR}")


def tearDownModule():
    if _restore_stubs is not None:
        _restore_stubs()


class _FakeHelpers:
    """Only CPU (amd_hsmp) is initialized so ``--cpu``/``--core`` register."""

    def is_amdgpu_initialized(self):
        return True

    def is_amd_hsmp_initialized(self):
        return True

    def is_hypervisor(self):
        return False

    def is_ainic_initialized(self):
        return False

    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False


class TestCliDeviceArgsCpuCoreHidden(unittest.TestCase):
    def _build_parser(self, cpu_core_supported):
        """Invoke the real ``_add_device_arguments`` on a minimal fake ``self``
        and return the ``argparse.ArgumentParser`` it populated.
        """
        fake = types.SimpleNamespace()
        fake.helpers = _FakeHelpers()
        for name in (
            "gpu_choices_str",
            "cpu_choices_str",
            "nic_choices_str",
            "core_choices_str",
            "switch_choices_str",
        ):
            setattr(fake, name, "0")
        for name in (
            "gpu_choices",
            "cpu_choices",
            "core_choices",
            "nic_choices",
            "vf_choices",
            "switch_choices",
        ):
            setattr(fake, name, ["0"])
        # Action factories return a plain "store" action; the test only cares
        # about help visibility and parseability, not the custom select logic.
        for name in ("_gpu_select", "_cpu_select", "_core_select", "_nic_select", "_switch_select"):
            setattr(fake, name, lambda choices: "store")
        fake._validate_cpu_core = str
        fake._add_device_arguments = types.MethodType(AMDSMIParser._add_device_arguments, fake)

        parser = argparse.ArgumentParser(prog="amd-smi list")
        fake._add_device_arguments(parser, required=False, cpu_core_supported=cpu_core_supported)
        return parser

    @staticmethod
    def _help_for(parser, option):
        for action in parser._actions:
            if option in action.option_strings:
                return action.help
        return "__MISSING__"

    def test_cpu_core_hidden_from_help_but_still_parse(self):
        parser = self._build_parser(cpu_core_supported=False)

        # Registered (so the runtime guard can fire) ...
        self.assertEqual(self._help_for(parser, "--cpu"), argparse.SUPPRESS)
        self.assertEqual(self._help_for(parser, "--core"), argparse.SUPPRESS)

        # ... but absent from rendered help (usage line + Device Arguments).
        rendered = parser.format_help()
        self.assertNotIn("--cpu", rendered)
        self.assertNotIn("--core", rendered)

        # Still parseable, so `<command> --cpu ...` reaches the runtime guard
        # rather than argparse's generic "unrecognized arguments" error.
        self.assertEqual(parser.parse_args(["--cpu", "0"]).cpu, ["0"])
        self.assertEqual(parser.parse_args(["--core", "0"]).core, ["0"])

    def test_cpu_core_visible_when_supported(self):
        parser = self._build_parser(cpu_core_supported=True)

        self.assertNotEqual(self._help_for(parser, "--cpu"), argparse.SUPPRESS)
        self.assertNotEqual(self._help_for(parser, "--core"), argparse.SUPPRESS)

        rendered = parser.format_help()
        self.assertIn("--cpu", rendered)
        self.assertIn("--core", rendered)

    def test_gpu_always_visible(self):
        for supported in (False, True):
            parser = self._build_parser(cpu_core_supported=supported)
            self.assertIn("--gpu", parser.format_help())
