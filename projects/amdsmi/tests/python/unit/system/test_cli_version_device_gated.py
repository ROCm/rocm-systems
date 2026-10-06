#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for gating ``amd-smi version`` device flags on hardware presence.

The ``--cpu``/``--core`` device arguments and the CPU option groups in
``static``/``metric``/``set`` are only registered when the amd_hsmp/hsmp_acpi
driver is present (``AMDSMIHelpers.is_amd_hsmp_initialized``). The ``version``
command was an exception for CPU: it always advertised ``-c``/``--cpu_version``
in ``amd-smi version -h`` and accepted it, even on systems without a CPU.

``AMDSMIParser._add_version_parser`` now registers ``--cpu_version`` only when
CPU is initialized (``is_amd_hsmp_initialized``). ``--gpu_version`` and
``--nic_version`` are always available regardless of hardware presence.

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
    def __init__(self, hsmp_initialized=False, ainic_initialized=False):
        self._hsmp = hsmp_initialized
        self._ainic = ainic_initialized

    def is_amd_hsmp_initialized(self):
        return self._hsmp

    def is_ainic_initialized(self):
        return self._ainic


class TestCliVersionDeviceGated(unittest.TestCase):
    def _build_version_parser(self, hsmp_initialized=False, ainic_initialized=False):
        """Invoke the real ``_add_version_parser`` with a minimal fake ``self``
        and return the resulting ``version`` subparser's option strings.
        """
        fake = types.SimpleNamespace()
        fake.description = "AMD System Management Interface"
        fake.helpers = _FakeHelpers(hsmp_initialized, ainic_initialized)
        # _add_command_modifiers (called by _add_version_parser) needs a file
        # action factory; a plain "store" keeps the test self-contained.
        fake._check_output_file_path = lambda: "store"
        fake._add_command_modifiers = types.MethodType(AMDSMIParser._add_command_modifiers, fake)
        fake._add_version_parser = types.MethodType(AMDSMIParser._add_version_parser, fake)

        root = argparse.ArgumentParser()
        subparsers = root.add_subparsers()
        fake._add_version_parser(subparsers, func=lambda: None)

        version_parser = subparsers.choices["version"]
        option_strings = set()
        for action in version_parser._actions:
            option_strings.update(action.option_strings)
        return option_strings

    def test_gpu_version_always_available(self):
        for hsmp in (False, True):
            for ainic in (False, True):
                options = self._build_version_parser(hsmp, ainic)
                self.assertIn("--gpu_version", options)
                self.assertIn("-g", options)

    def test_cpu_version_hidden_when_cpu_absent(self):
        options = self._build_version_parser(hsmp_initialized=False, ainic_initialized=True)
        self.assertNotIn("--cpu_version", options)
        self.assertNotIn("-c", options)

    def test_cpu_version_present_when_cpu_initialized(self):
        options = self._build_version_parser(hsmp_initialized=True, ainic_initialized=False)
        self.assertIn("--cpu_version", options)
        self.assertIn("-c", options)

    def test_nic_version_always_available(self):
        # --nic_version is not gated on NIC presence; it is always registered.
        for hsmp in (False, True):
            for ainic in (False, True):
                options = self._build_version_parser(hsmp, ainic)
                self.assertIn("--nic_version", options)
                self.assertIn("-n", options)

    def test_only_gpu_and_nic_version_when_no_cpu(self):
        options = self._build_version_parser(hsmp_initialized=False, ainic_initialized=False)
        self.assertIn("--gpu_version", options)
        self.assertIn("--nic_version", options)
        self.assertNotIn("--cpu_version", options)
