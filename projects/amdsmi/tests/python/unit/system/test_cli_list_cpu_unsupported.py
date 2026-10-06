#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``amd-smi list --cpu``/``--core`` rejection.

``--cpu`` and ``--core`` are accepted by the shared device-argument group used
by every subcommand (see ``AMDSMIParser._add_device_arguments``), but ``list``
never implemented a CPU/core listing path. Passing either flag used to fall
through to ``list_gpu``'s dead cpu guard, which ``print("N/A")``'d straight to
stdout (bypassing ``AMDSMILogger`` entirely) and returned success, so
``amd-smi list --cpu all --file out.log`` silently wrote an empty file and
exited 0.

``ListDevicesCommands.list_devices`` now rejects ``--cpu``/``--core`` up front
with ``AmdSmiCommandNotSupportedException`` (``COMMAND_NOT_SUPPORTED`` exit
code), which the CLI entry point turns into a non-zero exit and a printed error
message instead of a silent empty success.

The CLI dir is resolved through ``common.find_cli_dir`` so the suite exercises
the install or the checkout as ``cli_search_order`` dictates, and skips cleanly
when no CLI is present.
"""

import argparse
import os
import sys
import unittest
from typing import Any

from common.common import (
    cli_search_order,
    fake_module,
    find_cli_dir,
    generated_version_stub,
    load_cli_module,
    stub_modules_at_import,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
if _CLI_DIR and _CLI_DIR not in sys.path:
    sys.path.append(_CLI_DIR)
_SUBCOMMANDS_DIR = os.path.join(_CLI_DIR, "subcommands") if _CLI_DIR else None
if _SUBCOMMANDS_DIR and _SUBCOMMANDS_DIR not in sys.path:
    sys.path.append(_SUBCOMMANDS_DIR)
_LIST_DEVICES_PATH = os.path.join(_SUBCOMMANDS_DIR, "list_devices.py") if _SUBCOMMANDS_DIR else None

# list_devices imports ``from amdsmi_helpers import AMDSMIHelpers``, and
# amdsmi_helpers does ``from amdsmi_init import *`` at load, which needs the
# driver-init module. Stub it (and the CMake-generated _version) at import;
# tearDownModule restores them.
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

cli_exc: Any = None
_CLI_IMPORT_ERROR: Any = None
if _CLI_DIR is None:
    _CLI_IMPORT_ERROR = "amd-smi CLI dir not found"
else:
    try:
        import amdsmi_cli_exceptions as _cli_exc_module  # noqa: E402

        cli_exc = _cli_exc_module
        if not hasattr(cli_exc, "AmdSmiExitCode"):
            _CLI_IMPORT_ERROR = (
                "amdsmi_cli_exceptions has no AmdSmiExitCode (CLI predates exit codes)"
            )
            cli_exc = None
    except ImportError as _err:
        _CLI_IMPORT_ERROR = _err


def setUpModule():
    if _CLI_IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"amd-smi CLI unusable ({_CLI_DIR}): {_CLI_IMPORT_ERROR}")


def tearDownModule():
    if _restore_stubs is not None:
        _restore_stubs()


class _FakeHelpers:
    def get_output_format(self):
        return "human"


class TestCliListCpuUnsupported(unittest.TestCase):
    CommandNotSupported: Any = (
        cli_exc.AmdSmiCommandNotSupportedException if cli_exc is not None else None
    )
    expected_exit_code: Any = (
        int(cli_exc.AmdSmiExitCode.COMMAND_NOT_SUPPORTED) if cli_exc is not None else None
    )

    def _make_commands(self):
        module = load_cli_module("list_devices_under_test", _LIST_DEVICES_PATH, _SUBCOMMANDS_DIR)
        commands = module.ListDevicesCommands()
        commands.helpers = _FakeHelpers()
        return commands

    def test_cpu_flag_raises_command_not_supported(self):
        commands = self._make_commands()
        args = argparse.Namespace(gpu=None, cpu=["0"], core=None)
        with self.assertRaises(self.CommandNotSupported) as ctx:
            commands.list_devices(args)
        self.assertEqual(ctx.exception.value, self.expected_exit_code)
        self.assertIn("--cpu and --core are not supported", str(ctx.exception))

    def test_core_flag_raises_command_not_supported(self):
        commands = self._make_commands()
        args = argparse.Namespace(gpu=None, cpu=None, core=["0"])
        with self.assertRaises(self.CommandNotSupported) as ctx:
            commands.list_devices(args)
        self.assertEqual(ctx.exception.value, self.expected_exit_code)

    def test_no_cpu_or_core_does_not_raise(self):
        # Namespace without cpu/core attrs at all (as e.g. a hypervisor-only
        # parser would produce) must not be mistaken for a cpu/core request.
        commands = self._make_commands()
        args = argparse.Namespace(gpu=None)
        try:
            commands.list_devices(args)
        except self.CommandNotSupported:
            self.fail("list_devices raised COMMAND_NOT_SUPPORTED for a GPU-only request")
        except Exception:
            # Expected: the GPU listing path itself needs further attributes
            # (device_handles_gpus, etc.) that this minimal stub doesn't
            # provide. Reaching that point proves the cpu/core guard didn't
            # fire, which is all this test verifies.
            pass
