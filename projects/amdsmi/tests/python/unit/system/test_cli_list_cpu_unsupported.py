#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``amd-smi list --cpu`` CPU identification and ``--core`` rejection.

``--cpu`` and ``--core`` are accepted by the shared device-argument group used
by every subcommand (see ``AMDSMIParser._add_device_arguments``). ``list --cpu``
reports the socket-level CPU identity (family/model/stepping, formatted as hex);
family and model come from the amd-smi APIs
(``amdsmi_get_cpu_family``/``amdsmi_get_cpu_model``) while stepping is read from
``/proc/cpuinfo`` via ``AMDSMIHelpers``. ``list --core`` is still not serviced
and is rejected up front with ``AmdSmiCommandNotSupportedException``
(``COMMAND_NOT_SUPPORTED`` exit code).

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

_CLI_IMPORT_ERROR: Any = None
cli_exc: Any = None
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


class _FakeHandle:
    def __init__(self, value):
        self.value = value


class _FakeLogger:
    def __init__(self):
        self.stored = {}

    def store_cpu_output(self, device_handle, argument, data):
        self.stored[argument] = data

    def store_multiple_device_output(self):
        pass

    def print_output(self, **kwargs):
        pass


class _FakeHelpers:
    def __init__(self, family=26, model=17, stepping=2):
        self._family = family
        self._model = model
        self._stepping = stepping

    def check_required_groups(self):
        pass

    def handle_cpus(self, args, logger, subcommand):
        if isinstance(args.cpu, list):
            args.cpu = args.cpu[0]
        return False, args.cpu

    def get_cpu_id_from_device_handle(self, device_handle):
        return 0

    def get_output_format(self):
        return "human"

    def get_cpu_stepping(self):
        return self._stepping


class TestCliListCpuIdentification(unittest.TestCase):
    CommandNotSupported: Any = (
        cli_exc.AmdSmiCommandNotSupportedException if cli_exc is not None else None
    )
    expected_exit_code: Any = (
        int(cli_exc.AmdSmiExitCode.COMMAND_NOT_SUPPORTED) if cli_exc is not None else None
    )

    def _make_commands(self, family=26, model=17, stepping=2):
        module = load_cli_module("list_devices_under_test", _LIST_DEVICES_PATH, _SUBCOMMANDS_DIR)
        # family/model come from the amd-smi interface; stub them on the module's
        # bound interface so the test is hardware-independent.
        module.amdsmi_interface.amdsmi_get_cpu_family = lambda: family
        module.amdsmi_interface.amdsmi_get_cpu_model = lambda: model
        commands = module.ListDevicesCommands()
        commands.helpers = _FakeHelpers(family=family, model=model, stepping=stepping)
        commands.logger = _FakeLogger()
        commands.group_check_printed = False
        commands.cpu_handles = [_FakeHandle(1)]
        return commands

    def test_cpu_flag_reports_family_model_stepping(self):
        commands = self._make_commands()
        args = argparse.Namespace(gpu=None, cpu=[_FakeHandle(1)], core=None)
        commands.list_devices(args)
        self.assertEqual(commands.logger.stored["family"], "0x1A")
        self.assertEqual(commands.logger.stored["model"], "0x11")
        self.assertEqual(commands.logger.stored["stepping"], "0x2")

    def test_core_flag_raises_command_not_supported(self):
        commands = self._make_commands()
        args = argparse.Namespace(gpu=None, cpu=None, core=[_FakeHandle(1)])
        with self.assertRaises(self.CommandNotSupported) as ctx:
            commands.list_devices(args)
        self.assertEqual(ctx.exception.value, self.expected_exit_code)
        self.assertIn("--core is not supported", str(ctx.exception))
        self.assertNotIn("family", commands.logger.stored)

    def test_family_model_fallback_to_na(self):
        commands = self._make_commands()
        module = load_cli_module("list_devices_under_test", _LIST_DEVICES_PATH, _SUBCOMMANDS_DIR)

        def _raise():
            raise module.amdsmi_exception.AmdSmiLibraryException(12)

        module.amdsmi_interface.amdsmi_get_cpu_family = _raise
        module.amdsmi_interface.amdsmi_get_cpu_model = _raise
        args = argparse.Namespace(gpu=None, cpu=[_FakeHandle(1)], core=None)
        commands.list_devices(args)
        self.assertEqual(commands.logger.stored["family"], "N/A")
        self.assertEqual(commands.logger.stored["model"], "N/A")
        self.assertEqual(commands.logger.stored["stepping"], "0x2")

    def test_no_cpu_or_core_does_not_list_cpu(self):
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
            # provide. Reaching that point proves the cpu/core path didn't
            # fire, which is all this test verifies.
            pass
        self.assertNotIn("family", commands.logger.stored)
