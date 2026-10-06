#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``--cpu``/``--core`` rejection across GPU-only subcommands.

``--cpu`` and ``--core`` are accepted by the shared device-argument group used
by every subcommand (see ``AMDSMIParser._add_device_arguments``), but
``bad-pages``, ``fabric``, ``firmware``, ``xgmi``, ``partition``, ``topology``,
``process``, and ``monitor`` never implemented a CPU/core path. Passing either
flag left ``args.gpu`` unset, so each command defaulted to "no GPU specified"
and quietly reported GPU data instead of rejecting the request.

Each entry point now rejects ``--cpu``/``--core`` up front with
``AmdSmiCommandNotSupportedException`` (``COMMAND_NOT_SUPPORTED`` exit code),
mirroring the fix already applied to ``amd-smi list`` (see
``test_cli_list_cpu_unsupported.py``).

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

# (module filename, class name, entry method name, command label).
CASES = [
    ("bad_pages.py", "BadPagesCommands", "bad_pages", "bad-pages"),
    ("fabric.py", "FabricCommands", "fabric", "fabric"),
    ("firmware.py", "FirmwareCommands", "firmware", "firmware"),
    ("xgmi.py", "XgmiCommands", "xgmi", "xgmi"),
    ("partition.py", "PartitionCommands", "partition", "partition"),
    ("topology.py", "TopologyCommands", "topology", "topology"),
    ("process.py", "ProcessCommands", "process", "process"),
    ("monitor.py", "MonitorCommands", "monitor", "monitor"),
]

# monitor.py does ``from amdsmi_helpers import AMDSMIHelpers``, and
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


class TestCliCpuUnsupportedSubcommands(unittest.TestCase):
    CommandNotSupported: Any = (
        cli_exc.AmdSmiCommandNotSupportedException if cli_exc is not None else None
    )
    expected_exit_code: Any = (
        int(cli_exc.AmdSmiExitCode.COMMAND_NOT_SUPPORTED) if cli_exc is not None else None
    )

    def _load_commands(self, filename, class_name, suffix):
        path = os.path.join(_SUBCOMMANDS_DIR, filename)
        module = load_cli_module(f"{class_name}_{suffix}", path, _SUBCOMMANDS_DIR)
        commands = getattr(module, class_name)()
        commands.helpers = _FakeHelpers()
        return commands

    def test_cpu_and_core_rejected_for_every_gpu_only_subcommand(self):
        for filename, class_name, method_name, command_label in CASES:
            with self.subTest(command=command_label):
                commands = self._load_commands(filename, class_name, "under_test")
                entry = getattr(commands, method_name)

                for device_kwargs in ({"cpu": ["0"], "core": None}, {"cpu": None, "core": ["0"]}):
                    args = argparse.Namespace(gpu=None, **device_kwargs)
                    with self.assertRaises(
                        self.CommandNotSupported,
                        msg=f"{command_label} did not reject {device_kwargs}",
                    ) as ctx:
                        entry(args)
                    self.assertEqual(ctx.exception.value, self.expected_exit_code)
                    self.assertIn("--cpu and --core are not supported", str(ctx.exception))

    def test_gpu_only_request_does_not_trip_the_cpu_core_guard(self):
        # A GPU-only Namespace (no cpu/core attrs at all) must not be mistaken
        # for a cpu/core request. Each entry point is expected to fail further
        # down (missing device_handles/etc. on this minimal stub) with
        # something other than AmdSmiCommandNotSupportedException.
        for filename, class_name, method_name, command_label in CASES:
            with self.subTest(command=command_label):
                commands = self._load_commands(filename, class_name, "gpu_only_under_test")
                entry = getattr(commands, method_name)

                args = argparse.Namespace(gpu=None)
                try:
                    entry(args)
                except self.CommandNotSupported:
                    self.fail(
                        f"{command_label} raised COMMAND_NOT_SUPPORTED for a GPU-only request"
                    )
                except Exception:
                    # Expected: the GPU path itself needs further attributes
                    # this minimal stub doesn't provide. Reaching that point
                    # proves the cpu/core guard didn't fire.
                    pass
