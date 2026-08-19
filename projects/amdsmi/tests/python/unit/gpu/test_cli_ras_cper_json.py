#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi ras --cper --json`` output validity.

These drive ``RasCommands.ras`` with the C library faked at the driver boundary
but a real ``AMDSMILogger`` (json/stdout) and real ``AMDSMIHelpers``, so the JSON
contract is verified without GPU hardware or the compiled package:

* no CPER entries -> exactly one ``[]`` document (``json.loads`` yields ``[]``)
* one entry per GPU across two GPUs -> a single top-level JSON list, not one
  array per GPU (which ``json.loads`` rejects with "Extra data")
* every GPU on a non-primary partition -> still one valid JSON document, with no
  human-readable ``WARNING`` text leaking into stdout

The CLI is loaded from the source tree (not the installed copy) so the suite
exercises the source CLI directly; it skips if that source file is absent.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest

from common.common import amdsmi_path


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Source tree: tests/python/unit/gpu -> repo root is four levels up.
_SRC_CLI_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
# Installed tree: <rocm>/share/amd_smi/amdsmi -> <rocm>/libexec/amdsmi_cli.
_INSTALLED_CLI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli"
)
# Prefer the source copy when present so a source-tree run tests local edits;
# otherwise fall back to the installed CLI (how the suite runs in CI).
_CLI_DIR = (
    _SRC_CLI_DIR
    if os.path.isfile(os.path.join(_SRC_CLI_DIR, "subcommands", "ras.py"))
    else _INSTALLED_CLI_DIR
)
_RAS_SRC = os.path.join(_CLI_DIR, "subcommands", "ras.py")

# Modules imported (directly or transitively) when the source CLI loads against
# the faked ``amdsmi`` package. They are snapshotted and cleared around the suite
# so a real/installed copy loaded by a sibling test is never shadowed.
_CLI_MODULES = (
    "amdsmi",
    "amdsmi.amdsmi_interface",
    "amdsmi.amdsmi_exception",
    "amdsmi_init",
    "amdsmi_helpers",
    "amdsmi_logger",
    "amdsmi_cli_exceptions",
    "BDF",
)


class _FakeHandle:
    """Processor-handle stand-in: ``get_gpu_id_from_device_handle`` keys on ``.value``."""

    def __init__(self, value):
        self.value = value


class _FakeLibraryException(Exception):
    def __init__(self, code=0, message="mock error"):
        super().__init__(message)
        self._code = code
        self._message = message

    def get_error_code(self):
        return self._code

    def get_error_info(self):
        return self._message


class _FakeParameterException(Exception):
    def __init__(self, *args):
        super().__init__("mock parameter error")


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeClkType:
    """Enough ``AmdSmiClkType`` members for ``AMDSMIHelpers.__init__`` to build its map."""

    SYS = "SYS"
    MEM = "MEM"
    DF = "DF"
    SOC = "SOC"
    DCEF = "DCEF"
    VCLK0 = "VCLK0"
    VCLK1 = "VCLK1"
    DCLK0 = "DCLK0"
    DCLK1 = "DCLK1"


class _FakeWrapper:
    # Mock status codes; values are arbitrary and only ever compared, never interpreted.
    AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF
    AMDSMI_STATUS_NO_PERM = 8
    AMDSMI_STATUS_NOT_SUPPORTED = 24
    AMDSMI_STATUS_FILE_NOT_FOUND = 25
    AMDSMI_STATUS_FILE_ERROR = 26
    AMDSMI_STATUS_INVAL = 3
    AMDSMI_STATUS_UNEXPECTED_SIZE = 27
    AMDSMI_STATUS_UNEXPECTED_DATA = 28
    AMDSMI_STATUS_NOT_INIT = 29
    AMDSMI_STATUS_DRIVER_NOT_LOADED = 30
    amdsmi_processor_handle = _FakeHandle


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so the source CLI imports without the
    compiled library, and return the fake ``amdsmi_interface`` for per-test wiring."""
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.amdsmi_wrapper = _FakeWrapper
    interface.AmdSmiInitFlags = _FakeInitFlags
    interface.AmdSmiClkType = _FakeClkType
    interface.AmdSmiLibraryException = _FakeLibraryException
    interface.AmdSmiParameterException = _FakeParameterException

    # Library lifecycle: amdsmi_init.py runs amdsmi_cli_init() at import and
    # registers amdsmi_shut_down() via atexit; both are inert here.
    interface.amdsmi_init = lambda _flag: None
    interface.amdsmi_shut_down = lambda: None

    # Driver entry points touched by the --cper --json path. Defaults keep the
    # path inert (no GPUs, no entries); individual tests override as needed.
    interface.amdsmi_get_processor_handles = lambda: []
    interface.amdsmi_get_gpu_kfd_info = lambda _h: {"current_partition_id": 0}
    interface.amdsmi_get_gpu_cper_entries = lambda _h, _m, _s, cursor: ({}, cursor, [], 0)
    interface.amdsmi_get_afids_from_cper = lambda _raw: ([], 0)

    def _bdf_unavailable(_handle):
        # Non-primary path falls back to "no primary id" when the BDF lookup fails.
        raise _FakeLibraryException(message="no bdf in mock")

    interface.amdsmi_get_gpu_device_bdf = _bdf_unavailable

    exception.AmdSmiLibraryException = _FakeLibraryException
    exception.AmdSmiParameterException = _FakeParameterException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    return interface


def _load_ras_module():
    spec = importlib.util.spec_from_file_location("ras_cper_json_under_test", _RAS_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_ras_args(handles, **overrides):
    defaults = dict(
        gpu=list(handles),
        cper=True,
        afid=False,
        severity=["fatal"],
        folder=None,
        file_limit=None,
        cper_file=None,
        follow=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliRasCperJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_RAS_SRC):
            raise unittest.SkipTest(f"amd-smi CLI ras.py not found at {_RAS_SRC}")

        # Clear the CLI modules so the fake amdsmi wins the import race, keeping a
        # snapshot to restore afterwards.
        cls._saved_modules = {name: sys.modules.get(name) for name in _CLI_MODULES}
        for name in _CLI_MODULES:
            sys.modules.pop(name, None)
        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        cls.interface = _install_fake_amdsmi()
        import amdsmi_helpers
        import amdsmi_logger

        cls.helpers_cls = amdsmi_helpers.AMDSMIHelpers
        cls.logger_cls = amdsmi_logger.AMDSMILogger
        cls.ras_module = _load_ras_module()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)
        for name, mod in getattr(cls, "_saved_modules", {}).items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def setUp(self):
        self._handles = [_FakeHandle(10), _FakeHandle(20)]
        self.interface.amdsmi_get_processor_handles = lambda: list(self._handles)
        # Default per test: primary partition, no CPER entries.
        self.interface.amdsmi_get_gpu_kfd_info = lambda _h: {"current_partition_id": 0}
        self.interface.amdsmi_get_gpu_cper_entries = lambda _h, _m, _s, cursor: ({}, cursor, [], 0)

    def _make_commands(self):
        commands = object.__new__(self.ras_module.RasCommands)
        commands.logger = self.logger_cls(format="json", destination="stdout")
        commands.helpers = self.helpers_cls()
        commands.group_check_printed = True  # skip the group-permission probe
        return commands

    def _run_ras(self, commands, args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            commands.ras(args)
        return buffer.getvalue()

    def test_no_entries_emits_empty_json_array(self):
        commands = self._make_commands()
        captured = self._run_ras(commands, _build_ras_args(self._handles))

        self.assertEqual(
            captured.strip(),
            "[]",
            f"--cper --json with no entries must print exactly '[]', got: {captured!r}",
        )
        self.assertEqual(json.loads(captured), [])

    def test_one_entry_per_gpu_is_single_json_document(self):
        # One CPER entry on the first fetch per handle, then empty to end the loop.
        counts = {}

        def _entries(handle, _mask, _size, cursor):
            seen = counts.get(handle.value, 0)
            counts[handle.value] = seen + 1
            if seen == 0:
                entry = {
                    "timestamp": "2026/07/18 00:00:00",
                    "error_severity": "fatal",
                    "notify_type": "RUNTIME",
                }
                return ({0: entry}, cursor + 1, [b""], 0)
            return ({}, cursor, [], 0)

        self.interface.amdsmi_get_gpu_cper_entries = _entries

        commands = self._make_commands()
        captured = self._run_ras(commands, _build_ras_args(self._handles))

        # A single top-level JSON value. This documents the aggregated-array
        # contract (one row per GPU in one document); json.loads would reject a
        # per-GPU-array layout with "Extra data".
        parsed = json.loads(captured)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2, f"expected one row per GPU, got: {parsed!r}")
        for row in parsed:
            self.assertEqual(row["severity"], "FATAL")

    def test_non_primary_partition_still_valid_json(self):
        # Every GPU on a non-primary partition: the human-readable warning must
        # not leak into JSON stdout, and an empty array is still emitted.
        self.interface.amdsmi_get_gpu_kfd_info = lambda _h: {"current_partition_id": 1}

        commands = self._make_commands()
        captured = self._run_ras(commands, _build_ras_args(self._handles))

        self.assertNotIn("WARNING", captured)
        self.assertEqual(json.loads(captured), [])

    def test_follow_mode_with_no_entries_stays_silent(self):
        # --follow with no entries must not spam "[]" every poll interval. Break
        # out of the otherwise-infinite poll loop by raising from time.sleep.
        class _StopLoop(Exception):
            pass

        original_sleep = self.ras_module.time.sleep
        self.ras_module.time.sleep = lambda _seconds: (_ for _ in ()).throw(_StopLoop())
        try:
            commands = self._make_commands()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                with self.assertRaises(_StopLoop):
                    commands.ras(_build_ras_args(self._handles, follow=True))
            self.assertEqual(buffer.getvalue(), "")
        finally:
            self.ras_module.time.sleep = original_sleep

    def test_afid_folder_all_files_skipped_emits_empty_json(self):
        # --afid --folder --json where every *.cper is a symlink: O_NOFOLLOW
        # skips them all, so results is empty. It must still print exactly `[]`
        # so json.loads consumers don't choke on empty output.
        with tempfile.TemporaryDirectory() as folder:
            link = os.path.join(folder, "planted.cper")
            os.symlink(os.devnull, link)

            commands = self._make_commands()
            args = argparse.Namespace(folder=folder)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                commands._decode_afid_folder(args)
            captured = buffer.getvalue()

        self.assertEqual(captured.strip(), "[]")
        self.assertEqual(json.loads(captured), [])
