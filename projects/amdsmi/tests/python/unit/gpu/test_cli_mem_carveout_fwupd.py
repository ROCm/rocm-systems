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

"""Mock-based unit tests for the fwupd fallback behind ``amd-smi ... --mem-carveout``.

On HP UEFI-HII systems (ZBook Ultra G1a, Z2 Mini G1a) the amdgpu
``/sys/class/drm/<card>/device/uma/carveout`` node is absent, so the C-backed
getter/setter returns NOT_SUPPORTED. The "Dedicated Graphics Memory" knob is
instead exposed through fwupd's BIOS-settings interface. These tests drive the
``fwupd_bios`` adapter and the CLI fallback paths in ``static.py`` /
``set_value.py`` with fwupd fully mocked, so they run with no GPU, no root, no
real fwupd, and no compiled ``amdsmi`` package.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

# Load the CLI straight from the source tree (not an installed copy) so the
# suite exercises the code under review without requiring an install, a GPU, or
# a compiled library.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CLI_DIR = _REPO_ROOT / "amdsmi_cli"
_STATIC_PATH = _CLI_DIR / "subcommands" / "static.py"
_SET_VALUE_PATH = _CLI_DIR / "subcommands" / "set_value.py"

_NOT_SUPPORTED = 2
_NO_PERM = 10

# HP "Dedicated Graphics Memory" enumeration. "32 GB" is the current value and
# sits at index 4 of the possible-values array.
_CARVEOUT_OPTIONS = ["512 MB", "4 GB", "8 GB", "16 GB", "32 GB", "64 GB", "96 GB"]
_DEFAULT_PAYLOAD = {
    "BiosSettings": [
        {
            # Decoy: an enumeration setting with a different name.
            "Name": "Thermal Profile",
            "BiosSettingCurrentValue": "Balanced",
            "BiosSettingType": 1,
            "BiosSettingReadOnly": "False",
            "BiosSettingPossibleValues": ["Quiet", "Balanced", "Performance"],
        },
        {
            "Name": "Dedicated Graphics Memory",
            "BiosSettingCurrentValue": "32 GB",
            "BiosSettingType": 1,
            "BiosSettingReadOnly": "False",
            "BiosSettingPossibleValues": list(_CARVEOUT_OPTIONS),
        },
    ]
}


class _FakeLibraryException(Exception):
    def __init__(self, error_code=_NOT_SUPPORTED, message="mock error"):
        super().__init__(message)
        self._error_code = error_code
        self._message = message

    def get_error_code(self):
        return self._error_code

    def get_error_info(self, detailed=True):
        return self._message


def _raise_not_supported(*_args, **_kwargs):
    raise _FakeLibraryException(_NOT_SUPPORTED, "uma carveout not supported")


def _install_fake_amdsmi():
    """Register stub ``amdsmi`` / ``amdsmi_helpers`` modules so the CLI loads.

    Returns the fake ``amdsmi_interface`` module plus a snapshot of the
    ``sys.modules`` entries we overwrote, so ``tearDownClass`` can restore them.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.amdsmi_wrapper = types.SimpleNamespace(
        AMDSMI_STATUS_NOT_SUPPORTED=_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM=_NO_PERM
    )
    # set_value.py binds these at import time.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 0
    interface.amdsmi_get_gpu_uma_carveout_info = _raise_not_supported
    interface.amdsmi_set_gpu_uma_carveout = lambda *_a, **_k: None

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    # static.py does ``from amdsmi_helpers import AMDSMIHelpers``; stub it so the
    # module loads without pulling in amdsmi_init and the compiled library.
    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = type("AMDSMIHelpers", (), {})

    new_modules = {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi_helpers": helpers_mod,
    }
    saved = {name: sys.modules.get(name) for name in new_modules}
    sys.modules.update(new_modules)
    return interface, saved


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def __init__(self, fmt="human"):
        self.fmt = fmt
        self.outputs = []
        self.printed = 0
        self.cleared = 0

    def is_json_format(self):
        return self.fmt == "json"

    def is_csv_format(self):
        return self.fmt == "csv"

    def is_human_readable_format(self):
        return self.fmt == "human"

    def store_output(self, _gpu, key, value):
        self.outputs.append((key, value))

    def print_output(self, *args, **kwargs):
        self.printed += 1

    def clear_multiple_devices_output(self):
        self.cleared += 1

    def last(self, key):
        for stored_key, value in reversed(self.outputs):
            if stored_key == key:
                return value
        return None


class _FakeHelpers:
    def __init__(self):
        self.reboot_prompts = 0

    def prompt_reboot(self):
        self.reboot_prompts += 1


def _fwupd_run(recorder, version="2.1.1", payload=None, set_result=(0, "", "")):
    """Build a ``subprocess.run`` stand-in that answers fwupdmgr invocations."""
    payload_json = json.dumps(_DEFAULT_PAYLOAD if payload is None else payload)

    def _run(cmd, *_args, **_kwargs):
        argv = list(cmd)
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "--version":
            return subprocess.CompletedProcess(argv, 0, f"client version:\t{version}\n", "")
        if sub == "get-bios-settings":
            return subprocess.CompletedProcess(argv, 0, payload_json, "")
        if sub == "set-bios-setting":
            recorder.append(argv[1:])
            returncode, stdout, stderr = set_result
            return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
        return subprocess.CompletedProcess(argv, 1, "", "unexpected command")

    return _run


class _CarveoutFwupdBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (_STATIC_PATH, _SET_VALUE_PATH):
            if not path.is_file():
                raise unittest.SkipTest(f"CLI source not found: {path}")
        if str(_CLI_DIR) not in sys.path:
            sys.path.insert(0, str(_CLI_DIR))
        cls.interface, cls._saved_modules = _install_fake_amdsmi()
        import fwupd_bios

        cls.fwupd_bios = fwupd_bios
        cls.static_mod = _load_module("static_under_test", str(_STATIC_PATH))
        cls.set_mod = _load_module("set_value_under_test", str(_SET_VALUE_PATH))

    @classmethod
    def tearDownClass(cls):
        for name, original in cls._saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def setUp(self):
        os.environ.pop("AMDSMI_DRY_RUN", None)

    def tearDown(self):
        os.environ.pop("AMDSMI_DRY_RUN", None)

    def _patch_fwupd(
        self,
        recorder=None,
        version="2.1.1",
        which="/usr/bin/fwupdmgr",
        payload=None,
        set_result=(0, "", ""),
    ):
        recorder = [] if recorder is None else recorder
        run = _fwupd_run(recorder, version=version, payload=payload, set_result=set_result)
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(self.fwupd_bios.subprocess, "run", run))
        stack.enter_context(mock.patch.object(self.fwupd_bios.shutil, "which", lambda _name: which))
        return recorder, stack


class TestFwupdBiosAdapter(_CarveoutFwupdBase):
    def test_get_parses_options_and_current_index(self):
        _recorder, stack = self._patch_fwupd()
        with stack:
            info = self.fwupd_bios.get_carveout_setting()
        self.assertIsNotNone(info)
        # (a) options come from BiosSettingPossibleValues, 0-indexed, array order.
        self.assertEqual(info["options"], _CARVEOUT_OPTIONS)
        # (b) the current index resolves from BiosSettingCurrentValue.
        self.assertEqual(info["current_value"], "32 GB")
        self.assertEqual(info["current_index"], 4)

    def test_get_rejects_same_name_wrong_type(self):
        payload = {
            "BiosSettings": [
                {
                    "Name": "Dedicated Graphics Memory",
                    "BiosSettingCurrentValue": "x",
                    "BiosSettingType": 2,
                    "BiosSettingReadOnly": "False",
                    "BiosSettingPossibleValues": ["x", "y"],
                }
            ]
        }
        _recorder, stack = self._patch_fwupd(payload=payload)
        with stack:
            self.assertIsNone(self.fwupd_bios.get_carveout_setting())

    def test_set_maps_index_to_value_string(self):
        # (c) index -> possible_values[index] STRING, passed to set-bios-setting.
        recorder, stack = self._patch_fwupd()
        with stack:
            self.fwupd_bios.set_carveout_setting(0)
        self.assertEqual(recorder, [["set-bios-setting", "Dedicated Graphics Memory", "512 MB"]])

    def test_dry_run_performs_no_write(self):
        # (d) AMDSMI_DRY_RUN suppresses the write.
        os.environ["AMDSMI_DRY_RUN"] = "1"
        recorder, stack = self._patch_fwupd()
        with stack:
            self.fwupd_bios.set_carveout_setting(1)
        self.assertEqual(recorder, [])

    def test_available_gates_on_version(self):
        # (e) read works from 1.8.4, write from 2.1.1.
        cases = [
            ("1.7.9", False, False),
            ("1.8.4", True, False),
            ("1.9.5", True, False),
            ("2.1.1", True, True),
        ]
        for version, read_ok, write_ok in cases:
            _recorder, stack = self._patch_fwupd(version=version)
            with stack:
                self.assertEqual(self.fwupd_bios.fwupd_available(), read_ok, version)
                self.assertEqual(
                    self.fwupd_bios.fwupd_available(require_write=True), write_ok, version
                )

    def test_absent_fwupdmgr_degrades(self):
        # (e) no fwupdmgr binary -> unavailable and no setting.
        _recorder, stack = self._patch_fwupd(which=None)
        with stack:
            self.assertFalse(self.fwupd_bios.fwupd_available())
            self.assertIsNone(self.fwupd_bios.get_carveout_setting())

    def test_set_refused_below_min_write_version(self):
        recorder, stack = self._patch_fwupd(version="1.9.5")
        with stack:
            with self.assertRaises(RuntimeError):
                self.fwupd_bios.set_carveout_setting(0)
        self.assertEqual(recorder, [])


class TestStaticCarveoutFwupdFallback(_CarveoutFwupdBase):
    def _run_static(self, fmt, **patch_kwargs):
        commands = object.__new__(self.static_mod.StaticCommands)
        commands.logger = _FakeLogger(fmt)
        static_dict = {}
        args = argparse.Namespace(mem_carveout=True, gpu=object())
        _recorder, stack = self._patch_fwupd(**patch_kwargs)
        with stack:
            commands._store_mem_carveout(args, static_dict, 0)
        return static_dict

    def test_static_human_shows_fwupd_options(self):
        static_dict = self._run_static("human")
        text = static_dict["mem_carveout"]
        for option in _CARVEOUT_OPTIONS:
            self.assertIn(option, text)
        self.assertIn("*[4] 32 GB", text)

    def test_static_json_tags_source_fwupd(self):
        static_dict = self._run_static("json")
        carveout = static_dict["mem_carveout"]
        self.assertEqual(carveout["current_index"], 4)
        self.assertEqual(carveout["source"], "fwupd")
        # Parity with the amdgpu path: each option carries a 0-based index
        # plus its description, in order.
        self.assertEqual(
            carveout["options"],
            [{"index": i, "description": d} for i, d in enumerate(_CARVEOUT_OPTIONS)],
        )

    def test_static_neither_interface_message(self):
        static_dict = self._run_static("human", which=None)
        self.assertIn("no UMA carveout interface", static_dict["mem_carveout"])


class TestSetCarveoutFwupdFallback(_CarveoutFwupdBase):
    def _run_set(self, index, **patch_kwargs):
        commands = object.__new__(self.set_mod.SetValueCommands)
        logger = _FakeLogger("human")
        helpers = _FakeHelpers()
        commands.logger = logger
        commands.helpers = helpers
        args = argparse.Namespace(mem_carveout=index, gpu=object())
        recorder, stack = self._patch_fwupd(**patch_kwargs)
        with stack:
            commands._set_mem_carveout(args)
        return logger, helpers, recorder

    def test_set_invokes_fwupdmgr_and_prompts_reboot(self):
        logger, helpers, recorder = self._run_set(0)
        self.assertEqual(recorder, [["set-bios-setting", "Dedicated Graphics Memory", "512 MB"]])
        self.assertIn("Successfully set", logger.last("mem_carveout"))
        self.assertEqual(helpers.reboot_prompts, 1)

    def test_set_already_set_no_write(self):
        logger, helpers, recorder = self._run_set(4)
        self.assertEqual(recorder, [])
        self.assertIn("already set", logger.last("mem_carveout"))
        self.assertEqual(helpers.reboot_prompts, 0)

    def test_set_write_version_too_old_refused(self):
        logger, helpers, recorder = self._run_set(0, version="1.9.5")
        self.assertEqual(recorder, [])
        self.assertIn("2.1.1", logger.last("mem_carveout"))
        self.assertEqual(helpers.reboot_prompts, 0)

    def test_set_dry_run_no_write(self):
        os.environ["AMDSMI_DRY_RUN"] = "1"
        _logger, _helpers, recorder = self._run_set(0)
        self.assertEqual(recorder, [])


if __name__ == "__main__":
    unittest.main()
