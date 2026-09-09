#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amd-smi version active DKMS output tests."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class _LibraryError(Exception):
    def get_error_info(self):
        return "not supported"


class _Logger:
    def __init__(self, human_readable=False):
        self.output = {}
        self.destination = "stdout"
        self._human_readable = human_readable

    def is_human_readable_format(self):
        return self._human_readable

    def is_json_format(self):
        return not self._human_readable

    def is_csv_format(self):
        return False

    def print_output(self):
        return None


def _load_version_module(dkms_getter):
    interface = types.SimpleNamespace(
        amdsmi_get_lib_version=lambda: {"major": 27, "minor": 1, "release": 0},
        amdsmi_get_rocm_version=lambda: (True, "10.1.0"),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_driver_info=lambda _gpu: {"driver_version": "6.19.20"},
        amdsmi_get_amdgpu_dkms_version=dkms_getter,
    )
    amdsmi = types.ModuleType("amdsmi")
    amdsmi.amdsmi_interface = interface
    amdsmi.amdsmi_exception = types.SimpleNamespace(AmdSmiLibraryException=_LibraryError)
    version_metadata = types.ModuleType("_version")
    version_metadata.__version__ = "1.0.0"

    old_amdsmi = sys.modules.get("amdsmi")
    old_version = sys.modules.get("_version")
    sys.modules["amdsmi"] = amdsmi
    sys.modules["_version"] = version_metadata
    try:
        path = Path(__file__).resolve().parents[4] / "amdsmi_cli/subcommands/version.py"
        spec = importlib.util.spec_from_file_location("version_dkms_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_amdsmi is None:
            sys.modules.pop("amdsmi", None)
        else:
            sys.modules["amdsmi"] = old_amdsmi
        if old_version is None:
            sys.modules.pop("_version", None)
        else:
            sys.modules["_version"] = old_version


def _run_version(dkms_getter, gpu_version=True, human_readable=False):
    module = _load_version_module(dkms_getter)
    commands = object.__new__(module.VersionCommands)
    commands.logger = _Logger(human_readable=human_readable)
    commands.helpers = types.SimpleNamespace()
    commands.group_check_printed = True
    args = types.SimpleNamespace(gpu_version=gpu_version, cpu_version=False, nic_version=False)
    commands.version(args)
    return commands.logger.output


class TestVersionDkmsOutput(unittest.TestCase):
    def test_reports_active_dkms_version(self):
        output = _run_version(lambda: "6.19.20-2450390.24.04")
        self.assertEqual(output["amdgpu_dkms_version"], "6.19.20-2450390.24.04")

    def test_reports_na_when_active_dkms_version_is_unsupported(self):
        def _unsupported():
            raise _LibraryError()

        output = _run_version(_unsupported)
        self.assertEqual(output["amdgpu_dkms_version"], "N/A")

    def test_keeps_na_when_gpu_version_is_disabled(self):
        def _should_not_run():
            raise AssertionError("dkms getter should not run")

        output = _run_version(_should_not_run, gpu_version=False)
        self.assertEqual(output["amdgpu_dkms_version"], "N/A")

    def test_human_readable_includes_amdgpu_dkms_version(self):
        with mock.patch("builtins.print") as printed:
            _run_version(lambda: "6.19.20-2450390.24.04", human_readable=True)
        printed.assert_called_once()
        self.assertIn("amdgpu dkms version: 6.19.20-2450390.24.04", printed.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
