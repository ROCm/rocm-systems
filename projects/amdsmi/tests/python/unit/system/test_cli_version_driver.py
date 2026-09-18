#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amd-smi version driver field tests."""

import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_date": "2026/09/16 00:00",
    "driver_kernel_version": "6.19.14",
    "driver_version": "31400000",
    "driver_build_version": "2370381",
    "driver_full_version": "6.19.14.31400000-2370381",
}


class _LibraryError(RuntimeError):
    def get_error_info(self) -> str:
        return "not supported"


class _Logger:
    def __init__(self, human_readable: bool = False) -> None:
        self.output = {}
        self.destination = "stdout"
        self._human_readable = human_readable

    def is_human_readable_format(self) -> bool:
        return self._human_readable

    def is_json_format(self) -> bool:
        return not self._human_readable

    def is_csv_format(self) -> bool:
        return False

    def print_output(self) -> None:
        return None


def _default_interface() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        amdsmi_get_lib_version=lambda: {"major": 27, "minor": 1, "release": 0},
        amdsmi_get_rocm_version=lambda: (True, "10.1.0"),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_driver_info=lambda _gpu: _DRIVER_INFO,
    )


def _load_version_module(interface: types.SimpleNamespace) -> types.ModuleType:
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
        spec = importlib.util.spec_from_file_location("version_driver_under_test", path)
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


def _run_version(gpu_version: bool = True, human_readable: bool = False) -> dict:
    return _run_version_with_interface(
        _default_interface(), gpu_version=gpu_version, human_readable=human_readable
    )


def _run_version_with_interface(
    interface: types.SimpleNamespace, gpu_version: bool = True, human_readable: bool = False
) -> dict:
    module = _load_version_module(interface)
    commands = object.__new__(module.VersionCommands)
    commands.logger = _Logger(human_readable=human_readable)
    commands.helpers = types.SimpleNamespace()
    commands.group_check_printed = True
    args = types.SimpleNamespace(gpu_version=gpu_version, cpu_version=False, nic_version=False)
    commands.version(args)
    return commands.logger.output


class TestVersionDriverOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[4] / "amdsmi_cli/subcommands/version.py"
        if not path.is_file():
            raise unittest.SkipTest(f"amd-smi CLI version.py not found at {path}")

    def test_json_reports_driver_fields(self) -> None:
        output = _run_version()
        self.assertEqual(output["driver_full_version"], "6.19.14.31400000-2370381")
        self.assertNotIn("driver_kernel_version", output)
        self.assertNotIn("driver_version", output)
        self.assertNotIn("driver_build_version", output)
        self.assertNotIn("amdgpu_version", output)
        self.assertNotIn("amdgpu_dkms_version", output)

    def test_human_readable_reports_driver_fields_in_order(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            _run_version(human_readable=True)

        line = stdout.getvalue()
        self.assertIn("AMDGPU Version: 6.19.14.31400000-2370381", line)
        self.assertNotIn("Kernel version:", line)
        self.assertNotIn("Driver version:", line)
        self.assertNotIn("Build version:", line)

    def test_gpu_fields_are_na_when_gpu_version_is_disabled(self) -> None:
        output = _run_version(gpu_version=False)
        self.assertEqual(output["driver_full_version"], "N/A")
        self.assertNotIn("driver_kernel_version", output)
        self.assertNotIn("driver_version", output)
        self.assertNotIn("driver_build_version", output)

    def test_empty_gpu_list_reports_full_version_as_na(self) -> None:
        interface = _default_interface()
        interface.amdsmi_get_processor_handles = lambda: []

        output = _run_version_with_interface(interface)

        self.assertEqual(output["driver_full_version"], "N/A")

    def test_driver_info_failure_reports_full_version_as_na(self) -> None:
        interface = _default_interface()

        def raise_driver_info(_gpu: object) -> None:
            raise _LibraryError("not supported")

        interface.amdsmi_get_gpu_driver_info = raise_driver_info

        output = _run_version_with_interface(interface)

        self.assertEqual(output["driver_full_version"], "N/A")


if __name__ == "__main__":
    unittest.main()
