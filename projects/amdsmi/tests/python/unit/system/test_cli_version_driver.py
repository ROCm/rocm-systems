#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amd-smi version driver field tests."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_date": "2026/09/16 00:00",
    "driver_kernel_version": "6.19.14",
    "driver_version": "31400000",
    "driver_build_version": "2370381",
    "driver_full_version": "6.19.14.31400000-2370381",
}


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


def _load_version_module() -> types.ModuleType:
    interface = types.SimpleNamespace(
        amdsmi_get_lib_version=lambda: {"major": 27, "minor": 1, "release": 0},
        amdsmi_get_rocm_version=lambda: (True, "10.1.0"),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_driver_info=lambda _gpu: _DRIVER_INFO,
    )
    amdsmi = types.ModuleType("amdsmi")
    amdsmi.amdsmi_interface = interface
    amdsmi.amdsmi_exception = types.SimpleNamespace(AmdSmiLibraryException=RuntimeError)
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
    module = _load_version_module()
    commands = object.__new__(module.VersionCommands)
    commands.logger = _Logger(human_readable=human_readable)
    commands.helpers = types.SimpleNamespace()
    commands.group_check_printed = True
    args = types.SimpleNamespace(gpu_version=gpu_version, cpu_version=False, nic_version=False)
    commands.version(args)
    return commands.logger.output


class TestVersionDriverOutput(unittest.TestCase):
    def test_json_reports_driver_fields(self) -> None:
        output = _run_version()
        self.assertEqual(output["driver_kernel_version"], "6.19.14")
        self.assertEqual(output["driver_version"], "31400000")
        self.assertEqual(output["driver_build_version"], "2370381")
        self.assertEqual(output["driver_full_version"], "6.19.14.31400000-2370381")
        self.assertNotIn("amdgpu_version", output)
        self.assertNotIn("amdgpu_dkms_version", output)

    def test_human_readable_reports_driver_fields_in_order(self) -> None:
        with mock.patch("builtins.print") as printed:
            _run_version(human_readable=True)

        line = printed.call_args[0][0]
        kernel_index = line.index("Kernel version: 6.19.14")
        driver_index = line.index("Driver version: 31400000")
        build_index = line.index("Build version: 2370381")
        self.assertLess(kernel_index, driver_index)
        self.assertLess(driver_index, build_index)

    def test_gpu_fields_are_na_when_gpu_version_is_disabled(self) -> None:
        output = _run_version(gpu_version=False)
        self.assertEqual(output["driver_kernel_version"], "N/A")
        self.assertEqual(output["driver_version"], "N/A")
        self.assertEqual(output["driver_build_version"], "N/A")
        self.assertEqual(output["driver_full_version"], "N/A")


if __name__ == "__main__":
    unittest.main()
