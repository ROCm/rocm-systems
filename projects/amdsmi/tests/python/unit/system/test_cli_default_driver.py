#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bare amd-smi missing-driver tests."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

from common.common import cli_search_order, find_cli_dir, stub_modules_at_import

# default.py imports the CLI's own exception module, so the CLI dir must be importable
_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
if _CLI_DIR and _CLI_DIR not in sys.path:
    sys.path.append(_CLI_DIR)

_DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_date": "2026/09/16 00:00",
    "driver_version": "6.19.14.31400000",
}


class _LibraryError(Exception):
    def get_error_info(self) -> str:
        return "not supported"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_module(relative_path: str, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, _project_root() / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raise(*_args: object, **_kwargs: object) -> None:
    raise _LibraryError()


def _load_default_module() -> types.ModuleType:
    interface = types.SimpleNamespace(
        amdsmi_get_rocm_version=lambda: (True, "10.1.0"),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_driver_info=lambda _gpu: _DRIVER_INFO,
        amdsmi_get_fw_info=_raise,
        amdsmi_get_gpu_vbios_info=_raise,
        amdsmi_get_gpu_metrics_info=_raise,
        amdsmi_get_gpu_memory_partition=_raise,
        amdsmi_get_gpu_accelerator_partition_profile=_raise,
        amdsmi_get_gpu_asic_info=_raise,
        amdsmi_get_gpu_device_bdf=_raise,
        amdsmi_get_gpu_enumeration_info=_raise,
        amdsmi_get_power_cap_info=_raise,
        amdsmi_get_gpu_memory_usage=_raise,
        amdsmi_get_gpu_memory_total=_raise,
        amdsmi_get_gpu_total_ecc_count=_raise,
        amdsmi_get_gpu_fan_speed=_raise,
        amdsmi_get_gpu_fan_speed_max=_raise,
        amdsmi_get_gpu_process_list=_raise,
        _NA_amdsmi_get_gpu_metrics_info=lambda: "N/A",
    )
    amdsmi = types.ModuleType("amdsmi")
    amdsmi.amdsmi_interface = interface
    amdsmi.amdsmi_exception = types.SimpleNamespace(AmdSmiLibraryException=_LibraryError)
    version_metadata = types.ModuleType("_version")
    version_metadata.__version__ = "1.0.0"
    helpers_module = types.ModuleType("amdsmi_helpers")
    helpers_module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})

    restore = stub_modules_at_import(
        {"amdsmi": amdsmi, "_version": version_metadata, "amdsmi_helpers": helpers_module}
    )
    try:
        return _load_module("amdsmi_cli/subcommands/default.py", "default_driver_under_test")
    finally:
        restore()


def _run_default() -> dict:
    module = _load_default_module()
    commands = object.__new__(module.DefaultCommands)

    class _Logger:
        def is_json_format(self) -> bool:
            return True

        def is_csv_format(self) -> bool:
            return False

        def print_output(self) -> None:
            return None

    commands.logger = _Logger()
    commands.helpers = types.SimpleNamespace(
        is_amdgpu_initialized=lambda: True,
        check_required_groups=lambda: None,
        get_gpu_id_from_device_handle=lambda _gpu: 0,
        get_apu_memory_type_and_name=lambda _gpu, _gpu_id: (0, "VRAM"),
        convert_SI_unit=lambda value, _unit: value,
        convert_bytes_to_readable=lambda value: str(value),
        unit_format=lambda _logger, value, _unit: str(value),
    )
    commands.group_check_printed = True
    commands.default(types.SimpleNamespace())
    return commands.logger.output


def _run_default_without_amdgpu(gpus_without_driver: list) -> dict:
    module = _load_default_module()
    commands = object.__new__(module.DefaultCommands)
    commands.logger = types.SimpleNamespace(format="human_readable", output={})
    commands.helpers = types.SimpleNamespace(
        is_amdgpu_initialized=lambda: False,
        get_devices_without_driver=lambda _driver: gpus_without_driver,
    )
    commands.group_check_printed = True
    commands.default(types.SimpleNamespace())
    return commands.logger.output


class TestDefaultDriverHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = _project_root()
        for relative_path in ("amdsmi_cli/subcommands/default.py", "amdsmi_cli/amdsmi_logger.py"):
            if not (root / relative_path).is_file():
                raise unittest.SkipTest(f"amd-smi CLI source not found at {root / relative_path}")

    def test_default_payload_uses_driver_info(self) -> None:
        output = _run_default()
        self.assertEqual(output["version_info"]["amdgpu version"], _DRIVER_INFO)
        self.assertNotIn("amdgpu dkms version", output["version_info"])

    def test_missing_amdgpu_with_a_gpu_present_names_the_driver(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _run_default_without_amdgpu(gpus_without_driver=["0001:01:00.0"])

        self.assertEqual(type(ctx.exception).__name__, "AmdSmiDriverNotLoadedException")
        self.assertIn("requires the amdgpu driver", str(ctx.exception))

    def test_missing_amdgpu_without_a_gpu_prints_nothing(self) -> None:
        self.assertEqual(_run_default_without_amdgpu(gpus_without_driver=[]), {})
