#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bare amd-smi driver version row tests."""

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

    old_modules = {name: sys.modules.get(name) for name in ("amdsmi", "_version", "amdsmi_helpers")}
    sys.modules["amdsmi"] = amdsmi
    sys.modules["_version"] = version_metadata
    sys.modules["amdsmi_helpers"] = helpers_module
    try:
        return _load_module("amdsmi_cli/subcommands/default.py", "default_driver_under_test")
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


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


def _load_logger_module() -> types.ModuleType:
    helpers_module = types.ModuleType("amdsmi_helpers")
    helpers_module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    old_helpers = sys.modules.get("amdsmi_helpers")
    sys.modules["amdsmi_helpers"] = helpers_module
    try:
        return _load_module("amdsmi_cli/amdsmi_logger.py", "logger_driver_under_test")
    finally:
        if old_helpers is None:
            sys.modules.pop("amdsmi_helpers", None)
        else:
            sys.modules["amdsmi_helpers"] = old_helpers


def _banner_payload() -> dict:
    return {
        "version_info": {
            "amd-smi": "27.1.0",
            "driver info": _DRIVER_INFO,
            "kernel version": "6.8.0-124-generic",
            "fw pldm version": "N/A",
            "vbios version": "N/A",
            "rocm version": (True, "10.1.0"),
        },
        "gpu_info_list": [],
        "processes": [],
    }


class TestDefaultDriverHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = _project_root()
        for relative_path in ("amdsmi_cli/subcommands/default.py", "amdsmi_cli/amdsmi_logger.py"):
            if not (root / relative_path).is_file():
                raise unittest.SkipTest(f"amd-smi CLI source not found at {root / relative_path}")

    def test_default_payload_uses_driver_info(self) -> None:
        output = _run_default()
        self.assertEqual(output["version_info"]["driver info"], _DRIVER_INFO)
        self.assertNotIn("amdgpu dkms version", output["version_info"])

    def test_banner_prints_driver_rows_in_order(self) -> None:
        module = _load_logger_module()
        logger = module.AMDSMILogger(
            helpers=types.SimpleNamespace(os_info=lambda: "Linux Baremetal")
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            logger.print_default_output(_banner_payload())

        lines = stdout.getvalue().splitlines()
        amdgpu_index = next(
            i for i, line in enumerate(lines) if line.startswith("| AMDGPU Version:")
        )
        self.assertIn("6.19.14.31400000-2370381", lines[amdgpu_index])
        self.assertTrue(lines[amdgpu_index + 1].startswith("| ROCm Version:"))
        self.assertFalse(any(line.startswith("| Kernel Version:") for line in lines))
        self.assertFalse(any(line.startswith("| Driver Version:") for line in lines))
        self.assertFalse(any(line.startswith("| Build Version:") for line in lines))

    def test_banner_uses_os_kernel_when_driver_info_is_unavailable(self) -> None:
        module = _load_logger_module()
        logger = module.AMDSMILogger(
            helpers=types.SimpleNamespace(os_info=lambda: "Linux Baremetal")
        )
        payload = _banner_payload()
        payload["version_info"]["driver info"] = "N/A"
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            logger.print_default_output(payload)

        lines = stdout.getvalue().splitlines()
        os_kernel_line = next(line for line in lines if line.startswith("| OS kernel Version:"))
        self.assertIn("6.8.0-124-generic", os_kernel_line)
        self.assertFalse(any(line.startswith("| AMDGPU Version:") for line in lines))


if __name__ == "__main__":
    unittest.main()
