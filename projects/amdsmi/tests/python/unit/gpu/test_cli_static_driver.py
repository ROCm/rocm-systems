#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi static --driver``: the driver version with its build, and the driver package.

Drives ``StaticCommands.static_gpu`` with the library, logger and helpers stubbed, so it runs
without GPU hardware.
"""

import argparse
import os
import unittest

from common.common import (
    amdsmi_path,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None

_DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_full_version": "6.19.14.31400000-2370381",
    "driver_package": "amdgpu-dkms",
}
_NO_PACKAGE_DRIVER_INFO = {
    **_DRIVER_INFO,
    "driver_full_version": "6.19.14.31400000",
    "driver_package": "N/A",
}


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)


class _FakeLogger:
    """Captures the ``values`` payload ``static_gpu`` stores per GPU."""

    def __init__(self, fmt):
        self._fmt = fmt
        self.captured_values = None
        self.store_gpu_json_output = []

    def is_json_format(self):
        return self._fmt == "json"

    def is_csv_format(self):
        return self._fmt == "csv"

    def is_human_readable_format(self):
        return self._fmt == "human"

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass


class _FakeHelpers:
    def __init__(self, driver_info):
        # What the real helper returns, or the exception it raises.
        self._driver_info = driver_info

    def get_gpu_driver_info(self, _handle):
        if isinstance(self._driver_info, Exception):
            raise self._driver_info
        return self._driver_info

    def handle_gpus(self, args, _logger, _func):
        return False, args.gpu

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def check_required_groups(self):
        pass

    def is_linux(self):
        return True

    def is_baremetal(self):
        return False

    def is_virtual_os(self):
        return True

    def is_hypervisor(self):
        return False


class TestCliStaticDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        interface = fake_module(
            "amdsmi.amdsmi_interface", amdsmi_get_gpu_device_bdf=lambda _handle: "0000:00:00.0"
        )
        exception = fake_module(
            "amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException
        )
        stub_modules(
            cls,
            {
                "amdsmi": fake_module(
                    "amdsmi", amdsmi_interface=interface, amdsmi_exception=exception
                ),
                "amdsmi.amdsmi_interface": interface,
                "amdsmi.amdsmi_exception": exception,
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
                "amdsmi_cli_exceptions": fake_module(
                    "amdsmi_cli_exceptions",
                    AmdSmiInvalidParameterException=type(
                        "AmdSmiInvalidParameterException", (Exception,), {}
                    ),
                ),
            },
        )
        cls.static_module = load_cli_module("static_driver_under_test", STATIC_PATH)

    def _run_driver(self, fmt, driver_info=_DRIVER_INFO):
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger(fmt)
        commands.helpers = _FakeHelpers(driver_info)
        commands.group_check_printed = True
        args = argparse.Namespace(
            gpu=object(),
            asic=False,
            bus=False,
            vbios=False,
            driver=True,
            ras=False,
            vram=False,
            cache=False,
            board=False,
            process_isolation=False,
            clock=False,
            mem_carveout=False,
            partition=False,
            numa=False,
        )

        commands.static_gpu(args)

        if fmt == "json":
            return commands.logger.store_gpu_json_output[-1]["driver"]
        return commands.logger.captured_values["driver"]

    def test_json_reports_full_version_and_package(self):
        driver = self._run_driver("json")
        self.assertEqual(list(driver), ["name", "version", "os_kernel_version", "package"])
        self.assertEqual(driver["name"], "amdgpu")
        self.assertEqual(driver["version"], "6.19.14.31400000-2370381")
        self.assertEqual(driver["package"], "amdgpu-dkms")

    def test_human_readable_reports_full_version_and_package(self):
        driver = self._run_driver("human")
        self.assertEqual(list(driver), ["name", "version", "os_kernel_version", "package"])
        self.assertEqual(driver["version"], "6.19.14.31400000-2370381")
        self.assertEqual(driver["package"], "amdgpu-dkms")

    def test_json_and_csv_keep_the_package_key_without_a_package(self):
        for fmt in ("json", "csv"):
            with self.subTest(fmt=fmt):
                driver = self._run_driver(fmt, _NO_PACKAGE_DRIVER_INFO)
                self.assertEqual(list(driver), ["name", "version", "os_kernel_version", "package"])
                self.assertEqual(driver["version"], "6.19.14.31400000")
                self.assertEqual(driver["package"], "N/A")

    def test_human_readable_hides_package_without_a_package(self):
        driver = self._run_driver("human", _NO_PACKAGE_DRIVER_INFO)
        self.assertEqual(list(driver), ["name", "version", "os_kernel_version"])

    def test_library_error_reports_na(self):
        driver = self._run_driver("json", _FakeLibraryException("mock: driver info unavailable"))
        self.assertEqual(driver["name"], "N/A")
        self.assertEqual(driver["version"], "N/A")
        self.assertEqual(driver["package"], "N/A")
