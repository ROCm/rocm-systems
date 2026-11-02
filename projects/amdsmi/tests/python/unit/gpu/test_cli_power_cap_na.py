#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_power_cap_info() reports unavailable caps as "N/A" (e.g. WSL);
the CLI power cap helpers must render that, not raise TypeError."""

import os
import sys
import types
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))

_STATUS_NOT_SUPPORTED = 2
_STATUS_NO_PERM = 10
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class _FakeLibraryException(Exception):
    def __init__(self, err_code=_STATUS_NOT_SUPPORTED):
        super().__init__(err_code)
        self.err_code = err_code

    def get_error_code(self):
        return self.err_code

    def get_error_info(self, detailed=True):
        return "AMDSMI_STATUS_NOT_SUPPORTED"


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeLogger:
    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False


def _fake_amdsmi_modules():
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper",
        AMDSMI_STATUS_NOT_SUPPORTED=_STATUS_NOT_SUPPORTED,
        AMDSMI_STATUS_NO_PERM=_STATUS_NO_PERM,
        AMDSMI_STATUS_INVAL=5,
        AMDSMI_STATUS_UNEXPECTED_SIZE=42,
        AMDSMI_STATUS_UNEXPECTED_DATA=43,
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        MaxUIntegerTypes=types.SimpleNamespace(UINT64_T=_UINT64_MAX),
        AmdSmiInitFlags=_FakeInitFlags,
        AmdSmiLibraryException=_FakeLibraryException,
        AmdSmiParameterException=type("AmdSmiParameterException", (Exception,), {}),
        amdsmi_init=lambda _flag: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_supported_power_cap=lambda _h: {"sensor_inds": [0]},
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


def _power_cap_info(power_cap="N/A"):
    return {
        "power_cap": power_cap,
        "default_power_cap": "N/A",
        "dpm_cap": "N/A",
        "min_power_cap": "N/A",
        "max_power_cap": "N/A",
    }


class TestPowerCapNA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        modules = _fake_amdsmi_modules()
        modules.update(
            {
                name: None
                for name in ("amdsmi_init", "amdsmi_helpers", "amdsmi_cli_exceptions", "BDF")
            }
        )
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        import amdsmi_helpers

        cls.helpers_class = amdsmi_helpers.AMDSMIHelpers

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def setUp(self):
        self.set_calls = []
        self.interface.amdsmi_set_power_cap = lambda *args: self.set_calls.append(args)
        self.helpers = object.__new__(self.helpers_class)
        self.helpers.get_gpu_id_from_device_handle = lambda _h: 0
        self.helpers.error_collector = types.SimpleNamespace(record_library_error=lambda _c: None)

    def test_convert_si_unit_passes_na_through(self):
        self.assertEqual(
            self.helpers.convert_SI_unit("N/A", self.helpers_class.SI_Unit.MICRO), "N/A"
        )
        self.assertEqual(
            self.helpers.convert_SI_unit(213000000, self.helpers_class.SI_Unit.MICRO), 213
        )

    def test_get_power_caps_reports_na_limits(self):
        self.interface.amdsmi_get_power_cap_info = lambda _h, _s: _power_cap_info(213000000)
        self.assertEqual(self.helpers.get_power_caps(), ("N/A", "N/A", "N/A", "N/A"))

    def test_validate_and_set_power_cap_rejects_na_without_setting(self):
        self.interface.amdsmi_get_power_cap_info = lambda _h, _s: _power_cap_info()
        message = self.helpers.validate_and_set_power_cap("gpu0", 0, "PPT0", 100, _FakeLogger())
        self.assertIn("Unable to set PPT0 power cap to 100W", message)
        self.assertEqual(self.set_calls, [])
