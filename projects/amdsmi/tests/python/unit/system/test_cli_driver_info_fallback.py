#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``AMDSMIHelpers.get_gpu_driver_info`` against libraries with and without the v2 call.

Imports the real ``amdsmi_cli/amdsmi_helpers.py`` over a faked ``amdsmi`` package (the
bootstrap ``test_amdsmi_helpers_ampp.py`` uses), since the commands' own tests stub the helper.
"""

import os
import sys
import unittest
import warnings

from common.common import (
    add_class_cleanup,
    cli_search_order,
    fake_module,
    find_cli_dir,
    stub_modules,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
_HELPERS_SRC = os.path.join(_CLI_DIR, "amdsmi_helpers.py") if _CLI_DIR else ""

# Removed up front so amdsmi_helpers and what it imports load against the fake package.
_CLI_MODULES = (
    "amdsmi",
    "amdsmi.amdsmi_interface",
    "amdsmi.amdsmi_exception",
    "amdsmi_init",
    "amdsmi_helpers",
    "amdsmi_cli_exceptions",
    "BDF",
)

_NOT_SUPPORTED = 2
_NO_PERM = 10
_V1_INFO = {
    "driver_name": "amdgpu",
    "driver_version": "6.19.14.31400000",
    "driver_date": "2015/01/01 00:00",
}
_V2_INFO = {
    **_V1_INFO,
    "driver_kernel_version": "6.19.14",
    "amdgpu_driver_version": "31400000",
    "driver_build_version": "2370381",
    "driver_full_version": "6.19.14.31400000-2370381",
    "driver_package": "amdgpu-dkms",
}


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeClkType:
    """Enough ``AmdSmiClkType`` members for ``amdsmi_helpers`` to import."""

    SYS = "SYS"
    MEM = "MEM"
    DF = "DF"
    SOC = "SOC"
    DCEF = "DCEF"
    VCLK0 = "VCLK0"
    VCLK1 = "VCLK1"
    DCLK0 = "DCLK0"
    DCLK1 = "DCLK1"


class _LibraryError(Exception):
    def __init__(self, code=_NOT_SUPPORTED):
        super().__init__(f"amdsmi error {code}")
        self.code = code

    def get_error_code(self):
        return self.code


def _build_fake_amdsmi_modules():
    # AMDSMIHelpers reads the status codes off amdsmi_wrapper while its class body runs.
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper",
        AMDSMI_STATUS_INVAL=1,
        AMDSMI_STATUS_NOT_SUPPORTED=_NOT_SUPPORTED,
        AMDSMI_STATUS_UNEXPECTED_SIZE=42,
        AMDSMI_STATUS_UNEXPECTED_DATA=43,
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        AmdSmiInitFlags=_FakeInitFlags,
        AmdSmiClkType=_FakeClkType,
        AmdSmiLibraryException=_LibraryError,
        AmdSmiParameterException=_LibraryError,
        AMDSMI_MAX_STRING_LENGTH=256,
        amdsmi_init=lambda _flag: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: [],
    )
    exception = fake_module(
        "amdsmi.amdsmi_exception",
        AmdSmiLibraryException=_LibraryError,
        AmdSmiParameterException=_LibraryError,
    )
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)

    modules = {name: None for name in _CLI_MODULES}
    modules["amdsmi"] = amdsmi_pkg
    modules["amdsmi.amdsmi_interface"] = interface
    modules["amdsmi.amdsmi_exception"] = exception
    return modules


class TestGetGpuDriverInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_HELPERS_SRC):
            raise unittest.SkipTest(f"amd-smi CLI amdsmi_helpers.py not found at {_HELPERS_SRC}")

        modules = _build_fake_amdsmi_modules()
        stub_modules(cls, modules)
        if _CLI_DIR not in sys.path:
            sys.path.insert(0, _CLI_DIR)
            add_class_cleanup(cls, sys.path.remove, _CLI_DIR)

        import amdsmi_helpers

        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.helpers = object.__new__(amdsmi_helpers.AMDSMIHelpers)

    def setUp(self):
        self.v1_calls = 0
        self.interface.amdsmi_get_gpu_driver_info_v2 = lambda _gpu: dict(_V2_INFO)
        self.interface.amdsmi_get_gpu_driver_info = self._v1

    def _v1(self, _gpu):
        self.v1_calls += 1
        warnings.warn("amdsmi_get_gpu_driver_info() is deprecated", DeprecationWarning)
        return dict(_V1_INFO)

    def _fail_v2(self, code):
        def raise_error(_gpu):
            raise _LibraryError(code)

        self.interface.amdsmi_get_gpu_driver_info_v2 = raise_error

    def _assert_v1_fallback(self, info):
        self.assertEqual(self.v1_calls, 1)
        self.assertEqual(
            info,
            {
                **_V1_INFO,
                "driver_kernel_version": "N/A",
                "amdgpu_driver_version": "N/A",
                "driver_build_version": "N/A",
                "driver_full_version": "6.19.14.31400000",
                "driver_package": "N/A",
            },
        )
        # The commands read the same keys whichever call answered.
        self.assertEqual(list(info), list(_V2_INFO))

    def test_uses_v2_when_the_library_has_it(self):
        self.assertEqual(self.helpers.get_gpu_driver_info("gpu0"), _V2_INFO)
        self.assertEqual(self.v1_calls, 0)

    def test_falls_back_to_v1_on_a_library_without_v2(self):
        self._fail_v2(_NOT_SUPPORTED)
        self._assert_v1_fallback(self.helpers.get_gpu_driver_info("gpu0"))

    def test_falls_back_to_v1_on_an_amdsmi_package_without_v2(self):
        del self.interface.amdsmi_get_gpu_driver_info_v2
        self._assert_v1_fallback(self.helpers.get_gpu_driver_info("gpu0"))

    def test_fallback_keeps_the_deprecation_warning_to_itself(self):
        self._fail_v2(_NOT_SUPPORTED)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._assert_v1_fallback(self.helpers.get_gpu_driver_info("gpu0"))

    def test_other_v2_errors_are_not_retried(self):
        self._fail_v2(_NO_PERM)
        with self.assertRaises(_LibraryError) as raised:
            self.helpers.get_gpu_driver_info("gpu0")
        self.assertEqual(raised.exception.get_error_code(), _NO_PERM)
        self.assertEqual(self.v1_calls, 0)

    def test_a_bug_inside_v2_is_not_mistaken_for_a_missing_call(self):
        def raise_attribute_error(_gpu):
            raise AttributeError("mock: bug inside amdsmi_get_gpu_driver_info_v2")

        self.interface.amdsmi_get_gpu_driver_info_v2 = raise_attribute_error
        with self.assertRaises(AttributeError):
            self.helpers.get_gpu_driver_info("gpu0")
        self.assertEqual(self.v1_calls, 0)

    def test_v1_error_reaches_the_caller(self):
        self._fail_v2(_NOT_SUPPORTED)

        def raise_error(_gpu):
            raise _LibraryError(_NOT_SUPPORTED)

        self.interface.amdsmi_get_gpu_driver_info = raise_error
        with self.assertRaises(_LibraryError):
            self.helpers.get_gpu_driver_info("gpu0")
