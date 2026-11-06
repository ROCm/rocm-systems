#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_rocm_version() status-branching unit tests."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_PACKAGE_DIR = Path(__file__).resolve().parents[4] / "py-interface"


def _load_source_package() -> types.ModuleType:
    package = types.ModuleType("amdsmi_rocm_version_under_test")
    package.__path__ = [str(_PACKAGE_DIR)]
    sys.modules[package.__name__] = package
    package.amdsmi_wrapper = importlib.import_module(f"{package.__name__}.amdsmi_wrapper")
    package.amdsmi_interface = importlib.import_module(f"{package.__name__}.amdsmi_interface")
    return package


amdsmi = _load_source_package() if _PACKAGE_DIR.is_dir() else None


class TestAmdSmiGetRocmVersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if amdsmi is None:
            raise unittest.SkipTest(f"amdsmi py-interface source not found at {_PACKAGE_DIR}")

    def test_success_formats_version_string(self) -> None:
        def _stub(version_ptr):
            version_ptr._obj.major = 7
            version_ptr._obj.minor = 0
            version_ptr._obj.release = 0
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_rocm_version", _stub):
            result = amdsmi.amdsmi_interface.amdsmi_get_rocm_version()

        self.assertEqual(result, (True, "7.0.0"))

    def test_not_supported_reports_missing_rocm_core(self) -> None:
        def _stub(_version_ptr):
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_rocm_version", _stub):
            result = amdsmi.amdsmi_interface.amdsmi_get_rocm_version()

        self.assertEqual(result, (False, "Could not find librocm-core.so"))

    def test_other_failure_status_reports_unpack_failure(self) -> None:
        def _stub(_version_ptr):
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_API_FAILED

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_rocm_version", _stub):
            result = amdsmi.amdsmi_interface.amdsmi_get_rocm_version()

        self.assertEqual(result, (False, "Failed to unpack ROCm version"))

    def test_os_error_reports_core_library_failed_to_load(self) -> None:
        # Mirrors the amdsmi_wrapper sentinel installed when libamd_smi.so itself
        # could not be dlopen'd: every symbol raises OSError on call.
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_get_rocm_version",
            side_effect=OSError("AMD SMI shared library could not be loaded."),
        ):
            result = amdsmi.amdsmi_interface.amdsmi_get_rocm_version()

        self.assertEqual(result, (False, "libamd_smi.so failed to load"))

    def test_attribute_error_reports_symbol_unavailable(self) -> None:
        # Mirrors an older libamd_smi.so build that predates this C API: the
        # wrapper module never binds the symbol at all.
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_get_rocm_version",
            side_effect=AttributeError("amdsmi_get_rocm_version"),
        ):
            result = amdsmi.amdsmi_interface.amdsmi_get_rocm_version()

        self.assertEqual(
            result,
            (False, "amdsmi_get_rocm_version is not available in this build of libamd_smi.so"),
        )
