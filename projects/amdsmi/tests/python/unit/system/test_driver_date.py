#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU driver date field tests."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_PACKAGE_DIR = Path(__file__).resolve().parents[4] / "py-interface"


def _load_source_package() -> types.ModuleType:
    package = types.ModuleType("amdsmi_driver_date_under_test")
    package.__path__ = [str(_PACKAGE_DIR)]
    sys.modules[package.__name__] = package
    package.amdsmi_wrapper = importlib.import_module(f"{package.__name__}.amdsmi_wrapper")
    package.amdsmi_interface = importlib.import_module(f"{package.__name__}.amdsmi_interface")
    return package


amdsmi = _load_source_package() if _PACKAGE_DIR.is_dir() else None


class TestDriverDate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if amdsmi is None:
            raise unittest.SkipTest(f"amdsmi py-interface source not found at {_PACKAGE_DIR}")

    def test_empty_driver_date_renders_na(self) -> None:
        handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _stub(_handle: object, info_ptr: object) -> int:
            info_ptr._obj.driver_name = b"amdgpu"
            info_ptr._obj.driver_version = b"6.8.0"
            info_ptr._obj.driver_date = b""
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_driver_info", _stub):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_driver_info(handle)

        self.assertEqual(info["driver_date"], "N/A")
        self.assertEqual(info["driver_version"], "6.8.0")
