#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU driver version field tests."""

import ctypes
import importlib
import sys
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock


_PACKAGE_DIR = Path(__file__).resolve().parents[4] / "py-interface"


def _load_amdsmi() -> Optional[types.SimpleNamespace]:
    """Load the wrapper and interface from the in-tree sources, else the installed package.

    The installed test layout ships no ``py-interface`` directory, so fall back to the
    installed ``amdsmi`` package rather than skipping.
    """
    if _PACKAGE_DIR.is_dir():
        package = types.ModuleType("amdsmi_driver_versions_under_test")
        package.__path__ = [str(_PACKAGE_DIR)]
        sys.modules[package.__name__] = package
        name = package.__name__
    else:
        name = "amdsmi"
    try:
        return types.SimpleNamespace(
            amdsmi_wrapper=importlib.import_module(f"{name}.amdsmi_wrapper"),
            amdsmi_interface=importlib.import_module(f"{name}.amdsmi_interface"),
        )
    except ImportError:
        if _PACKAGE_DIR.is_dir():
            raise
        return None


amdsmi = _load_amdsmi()


class TestDriverVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if amdsmi is None:
            raise unittest.SkipTest(
                "neither py-interface sources nor an installed amdsmi package found"
            )

    def test_struct_layout_is_frozen(self) -> None:
        struct_type = amdsmi.amdsmi_wrapper.struct_amdsmi_driver_info_t
        self.assertEqual(ctypes.sizeof(struct_type), 768)
        self.assertEqual(struct_type.driver_version.offset, 0)
        self.assertEqual(struct_type.driver_date.offset, 256)
        self.assertEqual(struct_type.driver_name.offset, 512)

    def test_reports_full_driver_version(self) -> None:
        handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _stub(_handle: object, info_ptr: object) -> int:
            info_ptr._obj.driver_name = b"amdgpu"
            info_ptr._obj.driver_version = b"6.19.14.31400000-2370381"
            info_ptr._obj.driver_date = b"2026/09/16 00:00"
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_driver_info", _stub):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_driver_info(handle)

        self.assertEqual(info["driver_version"], "6.19.14.31400000-2370381")
        self.assertEqual(set(info), {"driver_name", "driver_version", "driver_date"})

    def test_empty_versions_render_na(self) -> None:
        handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _stub(_handle: object, info_ptr: object) -> int:
            info_ptr._obj.driver_name = b"amdgpu"
            info_ptr._obj.driver_date = b""
            info_ptr._obj.driver_version = b""
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_driver_info", _stub):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_driver_info(handle)

        self.assertEqual(info["driver_version"], "N/A")
        self.assertEqual(info["driver_date"], "N/A")
