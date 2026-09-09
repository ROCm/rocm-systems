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
from unittest import mock


def _load_source_package() -> types.ModuleType:
    package_dir = Path(__file__).resolve().parents[4] / "py-interface"
    package = types.ModuleType("amdsmi_driver_versions_under_test")
    package.__path__ = [str(package_dir)]
    sys.modules[package.__name__] = package
    package.amdsmi_wrapper = importlib.import_module(f"{package.__name__}.amdsmi_wrapper")
    package.amdsmi_interface = importlib.import_module(f"{package.__name__}.amdsmi_interface")
    return package


amdsmi = _load_source_package()


class TestDriverVersions(unittest.TestCase):
    def test_struct_layout_includes_new_fields(self) -> None:
        struct_type = amdsmi.amdsmi_wrapper.struct_amdsmi_driver_info_t
        self.assertEqual(ctypes.sizeof(struct_type), 1536)
        self.assertEqual(struct_type.driver_kernel_version.offset, 768)
        self.assertEqual(struct_type.driver_build_version.offset, 1024)
        self.assertEqual(struct_type.driver_full_version.offset, 1280)

    def test_reports_split_driver_versions(self) -> None:
        handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _stub(_handle: object, info_ptr: object) -> int:
            info_ptr._obj.driver_name = b"amdgpu"
            info_ptr._obj.driver_version = b"31400000"
            info_ptr._obj.driver_date = b"2026/09/16 00:00"
            info_ptr._obj.driver_kernel_version = b"6.19.14"
            info_ptr._obj.driver_build_version = b"2370381"
            info_ptr._obj.driver_full_version = b"6.19.14.31400000-2370381"
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_driver_info", _stub):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_driver_info(handle)

        self.assertEqual(info["driver_kernel_version"], "6.19.14")
        self.assertEqual(info["driver_version"], "31400000")
        self.assertEqual(info["driver_build_version"], "2370381")
        self.assertEqual(info["driver_full_version"], "6.19.14.31400000-2370381")

    def test_empty_versions_render_na(self) -> None:
        handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _stub(_handle: object, info_ptr: object) -> int:
            info_ptr._obj.driver_name = b"amdgpu"
            info_ptr._obj.driver_date = b""
            info_ptr._obj.driver_kernel_version = b""
            info_ptr._obj.driver_version = b""
            info_ptr._obj.driver_build_version = b""
            info_ptr._obj.driver_full_version = b""
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_driver_info", _stub):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_driver_info(handle)

        self.assertEqual(info["driver_kernel_version"], "N/A")
        self.assertEqual(info["driver_version"], "N/A")
        self.assertEqual(info["driver_build_version"], "N/A")
        self.assertEqual(info["driver_full_version"], "N/A")


if __name__ == "__main__":
    unittest.main()
