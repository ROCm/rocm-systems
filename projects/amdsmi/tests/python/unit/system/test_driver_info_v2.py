#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Driver info v2 struct layout, field reporting and v1 deprecation (hardware-free)."""

import ctypes
import unittest
import warnings
from unittest import mock

from common.common import amdsmi

_V1_SIZE = 768
_V2_SIZE = 4096
_STRING_LENGTH = 256
_V2_STRINGS = (
    "driver_version",
    "driver_date",
    "driver_name",
    "driver_kernel_version",
    "amdgpu_driver_version",
    "driver_build_version",
    "driver_full_version",
)
_FULL_INFO = {
    "driver_version": "6.19.14.31400000",
    "driver_date": "2015/01/01 00:00",
    "driver_name": "amdgpu",
    "driver_kernel_version": "6.19.14",
    "amdgpu_driver_version": "31400000",
    "driver_build_version": "2370381",
    "driver_full_version": "6.19.14.31400000-2370381",
}

wrapper = amdsmi.amdsmi_wrapper
interface = amdsmi.amdsmi_interface


def _driver_info_v2(strings: dict, package: int) -> dict:
    """Return the public dict with a stub library that writes the given fields."""
    handle = wrapper.amdsmi_processor_handle()

    def _stub(_handle, info_ptr):
        for name, value in strings.items():
            setattr(info_ptr._obj, name, value.encode())
        info_ptr._obj.driver_package = package
        return wrapper.AMDSMI_STATUS_SUCCESS

    with mock.patch.object(wrapper, "amdsmi_get_gpu_driver_info_v2", _stub, create=True):
        return interface.amdsmi_get_gpu_driver_info_v2(handle)


class TestDriverInfoLayout(unittest.TestCase):
    """Binaries built against 27.0 pass a 768-byte v1 struct, so only v2 may hold new fields."""

    def test_v1_struct_size_unchanged(self):
        self.assertEqual(ctypes.sizeof(wrapper.struct_amdsmi_driver_info_t), _V1_SIZE)

    def test_v2_strings_follow_the_v1_fields(self):
        struct_type = wrapper.struct_amdsmi_driver_info_v2_t
        for index, name in enumerate(_V2_STRINGS):
            with self.subTest(field=name):
                self.assertEqual(getattr(struct_type, name).offset, index * _STRING_LENGTH)

    def test_v2_reserved_tail_fills_struct(self):
        struct_type = wrapper.struct_amdsmi_driver_info_v2_t
        self.assertEqual(ctypes.sizeof(struct_type), _V2_SIZE)
        self.assertEqual(struct_type.driver_package.offset, len(_V2_STRINGS) * _STRING_LENGTH)
        self.assertEqual(
            struct_type.reserved.offset,
            struct_type.driver_package.offset + struct_type.driver_package.size,
        )
        self.assertEqual(struct_type.reserved.offset + struct_type.reserved.size, _V2_SIZE)


class TestDriverInfoV2(unittest.TestCase):
    def test_reports_every_field(self):
        info = _driver_info_v2(_FULL_INFO, wrapper.AMDSMI_DRIVER_PACKAGE_DKMS)
        self.assertEqual(info, {**_FULL_INFO, "driver_package": "amdgpu-dkms"})

    def test_package_is_reported_by_name(self):
        packages = {
            wrapper.AMDSMI_DRIVER_PACKAGE_DKMS: "amdgpu-dkms",
            wrapper.AMDSMI_DRIVER_PACKAGE_KMOD: "amdgpu-kmod",
            wrapper.AMDSMI_DRIVER_PACKAGE_NONE: "N/A",
            # A package type from a newer library must not be mistaken for a known one.
            99: "N/A",
        }
        for package, name in packages.items():
            with self.subTest(package=package):
                self.assertEqual(_driver_info_v2({}, package)["driver_package"], name)

    def test_empty_values_render_na(self):
        info = _driver_info_v2({"driver_name": "amdgpu"}, wrapper.AMDSMI_DRIVER_PACKAGE_NONE)
        self.assertEqual(info.pop("driver_name"), "amdgpu")
        self.assertEqual(set(info), set(_V2_STRINGS[:2] + _V2_STRINGS[3:]) | {"driver_package"})
        self.assertEqual(set(info.values()), {"N/A"})

    def test_library_without_the_call_is_not_supported(self):
        handle = wrapper.amdsmi_processor_handle()
        with mock.patch.object(wrapper, "amdsmi_get_gpu_driver_info_v2", None, create=True):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as raised:
                interface.amdsmi_get_gpu_driver_info_v2(handle)
        self.assertEqual(raised.exception.get_error_code(), wrapper.AMDSMI_STATUS_NOT_SUPPORTED)

    def test_library_error_status_raises(self):
        handle = wrapper.amdsmi_processor_handle()

        def _stub(_handle, _info_ptr):
            return wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        # The CLI falls back to v1 on this exception, so a zeroed dict must never replace it.
        with mock.patch.object(wrapper, "amdsmi_get_gpu_driver_info_v2", _stub, create=True):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as raised:
                interface.amdsmi_get_gpu_driver_info_v2(handle)
        self.assertEqual(raised.exception.get_error_code(), wrapper.AMDSMI_STATUS_NOT_SUPPORTED)

    def test_rejects_non_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            interface.amdsmi_get_gpu_driver_info_v2(None)


class TestDriverInfoV1Deprecation(unittest.TestCase):
    def test_v1_warns_and_still_reports_its_fields(self):
        handle = wrapper.amdsmi_processor_handle()

        def _stub(_handle, info_ptr):
            info_ptr._obj.driver_name = b"amdgpu"
            info_ptr._obj.driver_version = b"6.19.14.31400000"
            return wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(wrapper, "amdsmi_get_gpu_driver_info", _stub):
            with self.assertWarnsRegex(
                DeprecationWarning, "amdsmi_get_gpu_driver_info_v2"
            ) as caught:
                info = interface.amdsmi_get_gpu_driver_info(handle)

        # The warning names the caller, or Python's default filter would hide it from scripts.
        self.assertEqual(caught.filename, __file__)
        self.assertEqual(
            info,
            {"driver_name": "amdgpu", "driver_version": "6.19.14.31400000", "driver_date": "N/A"},
        )

    def test_v2_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _driver_info_v2(_FULL_INFO, wrapper.AMDSMI_DRIVER_PACKAGE_DKMS)
