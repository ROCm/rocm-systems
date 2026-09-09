#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Active amdgpu DKMS version Python API tests."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_source_package():
    package_dir = Path(__file__).resolve().parents[4] / "py-interface"
    package = types.ModuleType("amdsmi_dkms_under_test")
    package.__path__ = [str(package_dir)]
    sys.modules[package.__name__] = package
    package.amdsmi_wrapper = importlib.import_module(f"{package.__name__}.amdsmi_wrapper")
    package.amdsmi_interface = importlib.import_module(f"{package.__name__}.amdsmi_interface")
    return package


amdsmi = _load_source_package()


class TestAmdgpuDkmsVersion(unittest.TestCase):
    def test_interface_exports_getter(self):
        self.assertTrue(hasattr(amdsmi.amdsmi_interface, "amdsmi_get_amdgpu_dkms_version"))

    def test_returns_decoded_version(self):
        expected = "6.19.20-2450390.24.04"
        captured = {}

        def _stub(version, length):
            captured["length"] = (
                int(length.value)
                if hasattr(length, "value")
                else int.from_bytes(bytes(length), "little")
            )
            version.value = expected.encode("utf-8")
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        # ctypes bind is skipped when the installed libamd_smi.so lacks the
        # symbol (generator except AttributeError). create=True still stubs
        # the interface call; drop it once this API is on the installed .so.
        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_amdgpu_dkms_version", _stub, create=True
        ):
            self.assertEqual(amdsmi.amdsmi_interface.amdsmi_get_amdgpu_dkms_version(), expected)
        self.assertEqual(captured["length"], amdsmi.amdsmi_interface.AMDSMI_MAX_STRING_LENGTH)

    def test_raises_when_library_returns_not_supported(self):
        def _stub(_version, _length):
            return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_amdgpu_dkms_version", _stub, create=True
        ):
            with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_interface.amdsmi_get_amdgpu_dkms_version()
        self.assertEqual(ctx.exception.err_code, amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED)


if __name__ == "__main__":
    unittest.main()
