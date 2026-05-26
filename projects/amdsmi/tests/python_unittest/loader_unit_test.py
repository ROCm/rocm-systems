#!/usr/bin/env python3

import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


class _DummySymbol:
    def __call__(self, *args, **kwargs):
        return None


class _DummyLibrary:
    def __getattr__(self, name):
        return _DummySymbol()


def _load_wrapper_module():
    test_dir = Path(__file__).resolve().parent
    wrapper_candidates = [
        test_dir.parents[1] / "py-interface/amdsmi_wrapper.py",
        test_dir.parents[1] / "amdsmi_wrapper.py",
        test_dir.parents[1] / "amdsmi/amdsmi_wrapper.py",
    ]

    wrapper_path = next((path for path in wrapper_candidates if path.exists()), None)
    if wrapper_path is None:
        raise FileNotFoundError("Could not locate amdsmi_wrapper.py")

    spec = importlib.util.spec_from_file_location("amdsmi_wrapper_loader_test", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("ctypes.CDLL", return_value=_DummyLibrary()):
        spec.loader.exec_module(module)
    return module


class TestFindSmiLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _load_wrapper_module()

    def test_non_root_uses_package_and_env_locations_only(self):
        attempts = []
        expected_success = "/custom/rocm/lib/libamd_smi.so"

        def fake_cdll(location):
            location = os.fspath(location)
            attempts.append(location)
            if location == expected_success:
                return _DummyLibrary()
            raise OSError("not found")

        base_dir = Path(self.wrapper.__file__).resolve().parents[3] / "lib"
        expected = [
            os.fspath(base_dir / "libamd_smi.so"),
            os.fspath(base_dir.parent / "lib64/libamd_smi.so"),
            "/custom/rocm/lib/libamd_smi.so",
        ]

        with mock.patch.dict(self.wrapper.os.environ, {"ROCM_PATH": "/custom/rocm"}, clear=True):
            with mock.patch.object(self.wrapper.os, "geteuid", return_value=1000):
                with mock.patch.object(self.wrapper, "_get_versioned_library_locations", return_value=[]):
                    with mock.patch.object(self.wrapper.ctypes, "CDLL", side_effect=fake_cdll):
                        _, location = self.wrapper.find_smi_library()

        self.assertEqual(attempts, expected)
        self.assertEqual(os.fspath(location), expected[-1])
        self.assertNotIn("libamd_smi.so", attempts[:-1])
        self.assertNotIn(os.fspath(Path.cwd() / "libamd_smi.so"), attempts)

    def test_root_rejects_untrusted_env_path(self):
        attempts = []

        def fake_cdll(location):
            attempts.append(os.fspath(location))
            raise OSError("not found")

        base_dir = Path(self.wrapper.__file__).resolve().parents[3] / "lib"
        expected = [
            os.fspath(base_dir / "libamd_smi.so"),
            os.fspath(base_dir.parent / "lib64/libamd_smi.so"),
        ]

        with mock.patch.dict(self.wrapper.os.environ, {"ROCM_PATH": "/tmp/evil"}, clear=True):
            with mock.patch.object(self.wrapper.os, "geteuid", return_value=0):
                with mock.patch.object(self.wrapper, "_get_versioned_library_locations", return_value=[]):
                    with mock.patch.object(self.wrapper.ctypes, "CDLL", side_effect=fake_cdll):
                        with self.assertRaises(OSError):
                            self.wrapper.find_smi_library()

        self.assertEqual(attempts, expected)

    def test_root_accepts_opt_rocm_env_path(self):
        attempts = []
        expected_success = "/opt/rocm/lib/libamd_smi.so"

        def fake_cdll(location):
            location = os.fspath(location)
            attempts.append(location)
            if location == expected_success:
                return _DummyLibrary()
            raise OSError("not found")

        with mock.patch.dict(self.wrapper.os.environ, {"ROCM_PATH": "/opt/rocm"}, clear=True):
            with mock.patch.object(self.wrapper.os, "geteuid", return_value=0):
                with mock.patch.object(self.wrapper, "_get_versioned_library_locations", return_value=[]):
                    with mock.patch.object(self.wrapper.ctypes, "CDLL", side_effect=fake_cdll):
                        _, location = self.wrapper.find_smi_library()

        self.assertEqual(attempts[-1], "/opt/rocm/lib/libamd_smi.so")
        self.assertEqual(os.fspath(location), "/opt/rocm/lib/libamd_smi.so")

    def test_root_falls_back_to_opt_rocm_lib64(self):
        attempts = []
        expected_success = "/opt/rocm/lib64/libamd_smi.so"

        def fake_cdll(location):
            location = os.fspath(location)
            attempts.append(location)
            if location == expected_success:
                return _DummyLibrary()
            raise OSError("not found")

        with mock.patch.dict(self.wrapper.os.environ, {"ROCM_PATH": "/opt/rocm"}, clear=True):
            with mock.patch.object(self.wrapper.os, "geteuid", return_value=0):
                with mock.patch.object(self.wrapper, "_get_versioned_library_locations", return_value=[]):
                    with mock.patch.object(self.wrapper.ctypes, "CDLL", side_effect=fake_cdll):
                        _, location = self.wrapper.find_smi_library()

        self.assertEqual(os.fspath(location), expected_success)
        self.assertIn(expected_success, attempts)

    def test_versioned_candidates_are_discovered_dynamically(self):
        lib_dir = Path("/trusted/lib")
        glob_results = [
            lib_dir / "libamd_smi.so",
            lib_dir / "libamd_smi.so.bad",
            lib_dir / "libamd_smi.so.27.1",
            lib_dir / "libamd_smi.so.27",
        ]

        with mock.patch.object(self.wrapper.Path, "glob", return_value=glob_results):
            candidates = self.wrapper._get_versioned_library_locations(lib_dir, "libamd_smi.so")

        self.assertEqual(
            [os.fspath(candidate) for candidate in candidates],
            [
                "/trusted/lib/libamd_smi.so.27",
                "/trusted/lib/libamd_smi.so.27.1",
            ],
        )


if __name__ == "__main__":
    unittest.main()