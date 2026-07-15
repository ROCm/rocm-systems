#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""WSL2 GPU detection unit tests for the AMD SMI CLI initializer."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def _find_amdsmi_init(test_file):
    test_path = Path(test_file).resolve()
    candidates = (
        test_path.parents[4] / "amdsmi_cli" / "amdsmi_init.py",
        test_path.parents[6] / "libexec" / "amdsmi_cli" / "amdsmi_init.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to find amdsmi_init.py in: {candidates}")


class TestAmdSmiCliWslDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FakeInitFlags:
            INIT_ALL_PROCESSORS = 0
            INIT_AMD_GPUS = 1
            INIT_AMD_CPUS = 2
            INIT_AMD_NICS = 4

        class FakeLibraryException(Exception):
            err_code = 0

        fake_wrapper = SimpleNamespace(AMDSMI_STATUS_NOT_INIT=1, AMDSMI_STATUS_DRIVER_NOT_LOADED=2)
        fake_interface = SimpleNamespace(
            AmdSmiInitFlags=FakeInitFlags,
            AmdSmiLibraryException=FakeLibraryException,
            AmdSmiParameterException=FakeLibraryException,
            amdsmi_wrapper=fake_wrapper,
            amdsmi_init=lambda _flags: None,
            amdsmi_shut_down=lambda: None,
        )
        fake_amdsmi = ModuleType("amdsmi")
        fake_amdsmi.amdsmi_interface = fake_interface
        fake_amdsmi.amdsmi_exception = SimpleNamespace(AmdSmiLibraryException=FakeLibraryException)

        module_path = _find_amdsmi_init(__file__)
        spec = importlib.util.spec_from_file_location("_amdsmi_init_wsl_test", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {module_path}")
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, {"amdsmi": fake_amdsmi}),
            patch("atexit.register"),
            patch("signal.signal"),
        ):
            spec.loader.exec_module(module)
        cls.amdsmi_init = module

    def check_wsl2_gpu(self, proc_version, dxg_exists, kfd_exists):
        proc_path = Mock()
        proc_path.read_text.return_value = proc_version
        dxg_path = Mock()
        dxg_path.exists.return_value = dxg_exists
        kfd_path = Mock()
        kfd_path.exists.return_value = kfd_exists
        paths = {"/proc/version": proc_path, "/dev/dxg": dxg_path, "/dev/kfd": kfd_path}
        with patch.object(self.amdsmi_init, "Path", side_effect=paths.__getitem__):
            return self.amdsmi_init.check_wsl2_gpu()

    def test_check_wsl2_gpu_detection_matrix(self):
        cases = (
            ("Linux microsoft-standard-WSL2", True, False, True),
            ("Linux WSL2 kernel", True, False, True),
            ("Linux version 6.8.0", True, False, False),
            ("Linux microsoft-standard-WSL2", False, False, False),
            ("Linux microsoft-standard-WSL2", True, True, False),
        )
        for proc_version, dxg_exists, kfd_exists, expected in cases:
            with self.subTest(
                proc_version=proc_version, dxg_exists=dxg_exists, kfd_exists=kfd_exists
            ):
                self.assertEqual(
                    self.check_wsl2_gpu(proc_version, dxg_exists, kfd_exists), expected
                )

    def test_check_wsl2_gpu_handles_proc_read_error(self):
        proc_path = Mock()
        proc_path.read_text.side_effect = OSError
        with patch.object(self.amdsmi_init, "Path", return_value=proc_path):
            self.assertFalse(self.amdsmi_init.check_wsl2_gpu())

    def test_find_amdsmi_init_in_install_layout(self):
        with tempfile.TemporaryDirectory() as prefix:
            prefix_path = Path(prefix)
            test_path = (
                prefix_path
                / "share"
                / "amd_smi"
                / "tests"
                / "python_unittest"
                / "unit"
                / "system"
                / "test_wsl_detection.py"
            )
            module_path = prefix_path / "libexec" / "amdsmi_cli" / "amdsmi_init.py"
            module_path.parent.mkdir(parents=True)
            module_path.touch()
            self.assertEqual(_find_amdsmi_init(test_path), module_path)
