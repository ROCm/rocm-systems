#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Count-then-read tests for amdsmi_get_gpu_compute_process_info() and
amdsmi_get_gpu_compute_process_gpus()."""

import ctypes
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_PACKAGE_DIR = Path(__file__).resolve().parents[4] / "py-interface"


def _load_source_package() -> types.ModuleType:
    package = types.ModuleType("amdsmi_compute_process_info_under_test")
    package.__path__ = [str(_PACKAGE_DIR)]
    sys.modules[package.__name__] = package
    package.amdsmi_wrapper = importlib.import_module(f"{package.__name__}.amdsmi_wrapper")
    package.amdsmi_interface = importlib.import_module(f"{package.__name__}.amdsmi_interface")
    return package


amdsmi = _load_source_package() if _PACKAGE_DIR.is_dir() else None


class _Library:
    """Stands in for the C call. Each call, count or read, sees the next list of
    running PIDs, as the library would while processes start and exit."""

    def __init__(self, pid_lists) -> None:
        self._pid_lists = pid_lists
        self.calls = 0

    def __call__(self, procs: object, num_items_ref: object) -> int:
        pids = self._pid_lists[self.calls]
        self.calls += 1
        num_items = num_items_ref._obj
        if isinstance(procs, ctypes.Array):
            if num_items.value < len(pids):
                return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
            for proc, pid in zip(procs, pids):
                proc.process_id = pid
                proc.vram_usage = pid * 4096
        num_items.value = len(pids)
        return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS


class TestComputeProcessInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if amdsmi is None:
            raise unittest.SkipTest(f"amdsmi py-interface source not found at {_PACKAGE_DIR}")

    def _read(self, library):
        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_compute_process_info", library
        ):
            return amdsmi.amdsmi_interface.amdsmi_get_gpu_compute_process_info()

    def test_counts_again_when_a_process_starts_before_the_read(self) -> None:
        procs = self._read(_Library([[10], [10, 11], [10, 11], [10, 11]]))
        self.assertEqual([p["process_id"] for p in procs], [10, 11])
        self.assertEqual([p["vram_usage"] for p in procs], [10 * 4096, 11 * 4096])

    def test_lists_only_the_processes_read(self) -> None:
        procs = self._read(_Library([[10, 11, 12], [10, 12]]))
        self.assertEqual([p["process_id"] for p in procs], [10, 12])

    def test_gives_up_after_three_counts(self) -> None:
        library = _Library([[1], [1, 2], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3, 4]])
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiLibraryException) as caught:
            self._read(library)
        self.assertEqual(
            caught.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
        )
        self.assertEqual(library.calls, 6)


class _GpuLibrary:
    """Stands in for the C call. Each call, count or read, sees the next list of
    GPU indices the process uses."""

    def __init__(self, gpu_lists) -> None:
        self._gpu_lists = gpu_lists
        self.calls = 0

    def __call__(self, pid: int, dv_indices: object, num_devices_ref: object) -> int:
        gpus = self._gpu_lists[self.calls]
        self.calls += 1
        num_devices = num_devices_ref._obj
        if isinstance(dv_indices, ctypes.Array):
            if num_devices.value < len(gpus):
                return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
            for i, gpu in enumerate(gpus):
                dv_indices[i] = gpu
        num_devices.value = len(gpus)
        return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS


class TestComputeProcessGpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if amdsmi is None:
            raise unittest.SkipTest(f"amdsmi py-interface source not found at {_PACKAGE_DIR}")

    def _read(self, library):
        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_compute_process_gpus", library
        ):
            return amdsmi.amdsmi_interface.amdsmi_get_gpu_compute_process_gpus(1234)

    def test_counts_then_reads_the_gpu_indices(self) -> None:
        library = _GpuLibrary([[0, 3], [0, 3]])
        self.assertEqual(self._read(library), [0, 3])
        self.assertEqual(library.calls, 2)

    def test_counts_again_when_the_process_uses_another_gpu_before_the_read(self) -> None:
        library = _GpuLibrary([[0], [0, 1], [0, 1], [0, 1]])
        self.assertEqual(self._read(library), [0, 1])
        self.assertEqual(library.calls, 4)

    def test_lists_only_the_gpus_read(self) -> None:
        self.assertEqual(self._read(_GpuLibrary([[0, 1], [1]])), [1])

    def test_gives_up_after_three_counts(self) -> None:
        library = _GpuLibrary([[0], [0, 1], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2, 3]])
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiLibraryException) as caught:
            self._read(library)
        self.assertEqual(
            caught.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
        )
        self.assertEqual(library.calls, 6)
