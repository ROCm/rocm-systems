#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free checks that the memory partition tests leave no NPS mode staged.

A staged NPS mode applies on the next driver reload, and requesting the active mode
does not cancel it. These run the functional tests against a fake driver with those
rules and check that the starting mode is still active, with nothing staged, after.
"""

import importlib.util
import os
import unittest
from unittest import mock

try:
    import common.common as common
    from common.common import amdsmi
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    common = None

_FUNCTIONAL_GPU_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "functional", "gpu"
)


def _load_functional(name):
    path = os.path.join(_FUNCTIONAL_GPU_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHive:
    """Hive-wide memory partition state that follows the driver's staging rules."""

    def __init__(self, active="NPS1", caps=("NPS1", "NPS2", "NPS4")):
        self.handles = [object(), object()]
        self.active = active
        self.staged = None
        self.caps = list(caps)
        self.reloads = 0
        self.busy = None  # (handle index, mode) that answers BUSY

    def set_mode(self, handle, mode):
        if handle not in self.handles:
            raise amdsmi.AmdSmiParameterException(handle, object)
        if self.busy == (self.handles.index(handle), mode.name):
            raise amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.BUSY)
        if mode.name not in self.caps:
            raise amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.INVAL)
        if mode.name != self.active:
            self.staged = mode.name

    def popen(self, argv):
        if argv == ["modprobe", "amdgpu"]:
            self.reloads += 1
            if self.staged:
                self.active, self.staged = self.staged, None
        return mock.Mock(wait=mock.Mock(return_value=0))

    def api(self):
        not_supported = amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.NOT_SUPPORTED)
        return {
            "amdsmi_init": mock.Mock(),
            "amdsmi_shut_down": mock.Mock(),
            "amdsmi_get_processor_handles": lambda: list(self.handles),
            "amdsmi_get_socket_handles": lambda: [object()],
            "amdsmi_get_gpu_virtualization_mode": mock.Mock(side_effect=not_supported),
            "amdsmi_get_gpu_asic_info": lambda _gpu: {},
            "amdsmi_get_gpu_board_info": lambda _gpu: {},
            "amdsmi_get_gpu_memory_partition": lambda _gpu: self.active,
            "amdsmi_get_gpu_memory_partition_config": lambda _gpu: {
                "partition_caps": list(self.caps)
            },
            "amdsmi_get_gpu_accelerator_partition_profile": lambda _gpu: {
                "partition_profile": {"profile_index": 0, "profile_type": "SPX"}
            },
            "amdsmi_set_gpu_accelerator_partition_profile": mock.Mock(),
            "amdsmi_set_gpu_memory_partition": self.set_mode,
            "amdsmi_set_gpu_memory_partition_mode": self.set_mode,
        }


@unittest.skipIf(common is None, "amdsmi test harness unavailable")
class TestGpuMemoryPartitionStaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partition = _load_functional("test_partition")
        cls.cycle = _load_functional("test_memory_partition_cycle")
        cls.benchmark = _load_functional("test_benchmark")

    def _run(self, test_class, name, hive, destructive=False):
        """Run one functional test against *hive*; return its unittest result."""
        result = unittest.TestResult()
        with mock.patch.multiple(amdsmi, **hive.api()), mock.patch.multiple(
            common.Common,
            amdsmi_smart_init=lambda _self: (None, 0),
            check_amdgpu_driver=lambda _self: True,
        ):
            harness = common.Common(common.VERBOSITY_QUIET)
            with mock.patch.object(test_class, "common", harness, create=True), mock.patch.object(
                self.cycle, "_ALLOW_DESTRUCTIVE_TESTS", destructive
            ), mock.patch.object(
                self.cycle.shutil, "which", return_value="/usr/sbin/modprobe"
            ), mock.patch.object(self.cycle.subprocess, "Popen", hive.popen), mock.patch.object(
                self.cycle.time, "sleep"
            ):
                test_class(name).run(result)
        return result

    def assertNothingStaged(self, hive, mode):
        self.assertIsNone(hive.staged, "a test left an NPS mode staged for the next reload")
        self.assertEqual(hive.active, mode)

    def test_return_codes_stages_nothing(self):
        hive = _FakeHive()
        result = self._run(
            self.partition.TestGpuPartition, "test_set_gpu_memory_partition_return_codes", hive
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")

    def test_benchmark_set_stages_nothing(self):
        for name in (
            "test_performance_set_gpu_memory_partition",
            "test_performance_set_gpu_memory_partition_mode",
        ):
            with self.subTest(name):
                hive = _FakeHive()
                result = self._run(self.benchmark.TestGpuBenchmark, name, hive)
                self.assertFalse(result.skipped)
                self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
                self.assertNothingStaged(hive, "NPS1")

    def test_set_without_reload_stages_nothing_by_default(self):
        hive = _FakeHive()
        self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_applying_memory_partition_needs_reload",
            hive,
        )
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.reloads, 0)

    def test_cycle_stages_nothing_by_default(self):
        hive = _FakeHive()
        self._run(self.cycle.TestGpuMemoryPartitionCycle, "test_cycle_memory_partition_modes", hive)
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.reloads, 0)

    def test_set_without_reload_applies_and_restores_when_destructive(self):
        hive = _FakeHive()
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_applying_memory_partition_needs_reload",
            hive,
            destructive=True,
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")

    def test_cycle_returns_to_start_when_destructive(self):
        hive = _FakeHive()
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.reloads, 3)

    def test_cycle_failure_before_reload_still_restores(self):
        hive = _FakeHive()
        hive.busy = (1, "NPS2")
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        self.assertFalse(result.wasSuccessful())
        self.assertNothingStaged(hive, "NPS1")
