#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free checks that the memory partition tests leave no NPS mode staged.

With current library and amdgpu releases a staged NPS mode applies on the next driver
reload, and requesting the active mode does not cancel it. These run the functional
tests against a fake driver with those rules and check that the starting mode is still
active, with nothing staged, after.
"""

import contextlib
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

_SET_HELP = """\
Set Arguments:
  --memory-partition PARTITION  Set the memory partition mode

Device Arguments:
  -g, --gpu GPU [GPU ...]       Select GPUs

"""


def _load_functional(name):
    path = os.path.join(_FUNCTIONAL_GPU_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHive:
    """Hive-wide memory partition state that follows the driver's staging rules.

    A reload that changes the mode resets every GPU to the first accelerator profile,
    the default after an NPS change.
    """

    def __init__(self, active="NPS1", caps=("NPS1", "NPS2", "NPS4")):
        self.handles = [object(), object()]
        self.active = active
        self.staged = None
        self.caps = list(caps)
        self.reloads = 0
        self.busy = None  # (handle index, mode) that answers BUSY
        self.reported = {}  # handle index -> mode it reports instead of the active one
        self.loaded = True
        self.load_fails = False
        self.unload_fails = False
        self.modprobe_calls = []
        self.profiles = [2, 2]  # accelerator profile index per handle (2 = QPX)
        self.profile_busy = 0  # profile sets that answer BUSY before one succeeds
        # Reload preconditions.
        self.modprobe_path = "/usr/sbin/modprobe"
        self.can_load_modules = True
        self.built_in = False
        self.refcnt = "0"

    def set_mode(self, handle, mode):
        if handle not in self.handles:
            raise amdsmi.AmdSmiParameterException(handle, object)
        if self.busy == (self.handles.index(handle), mode.name):
            raise amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.BUSY)
        if mode.name not in self.caps:
            raise amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.INVAL)
        if mode.name != self.active:
            self.staged = mode.name

    def modprobe(self, *args):
        self.modprobe_calls.append(args)
        returncode = 0
        if args == ("-r", "amdgpu") and self.unload_fails:
            returncode = 1
        elif args == ("-r", "amdgpu"):
            self.loaded = False
        elif args == ("amdgpu",) and self.load_fails:
            returncode = 1
        elif args == ("amdgpu",):
            self.reloads += 1
            self.loaded = True
            if self.staged:
                self.active, self.staged = self.staged, None
                self.profiles = [0] * len(self.handles)
        return mock.Mock(wait=mock.Mock(return_value=returncode))

    def read_first_line(self, path):
        if self.built_in or not self.loaded:
            return ""
        if path.endswith("/refcnt"):
            return self.refcnt
        return "live" if path.endswith("/initstate") else ""

    def get_profile(self, gpu):
        index = self.profiles[self.handles.index(gpu)]
        return {
            "partition_profile": {
                "profile_index": index,
                "profile_type": ["SPX", "DPX", "QPX"][index],
            }
        }

    def set_profile(self, gpu, index):
        if self.profile_busy:
            self.profile_busy -= 1
            raise amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.BUSY)
        self.profiles[self.handles.index(gpu)] = index

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
            "amdsmi_get_gpu_memory_partition": lambda gpu: self.reported.get(
                self.handles.index(gpu), self.active
            ),
            "amdsmi_get_gpu_memory_partition_config": lambda _gpu: {
                "partition_caps": list(self.caps)
            },
            "amdsmi_get_gpu_accelerator_partition_profile": self.get_profile,
            "amdsmi_set_gpu_accelerator_partition_profile": self.set_profile,
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

    def _run(self, test_class, name, hive, destructive=False, **harness_attrs):
        """Run one functional test against *hive*; return its unittest result."""
        result = unittest.TestResult()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.multiple(amdsmi, **hive.api()))
            stack.enter_context(
                mock.patch.multiple(
                    common.Common,
                    amdsmi_smart_init=lambda _self: (None, 0),
                    check_amdgpu_driver=lambda _self: hive.loaded,
                )
            )
            harness = common.Common(common.VERBOSITY_QUIET)
            for attr, value in harness_attrs.items():
                setattr(harness, attr, value)
            cycle = self.cycle
            for target, attr, value in (
                (test_class, "common", harness),
                (cycle, "_ALLOW_DESTRUCTIVE_TESTS", destructive),
                (cycle, "_modprobe_path", lambda: hive.modprobe_path),
                (cycle, "_can_load_modules", lambda: hive.can_load_modules),
                (cycle, "_read_first_line", hive.read_first_line),
                (cycle, "_run_modprobe", hive.modprobe),
                (cycle.time, "sleep", mock.Mock()),
            ):
                stack.enter_context(mock.patch.object(target, attr, value, create=True))
            test_class(name).run(result)
        return result

    def _failures(self, result):
        return [text for _, text in result.failures + result.errors]

    def assertNothingStaged(self, hive, mode):
        self.assertIsNone(hive.staged, "a test left an NPS mode staged for the next reload")
        self.assertEqual(hive.active, mode)

    def assertStartingState(self, hive):
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.profiles, [2, 2], "accelerator profiles not restored")

    def assertSkippedWithoutFlag(self, result):
        self.assertEqual(len(result.skipped), 1, result.failures + result.errors)
        self.assertIn("AMDSMI_ALLOW_DESTRUCTIVE_TESTS", result.skipped[0][1])

    def test_return_codes_stages_nothing(self):
        hive = _FakeHive()
        result = self._run(
            self.partition.TestGpuPartition, "test_set_gpu_memory_partition_return_codes", hive
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")

    def test_mode_setter_stages_nothing_once_enabled(self):
        hive = _FakeHive()
        result = self._run(
            self.partition.TestGpuPartition,
            "test_set_gpu_memory_partition_mode",
            hive,
            TODO_SKIP_FAIL=False,
        )
        self.assertFalse(result.skipped)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")

    def _cli_memory_partition_cmds(self, modes):
        """The ``set --memory-partition`` commands the CLI sweep builds for GPUs on *modes*."""
        try:
            from cli.base import TestCliBase
        except (ImportError, AttributeError) as exc:  # needs the amd-smi CLI under test
            self.skipTest(f"amd-smi CLI unavailable: {exc}")
        sweep = TestCliBase()
        sweep.util = mock.Mock(RunCmdSync=lambda _cmd, **_: (0, _SET_HELP, ""))
        sweep.common = mock.Mock()
        sweep.sub_args = {"GPU": ["all", *range(len(modes))]}
        sweep.partition_data = {"current_partition": [{"memory": mode} for mode in modes]}
        cmds = sweep.CreateCmds("set", "Set Arguments:", "Device Arguments:", "", "")
        requested = set()
        for cmd, _ in cmds:
            args = cmd.split()
            if "--memory-partition" in args:
                gpu = args[args.index("--gpu") + 1] if "--gpu" in args else None
                requested.add((gpu, args[args.index("--memory-partition") + 1]))
        return requested

    def test_cli_sweep_requests_only_the_active_mode(self):
        requested = self._cli_memory_partition_cmds(["NPS1", "NPS1"])
        self.assertEqual(requested, {(None, "NPS1"), ("all", "NPS1"), ("0", "NPS1"), ("1", "NPS1")})

    def test_cli_sweep_targets_each_gpu_when_modes_differ(self):
        requested = self._cli_memory_partition_cmds(["NPS1", "NPS4"])
        self.assertEqual(requested, {("0", "NPS1"), ("1", "NPS4")})

    def test_cli_sweep_ignores_handles_without_a_mode(self):
        requested = self._cli_memory_partition_cmds(["NPS1", "N/A"])
        self.assertEqual(requested, {(None, "NPS1"), ("all", "NPS1"), ("0", "NPS1")})

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
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_applying_memory_partition_needs_reload",
            hive,
        )
        self.assertSkippedWithoutFlag(result)
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.modprobe_calls, [])

    def test_cycle_stages_nothing_by_default(self):
        hive = _FakeHive()
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle, "test_cycle_memory_partition_modes", hive
        )
        self.assertSkippedWithoutFlag(result)
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.modprobe_calls, [])

    def test_reload_requires_the_destructive_flag(self):
        hive = _FakeHive()
        case = self.cycle.TestGpuMemoryPartitionCycle("test_cycle_memory_partition_modes")
        with mock.patch.object(self.cycle, "_ALLOW_DESTRUCTIVE_TESTS", False), mock.patch.object(
            self.cycle, "_run_modprobe", hive.modprobe
        ):
            with self.assertRaises(case.failureException):
                case._reload_driver()
        self.assertEqual(hive.modprobe_calls, [])

    def test_cycle_skips_when_devices_disagree(self):
        hive = _FakeHive()
        hive.reported = {1: "NPS2"}
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        self.assertEqual(len(result.skipped), 1, result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.modprobe_calls, [])

    def test_rejected_modes_need_no_reload(self):
        hive = _FakeHive(caps=("NPS1",))
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_applying_memory_partition_needs_reload",
            hive,
            destructive=True,
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertNothingStaged(hive, "NPS1")
        self.assertEqual(hive.modprobe_calls, [])

    def test_restore_fails_when_amdgpu_is_left_unloaded_with_a_staged_mode(self):
        hive = _FakeHive()
        hive.load_fails = True
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        messages = self._failures(result)
        self.assertTrue(any("up in NPS2" in text for text in messages), messages)
        self.assertEqual(hive.modprobe_calls, [("-r", "amdgpu"), ("amdgpu",), ("amdgpu",)])

    def test_failed_unload_names_the_staged_mode(self):
        hive = _FakeHive()
        hive.unload_fails = True
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        messages = self._failures(result)
        self.assertTrue(any("NPS2 is still staged" in text for text in messages), messages)

    def test_every_reload_precondition_is_checked_before_staging(self):
        blockers = {
            "modprobe not found": lambda hive: setattr(hive, "modprobe_path", None),
            "CAP_SYS_MODULE": lambda hive: setattr(hive, "can_load_modules", False),
            "not a loaded module": lambda hive: setattr(hive, "built_in", True),
            "in use by another process (refcnt 3)": lambda hive: setattr(hive, "refcnt", "3"),
            "refcnt is unreadable": lambda hive: setattr(hive, "refcnt", "busy"),
        }
        for name in (
            "test_cycle_memory_partition_modes",
            "test_applying_memory_partition_needs_reload",
        ):
            for reason, block in blockers.items():
                with self.subTest(test=name, blocker=reason):
                    hive = _FakeHive()
                    block(hive)
                    result = self._run(
                        self.cycle.TestGpuMemoryPartitionCycle, name, hive, destructive=True
                    )
                    messages = self._failures(result)
                    self.assertTrue(any(reason in text for text in messages), messages)
                    self.assertTrue(any("nothing was staged" in text for text in messages))
                    self.assertNothingStaged(hive, "NPS1")
                    self.assertEqual(hive.modprobe_calls, [])

    def test_profile_restore_retries_busy(self):
        hive = _FakeHive()
        hive.profile_busy = 2
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertStartingState(hive)

    def test_profile_restore_fails_when_it_stays_busy(self):
        hive = _FakeHive()
        hive.profile_busy = 1000
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        messages = self._failures(result)
        self.assertTrue(any("accelerator profiles not restored" in t for t in messages), messages)

    def test_profile_restore_is_skipped_after_a_failed_memory_restore(self):
        hive = _FakeHive()
        hive.load_fails = True
        unload_and_load = hive.modprobe

        def modprobe(*args):
            if args == ("-r", "amdgpu"):
                hive.profiles = [0, 0]  # the unload drops the partitions
            return unload_and_load(*args)

        hive.modprobe = modprobe
        set_profile = mock.Mock(side_effect=hive.set_profile)
        hive.set_profile = set_profile
        self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        set_profile.assert_not_called()

    def test_set_without_reload_applies_and_restores_when_destructive(self):
        hive = _FakeHive()
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_applying_memory_partition_needs_reload",
            hive,
            destructive=True,
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertStartingState(hive)

    def test_cycle_returns_to_start_when_destructive(self):
        hive = _FakeHive()
        result = self._run(
            self.cycle.TestGpuMemoryPartitionCycle,
            "test_cycle_memory_partition_modes",
            hive,
            destructive=True,
        )
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertStartingState(hive)
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
