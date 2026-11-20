#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free replay of the set sweep's recorded starting values."""

import json
import sys
import unittest
from copy import deepcopy
from unittest import mock

from common.common import amdsmi

# cli.base puts the CLI dir on sys.path and imports amdsmi_cli_exceptions from it.
# Undo both so later test modules import the CLI as they would without this one.
_sys_path = list(sys.path)
_CLI_EXCEPTIONS = "amdsmi_cli_exceptions"
_cli_exceptions = sys.modules.get(_CLI_EXCEPTIONS)
try:
    from cli import test_set
finally:
    sys.path[:] = _sys_path
    if _cli_exceptions is None:
        sys.modules.pop(_CLI_EXCEPTIONS, None)
    else:
        sys.modules[_CLI_EXCEPTIONS] = _cli_exceptions

UNSUPPORTED = int(amdsmi.AmdSmiStatus.NOT_SUPPORTED)
UNEXPECTED_DATA = int(amdsmi.AmdSmiStatus.UNEXPECTED_DATA)
DGPU_FAN = {"speed": 80, "max": 255, "rpm": 1200, "usage": {"value": 31.37, "unit": "%"}}
FANLESS_DGPU_FAN = {"speed": "N/A", "max": "N/A", "rpm": "N/A", "usage": "N/A"}
APU_FAN = {"apu_fan_pwm": "N/A"}
POWER_CAP = {
    "min_power_limit": {"value": 261, "unit": "W"},
    "max_power_limit": {"value": 350, "unit": "W"},
    "socket_power_limit": {"value": 291, "unit": "W"},
}


class TestCliSetRestore(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep = test_set.TestSet(methodName="test_command")
        self.sweep.common = mock.Mock(processors=[object()])
        self.sweep.metric_data = {"gpu_data": [{"clock": {}}]}
        self.sweep.metric_named_data = {
            "gpu_data": [{"fan": dict(FANLESS_DGPU_FAN), "perf_level": "AUTO"}]
        }
        self.sweep.static_data = {
            "gpu_data": [
                {
                    "profile": {"current": "COMPUTE", "available_profiles": ["COMPUTE"]},
                    "clock": {"sys": "N/A"},
                    "partition": {"compute_partition_mem_alloc_mode": "N/A"},
                    "soc_pstate": "N/A",
                    "xgmi_plpd": "N/A",
                    "limit": {
                        "ppt0": "N/A",
                        "ppt1": "N/A",
                        "ptl_state": "N/A",
                        "ptl_format": "N/A",
                    },
                    "process_isolation": "Disabled",
                }
            ]
        }
        self.sweep.partition_data = {
            "current_partition": [{"memory": "N/A", "accelerator_type": "N/A"}]
        }

    def _expected_commands(self, index: int = 0) -> list:
        return [
            f"amd-smi reset --fans --gpu {index}",
            f"amd-smi set --profile COMPUTE --gpu {index}",
            f"amd-smi reset --power-cap --gpu {index}",
            f"amd-smi reset --clocks --gpu {index}",
            f"amd-smi set --perf-level AUTO --gpu {index}",
            f"amd-smi set --process-isolation 0 --gpu {index}",
        ]

    def _mock_commands(self, expected: list, overrides: dict = None) -> None:
        # The default baseline has no fan max and no power cap, so both resets
        # answer NOT_SUPPORTED, as they do on that hardware.
        codes = {
            command: UNSUPPORTED
            if command.startswith(("amd-smi reset --fans", "amd-smi reset --power-cap"))
            else 0
            for command in expected
        }
        if overrides:
            codes.update(overrides)

        def run(command: str, *, msg_in: str = None) -> tuple:
            return codes[command], "", ""

        self.sweep.util = mock.Mock(RunCmdSync=mock.Mock(side_effect=run))

    def _assert_commands(self, expected: list) -> None:
        self.assertEqual(
            self.sweep.util.RunCmdSync.call_args_list,
            [mock.call(command, msg_in=None) for command in expected],
        )

    def _restore(self, expected: list, overrides: dict = None) -> None:
        self._mock_commands(expected, overrides)
        self._run_restore()
        self._assert_commands(expected)

    def _run_restore(self) -> None:
        self.sweep.RunCmds(self.sweep._restore_commands())

    def _assert_reset_codes(self, command: str, accepted: tuple, rejected: tuple) -> None:
        """Run the restore with *command* answering each code; only *accepted* pass."""
        expected = self._expected_commands()
        for code in accepted:
            with self.subTest(command=command, code=code):
                self._restore(expected, {command: code})
        for code in rejected:
            with self.subTest(command=command, code=code):
                self._mock_commands(expected, {command: code})
                with self.assertRaises(AssertionError):
                    self._run_restore()
                self._assert_commands(expected)

    def _set_fan(self, fan: dict) -> None:
        self.sweep.metric_named_data["gpu_data"][0]["fan"] = fan

    def test_apu_default_dump_without_fan_or_perf_level_runs_cleanup(self) -> None:
        # The APU default metric dump omits fan and an N/A perf_level; the restore
        # reads both from the named sections instead.
        self._set_fan(dict(APU_FAN))
        self._restore(self._expected_commands())

    def test_restore_commands_leave_baseline_unchanged(self) -> None:
        self._set_fan(dict(APU_FAN))
        baseline = deepcopy(
            (self.sweep.metric_data, self.sweep.metric_named_data, self.sweep.static_data)
        )
        first = self.sweep._restore_commands()
        self.assertEqual(self.sweep._restore_commands(), first)
        self.assertEqual(
            (self.sweep.metric_data, self.sweep.metric_named_data, self.sweep.static_data), baseline
        )

    def test_missing_sections_are_rejected_before_writes(self) -> None:
        for data, field in (
            (self.sweep.metric_named_data, "fan"),
            (self.sweep.metric_named_data, "perf_level"),
            (self.sweep.metric_data, "clock"),
        ):
            with self.subTest(field=field):
                original = data["gpu_data"][0].pop(field)
                try:
                    self._mock_commands([])
                    with self.assertRaisesRegex(KeyError, field):
                        self._run_restore()
                    self.sweep.util.RunCmdSync.assert_not_called()
                finally:
                    data["gpu_data"][0][field] = original

    def test_invalid_restore_data_prevents_sweep_writes(self) -> None:
        self.sweep.static_data["gpu_data"][0]["profile"] = {}
        self.sweep.CreateCmds = mock.Mock(return_value=[("amd-smi set --fan 50% --gpu 0", 0)])
        self.sweep.util = mock.Mock(RunCmdSync=mock.Mock(return_value=(0, "", "")))
        with self.assertRaisesRegex(KeyError, "current"):
            self.sweep.test_command()
        self.sweep.util.RunCmdSync.assert_not_called()

    def test_baseline_records_named_metric_sections(self) -> None:
        outputs = {
            "amd-smi metric --json": {"gpu_data": [{"clock": {}}]},
            "amd-smi metric --fan --perf-level --json": {
                "gpu_data": [{"fan": APU_FAN, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]
            },
            "amd-smi static --json": {"gpu_data": [{}]},
            "amd-smi list --json": [{"gpu": 0, "bdf": "0000:c4:00.0", "uuid": "u"}],
            "amd-smi partition --current --json": {},
            "amd-smi list --nic all --json": [],
        }
        util = mock.Mock(
            RunCmdSync=mock.Mock(side_effect=lambda cmd: (0, json.dumps(outputs[cmd]), ""))
        )
        with mock.patch.object(test_set.TestSet, "util", util, create=True):
            with mock.patch.object(test_set.TestSet, "_read_clk_freq", return_value=[]):
                baseline = test_set.TestSet._build_baseline()
        self.assertEqual(
            baseline["metric_named_data"], outputs["amd-smi metric --fan --perf-level --json"]
        )

    def test_reported_power_cap_requires_reset_success(self) -> None:
        limits = self.sweep.static_data["gpu_data"][0]["limit"]
        # A current cap of 0 does not matter: reset writes the default cap.
        for power_type, current in (("ppt0", 291), ("ppt1", 291), ("ppt0", 0)):
            limits.update(ppt0="N/A", ppt1="N/A")
            limits[power_type] = deepcopy(POWER_CAP)
            limits[power_type]["socket_power_limit"]["value"] = current
            self._assert_reset_codes(
                "amd-smi reset --power-cap --gpu 0", accepted=(0,), rejected=(UNSUPPORTED,)
            )

    def test_missing_power_cap_requires_reset_not_supported(self) -> None:
        for sensor in ("N/A", {"min_power_limit": "N/A", "max_power_limit": "N/A"}):
            self.sweep.static_data["gpu_data"][0]["limit"]["ppt0"] = sensor
            self._assert_reset_codes(
                "amd-smi reset --power-cap --gpu 0", accepted=(UNSUPPORTED,), rejected=(0,)
            )

    def test_perf_determinism_restore_follows_clock_levels(self) -> None:
        static = self.sweep.static_data["gpu_data"][0]
        for sys_clock in ("N/A", {"frequency_levels": {}}):
            with self.subTest(sys_clock=sys_clock):
                static["clock"] = {"sys": sys_clock}
                self._restore(self._expected_commands())
        static["clock"] = {
            "sys": {
                "frequency_levels": {
                    "Level 0": {"value": "500", "unit": "MHz"},
                    "Level 1": {"value": "2100", "unit": "MHz"},
                }
            }
        }
        expected = self._expected_commands()
        expected.insert(2, "amd-smi set --perf-determinism 2100 --gpu 0")
        self._restore(expected)

    def test_unreadable_perf_level_is_not_restored(self) -> None:
        for perf_level in ("N/A", "AMDSMI_DEV_PERF_LEVEL_UNKNOWN"):
            with self.subTest(perf_level=perf_level):
                self.sweep.metric_named_data["gpu_data"][0]["perf_level"] = perf_level
                expected = [cmd for cmd in self._expected_commands() if "--perf-level" not in cmd]
                self._restore(expected)

    def test_fan_reset_without_reported_max_requires_not_supported(self) -> None:
        # An APU reports only apu_fan_pwm; a fanless dGPU reports max as N/A.
        for fan in (
            APU_FAN,
            {"apu_fan_pwm": {"value": 38, "unit": "%"}},
            FANLESS_DGPU_FAN,
            {"speed": 80, "max": "N/A", "rpm": "N/A", "usage": "N/A"},
        ):
            with self.subTest(fan=fan):
                self._set_fan(dict(fan))
                self._assert_reset_codes(
                    "amd-smi reset --fans --gpu 0", accepted=(UNSUPPORTED,), rejected=(0,)
                )

    def test_fan_reset_with_reported_max_allows_pass(self) -> None:
        # A runtime-suspended dGPU reports speed as N/A; max decides whether a fan exists.
        for speed in ("N/A", 80, 0):
            with self.subTest(speed=speed):
                self._set_fan(dict(DGPU_FAN, speed=speed))
                self._assert_reset_codes(
                    "amd-smi reset --fans --gpu 0",
                    accepted=(0, UNSUPPORTED),
                    rejected=(UNEXPECTED_DATA,),
                )

    def test_fan_max_out_of_range_fails_before_writes(self) -> None:
        for maximum in (0, 256):
            with self.subTest(maximum=maximum):
                self._set_fan(dict(DGPU_FAN, max=maximum))
                self._mock_commands([])
                with self.assertRaises(AssertionError):
                    self._run_restore()
                self.sweep.util.RunCmdSync.assert_not_called()

    def test_restore_mixed_apu_dgpu_grades_each_gpu(self) -> None:
        static = deepcopy(self.sweep.static_data["gpu_data"][0])
        self.sweep.common.processors.append(object())
        self.sweep.partition_data["current_partition"] *= 2
        self.sweep.metric_data["gpu_data"] *= 2
        for dgpu_index in (0, 1):
            with self.subTest(dgpu_index=dgpu_index):
                apu_index = 1 - dgpu_index
                named = [None, None]
                named[dgpu_index] = {"fan": dict(DGPU_FAN), "perf_level": "HIGH"}
                named[apu_index] = {"fan": dict(APU_FAN), "perf_level": "N/A"}
                self.sweep.metric_named_data = {"gpu_data": named}
                self.sweep.static_data["gpu_data"] = [deepcopy(static), deepcopy(static)]
                self.sweep.static_data["gpu_data"][dgpu_index]["limit"]["ppt0"] = deepcopy(
                    POWER_CAP
                )
                expected = []
                for index in range(2):
                    commands = self._expected_commands(index)
                    if index == dgpu_index:
                        commands = [
                            cmd.replace("--perf-level AUTO", "--perf-level HIGH")
                            for cmd in commands
                        ]
                    else:
                        commands = [cmd for cmd in commands if "--perf-level" not in cmd]
                    expected.extend(commands)
                dgpu_resets = {
                    f"amd-smi reset --fans --gpu {dgpu_index}": 0,
                    f"amd-smi reset --power-cap --gpu {dgpu_index}": 0,
                }
                self._restore(expected, dgpu_resets)
                for command in (
                    f"amd-smi reset --fans --gpu {apu_index}",
                    f"amd-smi reset --power-cap --gpu {apu_index}",
                ):
                    self._mock_commands(expected, dict(dgpu_resets, **{command: 0}))
                    with self.assertRaisesRegex(AssertionError, command):
                        self._run_restore()

    def test_restore_unexpected_command_failure_still_runs_remaining_cleanup(self) -> None:
        self._set_fan(dict(DGPU_FAN))
        expected = self._expected_commands()
        for failed_command in expected:
            with self.subTest(command=failed_command):
                self._mock_commands(expected, {failed_command: UNEXPECTED_DATA})
                with self.assertRaisesRegex(AssertionError, "UNEXPECTED_DATA"):
                    self._run_restore()
                self._assert_commands(expected)

    def test_restore_does_not_hide_missing_required_data(self) -> None:
        for section, value, missing in (("profile", {}, "current"), ("limit", {}, "ppt0")):
            with self.subTest(section=section):
                original = self.sweep.static_data["gpu_data"][0][section]
                self.sweep.static_data["gpu_data"][0][section] = value
                try:
                    self._mock_commands([])
                    with self.assertRaisesRegex(KeyError, missing):
                        self._run_restore()
                    self.sweep.util.RunCmdSync.assert_not_called()
                finally:
                    self.sweep.static_data["gpu_data"][0][section] = original

    def test_failed_set_sweep_registers_actual_restore_cleanup(self) -> None:
        expected = self._expected_commands()
        self._mock_commands(expected)
        run_restore = self.sweep.util.RunCmdSync.side_effect

        def run(command: str, *, msg_in: str = None) -> tuple:
            if command == "amd-smi set --help":
                return 0, "Set Arguments:\n  --fan %\n\n", ""
            if command.startswith("amd-smi set --fan "):
                return UNEXPECTED_DATA, "", ""
            return run_restore(command, msg_in=msg_in)

        self.sweep.util.RunCmdSync.side_effect = run
        self.sweep.sub_args = {}
        with self.assertRaisesRegex(AssertionError, "UNEXPECTED_DATA"):
            self.sweep.test_command()
        self.sweep.util.RunCmdSync.reset_mock()
        self.assertTrue(self.sweep.doCleanups())
        self._assert_commands(expected)
