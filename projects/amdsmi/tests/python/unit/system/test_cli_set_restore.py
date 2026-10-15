#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free replay of the set sweep's recorded starting values."""

import unittest
from copy import deepcopy
from unittest import mock

from common.common import amdsmi

try:
    from cli import test_set
except (ImportError, AttributeError) as exc:
    _CLI_SKIP = f"amd-smi CLI unavailable: {exc}"
else:
    _CLI_SKIP = None


@unittest.skipIf(_CLI_SKIP, _CLI_SKIP)
class TestCliSetRestore(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep = test_set.TestSet(methodName="test_command")
        self.sweep.common = mock.Mock(processors=[object()])
        self.sweep.list_data = [{"gpu": 0, "bdf": "0000:01:00.0"}]
        self.sweep.metric_data = {"gpu_data": [{"clock": {}, "perf_level": "AUTO"}]}
        self.sweep.static_data = {
            "gpu_data": [
                {
                    "profile": {"current": "COMPUTE", "available_profiles": ["COMPUTE"]},
                    "clock": {"sys": "N/A"},
                    "partition": {"compute_partition_mem_alloc_mode": "N/A"},
                    "soc_pstate": "N/A",
                    "xgmi_plpd": "N/A",
                    "limit": {"ptl_state": "N/A", "ptl_format": "N/A"},
                    "process_isolation": "Disabled",
                }
            ]
        }
        self.sweep.partition_data = {
            "current_partition": [{"memory": "NPS1", "accelerator_type": "SPX"}]
        }
        patcher = mock.patch.object(test_set.common, "has_gpu_od_interface", return_value=False)
        self.addCleanup(patcher.stop)
        self.gpu_od = patcher.start()

    def _remaining_commands(self, index: int = 0) -> list:
        return [
            f"amd-smi set --profile COMPUTE --gpu {index}",
            f"amd-smi set --memory-partition NPS1 --gpu {index}",
            f"amd-smi set --compute-partition SPX --gpu {index}",
            f"amd-smi reset --power-cap --gpu {index}",
            f"amd-smi reset --clocks --gpu {index}",
            f"amd-smi set --perf-level AUTO --gpu {index}",
            f"amd-smi set --process-isolation 0 --gpu {index}",
        ]

    def _mock_commands(self, expected: list, failures: dict = None) -> None:
        codes = {command: 0 for command in expected}
        for command in expected:
            if "reset" in command:
                codes[command] = int(amdsmi.AmdSmiStatus.NOT_SUPPORTED)
        if failures:
            codes.update(failures)

        def run(command: str, *, msg_in: str = None) -> tuple:
            return codes[command], "", ""

        self.sweep.util = mock.Mock(RunCmdSync=mock.Mock(side_effect=run))

    def _assert_commands(self, expected: list) -> None:
        self.assertEqual(
            self.sweep.util.RunCmdSync.call_args_list,
            [
                mock.call(command, msg_in="y\n" if "--memory-partition" in command else None)
                for command in expected
            ],
        )

    def _restore(self, expected: list) -> None:
        self._mock_commands(expected)
        self.sweep._restore_starting_values()
        self._assert_commands(expected)

    def test_restore_with_omitted_fan_runs_remaining_cleanup(self) -> None:
        self._restore(self._remaining_commands())
        self.gpu_od.assert_not_called()

    def test_restore_with_na_fan_runs_remaining_cleanup(self) -> None:
        self.sweep.metric_data["gpu_data"][0]["fan"] = "N/A"
        self._restore(self._remaining_commands())
        self.gpu_od.assert_not_called()

    def test_restore_without_static_clock_levels_runs_remaining_cleanup(self) -> None:
        for fields in (
            {},
            {"clock": "N/A"},
            {"clock": {}},
            {"clock": {"sys": "N/A"}},
            {"clock": {"sys": {"frequency_levels": {}}}},
        ):
            with self.subTest(fields=fields):
                self.sweep.static_data["gpu_data"][0].pop("clock", None)
                self.sweep.static_data["gpu_data"][0].update(fields)
                self._restore(self._remaining_commands())

    def test_restore_without_metric_clock_runs_remaining_cleanup(self) -> None:
        for fields in (
            {},
            {"clock": "N/A"},
            {"clock": {}},
            {"clock": {"gfx_0": "N/A"}},
            {"clock": {"apu_gfxclk": {"value": 400, "unit": "MHz"}}},
        ):
            with self.subTest(fields=fields):
                self.sweep.metric_data["gpu_data"][0].pop("clock", None)
                self.sweep.metric_data["gpu_data"][0].update(fields)
                self._restore(self._remaining_commands())

    def test_restore_without_perf_level_runs_remaining_cleanup(self) -> None:
        for fields in ({}, {"perf_level": "N/A"}, {"perf_level": "AMDSMI_DEV_PERF_LEVEL_UNKNOWN"}):
            with self.subTest(fields=fields):
                self.sweep.metric_data["gpu_data"][0].pop("perf_level", None)
                self.sweep.metric_data["gpu_data"][0].update(fields)
                expected = [cmd for cmd in self._remaining_commands() if "--perf-level" not in cmd]
                self._restore(expected)

    def test_restore_with_apu_or_unavailable_fan_has_no_dgpu_fan_reset(self) -> None:
        for fan in (
            {},
            {"max": "N/A", "speed": "N/A"},
            {"apu_fan_pwm": "N/A"},
            {"apu_fan_pwm": {"value": 38, "unit": "%"}},
        ):
            with self.subTest(fan=fan):
                self.sweep.metric_data["gpu_data"][0]["fan"] = fan
                self._restore(self._remaining_commands())
        self.gpu_od.assert_not_called()

    def test_restore_dgpu_fan_fields_are_independently_readable(self) -> None:
        for maximum, speed in (("N/A", 80), (255, "N/A"), (255, 80), (255, 0)):
            with self.subTest(maximum=maximum, speed=speed):
                self.sweep.metric_data["gpu_data"][0]["fan"] = {"max": maximum, "speed": speed}
                expected = self._remaining_commands()
                if speed != "N/A":
                    expected.insert(0, "amd-smi reset --fans --gpu 0")
                self._restore(expected)

    def test_restore_rejects_invalid_dgpu_fan_max(self) -> None:
        for gpu_od, maximum in ((False, 0), (False, -1), (False, 256), (True, 101)):
            with self.subTest(gpu_od=gpu_od, maximum=maximum):
                self.gpu_od.return_value = gpu_od
                self.sweep.metric_data["gpu_data"][0]["fan"] = {"max": maximum, "speed": 80}
                self._mock_commands([])
                with self.assertRaisesRegex(AssertionError, "max fan speed must be"):
                    self.sweep._restore_starting_values()
                self.sweep.util.RunCmdSync.assert_not_called()

    def test_restore_gpu_od_fan_accepts_percentage_maximum(self) -> None:
        self.gpu_od.return_value = True
        self.sweep.metric_data["gpu_data"][0]["fan"] = {"max": 100, "speed": 50}
        self._restore(["amd-smi reset --fans --gpu 0"] + self._remaining_commands())
        self.gpu_od.assert_called_once_with("0000:01:00.0")

    def test_restore_mixed_apu_dgpu_preserves_clock_and_cleanup_order(self) -> None:
        dgpu = {
            "fan": {"max": 255, "speed": 80},
            "clock": {"gfx_0": {"min_clk": {"value": 100}, "max_clk": {"value": 900}}},
            "perf_level": "AMDSMI_DEV_PERF_LEVEL_HIGH",
        }
        static = deepcopy(self.sweep.static_data["gpu_data"][0])
        static.pop("clock")
        self.sweep.common.processors.append(object())
        self.sweep.list_data.append({"gpu": 1, "bdf": "0000:02:00.0"})
        self.sweep.partition_data["current_partition"] *= 2
        for dgpu_index in (0, 1):
            with self.subTest(dgpu_index=dgpu_index):
                metrics = [{}, {}]
                metrics[dgpu_index] = dgpu
                self.sweep.metric_data["gpu_data"] = metrics
                self.sweep.static_data["gpu_data"] = [deepcopy(static), deepcopy(static)]
                self.sweep.static_data["gpu_data"][dgpu_index]["clock"] = {
                    "sys": {"frequency_levels": {"Level 0": {"value": "900"}}}
                }
                expected = []
                for index in range(2):
                    commands = self._remaining_commands(index)
                    if index == dgpu_index:
                        commands.insert(0, f"amd-smi reset --fans --gpu {index}")
                        commands.insert(2, f"amd-smi set --perf-determinism 900 --gpu {index}")
                        commands[6:6] = [
                            f"amd-smi set --clk-limit SCLK MAX 900 --gpu {index}",
                            f"amd-smi set --clk-limit SCLK MIN 100 --gpu {index}",
                        ]
                        commands = [
                            cmd.replace("--perf-level AUTO", "--perf-level HIGH")
                            for cmd in commands
                        ]
                    else:
                        commands = [cmd for cmd in commands if "--perf-level" not in cmd]
                    expected.extend(commands)
                self._restore(expected)

    def test_restore_unexpected_command_failure_still_runs_remaining_cleanup(self) -> None:
        self.sweep.metric_data["gpu_data"][0]["fan"] = {"max": 255, "speed": 80}
        expected = ["amd-smi reset --fans --gpu 0"] + self._remaining_commands()
        for failed_command in expected:
            with self.subTest(command=failed_command):
                self._mock_commands(
                    expected, {failed_command: int(amdsmi.AmdSmiStatus.UNEXPECTED_DATA)}
                )
                with self.assertRaisesRegex(AssertionError, "UNEXPECTED_DATA"):
                    self.sweep._restore_starting_values()
                self._assert_commands(expected)

    def test_restore_does_not_hide_missing_required_data(self) -> None:
        for section, value, missing in (
            ("profile", {}, "current"),
            ("clock", {"sys": {}}, "frequency_levels"),
            ("limit", {}, "ptl_state"),
        ):
            with self.subTest(section=section):
                original = self.sweep.static_data["gpu_data"][0][section]
                self.sweep.static_data["gpu_data"][0][section] = value
                self._mock_commands([])
                with self.assertRaisesRegex(KeyError, missing):
                    self.sweep._restore_starting_values()
                self.sweep.util.RunCmdSync.assert_not_called()
                self.sweep.static_data["gpu_data"][0][section] = original

    def test_restore_does_not_hide_missing_clock_bound(self) -> None:
        self.sweep.metric_data["gpu_data"][0]["clock"] = {"gfx_0": {"min_clk": {"value": 100}}}
        self._mock_commands([])
        with self.assertRaisesRegex(KeyError, "max_clk"):
            self.sweep._restore_starting_values()
        self.sweep.util.RunCmdSync.assert_not_called()

    def test_restore_does_not_hide_malformed_optional_sections(self) -> None:
        for section in ("fan", "clock"):
            with self.subTest(section=section):
                self.sweep.metric_data["gpu_data"][0][section] = None
                self._mock_commands([])
                with self.assertRaises(AttributeError):
                    self.sweep._restore_starting_values()
                self.sweep.util.RunCmdSync.assert_not_called()
                self.sweep.metric_data["gpu_data"][0].pop(section)

    def test_failed_set_sweep_registers_actual_restore_cleanup(self) -> None:
        expected = self._remaining_commands()
        self._mock_commands(expected)
        run_restore = self.sweep.util.RunCmdSync.side_effect

        def run(command: str, *, msg_in: str = None) -> tuple:
            if command == "amd-smi set --help":
                return 0, "Set Arguments:\n  --fan %\n\n", ""
            if command.startswith("amd-smi set --fan "):
                return int(amdsmi.AmdSmiStatus.UNEXPECTED_DATA), "", ""
            return run_restore(command, msg_in=msg_in)

        self.sweep.util.RunCmdSync.side_effect = run
        self.sweep.sub_args = {}
        with self.assertRaisesRegex(AssertionError, "UNEXPECTED_DATA"):
            self.sweep.test_command()
        self.sweep.util.RunCmdSync.reset_mock()
        self.assertTrue(self.sweep.doCleanups())
        self._assert_commands(expected)
