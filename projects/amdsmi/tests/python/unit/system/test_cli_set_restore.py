#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free replay of the set sweep's recorded starting values."""

import unittest
from copy import deepcopy
from unittest import mock

from cli import test_set
from common.common import amdsmi


class TestCliSetRestore(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep = test_set.TestSet(methodName="test_command")
        self.sweep.common = mock.Mock(processors=[object()])
        self.sweep.clk_freq = [{"is_apu": False}]
        self.sweep.metric_data = {
            "gpu_data": [{"fan": {"max": "N/A", "speed": "N/A"}, "clock": {}, "perf_level": "AUTO"}]
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

    def _remaining_commands(self, index: int = 0) -> list:
        return [
            f"amd-smi set --profile COMPUTE --gpu {index}",
            f"amd-smi reset --power-cap --gpu {index}",
            f"amd-smi reset --clocks --gpu {index}",
            f"amd-smi set --perf-level AUTO --gpu {index}",
            f"amd-smi set --process-isolation 0 --gpu {index}",
        ]

    def _mock_commands(self, expected: list, failures: dict = None) -> None:
        codes = {command: 0 for command in expected}
        if failures:
            codes.update(failures)

        def run(command: str, *, msg_in: str = None) -> tuple:
            return codes[command], "", ""

        self.sweep.util = mock.Mock(RunCmdSync=mock.Mock(side_effect=run))

    def _assert_commands(self, expected: list) -> None:
        self.assertEqual(
            self.sweep.util.RunCmdSync.call_args_list,
            [mock.call(command, msg_in=None) for command in expected],
        )

    def _restore(self, expected: list) -> None:
        self._mock_commands(expected)
        self._run_restore()
        self._assert_commands(expected)

    def _run_restore(self) -> None:
        self.sweep.RunCmds(self.sweep._restore_commands())

    def test_restore_with_omitted_fan_runs_remaining_cleanup(self) -> None:
        self.sweep.clk_freq[0]["is_apu"] = True
        del self.sweep.metric_data["gpu_data"][0]["fan"]
        self._restore(self._remaining_commands())

    def test_missing_dgpu_sections_are_rejected(self) -> None:
        for field in ("fan", "clock", "perf_level"):
            with self.subTest(field=field):
                original = self.sweep.metric_data["gpu_data"][0].pop(field)
                self._mock_commands([])
                with self.assertRaisesRegex(KeyError, field):
                    self._run_restore()
                self.sweep.util.RunCmdSync.assert_not_called()
                self.sweep.metric_data["gpu_data"][0][field] = original

    def test_invalid_restore_data_prevents_sweep_writes(self) -> None:
        self.sweep.static_data["gpu_data"][0]["profile"] = {}
        self.sweep.CreateCmds = mock.Mock(return_value=[("amd-smi set --fan 50% --gpu 0", 0)])
        self.sweep.util = mock.Mock(RunCmdSync=mock.Mock(return_value=(0, "", "")))
        with self.assertRaisesRegex(KeyError, "current"):
            self.sweep.test_command()
        self.sweep.util.RunCmdSync.assert_not_called()

    def test_baseline_records_apu_status_from_metrics(self) -> None:
        processors = [object(), object(), object()]
        with mock.patch.object(test_set.TestSet, "common", self.sweep.common, create=True):
            with mock.patch.multiple(
                amdsmi,
                amdsmi_get_processor_handles=mock.Mock(return_value=processors),
                amdsmi_get_gpu_metrics_info=mock.Mock(
                    side_effect=[
                        {"is_apu": True},
                        {"is_apu": False},
                        amdsmi.AmdSmiLibraryException(amdsmi.AmdSmiStatus.NOT_SUPPORTED),
                    ]
                ),
                amdsmi_get_clk_freq=mock.Mock(return_value={"num_supported": 1}),
                amdsmi_get_clock_info=mock.Mock(return_value={"clk_deep_sleep": "N/A"}),
                amdsmi_shut_down=mock.DEFAULT,
            ) as functions:
                baseline = test_set.TestSet._read_clk_freq()
        self.assertEqual([entry.get("is_apu") for entry in baseline], [True, False, False])
        functions["amdsmi_shut_down"].assert_called_once_with()

    def test_supported_power_cap_rejects_not_supported_reset(self) -> None:
        limits = self.sweep.static_data["gpu_data"][0]["limit"]
        for power_type in ("ppt0", "ppt1"):
            with self.subTest(power_type=power_type):
                limits.update(ppt0="N/A", ppt1="N/A")
                limits[power_type] = {
                    "min_power_limit": {"value": 261, "unit": "W"},
                    "max_power_limit": {"value": 350, "unit": "W"},
                    "socket_power_limit": {"value": 291, "unit": "W"},
                }
                self.sweep.static_data["gpu_data"][0]["limit"] = limits
                self._restore(self._remaining_commands())
                self._mock_commands(
                    self._remaining_commands(),
                    {"amd-smi reset --power-cap --gpu 0": int(amdsmi.AmdSmiStatus.NOT_SUPPORTED)},
                )
                with self.assertRaisesRegex(AssertionError, "NOT_SUPPORTED"):
                    self._run_restore()

    def test_unsettable_power_cap_allows_not_supported_reset(self) -> None:
        for sensor in (
            "N/A",
            {"min_power_limit": "N/A", "max_power_limit": "N/A"},
            {
                "min_power_limit": {"value": 100},
                "max_power_limit": {"value": 300},
                "socket_power_limit": {"value": 0},
            },
        ):
            with self.subTest(sensor=sensor):
                self.sweep.static_data["gpu_data"][0]["limit"]["ppt0"] = sensor
                self._mock_commands(
                    self._remaining_commands(),
                    {"amd-smi reset --power-cap --gpu 0": int(amdsmi.AmdSmiStatus.NOT_SUPPORTED)},
                )
                self._run_restore()
                self._assert_commands(self._remaining_commands())

    def test_restore_with_na_fan_runs_remaining_cleanup(self) -> None:
        self.sweep.clk_freq[0]["is_apu"] = True
        self.sweep.metric_data["gpu_data"][0]["fan"] = "N/A"
        self._restore(self._remaining_commands())

    def test_restore_without_static_clock_levels_runs_remaining_cleanup(self) -> None:
        self.sweep.clk_freq[0]["is_apu"] = True
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
        self.sweep.clk_freq[0]["is_apu"] = True
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
        self.sweep.clk_freq[0]["is_apu"] = True
        for fields in ({}, {"perf_level": "N/A"}, {"perf_level": "AMDSMI_DEV_PERF_LEVEL_UNKNOWN"}):
            with self.subTest(fields=fields):
                self.sweep.metric_data["gpu_data"][0].pop("perf_level", None)
                self.sweep.metric_data["gpu_data"][0].update(fields)
                expected = [cmd for cmd in self._remaining_commands() if "--perf-level" not in cmd]
                self._restore(expected)

    def test_restore_with_apu_or_unavailable_fan_has_no_dgpu_fan_reset(self) -> None:
        self.sweep.clk_freq[0]["is_apu"] = True
        for fan in (
            {},
            {"max": "N/A", "speed": "N/A"},
            {"apu_fan_pwm": "N/A"},
            {"apu_fan_pwm": {"value": 38, "unit": "%"}},
        ):
            with self.subTest(fan=fan):
                self.sweep.metric_data["gpu_data"][0]["fan"] = fan
                self._restore(self._remaining_commands())

    def test_restore_dgpu_fan_fields_are_independently_readable(self) -> None:
        for maximum, speed in (("N/A", 80), (255, "N/A"), (255, 80), (255, 0)):
            with self.subTest(maximum=maximum, speed=speed):
                self.sweep.metric_data["gpu_data"][0]["fan"] = {"max": maximum, "speed": speed}
                expected = self._remaining_commands()
                if speed != "N/A":
                    expected.insert(0, "amd-smi reset --fans --gpu 0")
                self._restore(expected)

    def test_restore_mixed_apu_dgpu_preserves_cleanup_order(self) -> None:
        dgpu = {
            "fan": {"max": 255, "speed": 80},
            "clock": {},
            "perf_level": "AMDSMI_DEV_PERF_LEVEL_HIGH",
        }
        static = deepcopy(self.sweep.static_data["gpu_data"][0])
        self.sweep.common.processors.append(object())
        self.sweep.partition_data["current_partition"] *= 2
        for dgpu_index in (0, 1):
            with self.subTest(dgpu_index=dgpu_index):
                self.sweep.clk_freq = [{"is_apu": index != dgpu_index} for index in range(2)]
                metrics = [{}, {}]
                metrics[dgpu_index] = deepcopy(dgpu)
                self.sweep.metric_data["gpu_data"] = metrics
                self.sweep.static_data["gpu_data"] = [deepcopy(static), deepcopy(static)]
                self.sweep.static_data["gpu_data"][1 - dgpu_index].pop("clock")
                expected = []
                for index in range(2):
                    commands = self._remaining_commands(index)
                    if index == dgpu_index:
                        commands = [f"amd-smi reset --fans --gpu {index}"] + [
                            cmd.replace("--perf-level AUTO", "--perf-level HIGH")
                            for cmd in commands
                        ]
                    else:
                        commands = [cmd for cmd in commands if "--perf-level" not in cmd]
                    expected.extend(commands)
                self._restore(expected)
                self.assertEqual(metrics[1 - dgpu_index], {})

    def test_restore_unexpected_command_failure_still_runs_remaining_cleanup(self) -> None:
        self.sweep.metric_data["gpu_data"][0]["fan"] = {"max": 255, "speed": 80}
        expected = ["amd-smi reset --fans --gpu 0"] + self._remaining_commands()
        for failed_command in expected:
            with self.subTest(command=failed_command):
                self._mock_commands(
                    expected, {failed_command: int(amdsmi.AmdSmiStatus.UNEXPECTED_DATA)}
                )
                with self.assertRaisesRegex(AssertionError, "UNEXPECTED_DATA"):
                    self._run_restore()
                self._assert_commands(expected)

    def test_restore_does_not_hide_missing_required_data(self) -> None:
        for section, value, missing in (("profile", {}, "current"), ("limit", {}, "ppt0")):
            with self.subTest(section=section):
                original = self.sweep.static_data["gpu_data"][0][section]
                self.sweep.static_data["gpu_data"][0][section] = value
                self._mock_commands([])
                with self.assertRaisesRegex(KeyError, missing):
                    self._run_restore()
                self.sweep.util.RunCmdSync.assert_not_called()
                self.sweep.static_data["gpu_data"][0][section] = original

    def test_restore_does_not_hide_malformed_optional_sections(self) -> None:
        for section in ("fan", "clock"):
            with self.subTest(section=section):
                original = self.sweep.metric_data["gpu_data"][0][section]
                self.sweep.metric_data["gpu_data"][0][section] = None
                self._mock_commands([])
                with self.assertRaises(TypeError):
                    self._run_restore()
                self.sweep.util.RunCmdSync.assert_not_called()
                self.sweep.metric_data["gpu_data"][0][section] = original

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
