# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regressions for the CLI set sweep."""

import types
import unittest
from unittest.mock import Mock

from cli import base, test_set


class TestCliSetSweep(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep = base.TestCliBase()
        self.sweep.common = types.SimpleNamespace(
            processors=[object(), object()], print=Mock(), cleaner_shader_resets_gpu=lambda: False
        )

    def _commands(self, options, command="set"):
        arguments = {
            "options": options,
            "devices": ["--gpu all", "--gpu 0", "--gpu 1"],
            "modifiers": ["pass"],
            "": ["pass"],
        }
        self.sweep.FindArgs = lambda _command, section: arguments[section]
        return self.sweep.CreateCmds(command, "options", "devices", "modifiers", "")

    def test_every_set_option_runs_once_per_gpu(self) -> None:
        options = ["--perf-level AUTO", "--fan 50"]
        for reduced in (True, False):
            with self.subTest(reduced=reduced):
                self.sweep.ReduceCmds = reduced
                commands = self._commands(options)
                self.assertCountEqual(
                    [command for command, _ in commands],
                    [
                        f"amd-smi set {option} --loglevel DEBUG --gpu {gpu}"
                        for option in options
                        for gpu in range(2)
                    ],
                )
                for _, allowed in commands:
                    self.assertEqual(
                        allowed, [self.sweep.PASS, base.amdsmi.AmdSmiStatus.NOT_SUPPORTED]
                    )

    def test_memory_partition_changes_are_explicitly_excluded(self) -> None:
        commands = self._commands(
            ["--memory-partition NPS1", "--memory-partition NPS2", "--perf-level AUTO"]
        )
        self.assertTrue(commands)
        self.assertFalse(any("--memory-partition" in command for command, _ in commands))
        messages = [call[0][0] for call in self.sweep.common.print.call_args_list]
        exclusions = [message for message in messages if "memory-partition" in message]
        self.assertEqual(len(exclusions), 1)
        self.assertIn("cannot cancel", exclusions[0])

    def test_clock_values_come_from_the_selected_gpu(self) -> None:
        self.sweep.clk_freq = [
            {"FCLK": {"num_supported": 3, "has_deep_sleep": False}},
            {"FCLK": {"num_supported": 3, "has_deep_sleep": True}},
        ]
        self.sweep.metric_data = {
            "gpu_data": [
                {"clock": {"fclk_0": {"min_clk": {"value": low}, "max_clk": {"value": high}}}}
                for low, high in ((1200, 2000), (400, 800))
            ]
        }
        commands = self._commands(
            [
                "--clk-level {clk_level_fclk}",
                "--clk-limit {clk_limit_fclk_max}",
                "--clk-limit {clk_limit_fclk_min}",
            ]
        )
        self.assertCountEqual(
            [command for command, _ in commands],
            [
                f"amd-smi set {option} --loglevel DEBUG --gpu {gpu}"
                for gpu, options in (
                    (
                        0,
                        (
                            "--clk-level FCLK 0 1 2",
                            "--clk-limit FCLK MAX 1800",
                            "--clk-limit FCLK MIN 1400",
                        ),
                    ),
                    (
                        1,
                        (
                            "--clk-level FCLK 0 1",
                            "--clk-limit FCLK MAX 700",
                            "--clk-limit FCLK MIN 500",
                        ),
                    ),
                )
                for option in options
            ],
        )
        for _, allowed in commands:
            self.assertEqual(
                allowed,
                [
                    self.sweep.PASS,
                    base.amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                    base.AmdSmiExitCode.INVALID_PARAMETER_VALUE,
                ],
            )

    def test_unsupported_clock_does_not_hide_the_other_gpu(self) -> None:
        self.sweep.clk_freq = [
            {"FCLK": {"num_supported": 3, "has_deep_sleep": False}},
            {"FCLK": None},
        ]
        commands = self._commands(["--clk-level {clk_level_fclk}"])
        self.assertEqual(
            [command for command, _ in commands],
            ["amd-smi set --clk-level FCLK 0 1 2 --loglevel DEBUG --gpu 0"],
        )

    def test_mixed_supported_and_unsupported_devices_pass_individually(self) -> None:
        commands = self._commands(["--fan 50"])
        self.sweep.util = types.SimpleNamespace(
            RunCmdSync=Mock(side_effect=[(0, "", ""), (2, "", "")])
        )
        self.sweep.RunCmds(commands)
        self.assertEqual(self.sweep.util.RunCmdSync.call_count, 2)

    def test_io_and_aggregate_errors_are_not_accepted(self) -> None:
        commands = self._commands(["--fan 50"])
        for code in (12, 204):
            with self.subTest(code=code):
                self.sweep.util = types.SimpleNamespace(
                    RunCmdSync=Mock(side_effect=[(0, "", ""), (code, "", "")])
                )
                with self.assertRaisesRegex(AssertionError, f"--gpu 1.*rc={code}"):
                    self.sweep.RunCmds(commands)
                self.assertEqual(self.sweep.util.RunCmdSync.call_count, 2)

    def test_read_sweep_keeps_all_gpu_coverage(self) -> None:
        commands = self._commands(["--usage"], command="metric")
        self.assertIn(
            "amd-smi metric --usage --loglevel DEBUG", [command for command, _ in commands]
        )
        self.assertIn(
            "amd-smi metric --usage --gpu all --loglevel DEBUG",
            [command for command, _ in commands],
        )

    def test_cleanup_does_not_claim_to_restore_memory_partitions(self) -> None:
        self.sweep.metric_data = {
            "gpu_data": [
                {"fan": {"max": "N/A", "speed": "N/A"}, "clock": {}, "perf_level": "N/A"}
                for _ in range(2)
            ]
        }
        self.sweep.static_data = {
            "gpu_data": [
                {
                    "clock": {"sys": "N/A"},
                    "partition": {"compute_partition_mem_alloc_mode": "N/A"},
                    "soc_pstate": "N/A",
                    "xgmi_plpd": "N/A",
                    "limit": {"ptl_state": "N/A", "ptl_format": "N/A"},
                    "process_isolation": "N/A",
                }
                for _ in range(2)
            ]
        }
        self.sweep.partition_data = {
            "current_partition": [{"memory": "NPS1", "accelerator_type": "N/A"} for _ in range(2)]
        }
        self.sweep.RunCmds = Mock()
        test_set.TestSet._restore_starting_values(self.sweep)
        commands = self.sweep.RunCmds.call_args[0][0]
        self.assertTrue(commands)
        self.assertFalse(any("--memory-partition" in command for command, _ in commands))
