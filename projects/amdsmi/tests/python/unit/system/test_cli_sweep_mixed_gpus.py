#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free checks of the CLI sweep on hosts whose GPUs differ, such as a dGPU
next to an iGPU that lacks some clock tables and gpu_metrics.
"""

import unittest
from unittest import mock

from common.common import amdsmi

try:
    from cli.base import AmdSmiExitCode, TestCliBase
except (ImportError, AttributeError) as exc:  # needs the amd-smi CLI under test
    _CLI_SKIP = f"amd-smi CLI unavailable: {exc}"
else:
    _CLI_SKIP = None

NOT_SUPPORTED = int(amdsmi.AmdSmiStatus.NOT_SUPPORTED)
INVAL = int(amdsmi.AmdSmiStatus.INVAL)
UNEXPECTED_DATA = int(amdsmi.AmdSmiStatus.UNEXPECTED_DATA)


@unittest.skipIf(_CLI_SKIP, _CLI_SKIP)
class TestCliSweepMixedGpus(unittest.TestCase):
    def _run_cmd(self, cmd, cond, codes, gpu_count=2):
        """Grade *cmd* through RunCmds; *codes* maps each command line to its exit code.

        Returns:
            tuple: (whether RunCmds passed, the command lines it ran).
        """
        sweep = TestCliBase()
        sweep.common = mock.Mock()
        sweep.list_data = [{"gpu": index} for index in range(gpu_count)]
        ran = []

        def run(command, **_):
            ran.append(command)
            return codes[command], "", ""

        sweep.util = mock.Mock(RunCmdSync=run)
        try:
            sweep.RunCmds([(cmd, cond)])
        except AssertionError:
            return False, ran
        return True, ran

    def test_initial_mixed_error_fails_without_retries(self) -> None:
        for selector in ("", " --gpu all"):
            with self.subTest(selector=selector):
                base = "amd-smi set --clk-level FCLK 0 1"
                cmd = f"{base}{selector}"
                codes = {
                    cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS),
                    f"{base} --gpu 0": 0,
                    f"{base} --gpu 1": NOT_SUPPORTED,
                }
                passed, ran = self._run_cmd(cmd, [0, NOT_SUPPORTED], codes)
                self.assertEqual((passed, ran), (False, [cmd]))

    def test_exact_gpu_target_aliases_are_not_expanded(self) -> None:
        for selector in ("--gpu 0", "-g 0", "-g0", "--gpu 0000:01:00.0", "--gpu GPU-123"):
            with self.subTest(selector=selector):
                cmd = f"amd-smi reset --clocks {selector}"
                codes = {
                    cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS),
                    f"{cmd} --gpu 0": 0,
                    f"{cmd} --gpu 1": NOT_SUPPORTED,
                }
                passed, ran = self._run_cmd(cmd, [0, NOT_SUPPORTED], codes)
                self.assertEqual((passed, ran), (False, [cmd]))

    def test_first_execution_per_gpu_codes_are_graded_independently(self) -> None:
        commands = [f"amd-smi set --clk-level FCLK 0 --gpu {gpu}" for gpu in (0, 1)]
        for codes, accepted, failures in (
            ((INVAL, NOT_SUPPORTED), [0, INVAL, NOT_SUPPORTED], []),
            ((INVAL, NOT_SUPPORTED), [0, NOT_SUPPORTED], [commands[0]]),
            ((NOT_SUPPORTED, UNEXPECTED_DATA), [0, NOT_SUPPORTED], [commands[1]]),
        ):
            with self.subTest(codes=codes, accepted=accepted):
                sweep = TestCliBase()
                sweep.common = mock.Mock()
                sweep.util = mock.Mock()
                sweep.util.RunCmdSync.side_effect = [(code, "", "") for code in codes]
                if failures:
                    with self.assertRaises(AssertionError) as error:
                        sweep.RunCmds([(cmd, accepted) for cmd in commands])
                    for cmd in commands:
                        self.assertEqual(cmd in str(error.exception), cmd in failures)
                else:
                    sweep.RunCmds([(cmd, accepted) for cmd in commands])
                self.assertEqual(
                    sweep.util.RunCmdSync.call_args_list,
                    [mock.call(cmd, msg_in=None) for cmd in commands],
                )

    def test_mixed_codes_on_one_named_gpu_are_not_rerun(self):
        cmd = "amd-smi reset --clocks --gpu 0"
        passed, ran = self._run_cmd(
            cmd, [0, NOT_SUPPORTED], {cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS)}
        )
        self.assertFalse(passed)
        self.assertEqual(ran, [cmd])

    def test_mixed_codes_on_a_single_gpu_host_are_not_rerun(self):
        cmd = "amd-smi reset --clocks"
        passed, ran = self._run_cmd(
            cmd, [0, NOT_SUPPORTED], {cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS)}, gpu_count=1
        )
        self.assertFalse(passed)
        self.assertEqual(ran, [cmd])

    def test_all_gpu_clk_level_is_dropped_when_a_gpu_has_no_table(self):
        sweep = TestCliBase()
        sweep.clk_freq = [{"FCLK": {"num_supported": 3, "has_deep_sleep": False}}, {"FCLK": None}]
        self.assertEqual(sweep._clk_level_count("FCLK", 0, explicit_gpu=False), 0)
        self.assertEqual(sweep._clk_level_count("FCLK", 0, explicit_gpu=True), 3)

    def test_all_gpu_clk_level_fits_the_smallest_table(self):
        sweep = TestCliBase()
        sweep.clk_freq = [
            {"SOCCLK": {"num_supported": 4, "has_deep_sleep": True}},
            {"SOCCLK": {"num_supported": 2, "has_deep_sleep": False}},
        ]
        self.assertEqual(sweep._clk_level_count("SOCCLK", 0, explicit_gpu=False), 2)

    def test_clock_table_is_kept_when_clock_info_is_not_supported(self):
        dgpu, igpu = object(), object()

        def clk_freq(handle, clk_type):
            if handle is igpu and clk_type == amdsmi.AmdSmiClkType.DF:
                raise amdsmi.AmdSmiLibraryException(UNEXPECTED_DATA)
            return {"num_supported": 3}

        def clock_info(handle, _clk_type):
            if handle is igpu:
                raise amdsmi.AmdSmiLibraryException(NOT_SUPPORTED)
            return {"clk_deep_sleep": 28}

        library = mock.patch.multiple(
            amdsmi,
            amdsmi_get_processor_handles=lambda: [dgpu, igpu],
            amdsmi_get_clk_freq=clk_freq,
            amdsmi_get_clock_info=clock_info,
            amdsmi_shut_down=lambda: None,
        )
        with library, mock.patch.object(TestCliBase, "common", mock.Mock(), create=True):
            tables = TestCliBase._read_clk_freq()

        self.assertTrue(tables[0]["SOCCLK"]["has_deep_sleep"])
        self.assertEqual(tables[1]["SOCCLK"], {"num_supported": 3, "has_deep_sleep": False})
        self.assertIsNone(tables[1]["FCLK"])


@unittest.skipIf(_CLI_SKIP, _CLI_SKIP)
class TestCliClockCommandGeneration(unittest.TestCase):
    def _commands(self, *, set_args: str, metrics: list, static: list = None) -> list:
        sweep = TestCliBase()
        sweep.common = mock.Mock()
        sweep.common.cleaner_shader_resets_gpu.return_value = False
        sweep.metric_data = {"gpu_data": metrics}
        sweep.static_data = {"gpu_data": static}
        sweep.sub_args = {"GPU": [0, 1]}
        sweep.ReduceCmds = False
        sweep.AddLogLevel = ""
        help_text = (
            f"Set Arguments:\n{set_args}\n  --process-isolation STATUS\n\n"
            "Device Arguments:\n  --gpu GPU\n\n"
        )

        def run(command: str) -> tuple:
            self.assertEqual(command, "amd-smi set --help")
            return 0, help_text, ""

        sweep.util = mock.Mock(RunCmdSync=run)
        return sweep.CreateCmds("set", "Set Arguments:", "Device Arguments:", "", "")

    def _isolation_commands(self) -> list:
        return [
            (f"amd-smi set --process-isolation {value}{gpu}", [0, NOT_SUPPORTED])
            for value in (0, 1)
            for gpu in ("", " --gpu 0", " --gpu 1")
        ]

    def test_create_clock_limits_without_first_gpu_clock(self) -> None:
        dgpu = {"clock": {"gfx_0": {"min_clk": {"value": 100}, "max_clk": {"value": 900}}}}
        for apu in (
            {},
            {"clock": "N/A"},
            {"clock": {}},
            {"clock": {"gfx_0": "N/A"}},
            {"clock": {"apu_gfxclk": {"value": 400, "unit": "MHz"}}},
        ):
            with self.subTest(apu=apu):
                commands = self._commands(set_args="  --clk-limit CLK_TYPE", metrics=[apu, dgpu])
                expected = [
                    (
                        f"amd-smi set --clk-limit SCLK {bound} {value} --gpu 1",
                        [0, NOT_SUPPORTED, AmdSmiExitCode.INVALID_PARAMETER_VALUE],
                    )
                    for bound, value in (("MAX", 700), ("MIN", 300))
                ]
                self.assertEqual(commands, expected + self._isolation_commands())

    def test_create_determinism_without_first_gpu_clock(self) -> None:
        dgpu = {"clock": {"sys": {"frequency_levels": {"Level 0": {"value": "900"}}}}}
        for apu in (
            {},
            {"clock": "N/A"},
            {"clock": {}},
            {"clock": {"sys": "N/A"}},
            {"clock": {"sys": {"frequency_levels": {}}}},
        ):
            with self.subTest(apu=apu):
                commands = self._commands(
                    set_args="  --perf-determinism SCLKMAX", metrics=[{}, {}], static=[apu, dgpu]
                )
                expected = [("amd-smi set --perf-determinism 950 --gpu 1", [0, NOT_SUPPORTED])]
                self.assertEqual(commands, expected + self._isolation_commands())

    def test_create_clock_limits_rejects_missing_bound(self) -> None:
        with self.assertRaisesRegex(KeyError, "max_clk"):
            self._commands(
                set_args="  --clk-limit CLK_TYPE",
                metrics=[{"clock": {"gfx_0": {"min_clk": {"value": 100}}}}],
            )

    def test_create_determinism_rejects_missing_frequency_levels(self) -> None:
        with self.assertRaisesRegex(KeyError, "frequency_levels"):
            self._commands(
                set_args="  --perf-determinism SCLKMAX",
                metrics=[{}],
                static=[{"clock": {"sys": {}}}],
            )


@unittest.skipIf(_CLI_SKIP, _CLI_SKIP)
class TestCliDeviceLocalSetGeneration(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.sweep = TestCliBase()
        self.sweep.common = mock.Mock()
        self.sweep.common.cleaner_shader_resets_gpu.return_value = False
        self.sweep.AddLogLevel = ""
        self.sweep.sub_args = {"GPU": ["all", 0, "0000:01:00.0", "GPU-123", 1]}

    def _commands(self, command: str, arguments: str) -> list:
        self.assertTrue(self.sweep.ReduceCmds)
        title = f"{command.title()} Arguments:"
        help_text = f"{title}\n{arguments}\n\nDevice Arguments:\n  -g, --gpu GPU\n\n"
        self.sweep.util = mock.Mock()
        self.sweep.util.RunCmdSync.return_value = (0, help_text, "")
        return self.sweep.CreateCmds(command, title, "Device Arguments:", "", "")

    def test_reduced_mixed_gpu_setters_keep_device_local_values(self) -> None:
        apu = {
            "metric": {"clock": {"gfx_0": {"min_clk": {"value": 100}, "max_clk": {"value": 500}}}},
            "static": {
                "clock": {"sys": {"frequency_levels": {"Level 0": {"value": "500"}}}},
                "bus": {"pcie_levels": "N/A"},
                "profile": "N/A",
                "limit": {
                    "ppt0": {
                        "min_power_limit": {"value": 10},
                        "max_power_limit": {"value": 30},
                        "socket_power_limit": {"value": 20},
                    },
                    "ppt1": "N/A",
                },
            },
            "clk_freq": {"SCLK": {"num_supported": 1, "has_deep_sleep": False}, "FCLK": None},
            "expected": [
                "--clk-level SCLK 0",
                "--clk-limit SCLK MAX 400",
                "--clk-limit SCLK MIN 200",
                "--perf-determinism 550",
                "--power-cap 10 ppt0",
                "--power-cap 20 ppt0",
                "--power-cap 30 ppt0",
            ],
        }
        dgpu = {
            "metric": {"clock": {"gfx_0": {"min_clk": {"value": 100}, "max_clk": {"value": 900}}}},
            "static": {
                "clock": {"sys": {"frequency_levels": {"Level 0": {"value": "900"}}}},
                "bus": {"pcie_levels": {"Level 0": {}, "Level 1": {}}},
                "profile": {"available_profiles": ["VIDEO", "COMPUTE"]},
                "limit": {
                    "ppt0": {
                        "min_power_limit": {"value": 100},
                        "max_power_limit": {"value": 300},
                        "socket_power_limit": {"value": 200},
                    },
                    "ppt1": "N/A",
                },
            },
            "clk_freq": {
                "SCLK": {"num_supported": 4, "has_deep_sleep": True},
                "FCLK": {"num_supported": 3, "has_deep_sleep": False},
            },
            "expected": [
                "--clk-level SCLK 0 1 2",
                "--clk-level FCLK 0 1 2",
                "--clk-level PCIE 0 1",
                "--clk-limit SCLK MAX 700",
                "--clk-limit SCLK MIN 300",
                "--profile VIDEO",
                "--profile COMPUTE",
                "--perf-determinism 950",
                "--power-cap 100 ppt0",
                "--power-cap 200 ppt0",
                "--power-cap 300 ppt0",
            ],
        }
        arguments = (
            "  --fan %\n  --clk-level CLK_TYPE\n  --clk-limit CLK_TYPE\n"
            "  --profile PROFILE_LEVEL\n  --perf-determinism SCLKMAX\n"
            "  --power-cap WATTS\n  --memory-partition PARTITION"
        )
        for devices in ((apu, dgpu), (dgpu, apu)):
            with self.subTest(apu_first=devices[0] is apu):
                self.sweep.metric_data = {"gpu_data": [device["metric"] for device in devices]}
                self.sweep.static_data = {"gpu_data": [device["static"] for device in devices]}
                self.sweep.clk_freq = [device["clk_freq"] for device in devices]
                commands = self._commands("set", arguments)
                expected = []
                for gpu, device in enumerate(devices):
                    for option in device["expected"]:
                        codes = [0, NOT_SUPPORTED]
                        if option.startswith("--clk-limit") or (
                            option.startswith("--clk-level") and "PCIE" not in option
                        ):
                            codes.append(AmdSmiExitCode.INVALID_PARAMETER_VALUE)
                        expected.append((f"amd-smi set {option} --gpu {gpu}", codes))
                    expected.extend(
                        (
                            f"amd-smi set --memory-partition {mode} --gpu {gpu}",
                            [0, NOT_SUPPORTED, INVAL],
                        )
                        for mode in ("NPS1", "NPS2", "NPS4", "NPS8")
                    )
                self.assertCountEqual(
                    [(cmd, codes) for cmd, codes in commands if "--fan" not in cmd], expected
                )
                self.assertIn(("amd-smi set --fan 50% --gpu all", [0, NOT_SUPPORTED]), commands)
                self.assertTrue(
                    all(
                        call[0] == ("amd-smi set --help",)
                        for call in self.sweep.util.RunCmdSync.call_args_list
                    )
                )

    def test_reduced_clock_setters_keep_gpu_with_only_readable_table(self) -> None:
        dgpu_metric = {"clock": {"gfx_0": {"min_clk": {"value": 100}, "max_clk": {"value": 900}}}}
        dgpu_static = {"clock": {"sys": {"frequency_levels": {"Level 0": {"value": "900"}}}}}
        for gpu in (0, 1):
            for apu in ({}, {"clock": "N/A"}, {"clock": {}}):
                with self.subTest(gpu=gpu, apu=apu):
                    metrics, static = [apu, apu], [apu, apu]
                    metrics[gpu], static[gpu] = dgpu_metric, dgpu_static
                    self.sweep.metric_data = {"gpu_data": metrics}
                    self.sweep.static_data = {"gpu_data": static}
                    commands = self._commands(
                        "set", "  --fan %\n  --clk-limit CLK_TYPE\n  --perf-determinism SCLKMAX"
                    )
                    self.assertEqual(
                        [(cmd, codes) for cmd, codes in commands if "--fan" not in cmd],
                        [
                            (
                                f"amd-smi set --clk-limit SCLK {bound} {value} --gpu {gpu}",
                                [0, NOT_SUPPORTED, AmdSmiExitCode.INVALID_PARAMETER_VALUE],
                            )
                            for bound, value in (("MAX", 700), ("MIN", 300))
                        ]
                        + [(f"amd-smi set --perf-determinism 950 --gpu {gpu}", [0, NOT_SUPPORTED])],
                    )

    def test_read_sweep_keeps_all_gpu_cases(self) -> None:
        commands = self._commands("static", "  --clock CLOCK\n  --profile PROFILE")
        self.assertEqual(
            commands,
            [
                ("amd-smi static --clock", [0]),
                ("amd-smi static --clock --gpu all", [0]),
                ("amd-smi static --clock --gpu 0", [0]),
                ("amd-smi static --clock --gpu 0000:01:00.0", [0]),
                ("amd-smi static --clock --gpu GPU-123", [0]),
                ("amd-smi static --clock --gpu 1", [0]),
                ("amd-smi static --profile", [0]),
            ],
        )

    def test_other_set_targets_and_system_controls_are_not_expanded(self) -> None:
        options = [
            "--cpu-pwr-limit 100",
            "--core-boost-limit 2000",
            "--nic foo",
            "--gtt 4",
            "--node-balancing-mode POWER_BALANCING",
            "--node-power-limit 300",
        ]
        for selectors in (("--cpu all", "--cpu 0"), ("--core 0",), ("--nic 0",), ("pass",)):
            with self.subTest(selectors=selectors):
                self.sweep.FindArgs = mock.Mock(
                    side_effect=[options, selectors, ["pass"], ["pass"]]
                )
                commands = self.sweep.CreateCmds(
                    "set", "Set Arguments:", "Device Arguments:", "", ""
                )
                self.assertEqual(
                    commands,
                    [
                        (f"amd-smi set {option}{selector}", [0, NOT_SUPPORTED])
                        for option in options
                        for selector in ["", *(f" {item}" for item in selectors if item != "pass")]
                    ],
                )
