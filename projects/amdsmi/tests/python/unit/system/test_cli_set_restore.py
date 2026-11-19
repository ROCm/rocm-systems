#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Driver-free unit tests for the ``amd-smi set`` CLI test's restore commands.

The restore must invoke every reset on every GPU and assert the documented result,
including NOT_SUPPORTED, instead of skipping GPUs that report no fan or no power cap.
"""

import sys
import unittest
from unittest import mock

from common.common import amdsmi

# cli.base appends the installed CLI dir to sys.path and imports amdsmi_cli_exceptions
# from it to load its exit codes. Restore both sys.path and the sys.modules entry for
# amdsmi_cli_exceptions once it has, so later test modules import CLI modules (e.g. the
# generated _version, or the exceptions from the checkout) the same way they would
# without this module in the run. cli.base keeps its own reference to the exit codes.
_sys_path = list(sys.path)
_CLI_EXCEPTIONS = "amdsmi_cli_exceptions"
_had_cli_exceptions = _CLI_EXCEPTIONS in sys.modules
_cli_exceptions = sys.modules.get(_CLI_EXCEPTIONS)
try:
    # Import the module, not the class: a TestCase class in this module's namespace
    # would be collected by discovery and run the real set sweep.
    from cli import test_set as cli_test_set
finally:
    sys.path[:] = _sys_path
    if _had_cli_exceptions:
        sys.modules[_CLI_EXCEPTIONS] = _cli_exceptions
    else:
        sys.modules.pop(_CLI_EXCEPTIONS, None)

PASS = amdsmi.AmdSmiStatus.SUCCESS
NOT_SUPPORTED = amdsmi.AmdSmiStatus.NOT_SUPPORTED
_ABSENT = object()

READABLE_FAN = {"speed": 120, "max": 255, "rpm": 1500, "usage": "N/A"}
# Fan present but its speed unread, as on a runtime-suspended dGPU: pwm1_max is a
# constant, pwm1 needs the device active.
SUSPENDED_FAN = {"speed": "N/A", "max": 255, "rpm": "N/A", "usage": "N/A"}
UNREADABLE_FAN = {"speed": "N/A", "max": "N/A", "rpm": "N/A", "usage": "N/A"}
# What `amd-smi static` reports for a power-cap sensor the GPU does not support.
NO_CAP = {"max_power_limit": "N/A", "min_power_limit": "N/A", "socket_power_limit": "N/A"}


class _FakeCommon:
    def __init__(self, count):
        self.processors = list(range(count))
        self.amdsmi_smart_init = mock.Mock()

    def print(self, *_args, **_kwargs):
        pass


def _cap(max_w, current_w):
    return {
        "min_power_limit": {"value": 0, "unit": "W"},
        "max_power_limit": {"value": max_w, "unit": "W"},
        "socket_power_limit": {"value": current_w, "unit": "W"},
    }


def _gpu(is_apu, fan=_ABSENT, ppt0=NO_CAP, ppt1=NO_CAP):
    metric = {"clock": {}, "perf_level": "N/A"}
    if fan is not _ABSENT:
        metric["fan"] = fan
    static = {
        "clock": {"sys": "N/A"},
        "limit": {"ppt0": ppt0, "ppt1": ppt1, "ptl_state": "N/A", "ptl_format": "N/A"},
        "soc_pstate": "N/A",
        "xgmi_plpd": "N/A",
        "process_isolation": "N/A",
    }
    return {"is_apu": is_apu, "metric": metric, "static": static}


def _make_test(gpus):
    test = cli_test_set.TestSet("test_command")
    test.common = _FakeCommon(len(gpus))
    test.is_apu = [gpu["is_apu"] for gpu in gpus]
    test.metric_data = {"gpu_data": [gpu["metric"] for gpu in gpus]}
    test.static_data = {"gpu_data": [gpu["static"] for gpu in gpus]}
    test.partition_data = {
        "current_partition": [{"memory": "N/A", "accelerator_type": "N/A"} for _ in gpus]
    }
    return test


def _cond(cmds, cmd):
    matches = [cond for command, cond in cmds if command == cmd]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {cmd!r}, got {len(matches)}")
    return matches[0]


class TestSetRestoreCommands(unittest.TestCase):
    def test_apu_without_fan_section_asserts_not_supported_resets(self):
        cmds, problems = _make_test([_gpu(True)])._restore_commands()
        self.assertEqual(
            cmds,
            [
                ("amd-smi reset --fans --gpu 0", [NOT_SUPPORTED]),
                ("amd-smi reset --power-cap --gpu 0", [NOT_SUPPORTED]),
                ("amd-smi reset --clocks --gpu 0", [PASS, NOT_SUPPORTED]),
            ],
        )
        self.assertEqual(problems, [])

    def test_dgpu_readable_fan_with_cap_keeps_baseline_expectations(self):
        cmds, problems = _make_test(
            [_gpu(False, fan=READABLE_FAN, ppt0=_cap(290, 290))]
        )._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [PASS, NOT_SUPPORTED])
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 0"), PASS)
        self.assertEqual(problems, [])

    def test_dgpu_unreadable_fan_asserts_not_supported_fan_reset(self):
        cmds, problems = _make_test(
            [_gpu(False, fan=UNREADABLE_FAN, ppt0=_cap(300, 300))]
        )._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [NOT_SUPPORTED])
        self.assertEqual(problems, [])

    def test_dgpu_fan_with_max_but_no_speed_allows_pass(self):
        # A suspended dGPU cannot report fan speed, yet still has a controllable fan.
        cmds, problems = _make_test(
            [_gpu(False, fan=SUSPENDED_FAN, ppt0=_cap(300, 300))]
        )._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [PASS, NOT_SUPPORTED])
        self.assertEqual(problems, [])

    def test_zero_current_cap_still_expects_pass(self):
        cmds, _ = _make_test([_gpu(False, fan=READABLE_FAN, ppt0=_cap(290, 0))])._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 0"), PASS)

    def test_cap_on_second_sensor_only_expects_pass(self):
        cmds, _ = _make_test(
            [_gpu(False, fan=READABLE_FAN, ppt1=_cap(340, 340))]
        )._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 0"), PASS)

    def test_dgpu_without_power_cap_asserts_not_supported(self):
        cmds, _ = _make_test([_gpu(False, fan=READABLE_FAN)])._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 0"), [NOT_SUPPORTED])

    def test_bare_na_limits_assert_not_supported_power_cap_reset(self):
        cmds, _ = _make_test(
            [_gpu(False, fan=READABLE_FAN, ppt0="N/A", ppt1="N/A")]
        )._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 0"), [NOT_SUPPORTED])

    def test_dgpu_missing_fan_section_is_a_problem_and_reset_still_queued(self):
        cmds, problems = _make_test([_gpu(False, ppt0=_cap(290, 290))])._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [PASS, NOT_SUPPORTED])
        self.assertEqual(len(problems), 1)
        self.assertIn("GPU 0", problems[0])

    def test_apu_with_fan_section_is_a_problem(self):
        cmds, problems = _make_test([_gpu(True, fan={"apu_fan_pwm": "N/A"})])._restore_commands()
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [NOT_SUPPORTED])
        self.assertEqual(len(problems), 1)
        self.assertIn("GPU 0", problems[0])

    def test_fan_max_out_of_range_is_a_problem_not_an_exception(self):
        for fan_max in (0, 300, "bogus"):
            with self.subTest(fan_max=fan_max):
                fan = dict(READABLE_FAN, max=fan_max)
                cmds, problems = _make_test(
                    [_gpu(False, fan=fan, ppt0=_cap(290, 290))]
                )._restore_commands()
                self.assertEqual(len(problems), 1)
                self.assertIn("GPU 0", problems[0])
                self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [PASS, NOT_SUPPORTED])

    def test_mixed_host_queues_resets_for_every_gpu(self):
        cmds, problems = _make_test(
            [
                _gpu(False, fan=UNREADABLE_FAN, ppt0=_cap(290, 290)),
                _gpu(False, fan=READABLE_FAN, ppt0=_cap(272, 272)),
                _gpu(True),
            ]
        )._restore_commands()
        self.assertEqual(problems, [])
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 0"), [NOT_SUPPORTED])
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 1"), [PASS, NOT_SUPPORTED])
        self.assertEqual(_cond(cmds, "amd-smi reset --fans --gpu 2"), [NOT_SUPPORTED])
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 0"), PASS)
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 1"), PASS)
        self.assertEqual(_cond(cmds, "amd-smi reset --power-cap --gpu 2"), [NOT_SUPPORTED])

    def test_fully_populated_dgpu_restores_everything_in_order(self):
        gpu = _gpu(False, fan=READABLE_FAN, ppt0=_cap(750, 700))
        gpu["metric"]["clock"] = {
            "gfx_0": {
                "min_clk": {"value": 500, "unit": "MHz"},
                "max_clk": {"value": 2100, "unit": "MHz"},
            },
            "mem_0": {
                "min_clk": {"value": 900, "unit": "MHz"},
                "max_clk": {"value": 1300, "unit": "MHz"},
            },
            "fclk_0": {
                "min_clk": {"value": 1200, "unit": "MHz"},
                "max_clk": {"value": 2000, "unit": "MHz"},
            },
        }
        gpu["metric"]["perf_level"] = "AMDSMI_DEV_PERF_LEVEL_AUTO"
        gpu["static"].update(
            {
                "profile": {
                    "available_profiles": ["BOOTUP_DEFAULT", "COMPUTE"],
                    "current": "COMPUTE",
                    "num_profiles": 2,
                },
                "clock": {
                    "sys": {
                        "frequency_levels": {
                            "Level 0": {"value": 500, "unit": "MHz"},
                            "Level 1": {"value": 1400, "unit": "MHz"},
                            "Level 2": {"value": 2100, "unit": "MHz"},
                        }
                    }
                },
                "partition": {"compute_partition_mem_alloc_mode": "NPS_FIRST"},
                "soc_pstate": {"num_supported": 3, "current_id": 2},
                "xgmi_plpd": {"num_supported": 2, "current_id": 1},
                "process_isolation": "Enabled",
            }
        )
        gpu["static"]["limit"].update({"ptl_state": "Enabled", "ptl_format": "I8,F16"})
        test = _make_test([gpu])
        test.partition_data["current_partition"][0] = {"memory": "NPS1", "accelerator_type": "SPX"}

        cmds, problems = test._restore_commands()

        either = [PASS, NOT_SUPPORTED]
        self.assertEqual(
            cmds,
            [
                ("amd-smi reset --fans --gpu 0", either),
                ("amd-smi set --profile COMPUTE --gpu 0", PASS),
                ("amd-smi set --perf-determinism 2100 --gpu 0", either),
                ("amd-smi set --memory-partition NPS1 --gpu 0", PASS),
                ("amd-smi set --compute-partition SPX --gpu 0", PASS),
                ("amd-smi set --compute-partition-mem-alloc-mode NPS_FIRST --gpu 0", PASS),
                ("amd-smi reset --power-cap --gpu 0", PASS),
                ("amd-smi set --soc-pstate 2 --gpu 0", PASS),
                ("amd-smi set --xgmi-plpd 1 --gpu 0", PASS),
                ("amd-smi set --ptl-status 1 --gpu 0", PASS),
                ("amd-smi set --ptl-format I8,F16 --gpu 0", PASS),
                ("amd-smi set --clk-limit SCLK MAX 2100 --gpu 0", either),
                ("amd-smi set --clk-limit SCLK MIN 500 --gpu 0", either),
                ("amd-smi set --clk-limit MCLK MAX 1300 --gpu 0", either),
                ("amd-smi set --clk-limit MCLK MIN 900 --gpu 0", either),
                ("amd-smi set --clk-limit FCLK MAX 2000 --gpu 0", either),
                ("amd-smi set --clk-limit FCLK MIN 1200 --gpu 0", either),
                ("amd-smi reset --clocks --gpu 0", either),
                ("amd-smi set --perf-level AUTO --gpu 0", PASS),
                ("amd-smi set --process-isolation 1 --gpu 0", either),
            ],
        )
        self.assertEqual(problems, [])


class TestSetRestoreRun(unittest.TestCase):
    def test_commands_run_before_problems_are_reported(self):
        test = _make_test([_gpu(False, ppt0=_cap(290, 290))])
        with mock.patch.object(test, "RunCmds") as run_cmds:
            with self.assertRaises(AssertionError) as raised:
                test._restore_starting_values()
        run_cmds.assert_called_once()
        ran = [cmd for cmd, _ in run_cmds.call_args[0][0]]
        self.assertIn("amd-smi reset --fans --gpu 0", ran)
        self.assertIn("GPU 0", str(raised.exception))

    def test_command_failures_and_problems_are_reported_together(self):
        test = _make_test([_gpu(False, ppt0=_cap(290, 290))])
        with mock.patch.object(test, "RunCmds", side_effect=AssertionError("reset failed")):
            with self.assertRaises(AssertionError) as raised:
                test._restore_starting_values()
        self.assertIn("reset failed", str(raised.exception))
        self.assertIn("GPU 0", str(raised.exception))

    def test_clean_restore_does_not_fail(self):
        test = _make_test([_gpu(True)])
        with mock.patch.object(test, "RunCmds") as run_cmds:
            test._restore_starting_values()
        run_cmds.assert_called_once()


class TestReadApuFlags(unittest.TestCase):
    def test_flags_follow_gpu_metrics_is_apu(self):
        metrics = [{"is_apu": True}, {"is_apu": False}, amdsmi.AmdSmiLibraryException(2)]
        calls = mock.Mock()
        fake_common = _FakeCommon(3)
        calls.attach_mock(fake_common.amdsmi_smart_init, "init")
        with mock.patch.object(cli_test_set.TestSet, "common", fake_common, create=True):
            with mock.patch.object(
                amdsmi, "amdsmi_get_processor_handles", return_value=[0, 1, 2]
            ) as handles:
                calls.attach_mock(handles, "handles")
                with mock.patch.object(amdsmi, "amdsmi_get_gpu_metrics_info", side_effect=metrics):
                    with mock.patch.object(amdsmi, "amdsmi_shut_down") as shut_down:
                        flags = cli_test_set.TestSet._read_apu_flags()
        self.assertEqual(flags, [True, False, False])
        # The library is shut down after Common's setup, so it must be reopened first.
        self.assertEqual(calls.mock_calls, [mock.call.init(), mock.call.handles()])
        shut_down.assert_called_once()


class TestBuildBaseline(unittest.TestCase):
    def test_baseline_records_apu_flags(self):
        outputs = {
            "amd-smi metric --json": "{}",
            "amd-smi static --json": "{}",
            "amd-smi list --json": '[{"gpu": 0, "bdf": "0000:01:00.0", "uuid": "u0"}]',
            "amd-smi partition --current --json": "{}",
        }
        util = mock.Mock()
        util.RunCmdSync.side_effect = lambda cmd: (0, outputs[cmd], "")
        flags = [True]
        test_cls = cli_test_set.TestSet
        with mock.patch.object(test_cls, "util", util, create=True), mock.patch.object(
            test_cls, "_nic_sweep_values", return_value=[]
        ), mock.patch.object(test_cls, "_read_clk_freq", return_value=[{}]), mock.patch.object(
            test_cls, "_read_apu_flags", return_value=flags
        ):
            baseline = test_cls._build_baseline()
        self.assertIs(baseline["is_apu"], flags)
