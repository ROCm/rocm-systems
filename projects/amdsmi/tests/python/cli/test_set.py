#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: set command."""

import json

from common.common import amdsmi

from cli.base import AmdSmiExitCode, TestCliBase


def _strip_prefix(value, prefix):
    # Backport of str.removeprefix() (added in Python 3.9) because the test suite
    # still supports Python 3.8. Strips a leading enum-name prefix (e.g.
    # "AMDSMI_DEV_PERF_LEVEL_") when present, otherwise returns value unchanged.
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


class TestSet(TestCliBase):
    def _requested_gpu_data(self, command: str) -> dict:
        rc, output, stderr = self.util.RunCmdSync(command)
        self.assertEqual(rc, self.PASS, f"{command}: {stderr}")
        entries = json.loads(output)["gpu_data"]
        self.assertEqual(len(entries), 1, command)
        return entries[0]

    def _requested_write(self, command: str) -> bool:
        rc, _, stderr = self.util.RunCmdSync(command, msg_in=self._prompt_answer_for(command))
        if rc == amdsmi.AmdSmiStatus.NOT_SUPPORTED:
            return False
        self.assertEqual(rc, self.PASS, f"{command}: {stderr}")
        return True

    def test_fan_set_reset(self) -> None:
        tested = 0
        for entry in self.list_data:
            gpu = entry["gpu"]
            cmd = f"amd-smi metric --fan --json -g {gpu}"
            if self._requested_gpu_data(cmd)["fan"]["max"] == "N/A":
                continue
            with self.subTest(gpu=gpu):
                reset = f"amd-smi reset --fans -g {gpu}"
                self.addCleanup(
                    self.RunCmds, [(reset, [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED])]
                )
                if not self._requested_write(f"amd-smi set --fan 50% -g {gpu}"):
                    continue
                tested += 1
                self.assertAlmostEqual(
                    self._requested_gpu_data(cmd)["fan"]["usage"]["value"], 50, delta=1
                )
                self.RunCmds([(reset, self.PASS)])
        if not tested:
            self.skipTest("no GPU supports fan control")

    def test_power_cap_after_static_limit(self) -> None:
        tested = 0
        for entry in self.list_data:
            gpu = entry["gpu"]
            cmd = f"amd-smi static --limit --json -g {gpu}"
            limits = self._requested_gpu_data(cmd)["limit"]
            for sensor in self.power_types:
                limit = limits[sensor]
                if not isinstance(limit, dict) or not all(
                    isinstance(limit[key], dict)
                    for key in ("min_power_limit", "max_power_limit", "socket_power_limit")
                ):
                    continue
                original = limit["socket_power_limit"]["value"]
                low = max(1, int(limit["min_power_limit"]["value"]))
                high = int(limit["max_power_limit"]["value"])
                if not original or low >= high:
                    continue
                value = low if original != low else high
                with self.subTest(gpu=gpu, sensor=sensor):
                    restore = f"amd-smi set --power-cap {original} {sensor} -g {gpu}"
                    self.addCleanup(self.RunCmds, [(restore, self.PASS)])
                    if not self._requested_write(
                        f"amd-smi set --power-cap {value} {sensor} -g {gpu}"
                    ):
                        continue
                    tested += 1
                    self.assertEqual(
                        self._requested_gpu_data(cmd)["limit"][sensor]["socket_power_limit"][
                            "value"
                        ],
                        value,
                    )
                    self.RunCmds([(restore, self.PASS)])
        if not tested:
            self.skipTest("no GPU supports changing a power cap")

    def test_power_cap_above_max(self) -> None:
        tested = 0
        for entry in self.list_data:
            gpu = entry["gpu"]
            cmd = f"amd-smi static --limit --json -g {gpu}"
            for sensor, limit in self._requested_gpu_data(cmd)["limit"].items():
                if sensor not in self.power_types or not isinstance(limit, dict):
                    continue
                if not isinstance(limit["max_power_limit"], dict) or not isinstance(
                    limit["socket_power_limit"], dict
                ):
                    continue
                original = limit["socket_power_limit"]["value"]
                if not original:
                    continue
                with self.subTest(gpu=gpu, sensor=sensor):
                    restore = f"amd-smi set --power-cap {original} {sensor} -g {gpu}"
                    self.addCleanup(self.RunCmds, [(restore, self.PASS)])
                    value = int(limit["max_power_limit"]["value"]) + 1
                    self.RunCmds(
                        [
                            (
                                f"amd-smi set --power-cap {value} {sensor} -g {gpu}",
                                AmdSmiExitCode.INVALID_PARAMETER_VALUE,
                            )
                        ]
                    )
                    self.assertEqual(
                        self._requested_gpu_data(cmd)["limit"][sensor]["socket_power_limit"][
                            "value"
                        ],
                        original,
                    )
                    tested += 1
        if not tested:
            self.skipTest("no GPU reports power cap limits")

    def test_xgmi_plpd_static(self) -> None:
        tested = 0
        for entry in self.list_data:
            gpu = entry["gpu"]
            cmd = f"amd-smi static --json -g {gpu}"
            policy = self._requested_gpu_data(cmd)["xgmi_plpd"]
            if not isinstance(policy, dict):
                continue
            original = int(policy["current_id"])
            for option in policy["policies"]:
                value = int(option["policy_id"])
                if value == original:
                    continue
                with self.subTest(gpu=gpu, policy=value):
                    restore = f"amd-smi set --xgmi-plpd {original} -g {gpu}"
                    self.addCleanup(self.RunCmds, [(restore, self.PASS)])
                    if not self._requested_write(f"amd-smi set --xgmi-plpd {value} -g {gpu}"):
                        continue
                    tested += 1
                    self.assertEqual(
                        int(self._requested_gpu_data(cmd)["xgmi_plpd"]["current_id"]), value
                    )
                    self.RunCmds([(restore, self.PASS)])
        if not tested:
            self.skipTest("no GPU supports changing XGMI PLPD policy")

    def test_ptl_static_limit_json(self) -> None:
        tested = 0
        for entry in self.list_data:
            gpu = entry["gpu"]
            cmd = f"amd-smi static -l --json -g {gpu}"
            limits = self._requested_gpu_data(cmd)["limit"]
            settings = []
            if limits["ptl_state"] in ("Enabled", "Disabled"):
                original = int(limits["ptl_state"] == "Enabled")
                settings.append(
                    (
                        "--ptl-status",
                        "ptl_state",
                        original,
                        1 - original,
                        "Disabled" if original else "Enabled",
                    )
                )
            if limits["ptl_format"] != "N/A":
                original = limits["ptl_format"]
                value = "I8,F32" if original != "I8,F32" else "F16,F32"
                settings.append(("--ptl-format", "ptl_format", original, value, value))
            for option, key, original, value, expected in settings:
                with self.subTest(gpu=gpu, option=option):
                    restore = f"amd-smi set {option} {original} -g {gpu}"
                    self.addCleanup(self.RunCmds, [(restore, self.PASS)])
                    if not self._requested_write(f"amd-smi set {option} {value} -g {gpu}"):
                        continue
                    tested += 1
                    self.assertEqual(self._requested_gpu_data(cmd)["limit"][key], expected)
                    self.RunCmds([(restore, self.PASS)])
        if not tested:
            self.skipTest("no GPU supports PTL settings")

    def test_process_isolation_clean_local_data(self) -> None:
        tested = 0
        for entry in self.list_data:
            gpu = entry["gpu"]
            cmd = f"amd-smi static --json -g {gpu}"
            state = self._requested_gpu_data(cmd)["process_isolation"]
            if state not in ("Enabled", "Disabled"):
                continue
            with self.subTest(gpu=gpu):
                restore = f"amd-smi set --process-isolation {int(state == 'Enabled')} -g {gpu}"
                self.addCleanup(self.RunCmds, [(restore, self.PASS)])
                if not self._requested_write(f"amd-smi set --process-isolation 1 -g {gpu}"):
                    continue
                tested += 1
                self.assertEqual(self._requested_gpu_data(cmd)["process_isolation"], "Enabled")
                self.RunCmds(
                    [(f"amd-smi reset -l -g {gpu}", [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED])]
                )
                self.RunCmds([(restore, self.PASS)])
        if not tested:
            self.skipTest("no GPU supports process isolation")

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "set", "Set Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        # Registered before the sweep: RunCmds raises on the first failure, and a
        # sweep that aborts partway is exactly when the GPU is left mid-change.
        self.addCleanup(self._restore_starting_values)
        self.RunCmds(cmds)

        return

    def _restore_starting_values(self):
        """Put the values the sweep changed back to what setUpClass recorded."""
        cmds = []
        for index, gpu in enumerate(self.common.processors):
            # max fan speed is the hwmon pwm1_max ceiling on every interface
            fan_max = self.metric_data["gpu_data"][index]["fan"]["max"]
            if fan_max != "N/A":
                self.assertGreater(fan_max, 0, f"GPU {index}: max fan speed must be > 0")
                self.assertLessEqual(fan_max, 255, f"GPU {index}: max fan speed must be <= 255")

            # reset --fans (works for both legacy hwmon and gpu_od interfaces)
            # A readable fan speed does not imply the fan can be controlled.
            fan_speed = self.metric_data["gpu_data"][index]["fan"]["speed"]
            if fan_speed != "N/A":
                cmds.append(
                    (
                        f"amd-smi reset --fans --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

            # set --profile defaults
            # Read from the static baseline: the library is shut down once setup
            # ends, so querying it here fails and the restore was skipped.
            profile = self.static_data["gpu_data"][index].get("profile")
            if isinstance(profile, dict) and profile["current"] in profile["available_profiles"]:
                cmds.append(
                    (f"amd-smi set --profile {profile['current']} --gpu {index}", self.PASS)
                )

            # set --perf-determinism defaults
            clock_sys = self.static_data["gpu_data"][index]["clock"]["sys"]
            if clock_sys != "N/A":
                num = len(clock_sys["frequency_levels"])
                level = f"Level {num - 1}"
                clock_freq = int(clock_sys["frequency_levels"][level]["value"])
                # Readable clock levels do not imply the determinism feature exists.
                cmds.append(
                    (
                        f"amd-smi set --perf-determinism {clock_freq} --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

            # set --memory-partition defaults
            # Safe to write back: the mode only takes effect after a driver reload.
            memory_partition = self.partition_data["current_partition"][index]["memory"]
            if memory_partition != "N/A":
                cmds.append(
                    (f"amd-smi set --memory-partition {memory_partition} --gpu {index}", self.PASS)
                )

            # set --compute-partition defaults
            # Must come after memory partition sets, since the memory partition set can override
            # this setting (their defaults may differ).
            accelerator_type = self.partition_data["current_partition"][index]["accelerator_type"]
            if accelerator_type != "N/A":
                cmds.append(
                    (f"amd-smi set --compute-partition {accelerator_type} --gpu {index}", self.PASS)
                )

            # set --compute-partition-mem-alloc-mode defaults
            try:
                mem_alloc_mode = self.static_data["gpu_data"][index]["partition"][
                    "compute_partition_mem_alloc_mode"
                ]
            except (KeyError, TypeError):
                mem_alloc_mode = "N/A"
            if mem_alloc_mode not in ("N/A", "INVALID"):
                cmds.append(
                    (
                        f"amd-smi set --compute-partition-mem-alloc-mode {mem_alloc_mode} --gpu {index}",
                        self.PASS,
                    )
                )

            # reset --power-cap
            # Writes each supported sensor's default_power_cap. Replaying the recorded
            # value did not restore it: min and max were written after it, so the cap
            # was left at max.
            cmds.append((f"amd-smi reset --power-cap --gpu {index}", self.PASS))

            # set --soc-pstate defaults
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                current = int(soc_pstate["current_id"])
                cmds.append((f"amd-smi set --soc-pstate {current} --gpu {index}", self.PASS))

            # set --xgmi-plpd defaults
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                current = int(xgmi_plpd["current_id"])
                cmds.append((f"amd-smi set --xgmi-plpd {current} --gpu {index}", self.PASS))

            # set --ptl-status defaults
            ptl_state = self.static_data["gpu_data"][index]["limit"]["ptl_state"]
            if ptl_state != "N/A":
                if ptl_state == "Disabled":
                    ptl_state_value = 0
                else:
                    ptl_state_value = 1
                cmds.append(
                    (f"amd-smi set --ptl-status {ptl_state_value} --gpu {index}", self.PASS)
                )

            # set --ptl-format defaults
            ptl_format = self.static_data["gpu_data"][index]["limit"]["ptl_format"]
            if ptl_format != "N/A":
                # TODO: get the right ptl-format
                cmds.append((f"amd-smi set --ptl-format {ptl_format} --gpu {index}", self.PASS))

            # set --clk-limit defaults
            clock = self.metric_data["gpu_data"][index]["clock"]
            for clk_type in self.clk_limits:
                for limit_type in self.limit_types:
                    value = self._clk_limit_value(clock, clk_type, limit_type)
                    if value is not None:
                        # A readable min/max clock does not imply the device
                        # supports setting a limit on it.
                        cmds.append(
                            (
                                f"amd-smi set --clk-limit {clk_type} {limit_type} {value} --gpu {index}",
                                [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                            )
                        )

            # reset --clocks
            # A clk-level write is a bitmask, so replaying the recorded level would pin DPM
            # to that one level; only AUTO hands every level back. Overdrive is reset too,
            # and reports NOT_SUPPORTED on parts that lack it even though the rest worked.
            cmds.append(
                (
                    f"amd-smi reset --clocks --gpu {index}",
                    [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                )
            )

            # set --perf-level defaults
            # Last of the clock writes: the reset above forces AUTO, and --clk-limit and
            # --perf-determinism each leave the device on a level of their own.
            perf_level = self.metric_data["gpu_data"][index]["perf_level"]
            if perf_level not in ("N/A", "AMDSMI_DEV_PERF_LEVEL_UNKNOWN"):
                perf_level = _strip_prefix(perf_level, "AMDSMI_DEV_PERF_LEVEL_")
                cmds.append((f"amd-smi set --perf-level {perf_level} --gpu {index}", self.PASS))
            # set --process-isolation defaults
            process_isolation = self.static_data["gpu_data"][index]["process_isolation"]
            # Put back whatever value the sweep left; "N/A" means unreadable, so
            # there is nothing to restore.
            if process_isolation != "N/A":
                original = 0 if process_isolation == "Disabled" else 1
                cmds.append(
                    (
                        f"amd-smi set --process-isolation {original} --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

        self.common.print("Restore Starting Values")
        self.RunCmds(cmds)

        return

    def _gfx_clock_range(self, gpu):
        """(min, max) SCLK limits `amd-smi metric -c` reports for *gpu*."""
        data = self._requested_gpu_data(f"amd-smi metric -c --json --gpu {gpu}")
        clocks = data["clock"]
        return (
            self._clk_limit_value(clocks, "SCLK", "MIN"),
            self._clk_limit_value(clocks, "SCLK", "MAX"),
        ), clocks

    def _restore_gfx_clock(self, gpu, low, high):
        self.util.RunCmdSync(f"amd-smi reset --clocks --gpu {gpu}")
        if self._gfx_clock_range(gpu)[0] != (low, high):
            self.util.RunCmdSync(f"amd-smi set -L sclk max {high} --gpu {gpu}")
            self.util.RunCmdSync(f"amd-smi set -L sclk min {low} --gpu {gpu}")

    def test_gfx_clock(self):
        """Set SCLK min/max limits, verify them via `metric -c`, then reset clocks."""
        self.common.print_func_name("")
        tested = 0
        for index, entry in enumerate(self.list_data):
            gpu = entry["gpu"]
            target = str(self.static_data["gpu_data"][index]["asic"]["target_graphics_version"])
            # Navi (RDNA) parts do not support writing gfx clock limits.
            if target.startswith(("gfx10", "gfx11", "gfx12")):
                continue
            (low, high), clocks = self._gfx_clock_range(gpu)
            new_min = self._clk_limit_probe_value(clocks, "SCLK", "MIN")
            new_max = self._clk_limit_probe_value(clocks, "SCLK", "MAX")
            if None in (low, high, new_min, new_max):
                continue

            with self.subTest(gpu=gpu):
                self.addCleanup(self._restore_gfx_clock, gpu, low, high)
                # Narrow the ceiling first so MIN never has to exceed the current MAX.
                supported = True
                for bound, value in (("max", new_max), ("min", new_min)):
                    cmd = f"amd-smi set -L sclk {bound} {value} --gpu {gpu}"
                    (rc, _, std_err) = self.util.RunCmdSync(cmd)
                    if rc == amdsmi.AmdSmiStatus.NOT_SUPPORTED:
                        supported = False
                        break
                    self.assertEqual(rc, self.PASS, f"'{cmd}' failed with rc={rc}: {std_err}")
                if not supported:
                    continue
                tested += 1

                # sclk limits are applied exactly, unlike mclk/fclk which snap to DPM levels.
                self.assertEqual(self._gfx_clock_range(gpu)[0], (new_min, new_max))

                cmd = f"amd-smi reset --clocks --gpu {gpu}"
                (rc, _, std_err) = self.util.RunCmdSync(cmd)
                # Overdrive reset reports NOT_SUPPORTED on parts without it; clocks still reset.
                self.assertIn(rc, (self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED), std_err)
                reset_min, reset_max = self._gfx_clock_range(gpu)[0]
                self.assertLessEqual(reset_min, low, f"GPU {gpu}: reset did not restore min")
                self.assertGreaterEqual(reset_max, high, f"GPU {gpu}: reset did not restore max")

        if not tested:
            self.skipTest("no GPU supports setting gfx clock limits")
