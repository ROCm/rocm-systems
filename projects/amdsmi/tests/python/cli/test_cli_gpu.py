#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""CLI GPU commands: static, firmware, bad_pages, metric, process, event, set, reset, monitor, static_mem_carveout_gtt."""

import ctypes
import json
import stat
import unittest

import common.common as common
import common.runcmd as runcmd
from cli.base import TestCliBase

# common.common owns path resolution, sys.path setup, and amdsmi loading — borrow the
# reference so AMDSMI_PATH/ROCM_HOME/ROCM_PATH resolution and the stale-package check
# (see ROCM-1552 / PR #6359) are not duplicated or bypassed here.
from common.common import amdsmi


class TestCliGpu(TestCliBase):
    TMP_FILENAME = "_tmp.log"
    TMP_FOLDER = "_tmp"

    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)
        cls.util = runcmd.Util("WARNING")

        # Record starting values; running here (once per class) rather than in
        # __init__ (once per test method) reduces setup overhead from O(N) to
        # O(1) — N being the number of test methods in this class.
        cmds = [
            ("metric", "amd-smi metric --json"),
            ("static", "amd-smi static --json"),
            ("list", "amd-smi list --json"),
            ("partition", "amd-smi partition --current --json"),
        ]
        for name, cmd in cmds:
            (rc, data, std_err) = cls.util.RunCmdSync(cmd)
            if rc:
                raise RuntimeError(f'Error executing "{cmd}": {std_err}')
            if not data:
                raise RuntimeError(f'Empty JSON output from "{cmd}". stderr: {std_err}')
            try:
                setattr(cls, f"{name}_data", json.loads(data))
            except (json.JSONDecodeError, TypeError) as e:
                # TODO(amdsmi_team): Known issue — several AI NIC and CPU commands can produce
                # malformed JSON/CSV/error output, causing parsing & other failures.
                # We need to log tickets on these issues.

                # Log warning but continue — malformed JSON output is a CLI bug,
                # not a test infrastructure failure; tests that depend on this
                # data will fail individually with a KeyError pointing to the
                # missing key, making the root cause clear.
                cls.common.print(f'\n\tERROR: Could not parse JSON from "{cmd}": {e}')
                setattr(cls, f"{name}_data", {})

        cls.gpus = ["all"]
        for entry in cls.list_data:
            cls.gpus.append(entry["gpu"])
            if entry["gpu"] == 0:
                # Only test bdf and uuid when gpu=0
                cls.gpus.append(entry["bdf"])
                cls.gpus.append(entry["uuid"])

        # When parsing, expand each arg with array element
        cls.sub_args = {
            "CLOCK": ["SYS", "DF", "DCEF", "SOC", "MEM", "VCLK0", "VCLK1", "DCLK0", "DCLK1", "ALL"],
            "PID": [123],
            "NAME": ["AMD"],
            "GPU": cls.gpus,
            "FILE": [
                cls.TMP_FILENAME,
                f"{cls.TMP_FILENAME} --overwrite",
                f"{cls.TMP_FILENAME} --append",
            ],
            "SEVERITY": ["nonfatal-uncorrected", "fatal", "nonfatal-corrected", "all"],
            "FOLDER": [cls.TMP_FOLDER],
            "FILE_LIMIT": [10],
            #'LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        }

    def test_bad_pages(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi bad-pages"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "bad-pages", "Bad Pages Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_event(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi event"
        self.common.print(msg)

        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs input"
                self.common.print(msg)
                self.skipTest(msg)

        # Start process with "amd-smi event"
        # In another process create an event with like "amd-smi reset --gpureset"
        cmds = self.CreateCmds(
            "event", "Event Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_firmware(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi firmware"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "firmware", "Firmware Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        cmds = self.CreateCmds(
            "ucode", "Firmware Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_metric(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi metric"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "metric",
            "Metric arguments:",
            "Device Arguments:",
            "Command Modifiers:",
            "Watch Arguments:",
        )
        self.RunCmds(cmds)
        return

    def test_monitor(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "monitor",
            "Monitor Arguments:",
            "Device Arguments:",
            "Command Modifiers:",
            "Watch Arguments:",
        )
        self.RunCmds(cmds)
        return

    def test_process(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi process"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "process",
            "Process arguments:",
            "Device Arguments:",
            "Command Modifiers:",
            "Watch Arguments:",
        )
        self.RunCmds(cmds)
        return

    def test_reset(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi reset"
        self.common.print(msg)

        # TODO allow reset commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs Testing, Not Yet Implemented"
                # self.common.print(msg)
                self.skipTest(msg)

        cmds = self.CreateCmds(
            "reset", "Reset Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_set(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set"
        self.common.print(msg)

        # TODO allow set commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs input"
                # self.common.print(msg)
                self.skipTest(msg)

        # Get current settings
        power_profile = {}
        for index, gpu in enumerate(self.common.processors):
            try:
                power_profile[index] = amdsmi.amdsmi_get_gpu_power_profile_presets(gpu, 0)
            except amdsmi.AmdSmiLibraryException as e:
                power_profile[index] = None

        cmds = self.CreateCmds(
            "set", "Set Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)

        # Restore starting values
        cmds = []
        for index, gpu in enumerate(self.common.processors):
            # Validate max fan speed is sensible; gpu_od GPUs must report <= 100
            fan_max = self.metric_data["gpu_data"][index]["fan"]["max"]
            if fan_max != "N/A":
                self.assertGreater(fan_max, 0, f"GPU {index}: max fan speed must be > 0")
                # Detect gpu_od interface via sysfs for this GPU
                gpu_bdf = self.list_data[index]["bdf"]
                has_gpu_od = common.has_gpu_od_interface(gpu_bdf)
                if has_gpu_od:
                    self.assertLessEqual(
                        fan_max, 100, f"GPU {index}: gpu_od max fan speed must be <= 100"
                    )
                else:
                    self.assertLessEqual(fan_max, 255, f"GPU {index}: max fan speed must be <= 255")

            # reset --fans (works for both legacy hwmon and gpu_od interfaces)
            fan_speed = self.metric_data["gpu_data"][index]["fan"]["speed"]
            if fan_speed != "N/A":
                cmds.append((f"amd-smi reset --fans --gpu {index}", self.PASS))

            # set --perf-level defaults
            perf_level = self.metric_data["gpu_data"][index]["perf_level"]
            if perf_level != "N/A":
                perf_level = perf_level.removeprefix("AMDSMI_DEV_PERF_LEVEL_")
                cmds.append((f"amd-smi set --perf-level {perf_level} --gpu {index}", self.PASS))

            # set --profile defaults
            if power_profile[index]:
                profile = power_profile[index]["current"].removeprefix("AMDSMI_PWR_PROF_PRST_")
                cmds.append((f"amd-smi set --profile {profile} --gpu {index}", self.PASS))

            # set --perf-determinism defaults
            clock_sys = self.static_data["gpu_data"][index]["clock"]["sys"]
            if clock_sys != "N/A":
                num = len(clock_sys["frequency_levels"])
                level = f"Level {num - 1}"
                clock_freq = int(clock_sys["frequency_levels"][level].split()[0].strip())
                cmds.append(
                    (f"amd-smi set --perf-determinism {clock_freq} --gpu {index}", self.PASS)
                )

            # set --compute-partition defaults
            accelerator_type = self.partition_data["current_partition"][index]["accelerator_type"]
            if accelerator_type != "N/A":
                cmds.append(
                    (f"amd-smi set --compute-partition {accelerator_type} --gpu {index}", self.PASS)
                )

            # set --memory-partition defaults
            memory_partition = self.partition_data["current_partition"][index]["memory"]
            if memory_partition != "N/A":
                cmds.append(
                    (f"amd-smi set --memory-partition {memory_partition} --gpu {index}", self.PASS)
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

            # set --power-cap defaults
            for power_type in self.power_types:
                socket_power_limit = self.static_data["gpu_data"][index]["limit"][power_type][
                    "socket_power_limit"
                ]
                if socket_power_limit != "N/A":
                    socket_power = socket_power_limit["value"]
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {socket_power} {power_type} --gpu {index}",
                            self.PASS,
                        )
                    )

            # set --soc-pstate defaults
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                current = int(soc_pstate["current"])
                cmds.append((f"amd-smi set --soc-pstate {current} --gpu {index}", self.PASS))

            # set --xgmi-plpd defaults
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                current = int(xgmi_plpd["current"])
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
                if clk_type == "SCLK":
                    clk_type_name = "socclk_0"
                else:
                    clk_type_name = "mem_0"
                for limit_type in self.limit_types:
                    if limit_type == "MIN":
                        clk_limit_name = "min_clk"
                    else:
                        clk_limit_name = "max_clk"
                    clk_type_limit_name = clock[clk_type_name][clk_limit_name]
                    if type(clk_type_limit_name) is dict:
                        value = clk_type_limit_name["value"]
                        cmds.append(
                            (
                                f"amd-smi set --clk-limit {clk_type} {limit_type} {value} --gpu {index}",
                                self.PASS,
                            )
                        )

            # set --clk-level defaults
            clock = self.static_data["gpu_data"][index]["clock"]
            for clk_type in self.clk_levels:
                value = -1
                if clk_type == "SCLK":
                    clk_type_name = "sys"
                elif clk_type == "MCLK":
                    clk_type_name = "mem"
                elif clk_type == "FCLK":
                    clk_type_name = "df"
                elif clk_type == "SOCCLK":
                    clk_type_name = "soc"
                else:
                    bus = self.static_data["gpu_data"][index]["bus"]
                    pcie_levels = bus["pcie_levels"]
                    if type(pcie_levels) is dict:
                        value = len(pcie_levels)
                        if value > 0:
                            value -= 1
                if clk_type != "PCIE" and value < 0:
                    clk_type_name = clock[clk_type_name]
                    if type(clk_type_name) is dict:
                        current_level = clk_type_name["current_level"]
                        value = current_level
                if value >= 0:
                    cmds.append(
                        (f"amd-smi set --clk-level {clk_type} {value} --gpu {index}", self.PASS)
                    )
            # set --process-isolation defaults
            process_isolation = self.static_data["gpu_data"][index]["process_isolation"]
            if process_isolation == "Disabled":
                process_isolation_value = 0
            else:
                process_isolation_value = 1
            cmds.append(
                (
                    f"amd-smi set --process-isolation {process_isolation_value} --gpu {index}",
                    self.PASS,
                )
            )

        print("Restore Starting Values")
        self.RunCmds(cmds)

        return

    def test_static(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "static", "Static Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_static_mem_carveout_gtt(self):
        """Test static --mem-carveout and node --gtt flags (display mode only)"""
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static --mem-carveout and node --gtt"
        self.common.print(msg)

        # Test mem-carveout display (static subcommand)
        cmd = "amd-smi static --mem-carveout"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test GTT display (node subcommand — GTT is system-wide, not per-GPU)
        cmd = "amd-smi node --gtt"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test mem-carveout with JSON output
        cmd = "amd-smi static --mem-carveout --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test GTT with JSON output (node subcommand)
        cmd = "amd-smi node --gtt --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test mem-carveout with CSV output
        cmd = "amd-smi static --mem-carveout --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test GTT with CSV output (node subcommand)
        cmd = "amd-smi node --gtt --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Note: We do NOT test set/reset operations (--mem-carveout in set, --gtt in set/reset) because:
        # 1. They require root/sudo permissions
        # 2. They require system reboot to take effect
        # 3. They could interfere with the test system configuration
        # These operations should be tested manually or in dedicated integration test environments

        msg = f"{self.tab}Static mem-carveout and node GTT tests passed (display mode only)"
        self.common.print(msg)
        return
