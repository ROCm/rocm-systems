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
"""CLI leaf test: set command."""

import common.common as common
from common.common import amdsmi

from cli.base import TestCliBase


def _strip_prefix(value, prefix):
    # Backport of str.removeprefix() (added in Python 3.9) because the test suite
    # still supports Python 3.8. Strips a leading enum-name prefix (e.g.
    # "AMDSMI_DEV_PERF_LEVEL_") when present, otherwise returns value unchanged.
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


class TestSet(TestCliBase):
    def test_command(self):
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
            except amdsmi.AmdSmiLibraryException:
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
                perf_level = _strip_prefix(perf_level, "AMDSMI_DEV_PERF_LEVEL_")
                cmds.append((f"amd-smi set --perf-level {perf_level} --gpu {index}", self.PASS))

            # set --profile defaults
            if power_profile[index]:
                profile = _strip_prefix(power_profile[index]["current"], "AMDSMI_PWR_PROF_PRST_")
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
                _power_type = self.static_data["gpu_data"][index]["limit"][power_type]
                socket_power_limit = _power_type["socket_power_limit"]
                if socket_power_limit != "N/A":
                    socket_power = socket_power_limit["value"]
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {socket_power} {power_type} --gpu {index}",
                            self.PASS,
                        )
                    )
                    # Both bounds are inclusive: the exact min and max must
                    # succeed. A reported min of 0 means the technical minimum
                    # is 1, since setting 0 reads back the current cap.
                    min_power = max(_power_type["min_power_limit"]["value"], 1)
                    max_power = _power_type["max_power_limit"]["value"]
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {min_power} {power_type} --gpu {index}",
                            self.PASS,
                        )
                    )
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {max_power} {power_type} --gpu {index}",
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
                clk_type_name = ""
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
