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

import argparse
import json
import locale
import multiprocessing
import os
import select
import shutil
import stat
import sys
import time

import unittest

import common
import runcmd

amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.'
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as e:
    raise ImportError(f'Could not import the "amdsmi" module from "{amdsmi_path}"') from e

# Module-level constant shared by both the class body and __main__.
DIAGNOSTIC_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Module-level fallback for my_args: used when the file is loaded by an external
# test runner (e.g. pytest, unittest discover) that does not enter __main__.
# __main__ overwrites these with values parsed from the command line.
my_args = argparse.Namespace(
    diagnostic="WARNING",
    diagnostic_index=DIAGNOSTIC_CHOICES.index("WARNING"),
    output=None,
    printStreamInfo=False,
    printCmdsOnly=False,
    reduceCmds=True,
    useAllCmdOptions=False,
    addLogLevel="",
    verbose=common.VERBOSITY_NORMAL,
)


class TestAmdSmiCli(unittest.TestCase):
    # event listener timing constants
    # Maximum time (s) to wait for the event process to print "EVENT LISTENING:",
    # confirming AMDSMI is initialized and listener.read() is active.
    EVENT_LISTENER_STARTUP_WAIT_SEC = 10
    # listener.read() blocks for up to 2000 ms, so poll more often than that.
    EVENT_FILE_POLL_INTERVAL_SEC = 0.1
    # Maximum time (s) to wait for the output file to become non-empty after
    # the trigger command completes.
    EVENT_FILE_POLL_TIMEOUT_SEC = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_encoding = locale.getpreferredencoding()

        self.AddCmdMods = True
        self.AddDeviceArgs = True
        self.AddWatchArgs = True
        self.AddLogLevel = ""
        if my_args.addLogLevel:
            self.AddLogLevel = f"--loglevel {my_args.addLogLevel}"
        return

    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(my_args.verbose)
        cls.util = runcmd.Util(my_args.diagnostic)

        cls.PASS = 0
        cls.FAIL = 1
        cls.tab = "    "

        cls.openBracket = "["
        cls.closeBracket = "]"
        cls.openCurlyBrace = "{"
        cls.closeCurlyBrace = "}"

        # Record starting values
        cmd = "amd-smi metric --json"
        (rc, data, std_err) = cls.util.RunCmdSync(cmd)
        if rc:
            raise RuntimeError(f'Error executing "{cmd}": {std_err}')
        cls.metric_data = json.loads(data)

        cmd = "amd-smi static --json"
        (rc, data, std_err) = cls.util.RunCmdSync(cmd)
        if rc:
            raise RuntimeError(f'Error executing "{cmd}": {std_err}')
        cls.static_data = json.loads(data)

        cmd = "amd-smi list --json"
        (rc, data, std_err) = cls.util.RunCmdSync(cmd)
        if rc:
            raise RuntimeError(f'Error executing "{cmd}": {std_err}')
        cls.list_data = json.loads(data)

        cmd = "amd-smi partition --current --json"
        (rc, data, std_err) = cls.util.RunCmdSync(cmd)
        if rc:
            raise RuntimeError(f'Error executing "{cmd}": {std_err}')
        cls.partition_data = json.loads(data)

        if my_args.verbose >= common.VERBOSITY_VERBOSE:
            # Execute the following to print the asic and board info once per test run
            if my_args.diagnostic == "DEBUG":
                for i, _ in enumerate(cls.common.processors):
                    msg = f"gpu={i}"
                    cls.common.print(msg)
                    msg = f"virtualization mode(gpu={i})"
                    cls.common.print(msg, cls.common.virt_mode[i])
                    msg = f"asic info(gpu={i})"
                    cls.common.print(msg, cls.common.asic_info[i])
                    msg = f"board info(gpu={i})"
                    cls.common.print(msg, cls.common.board_info[i])
                    cls.common.print("")

        cls.tmp_filename = "_tmp.log"
        cls.tmp_folder = "_tmp"

        # TODO: Need to be able to get automatic answers
        cls.skip_args_require_input = {
            "reset --gtt",
            "set --fan",
            "set --memory-partition",
        }

        cls.gpus = ["all"]
        for data in cls.list_data:
            cls.gpus.append(data["gpu"])
            if data["gpu"] == 0:
                # Only test bdf and uuid when gpu=0
                cls.gpus.append(data["bdf"])
                cls.gpus.append(data["uuid"])

        # When parsing, expand each arg with array element
        cls.sub_args = {
            "CLOCK": ["SYS", "DF", "DCEF", "SOC", "MEM", "VCLK0", "VCLK1", "DCLK0", "DCLK1", "ALL"],
            "PID": [123],
            "NAME": ["AMD"],
            "GPU": cls.gpus,
            "FILE": [
                cls.tmp_filename,
                f"{cls.tmp_filename} --overwrite",
                f"{cls.tmp_filename} --append",
            ],
            "SEVERITY": ["nonfatal-uncorrected", "fatal", "nonfatal-corrected", "all"],
            "FOLDER": [cls.tmp_folder],
            "FILE_LIMIT": [10],
            #'LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        }

        cls.perf_levels = [
            "AUTO",
            "LOW",
            "HIGH",
            "MANUAL",
            "STABLE_STD",
            "STABLE_PEAK",
            "STABLE_MIN_MCLK",
            "STABLE_MIN_SCLK",
            "DETERMINISM",
        ]
        cls.profile_levels = [
            "CUSTOM_MASK",
            "VIDEO_MASK",
            "POWER_SAVING_MASK",
            "COMPUTE_MASK",
            "VR_MASK",
            "THREE_D_FULL_SCR_MASK",
            "BOOTUP_DEFAULT",
        ]
        cls.compute_partition_modes = ["SPX", "DPX", "TPX", "QPX", "CPX"]
        cls.memory_partition_modes = ["NPS1", "NPS2", "NPS4", "NPS8"]
        cls.power_types = ["ppt0", "ppt1"]
        cls.ptl_formats = ["I8", "F16", "BF16", "F32", "F64", "F8", "VECTOR"]
        cls.clk_limits = ["SCLK", "MCLK"]
        cls.limit_types = ["MIN", "MAX"]
        cls.clk_levels = ["SCLK", "MCLK", "FCLK", "SOCCLK", "PCIE"]

        # When parsing, ignore these entries as they are abnormal
        cls.cmd_arg_exceptions = ["--voltage"]

        # When parsing, change these args into something else or add to arg
        cls.cmd_arg_changes = [
            "--loglevel",
            "--json",
            "--csv",
            "--append",
            "--overwrite",
            "--ucode-list",
            "--watch",
            "--watch_time",
            "--iterations",
        ]

        return

    @classmethod
    def StrToNumber(cls, num_str):
        rc = 0
        num_str = num_str.strip()
        try:
            value = int(num_str)
        except ValueError:
            try:
                value = float(num_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                rc = 1
                value = num_str
        return (rc, value)

    def _PrintResults(self, results, fail_on_results=False):
        if results:
            cmd_len = 0
            for cmd, _ in results:
                num = len(cmd)
                if num > cmd_len:
                    cmd_len = num
            cmd_len += 2

            msg = ""
            for cmd, cmd_out in results:
                msg += f"\n{self.tab}{cmd:{cmd_len}s} : {cmd_out}"
            msg = msg.strip()
            msg = f"{self.tab}{msg}"

            # Output to file if set
            if my_args.output:
                with open(my_args.output, "a", encoding=self.use_encoding) as fout:
                    fout.write(f"{msg}\n")

            # Output to std_out
            self.common.print(f"{msg}")

            # Output to std_err
            if fail_on_results:
                self.fail(f"Fail:\n\n{msg}")
        return

    def _get_monitor_metric_data(self, monitor1, monitor2, metric):
        data = []
        for i in range(len(monitor1)):
            data.append(
                {
                    "power_usage": None,
                    "hotspot_temperature": None,
                    "memory_temperature": None,
                    "gfx_clk": None,
                    "gfx": None,
                    "mem": None,
                    "vram_used": None,
                    "vram_total": None,
                }
            )
            for key in data[i]:
                if isinstance(monitor1[i][key], str) and monitor1[i][key] == "N/A":
                    unit = "N/A"
                    data1 = 0
                    data2 = 0
                else:
                    unit = monitor1[i][key]["unit"]
                    if key == "vram_used" or key == "vram_total":
                        # Monitor data reported in GB, Metric data in GB
                        # Convert GB to MB for comparison
                        unit = monitor1[i][key]["unit"] = "MB"
                        data1 = int(monitor1[i][key]["value"] * 1000)
                    else:
                        data1 = int(monitor1[i][key]["value"])
                    if monitor2 != None:
                        if key == "vram_used" or key == "vram_total":
                            data2 = int(monitor2[i][key]["value"] * 1000)
                        else:
                            data2 = int(monitor2[i][key]["value"])
                    else:
                        if key == "power_usage":
                            data2 = metric["gpu_data"][i]["power"]["socket_power"]["value"]
                        elif key == "hotspot_temperature":
                            data2 = metric["gpu_data"][i]["temperature"]["hotspot"]["value"]
                        elif key == "memory_temperature":
                            data2 = metric["gpu_data"][i]["temperature"]["mem"]["value"]
                        elif key == "gfx_clk":
                            data2 = metric["gpu_data"][i]["clock"]["gfx_0"]["clk"]["value"]
                        elif key == "gfx":
                            data2 = metric["gpu_data"][i]["usage"]["gfx_activity"]["value"]
                        elif key == "mem":
                            data2 = metric["gpu_data"][i]["usage"]["mm_activity"]["value"]
                        elif key == "vram_used":
                            data2 = int(metric["gpu_data"][i]["mem_usage"]["used_vram"]["value"])
                        elif key == "vram_total":
                            data2 = int(metric["gpu_data"][i]["mem_usage"]["total_vram"]["value"])
                data[i][key] = [data1, data2, abs(data1 - data2), unit]
        return data

    def _compare_monitor_metric_data(self, component, data):
        failures = []
        for i in range(len(data)):
            msg_title = f"Monitor to {component}: gpu={i}"
            msg_header = f"{'key':>20s} ({'Unit':>4s}): {'Monitor':>8s} {component:>8s}  {'Diff':>8s}   {'Threshold':>8s} {'Status':>7s}"
            self.common.print(f"Compare {msg_title}")
            self.common.print(msg_header)
            for key in data[i]:
                if data[i][key][3] == "N/A":
                    continue
                max_diff = max(data[i][key][0], data[i][key][1])
                max_diff = max(data[i][key][0], data[i][key][1]) * 0.1
                if data[i][key][2] > max_diff:
                    status = "Failure"
                    compare = ">"
                else:
                    status = "Success"
                    compare = "<"
                _msg = f"{key:>20s} ({data[i][key][3]:>4s}): {data[i][key][0]:>8d} {data[i][key][1]:>8d} ({data[i][key][2]:>8d} {compare} {max_diff:>8.2f}) {status:>7s}"
                self.common.print(_msg)
                if status == "Failure":
                    cmd = msg_title
                    if len(failures) == 0:
                        failures.append(("*" * len(cmd), msg_header))
                    failures.append((cmd, _msg))
        return failures

    def FindArgs(self, cmd, match_str):
        if (
            (not match_str)
            or (not self.AddDeviceArgs and "Device" in match_str)
            or (not self.AddWatchArgs and "Watch" in match_str)
            or (not self.AddCmdMods and "Command" in match_str)
        ):
            return ["pass"]

        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        if rc:
            raise RuntimeError(f'Error executing "{cmd}": {std_err}')
        lines = std_out.split("\n")

        found = False
        options = []
        for line in lines:
            if found:
                if not line:
                    break
                items = line.split()
                for item_index, item in enumerate(items):
                    items[item_index] = item.strip()
                item_index = -1
                if "-h" == items[0][0:2]:
                    # Turn help into command without an option
                    if "Set" in match_str or "Reset" in match_str or "RAS" in match_str:
                        pass  # These require an option
                    else:
                        options.append("")
                elif "--" in items[0][0:2]:
                    item_index = 0
                elif len(items) > 1 and "--" == items[1][0:2]:
                    item_index = 1
                elif "-" == items[0][0:1]:
                    item_index = 0

                sub_found = False
                if item_index >= 0:
                    if items[item_index][-1:] == ",":
                        items[item_index] = items[item_index][:-1]
                    if items[item_index] in self.cmd_arg_exceptions:
                        pass
                    elif items[item_index] in self.cmd_arg_changes:
                        sub_found = True
                        if "--ucode-list" == items[item_index]:
                            options.append(f"{items[item_index]}")
                            options.append("--fw-list")
                        elif "--json" == items[item_index]:
                            options.append("{json}")
                            options.append("{json_file}")
                            options.append("{json_file_append}")
                            options.append("{json_file_overwrite}")
                        elif "--csv" == items[item_index]:
                            options.append("{csv}")
                            options.append("{csv_file}")
                            options.append("{csv_file_append}")
                            options.append("{csv_file_overwrite}")
                        elif "--append" == items[item_index] or "--overwrite" == items[item_index]:
                            pass
                        elif "--watch" == items[item_index]:
                            options.append("{watch_time}")
                            options.append("{watch_iterations}")
                        elif (
                            "--watch_time" == items[item_index]
                            or "--iterations" == items[item_index]
                        ):
                            pass
                        elif "--loglevel" == items[item_index]:
                            pass
                        else:
                            print(f"ERROR: bad sub arg {items[item_index]}")
                    elif len(items) > item_index:
                        if items[item_index + 1][0:1] == self.openBracket:
                            items[item_index + 1] = items[item_index + 1][1:]
                        sub_arg = items[item_index + 1]
                        # Expand out sub_args
                        if sub_arg.isupper() and sub_arg in self.sub_args:
                            sub_found = True
                            for item in self.sub_args[sub_arg]:
                                options.append(f"{items[item_index]} {item}")
                        elif "Set" in match_str:
                            if sub_arg == "%":  # arg --fan
                                options.append(f"{items[item_index]} 50%")
                                options.append(f"{items[item_index]} 50")
                            elif sub_arg == "LEVEL":  # arg --perf-level
                                for perf_level in self.perf_levels:
                                    options.append(f"{items[item_index]} {perf_level}")
                            elif sub_arg == "PROFILE_LEVEL":  # arg --profile
                                for profile_level in self.profile_levels:
                                    options.append(f"{items[item_index]} {profile_level}")
                            elif sub_arg == "SCLKMAX":  # arg --perf-determinism
                                options.append("{perf_determinism}")
                            elif sub_arg == "TYPE/INDEX":  # arg
                                for compute_partition_mode in self.compute_partition_modes:
                                    options.append(f"{items[item_index]} {compute_partition_mode}")
                            elif sub_arg == "PARTITION":  # arg --memory-partition
                                for memory_partition_mode in self.memory_partition_modes:
                                    options.append(f"{items[item_index]} {memory_partition_mode}")
                            elif sub_arg == "WATTS":  # arg --power-cap
                                for power_type in self.power_types:
                                    options.append(f"--power-cap {{min_power}} {power_type}")
                                    options.append(f"--power-cap {{avg_power}} {power_type}")
                                    options.append(f"--power-cap {{max_power}} {power_type}")
                            elif (
                                sub_arg == "POLICY_ID" and "soc" in items[item_index]
                            ):  # arg --soc-pstate
                                options.append(f"{items[item_index]} {{soc_pstate}}")
                            elif (
                                sub_arg == "POLICY_ID" and "xgmi" in items[item_index]
                            ):  # arg --xgmi-plpd
                                options.append(f"{items[item_index]} {{xgmi_plpd}}")
                            elif (
                                sub_arg == "CLK_TYPE" and "level" in items[item_index]
                            ):  # arg --clk-level
                                options.append(f"{items[item_index]} {{clk_level_sclk}}")
                                options.append(f"{items[item_index]} {{clk_level_mclk}}")
                                options.append(f"{items[item_index]} {{clk_level_fclk}}")
                                options.append(f"{items[item_index]} {{clk_level_socclk}}")
                                options.append(f"{items[item_index]} {{clk_level_pcie}}")
                            elif (
                                sub_arg == "STATUS" and "ptl" in items[item_index]
                            ):  # arg --ptl-status
                                options.append(f"{items[item_index]} 0")
                                options.append(f"{items[item_index]} 1")
                            elif sub_arg == "FRMT1,FRMT2":  # arg --ptl-format
                                for fmt1 in self.ptl_formats:
                                    for fmt2 in self.ptl_formats:
                                        if fmt1 == fmt2:
                                            continue
                                        options.append(f"{items[item_index]} {fmt1},{fmt2}")
                            elif (
                                sub_arg == "CLK_TYPE" and "limit" in items[item_index]
                            ):  # arg --clk-limit
                                options.append(f"{items[item_index]} {{clk_limit_sclk_min}}")
                                options.append(f"{items[item_index]} {{clk_limit_sclk_max}}")
                                options.append(f"{items[item_index]} {{clk_limit_mclk_min}}")
                                options.append(f"{items[item_index]} {{clk_limit_mclk_max}}")
                            elif (
                                sub_arg == "STATUS" and "process" in items[item_index]
                            ):  # arg --process-isolation
                                options.append(f"{items[item_index]} 0")
                                options.append(f"{items[item_index]} 1")
                            else:
                                print(
                                    f"TODO: set {items[item_index]} sub_arg={sub_arg}  match_str={match_str}"
                                )
                    if not sub_found:
                        # Put in sub_arg if it was not found
                        if "Set" in match_str:
                            pass
                        else:
                            options.append(items[item_index])
            if match_str in line:
                found = True
        if not options:
            return ["pass"]
        return options

    def CreateCmds(self, cmd_name, list1_name, list2_name, list3_name, list4_name):
        cmd = f"amd-smi {cmd_name} --help"
        list1_args = self.FindArgs(cmd, list1_name)
        list2_args = self.FindArgs(cmd, list2_name)
        list3_args = self.FindArgs(cmd, list3_name)
        list4_args = self.FindArgs(cmd, list4_name)
        if my_args.diagnostic == "DEBUG":
            print(f"{list1_name}: {'*' * 80}")
            print(json.dumps(list1_args, sort_keys=False, indent=4), flush=True)
            print(f"{list2_name}: {'*' * 80}")
            print(json.dumps(list2_args, sort_keys=False, indent=4), flush=True)
            print(f"{list3_name}: {'*' * 80}")
            print(json.dumps(list3_args, sort_keys=False, indent=4), flush=True)
            print(f"{list4_name}: {'*' * 80}")
            print(json.dumps(list4_args, sort_keys=False, indent=4), flush=True)

        cmds = []
        cmd = f"amd-smi {cmd_name}"
        for list1_arg in list1_args:
            if list1_arg != "pass":
                cmds.append((f"{cmd} {list1_arg} {self.AddLogLevel}", self.PASS))
                if not list1_arg:
                    cmds.append((f"{cmd} --file {self.tmp_filename} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{json}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{json_file}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{json_file_append}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{json_file_overwrite}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{csv}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{csv_file}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{csv_file_append}} {self.AddLogLevel}", self.PASS))
                    cmds.append((f"{cmd} {{csv_file_overwrite}} {self.AddLogLevel}", self.PASS))
            else:
                list1_arg = ""
            for list2_arg in list2_args:
                if list2_arg != "pass":
                    cmds.append((f"{cmd} {list1_arg} {list2_arg} {self.AddLogLevel}", self.PASS))
                else:
                    list2_arg = ""
                for list3_arg in list3_args:
                    if list3_arg != "pass":
                        cmds.append(
                            (
                                f"{cmd} {list1_arg} {list2_arg} {list3_arg} {self.AddLogLevel}",
                                self.PASS,
                            )
                        )
                    else:
                        list3_arg = ""
                    for list4_arg in list4_args:
                        if list4_arg != "pass":
                            cmds.append(
                                (
                                    f"{cmd} {list1_arg} {list2_arg} {list3_arg} {list4_arg} {self.AddLogLevel}",
                                    self.PASS,
                                )
                            )

        # Calculate and substitute in dependent values
        # Removes cmds that are invalid
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            while self.openCurlyBrace in cmd:
                items = cmd.split()
                # Find gpu index and mark when gpu=0
                try:
                    i = items.index("--gpu")
                    gpu = items[i + 1]
                    if gpu.isdigit():
                        gpu_index = int(gpu)
                    else:
                        gpu_index = 0
                except ValueError:
                    gpu_index = 0

                # Find conditional arguments
                posOpen = cmd.find(self.openCurlyBrace)
                if posOpen < 0:
                    break
                posClose = cmd.find(self.closeCurlyBrace, posOpen)
                if posClose < 0:
                    break
                nameStr = cmd[posOpen : posClose + 1]

                if (
                    nameStr == "{json}"
                    or "json_file" in nameStr
                    or nameStr == "{csv}"
                    or "csv_file" in nameStr
                ):
                    # For adding file options
                    if nameStr == "{json}":
                        cmd = cmd.replace(nameStr, "--json", 1)
                    elif nameStr == "{json_file}":
                        cmd = cmd.replace(nameStr, f"--json --file {self.tmp_filename}", 1)
                    elif nameStr == "{json_file_append}":
                        cmd = cmd.replace(nameStr, f"--json --file {self.tmp_filename} --append", 1)
                    elif nameStr == "{json_file_overwrite}":
                        cmd = cmd.replace(
                            nameStr, f"--json --file {self.tmp_filename} --overwrite", 1
                        )
                    elif nameStr == "{csv}":
                        cmd = cmd.replace(nameStr, "--csv", 1)
                    elif nameStr == "{csv_file}":
                        cmd = cmd.replace(nameStr, f"--csv --file {self.tmp_filename}", 1)
                    elif nameStr == "{csv_file_append}":
                        cmd = cmd.replace(nameStr, f"--csv --file {self.tmp_filename} --append", 1)
                    elif nameStr == "{csv_file_overwrite}":
                        cmd = cmd.replace(
                            nameStr, f"--csv --file {self.tmp_filename} --overwrite", 1
                        )
                    else:
                        print(f"Error: could not replace json/csv options, {nameStr}  cmd={cmd}")
                        cmd = ""
                elif nameStr == "{watch_time}" or nameStr == "{watch_iterations}":
                    # For adding watch options
                    if nameStr == "{watch_time}":
                        cmd = cmd.replace(nameStr, "--watch 1 --watch_time 2", 1)
                    else:
                        cmd = cmd.replace(nameStr, "--watch 1 --iterations 2", 1)
                elif (
                    nameStr == "{min_power}" or nameStr == "{avg_power}" or nameStr == "{max_power}"
                ):
                    # For setting --power-cap
                    # Find power_type
                    for power_type in self.power_types:
                        if power_type in cmd:
                            power_type = self.static_data["gpu_data"][gpu_index]["limit"][
                                power_type
                            ]
                        else:
                            power_type = "N/A"
                    if (
                        power_type == "N/A"
                        or power_type["min_power_limit"] == "N/A"
                        or power_type["max_power_limit"] == "N/A"
                    ):
                        cmd = ""
                    else:
                        min_power = power_type["min_power_limit"]["value"]
                        max_power = power_type["max_power_limit"]["value"]
                        avg_power = int((min_power + max_power) / 2)
                        if nameStr == "{min_power}":
                            cmd = cmd.replace("{min_power}", str(min_power), 1)
                        elif nameStr == "{avg_power}":
                            cmd = cmd.replace("{avg_power}", str(avg_power), 1)
                        elif nameStr == "{max_power}":
                            cmd = cmd.replace("{max_power}", str(max_power), 1)
                elif nameStr == "{perf_determinism}":
                    clock_sys = self.static_data["gpu_data"][gpu_index]["clock"]["sys"]
                    if clock_sys != "N/A" and len(clock_sys["frequency_levels"]):
                        num = len(clock_sys["frequency_levels"])
                        level = f"Level {num - 1}"
                        clock_freq = int(clock_sys["frequency_levels"][level]["value"])
                        cmd = cmd.replace(
                            "{perf_determinism}", f"--perf-determinism {clock_freq + 50}", 1
                        )
                    else:
                        cmd = ""
                elif "clk_limit" in nameStr:
                    clock = self.metric_data["gpu_data"][gpu_index]["clock"]
                    clk_type = None
                    clk_type_name = None
                    limit_type = None
                    clk_limit_name = None
                    if nameStr == "{clk_limit_sclk_min}":
                        clk_type = "SCLK"
                        clk_type_name = "socclk_0"
                        limit_type = "MIN"
                        clk_limit_name = "min_clk"
                    elif nameStr == "{clk_limit_sclk_max}":
                        clk_type = "SCLK"
                        clk_type_name = "socclk_0"
                        limit_type = "MAX"
                        clk_limit_name = "max_clk"
                    elif nameStr == "{clk_limit_mclk_min}":
                        clk_type = "MCLK"
                        clk_type_name = "mem_0"
                        limit_type = "MAX"
                        clk_limit_name = "min_clk"
                    elif nameStr == "{clk_limit_mclk_max}":
                        clk_type = "MCLK"
                        clk_type_name = "mem_0"
                        limit_type = "MIN"
                        clk_limit_name = "max_clk"
                    clk_type_limit_name = clock[clk_type_name][clk_limit_name]
                    if isinstance(clk_type_limit_name, dict):
                        value = clk_type_limit_name["value"]
                        cmd = cmd.replace(nameStr, f"{clk_type} {limit_type} {value}", 1)
                    else:
                        cmd = ""
                elif "clk_level" in nameStr:
                    clock = self.static_data["gpu_data"][gpu_index]["clock"]
                    value = -1
                    if nameStr == "{clk_level_sclk}":
                        clk_type = "SCLK"
                        clk_type_name = "sys"
                    elif nameStr == "{clk_level_mclk}":
                        clk_type = "MCLK"
                        clk_type_name = "mem"
                    elif nameStr == "{clk_level_fclk}":
                        clk_type = "FCLK"
                        clk_type_name = "df"
                    elif nameStr == "{clk_level_socclk}":
                        clk_type = "SOCCLK"
                        clk_type_name = "soc"
                    elif nameStr == "{clk_level_pcie}":
                        bus = self.static_data["gpu_data"][gpu_index]["bus"]
                        clk_type = "PCIE"
                        pcie_levels = bus["pcie_levels"]
                        if isinstance(pcie_levels, dict):
                            value = len(pcie_levels)
                            if value > 0:
                                value = 0
                    if clk_type != "PCIE" and value < 0:
                        clk_type_name = clock[clk_type_name]
                        if isinstance(clk_type_name, dict):
                            current_level = clk_type_name["current_level"]
                            freq_levels = clk_type_name["frequency_levels"]
                            if current_level == 0:
                                value = len(freq_levels) - 1
                            else:
                                value = 0
                    if value >= 0:
                        cmd = cmd.replace(nameStr, f"{clk_type} {value}", 1)
                    else:
                        cmd = ""
                elif nameStr == "{soc_pstate}":
                    soc_pstate = self.static_data["gpu_data"][gpu_index]["soc_pstate"]
                    if isinstance(soc_pstate, dict):
                        num_supported = int(soc_pstate["num_supported"])
                        if num_supported > 0:
                            current = int(soc_pstate["current_id"])
                            if current == 0:
                                num = num_supported - 1
                            else:
                                num = 0
                            cmd = cmd.replace(nameStr, f"{num}", 1)
                        else:
                            cmd = ""
                    else:
                        cmd = ""
                elif nameStr == "{xgmi_plpd}":
                    xgmi_plpd = self.static_data["gpu_data"][gpu_index]["xgmi_plpd"]
                    if isinstance(xgmi_plpd, dict):
                        num_supported = int(xgmi_plpd["num_supported"])
                        if num_supported > 0:
                            current = int(xgmi_plpd["current_id"])
                            if current == 0:
                                num = num_supported - 1
                            else:
                                num = 0
                            cmd = cmd.replace(nameStr, f"{num}", 1)
                        else:
                            cmd = ""
                    else:
                        cmd = ""
            cmds[index] = (cmd, cond)

        # Remove cmds that require input
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            for skip_arg in self.skip_args_require_input:
                if skip_arg in cmd:
                    cmds[index] = ("", cond)
                    break

        # For event command, only take commands with a file
        if cmd_name == "event":
            for index, cmd_cond in enumerate(cmds):
                cmd, cond = cmd_cond
                if "--file" not in cmd:
                    cmds[index] = ("", cond)

        # Pare down commands
        if my_args.reduceCmds:
            file_mods = ["--file", "--json", "--csv"]
            watch_mods = ["--watch", "--watch_time", "--iterations"]

            found_sub_arg = False
            for index, cmd_cond in enumerate(cmds):
                cmd, cond = cmd_cond
                items = cmd.split()

                # Find the first sub_arg
                if not found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    for mod in file_mods + ["--gpu", "--loglevel"]:
                        if mod == sub_arg:
                            sub_arg = ""
                            break
                    found_sub_arg = sub_arg

                # No explicit gpu infers a gpu=0
                gpu_index = "0"
                if "--gpu" in cmd:
                    try:
                        i = items.index("--gpu")
                        gpu_index = items[i + 1]
                    except ValueError:
                        # condition where --gpu is not in the cmd
                        # will get default gpu_index=0
                        pass

                # Remove all --gpu for all sub_args except for the first sub_arg
                if cmd and found_sub_arg:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        if "--gpu" in cmd:
                            cmd = ""

                # Remove all file and watch modifiers except for gpu 0
                if cmd and gpu_index != "0":
                    for mod in file_mods + watch_mods:
                        if mod in cmd:
                            cmd = ""
                            break

                # Remove all --file and --watch combinations
                if cmd and "--file" in cmd and "--watch" in cmd:
                    cmd = ""

                # Remove all --watch mod for all sub_args except for the first sub_arg
                if cmd and found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        if "--watch" in cmd:
                            cmd = ""

                # Remove all file mod for all sub_args except for the first sub_arg
                if cmd and found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        for mod in file_mods:
                            if mod in cmd:
                                cmd = ""
                                break

                cmds[index] = (cmd, cond)

        # Remove empty (cmd,cond) arguments
        cmds = [cmd_cond for cmd_cond in cmds if cmd_cond[0] != ""]

        # Remove extra spaces between arguments
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            cmd = cmd.split()
            cmd = " ".join(cmd).strip()
            cmds[index] = (cmd, cond)
        if my_args.diagnostic == "DEBUG":
            print(f"cmds: {'*' * 80}")
            print(json.dumps(cmds, sort_keys=False, indent=4), flush=True)
        return cmds

    def _GetErrorCode(self, std_out, std_err, cond):
        error_code = 0
        items = []
        output_stream = None
        if std_out and "Error code" in std_out:
            output_stream = "std_out"
            items = std_out.strip().split()
        elif std_err and "Error code" in std_err:
            output_stream = "std_err"
            items = std_err.strip().split()
        elif cond != self.PASS:
            if std_out:
                output_stream = "std_out"
                items = std_out.strip().split()
            elif std_err:
                output_stream = "std_err"
                items = std_err.strip().split()
        if items:
            _, error_code = self.StrToNumber(items[-1])
        return (error_code, output_stream)

    def _GetCmdReturnMsg(self, rc, error_code, cond):
        msg = ""
        rc = str(rc)
        passed = False
        error_code = str(error_code)
        if rc != "0" and cond == self.PASS:
            msg = f"Failure: Received FAIL ({rc:>3s}), expected PASS (  0)"
        elif rc == "0" and cond != self.PASS:
            if error_code == "0":
                msg = "Failure: Received PASS (  0), expected FAIL ( !0)"
            else:
                msg = f"Failure: Received PASS (  0), returned FAIL ({error_code:>3s})"
        elif rc == "0" and error_code != "0":
            msg = f"Failure: Received rc = ({rc:>3s}), returned ec = ({error_code:>3s})"
        elif rc != "0" and f"-{rc}" != error_code:
            msg = f"Failure: Received rc = ({rc:>3s}), returned ec = ({error_code:>3s})"
        else:
            passed = True
            if rc == "0":
                expected = "PASS"
            else:
                expected = "FAIL"
            msg = f"Success: Received and Expected {expected} ({error_code:>3s})"
        return (msg, passed)

    def RunCmds(self, cmds):
        failures = []
        successes = []
        for cmd, cond in cmds:
            if my_args.diagnostic == "INFO" or my_args.printCmdsOnly:
                print(f"cmd={cmd}")
            if my_args.printCmdsOnly:
                continue

            # Remove output file if it exists
            if os.path.exists(self.tmp_filename):
                os.chmod(self.tmp_filename, stat.S_IWRITE)
                os.remove(self.tmp_filename)

            cmd_trigger = None
            cmd_subcmd = cmd.split()[1] if len(cmd.split()) > 1 else ""
            if cmd_subcmd == "event":
                (_, _, _, proc) = self.util.RunCmdAsync(cmd, capture_stdout=True)
                rc = 0
                std_out = ""
                std_err = ""

                # Wait until the process prints "EVENT LISTENING:", confirming
                # AMDSMI is initialized and listener.read() is active.
                event_listening = False
                deadline = time.monotonic() + self.EVENT_LISTENER_STARTUP_WAIT_SEC
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(
                        [proc.stdout], [], [], self.EVENT_FILE_POLL_INTERVAL_SEC
                    )
                    if ready:
                        line = proc.stdout.readline()
                        if "EVENT LISTENING" in line:
                            event_listening = True
                            break
                if not event_listening:
                    print(
                        f'{self.tab}Warning: event process did not print "EVENT LISTENING" '
                        f"within {self.EVENT_LISTENER_STARTUP_WAIT_SEC}s for cmd: {cmd}",
                        file=sys.stderr,
                    )

                # Run an event
                cmd_trigger = "amd-smi reset --gpureset"
                (rc_trigger, std_out_trigger, std_err_trigger) = self.util.RunCmdSync(cmd_trigger)

                # Wait for the event notification to propagate from the driver,
                # be received by listener.read() and written to the output file
                # before killing the process with SIGKILL.
                # listener.read() has a 2000ms polling timeout, so wait up to
                # that long (plus a small buffer) for output to appear.
                if "--file" in cmd:
                    deadline = time.monotonic() + self.EVENT_FILE_POLL_TIMEOUT_SEC
                    while time.monotonic() < deadline:
                        if (
                            os.path.exists(self.tmp_filename)
                            and os.path.getsize(self.tmp_filename) > 0
                        ):
                            break
                        time.sleep(self.EVENT_FILE_POLL_INTERVAL_SEC)

                # Terminate the monitoring
                proc.kill()
                proc.wait()
                if proc.stdout:
                    proc.stdout.close()
            else:
                (rc, std_out, std_err) = self.util.RunCmdSync(cmd)

            error_code, output_stream = self._GetErrorCode(std_out, std_err, cond)

            passed_file = True
            if "--file" in cmd:
                if not os.path.exists(self.tmp_filename):
                    msg = f"Failure: File {self.tmp_filename} does not exist"
                    failures.append((cmd, msg))
                else:
                    with open(self.tmp_filename, "r", encoding=self.use_encoding) as fin:
                        std_out = fin.read()
                    if len(std_out) == 0:
                        passed_file = False
                        msg = f"Failure: File {self.tmp_filename} is empty"
                        failures.append((cmd, msg))
                    elif "--json" in cmd:
                        try:
                            data = json.loads(std_out)
                        except json.JSONDecodeError:
                            passed_file = False
                            msg = f"Failure: File {self.tmp_filename} contains invalid json"
                            failures.append((cmd, msg))
                    os.chmod(self.tmp_filename, stat.S_IWRITE)
                    os.remove(self.tmp_filename)

            msg, passed = self._GetCmdReturnMsg(rc, error_code, cond)

            # When testing for the file, if file test failed mark test as fail
            # This keeps the fail and pass output together
            if not passed_file:
                passed = False

            if passed:
                successes.append((cmd, msg))
            else:
                if my_args.printStreamInfo:
                    msg = f"{msg}, stream={output_stream}"
                failures.append((cmd, msg))

            if cmd_trigger:
                error_code_trigger, _ = self._GetErrorCode(std_out_trigger, std_err_trigger, cond)
                msg_trigger, _ = self._GetCmdReturnMsg(rc_trigger, error_code_trigger, self.PASS)
                # Even if the trigger cmd failed, put the output in with the cmd fail or pass output
                if passed:
                    successes.append((f"{self.tab}trigger: {cmd_trigger}", msg_trigger))
                else:
                    failures.append((f"{self.tab}trigger: {cmd_trigger}", msg_trigger))

            if my_args.diagnostic == "DEBUG":
                print(f"{self.tab}rc={rc}")
                print(f"{self.tab}stream={output_stream}")
                print(f"{self.tab}error_code={error_code}")
                print(f"{self.tab}std_out={std_out}")
                print(f"{self.tab}std_err={std_err}")

        self._PrintResults(successes)
        self._PrintResults(failures, fail_on_results=True)
        return

    def tearDown(self):
        # Clean up temp file and folder if they exist
        if os.path.exists(self.tmp_filename):
            os.chmod(self.tmp_filename, stat.S_IWRITE)
            os.remove(self.tmp_filename)
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)
        self.common.print(
            f"Teardown triggered | Cleaned up file: {self.tmp_filename} and folder: {self.tmp_folder}"
        )

    def test_help(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi help"
        self.common.print(msg)

        cmd = "amd-smi --help"
        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        if rc:
            raise RuntimeError(f'Error executing "{cmd}": {std_err}')
        lines = std_out.split("\n")
        # Find all available command line args
        cmd_args = []
        found = False
        for line in lines:
            if found:
                if not line:
                    break
                items = line.split()
                cmd_args.append(items[0])
                continue
            if "Descriptions" in line:
                found = True

        cmds = [("amd-smi --help", self.PASS)]
        for cmd_arg in cmd_args:
            cmds.append((f"amd-smi {cmd_arg} --help", self.PASS))

        self.RunCmds(cmds)
        return

    def test_valid(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi VALID"
        self.common.print(msg)

        cmds = [
            ("amd-smi metric --json", self.PASS),
            ("amd-smi metric --gpu 0 --json", self.PASS),
            # ('amd-smi metric --watch 1 --json', self.PASS),
        ]

        self.RunCmds(cmds)
        return

    def test_invalid(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi INVALID"
        self.common.print(msg)

        # Create bad bdf and uuid gpus
        bdf = self.list_data[0]["bdf"]
        if bdf[-1] == "0":
            bad_bdf = self.list_data[0]["bdf"][:-1] + "1"
        else:
            bad_bdf = self.list_data[0]["bdf"][:-1] + "0"
        uuid = self.list_data[0]["uuid"]
        if uuid[-1] == "0":
            bad_uuid = self.list_data[0]["uuid"][:-1] + "1"
        else:
            bad_uuid = self.list_data[0]["uuid"][:-1] + "0"

        cmds = [
            # Test invalid command
            ("amd-smi invalid_cmd", self.FAIL),
            # Test invalid sub command
            ("amd-smi version --invalid", self.FAIL),
            ("amd-smi list --invalid", self.FAIL),
            ("amd-smi static --invalid", self.FAIL),
            ("amd-smi firmware --invalid", self.FAIL),
            ("amd-smi bad_pages --invalid", self.FAIL),
            ("amd-smi metric --invalid", self.FAIL),
            ("amd-smi process --invalid", self.FAIL),
            ("amd-smi event --invalid", self.FAIL),
            ("amd-smi topology --invalid", self.FAIL),
            ("amd-smi set --invalid", self.FAIL),
            ("amd-smi reset", self.FAIL),
            ("amd-smi reset --invalid", self.FAIL),
            ("amd-smi monitor --invalid", self.FAIL),
            ("amd-smi xgmi --invalid", self.FAIL),
            ("amd-smi partition --invalid", self.FAIL),
            ("amd-smi ras --invalid", self.FAIL),
            ("amd-smi node --invalid", self.FAIL),
            # Test invalid gpu value
            ("amd-smi version --gpu 0", self.FAIL),
            ("amd-smi version --gpu -1", self.FAIL),
            ("amd-smi version --gpu ALL", self.FAIL),
            (f"amd-smi version --gpu {len(self.common.processors)}", self.FAIL),
            ("amd-smi static --gpu -1", self.FAIL),
            ("amd-smi static --gpu _ALL", self.FAIL),
            (f"amd-smi static --gpu {len(self.common.processors)}", self.FAIL),
            (f"amd-smi static --gpu {bad_bdf}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['bdf'][:-1]}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['bdf'] + '0'}", self.FAIL),
            (f"amd-smi static --gpu {bad_uuid}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['uuid'][:-1]}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['uuid'] + '0'}", self.FAIL),
            # Test invalid loglevel
            ("amd-smi metric --loglevel DDEBUG", self.FAIL),
            ("amd-smi metric --loglevel DEBUGG", self.FAIL),
            ("amd-smi metric --loglevel BADLEVEL", self.FAIL),
            # Test invalid set options
            ("amd-smi set", self.FAIL),
            ("amd-smi set --fan", self.FAIL),
            ("amd-smi set --fan 500", self.FAIL),
            # ('amd-smi set --fan 150%', self.FAIL),  # requires y/n answer
            ("amd-smi set --perf-level", self.FAIL),
            ("amd-smi set --perf-level INVALID", self.FAIL),
            ("amd-smi set --profile", self.FAIL),
            ("amd-smi set --profile INVALID", self.FAIL),
            ("amd-smi set --perf-determinism", self.FAIL),
            ("amd-smi set --compute-partition", self.FAIL),
            ("amd-smi set --compute-partition INVALID", self.FAIL),
            ("amd-smi set --memory-partition", self.FAIL),
            ("amd-smi set --memory-partition NPS3", self.FAIL),
            ("amd-smi set --memory-partition INVALID", self.FAIL),
            ("amd-smi set --process-isolation", self.FAIL),
            ("amd-smi set --process-isolation 2", self.FAIL),
            ("amd-smi set --clk-limit", self.FAIL),
            ("amd-smi set --clk-limit INVALID", self.FAIL),
            ("amd-smi set --clk-limit SCLK INVALID", self.FAIL),
            ("amd-smi set --clk-limit MCLK INVALID", self.FAIL),
            ("amd-smi set --clk-limit SCLK MIN", self.FAIL),
            ("amd-smi set --clk-limit MCLK MAX", self.FAIL),
            ("amd-smi set --clk-level", self.FAIL),
            ("amd-smi set --clk-level INVALID", self.FAIL),
            ("amd-smi set --clk-level SCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level MCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level FCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level SOCCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level PCIE INVALID", self.FAIL),
            # Test invalid process PID, NAME
            ("amd-smi process --name", self.FAIL),
            ("amd-smi process --pid", self.FAIL),
            ("amd-smi process --pid NOT_A_NUMBER", self.FAIL),
            # Test invalid ras options
            ("amd-smi ras", self.FAIL),
            ("amd-smi ras --cper INVALID", self.FAIL),
            ("amd-smi ras --cper --severity INVALID", self.FAIL),
            ("amd-smi ras --afid", self.FAIL),
            ("amd-smi ras --afid INVALID", self.FAIL),
            # Test invalid watch order
            ("amd-smi monitor --interval 2 --watch 1", self.FAIL),
            ("amd-smi monitor --watch_time 2 --watch 1", self.FAIL),
        ]

        for index, _ in enumerate(self.common.processors):
            # Test invalid power-cap values
            cmds.append((f"amd-smi set --power-cap --gpu {index}", self.FAIL))
            for power_type in self.power_types:
                cmds.append((f"amd-smi set --power-cap {power_type} --gpu {index}", self.FAIL))
                _power_type = self.static_data["gpu_data"][index]["limit"][power_type]
                socket_power_limit = _power_type["socket_power_limit"]
                if socket_power_limit != "N/A":
                    min_power = _power_type["min_power_limit"]["value"]
                    if min_power > 0:
                        cmds.append(
                            (
                                f"amd-smi set --power-cap {min_power - 1} {power_type} --gpu {index}",
                                self.FAIL,
                            )
                        )
                    max_power = _power_type["max_power_limit"]["value"]
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {max_power + 1} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {int(max_power * 1.10)} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )

            # Test invalid soc-pstate values
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                cmds.append((f"amd-smi set --soc-pstate --gpu {index}", self.FAIL))
                num_supported = int(soc_pstate["num_supported"])
                cmds.append((f"amd-smi set --soc-pstate {num_supported} --gpu {index}", self.FAIL))

            # Test invalid xgmi-plpd values
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                cmds.append((f"amd-smi set --xgmi-plpd --gpu {index}", self.FAIL))
                num_supported = int(xgmi_plpd["num_supported"])
                cmds.append((f"amd-smi set --xgmi-plpd {num_supported} --gpu {index}", self.FAIL))

        self.RunCmds(cmds)
        return

    def test_default(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi"
        self.common.print(msg)

        cmds = [("amd-smi", self.PASS)]

        self.RunCmds(cmds)
        return

    def test_version(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi version"
        self.common.print(msg)

        cmds = [
            ("amd-smi version", self.PASS),
            ("amd-smi version --cpu_version", self.PASS),
            ("amd-smi version --gpu_version", self.PASS),
        ]

        self.RunCmds(cmds)
        return

    def test_list(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi list"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "list", "List Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
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

    def test_bad_pages(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi bad-pages"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "bad-pages", "Bad Pages Arguments:", "Device Arguments:", "Command Modifiers:", ""
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

    def test_event(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi event"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "event", "Event Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_topology(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi topology"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "topology", "Topology arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_set(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set"
        self.common.print(msg)

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
            # set --fan defaults
            fan_speed = self.metric_data["gpu_data"][index]["fan"]["speed"]
            if fan_speed != "N/A":
                cmds.append((f"amd-smi set --fan {fan_speed} --gpu {index}", self.PASS))

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
                freq_entry = clock_sys["frequency_levels"][level]
                clock_freq = int(
                    freq_entry["value"]
                    if isinstance(freq_entry, dict)
                    else freq_entry.split()[0].strip()
                )
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
                    if isinstance(clk_type_limit_name, dict):
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
                    if isinstance(pcie_levels, dict):
                        value = len(pcie_levels)
                        if value > 0:
                            value -= 1
                if clk_type != "PCIE" and value < 0:
                    clk_type_name = clock[clk_type_name]
                    if isinstance(clk_type_name, dict):
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

        self.common.print("Restore Starting Values")
        self.RunCmds(cmds)

        return

    def test_reset(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi reset"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "reset", "Reset Arguments:", "Device Arguments:", "Command Modifiers:", ""
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

    def test_monitor_serial(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor"
        self.common.print(msg)

        cmd = "amd-smi monitor --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)
        (rc, data2, std_err) = self.util.RunCmdSync(cmd)
        cmd = "amd-smi metric --json"
        (rc, data3, std_err) = self.util.RunCmdSync(cmd)
        monitor1 = json.loads(data1)
        monitor2 = json.loads(data2)
        metric3 = json.loads(data3)

        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_fails = self._compare_monitor_metric_data("Monitor", data)

        data = self._get_monitor_metric_data(monitor2, None, metric3)
        metric_fails = self._compare_monitor_metric_data("Metric", data)

        failures = monitor_fails + metric_fails
        if len(failures) > 0:
            self._PrintResults(failures, fail_on_results=True)
        return

    def test_monitor_parallel(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor"
        self.common.print(msg)

        def _Process(q, cmd):
            if my_args.diagnostic == "DEBUG":
                print(f"_Process pid={os.getpid()} received: cmd={cmd}")

            # Receive timestamp
            time_stamp = q.get()

            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            time_stamp2 = time.monotonic()
            q.put(data)
            q.put(time_stamp2)
            return

        # Setup queue between processes
        q = multiprocessing.Queue()

        # Monitor to Monitor
        cmd = "amd-smi monitor --json"
        p1 = multiprocessing.Process(target=_Process, args=(q, cmd))
        p1.start()
        # Send time_stamp
        time_stamp = time.monotonic()
        # print(f"Producer pid={os.getpid()}  sending: {time_stamp}")
        q.put(time_stamp)

        # Get monitor data
        cmd = "amd-smi monitor --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)
        time_stamp = time.monotonic()

        # Receive process data and time_stamp
        data2 = q.get()
        time_stamp_process = q.get()
        p1.join()

        if my_args.diagnostic == "DEBUG":
            print(f"Collection TimeStamp: Monitor2={time_stamp_process}  Monitor1={time_stamp}")
            print(f"          Difference: {abs(time_stamp_process - time_stamp)} seconds")

        monitor1 = json.loads(data1)
        monitor2 = json.loads(data2)
        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_fails = self._compare_monitor_metric_data("Monitor", data)

        # Monitor to Metric
        cmd = "amd-smi monitor --json"
        p1 = multiprocessing.Process(target=_Process, args=(q, cmd))
        p1.start()
        # Send time_stamp
        time_stamp = time.monotonic()
        # print(f"Producer pid={os.getpid()}  sending: {time_stamp}")
        q.put(time_stamp)

        # Get metric data
        cmd = "amd-smi metric --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)
        time_stamp = time.monotonic()

        # Receive process data and time_stamp
        data2 = q.get()
        time_stamp_process = q.get()
        p1.join()

        if my_args.diagnostic == "DEBUG":
            print(f"Collection TimeStamp: Monitor={time_stamp_process}  Metric={time_stamp}")
            print(f"          Difference: {abs(time_stamp_process - time_stamp)} seconds")

        monitor = json.loads(data2)
        metric3 = json.loads(data1)
        data = self._get_monitor_metric_data(monitor, None, metric3)
        metric_fails = self._compare_monitor_metric_data("Metric", data)

        # Report failures
        failures = monitor_fails + metric_fails
        if len(failures) > 0:
            self._PrintResults(failures, fail_on_results=True)
        return

    def test_monitor_with_workload(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor"
        self.common.print(msg)

        def _Process(q, cmd):
            if my_args.diagnostic == "DEBUG":
                print(f"_Process pid={os.getpid()} received: cmd={cmd}")

            # Receive timestamp
            time_stamp = q.get()

            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            time_stamp2 = time.monotonic()
            q.put(data)
            q.put(time_stamp2)
            return

        # Setup queue between processes
        q = multiprocessing.Queue()

        # Get baseline monitor data
        cmd = "amd-smi monitor --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)

        # Monitor to Monitor
        cmd = "rvs --json"
        p1 = multiprocessing.Process(target=_Process, args=(q, cmd))
        p1.start()
        # Send time_stamp
        time_stamp = time.monotonic()
        # print(f"Producer pid={os.getpid()}  sending: {time_stamp}")
        q.put(time_stamp)

        # Get monitor data under workload
        time.sleep(4)
        cmd = "amd-smi monitor --json"
        (rc, data2, std_err) = self.util.RunCmdSync(cmd)
        time_stamp = time.monotonic()

        # Receive process data and time_stamp
        process_data = q.get()
        print(process_data)
        process_time_stamp = q.get()
        p1.join()

        if my_args.diagnostic == "DEBUG":
            print(f"Collection TimeStamp: Monitor2={time_stamp_process}  Monitor1={time_stamp}")
            print(f"          Difference: {abs(time_stamp_process - time_stamp)} seconds")

        monitor1 = json.loads(data1)
        monitor2 = json.loads(data2)
        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_fails = self._compare_monitor_metric_data("Workload", data)

        # Report failures
        failures = monitor_fails
        if len(failures) > 0:
            self._PrintResults(failures, fail_on_results=True)
        return

    def test_xgmi(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi xgmi"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "xgmi", "XGMI arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_partition(self):
        """
                TODO: Can we add a test case for changing partitions while processes are accessing the GPUs. We can add the negative and the positive test case to show the behavior that processes need to be killed before partition changes.

        We need to make sure the messaging is correct, and extra credit idea would be to print out the PIDs that are currently testing.
        """
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi partition"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "partition", "Partition arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_ras(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi ras"
        self.common.print(msg)

        if not my_args.printCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = "Not Yet Implemented"
                self.common.print(msg)
                self.skipTest(msg)

        cmds = self.CreateCmds(
            "ras", "RAS arguments:", "CPER Arguments", "Device Arguments:", "Command Modifiers:"
        )
        self.RunCmds(cmds)
        return

    def test_node(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi node"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "node", "Node arguments:", "Device Arguments:", "Command Modifiers:", ""
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


if __name__ == "__main__":
    # exit_on_error was added in Python 3.9; fall back gracefully on older versions
    _parser_kwargs = {"add_help": False}
    if sys.version_info >= (3, 9):
        _parser_kwargs["exit_on_error"] = False
    parser = argparse.ArgumentParser(**_parser_kwargs)
    parser.add_argument(
        "--diagnostic",
        choices=DIAGNOSTIC_CHOICES,
        type=str,
        default="WARNING",
        help="Level of information to output, default=%(default)s",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="file for output, default=%(default)s"
    )
    parser.add_argument(
        "--printStreamInfo",
        action="store_true",
        default=False,
        help="Print where output is streamed, default=%(default)s",
    )
    parser.add_argument(
        "--printCmdsOnly",
        action="store_true",
        default=False,
        help="Print cmds only, default=%(default)s",
    )
    parser.add_argument(
        "--useAllCmdOptions",
        action="store_true",
        default=False,
        help="Use all cmd option combinations, default=%(default)s",
    )
    parser.add_argument(
        "--addLogLevel",
        choices=DIAGNOSTIC_CHOICES,
        type=str,
        default="",
        help="add --loglevel to cmd",
    )
    my_args, remaining = parser.parse_known_args(sys.argv[1:])

    my_args.reduceCmds = not my_args.useAllCmdOptions
    my_args.diagnostic_index = DIAGNOSTIC_CHOICES.index(my_args.diagnostic)
    my_args.verbose = common.VERBOSITY_NORMAL

    # Print argparse and unittest help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("AMD-SMI CLI Options:")
        parser.print_help()
        print("\nUnit tests Options:")
        unittest.main(
            argv=["unittest", "-h"],
            exit=False,
            testRunner=unittest.TextTestRunner(stream=sys.stdout),
        )
        sys.exit(0)

    # Detect if ran without sudo or root privileges
    # Run if only printing commands
    if os.geteuid() != 0 and not my_args.printCmdsOnly:
        print(
            "Some tests may require elevated privileges (sudo/root) to run completely.\n",
            file=sys.stderr,
        )
        print("Please relaunch with elevated privileges.\n", file=sys.stderr)
        sys.exit(1)

    # Parse verbosity from command line (updates the module-level default)
    if "-q" in sys.argv or "--quiet" in sys.argv:
        my_args.verbose = common.VERBOSITY_QUIET
    elif "-v" in sys.argv or "--verbose" in sys.argv:
        my_args.verbose = common.VERBOSITY_VERBOSE

    if my_args.verbose > common.VERBOSITY_QUIET:
        print("AMD SMI CLI Unit Tests\n", file=sys.stdout)

    # If no -k or --keyword argument is given, print all available tests
    if not ("-k" in sys.argv or "--keyword" in sys.argv):
        if my_args.verbose > common.VERBOSITY_QUIET:
            common.print_tests(__name__)

    # Print legend only when progress chars (., s, F, E), only show if in quiet mode
    if my_args.verbose == common.VERBOSITY_NORMAL:
        common.print_legend()

    verbosity_str = {
        common.VERBOSITY_QUIET: "QUIET",
        common.VERBOSITY_NORMAL: "NORMAL",
        common.VERBOSITY_VERBOSE: "VERBOSE",
    }

    print(f"Running with verbosity: {verbosity_str[my_args.verbose]}\n", file=sys.stderr)

    sys.argv[1:] = remaining
    unittest.main()
    sys.exit(0)
