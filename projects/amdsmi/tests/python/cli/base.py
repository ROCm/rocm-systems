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
"""Shared base class for CLI unit tests.

Provides __init__, FindArgs, CreateCmds, and RunCmds — the instance-level
helpers that all CLI test classes need.  Each concrete test class inherits
from TestCliBase and provides its own setUpClass / test_* methods.
"""

import json
import os
import stat
import unittest


class TestCliBase(unittest.TestCase):
    """Base class for CLI functional tests.

    Subclasses must define a setUpClass that sets cls.common, cls.util, and
    the JSON data attributes (cls.list_data, cls.static_data, cls.metric_data,
    cls.partition_data) before any test method runs.
    """

    TMP_FILENAME = "_tmp.log"
    TMP_FOLDER = "_tmp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Debug = False
        self.ReduceCmds = True
        self.PrintCmdsOnly = False

        self.AddCmdMods = True
        self.AddDeviceArgs = True
        self.AddWatchArgs = True
        self.AddLogLevel = "--loglevel DEBUG"

        self.PASS = 0
        self.FAIL = 1
        self.tab = "    "
        self.tmp_filename = self.TMP_FILENAME
        self.tmp_folder = self.TMP_FOLDER

        self.openBracket = "["
        self.closeBracket = "]"
        self.openCurlyBrace = "{"
        self.closeCurlyBrace = "}"

        self.perf_levels = [
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
        self.profile_levels = [
            "CUSTOM_MASK",
            "VIDEO_MASK",
            "POWER_SAVING_MASK",
            "COMPUTE_MASK",
            "VR_MASK",
            "THREE_D_FULL_SCR_MASK",
            "BOOTUP_DEFAULT",
        ]
        self.compute_partition_modes = ["SPX", "DPX", "TPX", "QPX", "CPX"]
        self.memory_partition_modes = ["NPS1", "NPS2", "NPS4", "NPS8"]
        self.power_types = ["ppt0", "ppt1"]
        self.ptl_formats = ["I8", "F16", "BF16", "F32", "F64", "F8", "VECTOR"]
        self.clk_limits = ["SCLK", "MCLK"]
        self.limit_types = ["MIN", "MAX"]
        self.clk_levels = ["SCLK", "MCLK", "FCLK", "SOCCLK", "PCIE"]

        # When parsing, ignore these entries as they are abnormal
        self.cmd_arg_exceptions = ["--voltage"]

        # When parsing, change these args into something else or add to arg
        self.cmd_arg_changes = [
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

    def setUp(self):
        # Called before each test by unittest framework
        return

    def tearDown(self):
        # Called after each test by unittest framework
        return

    def FindArgs(self, cmd, match_str):
        if (
            (not match_str)
            or (not self.AddDeviceArgs and "Device" in match_str)
            or (not self.AddWatchArgs and "Watch" in match_str)
            or (not self.AddCmdMods and "Command" in match_str)
        ):
            return ["pass"]

        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        lines = std_out.split("\n")

        found = False
        options = []
        for index, line in enumerate(lines):
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
                            options.append(f"{{json}}")
                            options.append(f"{{json_file}}")
                            options.append(f"{{json_file_append}}")
                            options.append(f"{{json_file_overwrite}}")
                        elif "--csv" == items[item_index]:
                            options.append(f"{{csv}}")
                            options.append(f"{{csv_file}}")
                            options.append(f"{{csv_file_append}}")
                            options.append(f"{{csv_file_overwrite}}")
                        elif "--append" == items[item_index] or "--overwrite" == items[item_index]:
                            pass
                        elif "--watch" == items[item_index]:
                            options.append(f"{{watch_time}}")
                            options.append(f"{{watch_iterations}}")
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
                                options.append(f"{{perf_determinism}}")
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
        if self.Debug:
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

        # Calculate and substitute in dependent values; remove invalid cmds
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            while self.openCurlyBrace in cmd:
                items = cmd.split()
                # Find gpu index and mark when gpu=0
                try:
                    i = items.index("--gpu")
                    gpu = items[i + 1]
                    gpu_index = int(gpu) if gpu.isdigit() else 0
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
                    if nameStr == "{watch_time}":
                        cmd = cmd.replace(nameStr, "--watch 1 --watch_time 2", 1)
                    else:
                        cmd = cmd.replace(nameStr, "--watch 1 --iterations 2", 1)
                elif (
                    nameStr == "{min_power}" or nameStr == "{avg_power}" or nameStr == "{max_power}"
                ):
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
                        clock_freq = int(clock_sys["frequency_levels"][level].split()[0].strip())
                        cmd = cmd.replace(
                            "{perf_determinism}", f"--perf-determinism {clock_freq + 50}", 1
                        )
                    else:
                        cmd = ""
                elif "clk_limit" in nameStr:
                    clock = self.metric_data["gpu_data"][gpu_index]["clock"]
                    if nameStr == "{clk_limit_sclk_min}":
                        clk_type, clk_type_name, limit_type, clk_limit_name = (
                            "SCLK",
                            "socclk_0",
                            "MIN",
                            "min_clk",
                        )
                    elif nameStr == "{clk_limit_sclk_max}":
                        clk_type, clk_type_name, limit_type, clk_limit_name = (
                            "SCLK",
                            "socclk_0",
                            "MAX",
                            "max_clk",
                        )
                    elif nameStr == "{clk_limit_mclk_min}":
                        clk_type, clk_type_name, limit_type, clk_limit_name = (
                            "MCLK",
                            "mem_0",
                            "MAX",
                            "min_clk",
                        )
                    elif nameStr == "{clk_limit_mclk_max}":
                        clk_type, clk_type_name, limit_type, clk_limit_name = (
                            "MCLK",
                            "mem_0",
                            "MIN",
                            "max_clk",
                        )
                    clk_type_limit_name = clock[clk_type_name][clk_limit_name]
                    if type(clk_type_limit_name) is dict:
                        value = clk_type_limit_name["value"]
                        cmd = cmd.replace(nameStr, f"{clk_type} {limit_type} {value}", 1)
                    else:
                        cmd = ""
                elif "clk_level" in nameStr:
                    clock = self.static_data["gpu_data"][gpu_index]["clock"]
                    value = -1
                    clk_type = ""
                    clk_type_name = ""
                    if nameStr == "{clk_level_sclk}":
                        clk_type, clk_type_name = "SCLK", "sys"
                    elif nameStr == "{clk_level_mclk}":
                        clk_type, clk_type_name = "MCLK", "mem"
                    elif nameStr == "{clk_level_fclk}":
                        clk_type, clk_type_name = "FCLK", "df"
                    elif nameStr == "{clk_level_socclk}":
                        clk_type, clk_type_name = "SOCCLK", "soc"
                    elif nameStr == "{clk_level_pcie}":
                        bus = self.static_data["gpu_data"][gpu_index]["bus"]
                        clk_type = "PCIE"
                        pcie_levels = bus["pcie_levels"]
                        if type(pcie_levels) is dict:
                            value = len(pcie_levels)
                            if value > 0:
                                value = 0
                    if clk_type != "PCIE" and value < 0:
                        clk_entry = clock[clk_type_name]
                        if type(clk_entry) is dict:
                            current_level = clk_entry["current_level"]
                            freq_levels = clk_entry["frequency_levels"]
                            value = len(freq_levels) - 1 if current_level == 0 else 0
                    if value >= 0:
                        cmd = cmd.replace(nameStr, f"{clk_type} {value}", 1)
                    else:
                        cmd = ""
                elif nameStr == "{soc_pstate}":
                    soc_pstate = self.static_data["gpu_data"][gpu_index]["soc_pstate"]
                    if type(soc_pstate) is dict:
                        num_supported = int(soc_pstate["num_supported"])
                        if num_supported > 0:
                            current = int(soc_pstate["current_id"])
                            num = num_supported - 1 if current == 0 else 0
                            cmd = cmd.replace(nameStr, f"{num}", 1)
                        else:
                            cmd = ""
                    else:
                        cmd = ""
                elif nameStr == "{xgmi_plpd}":
                    xgmi_plpd = self.static_data["gpu_data"][gpu_index]["xgmi_plpd"]
                    if type(xgmi_plpd) is dict:
                        num_supported = int(xgmi_plpd["num_supported"])
                        if num_supported > 0:
                            current = int(xgmi_plpd["current_id"])
                            num = num_supported - 1 if current == 0 else 0
                            cmd = cmd.replace(nameStr, f"{num}", 1)
                        else:
                            cmd = ""
                    else:
                        cmd = ""
            cmds[index] = (cmd, cond)

        # Pare down commands
        if self.ReduceCmds:
            file_mods = ["--file", "--json", "--csv"]
            watch_mods = ["--watch", "--watch_time", "--iterations"]

            found_sub_arg = False
            for index, cmd_cond in enumerate(cmds):
                cmd, cond = cmd_cond
                items = cmd.split()

                if not found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    for mod in file_mods + ["--gpu", "--loglevel"]:
                        if mod == sub_arg:
                            sub_arg = ""
                            break
                    found_sub_arg = sub_arg

                gpu_index = "0"
                if "--gpu" in cmd:
                    try:
                        i = items.index("--gpu")
                        gpu_index = items[i + 1]
                    except ValueError:
                        pass

                if cmd and found_sub_arg:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        if "--gpu" in cmd:
                            cmd = ""

                if cmd and gpu_index != "0":
                    for mod in file_mods + watch_mods:
                        if mod in cmd:
                            cmd = ""
                            break

                if cmd and "--file" in cmd and "--watch" in cmd:
                    cmd = ""

                if cmd and found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        if "--watch" in cmd:
                            cmd = ""

                if cmd and found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        for mod in file_mods:
                            if mod in cmd:
                                cmd = ""
                                break

                cmds[index] = (cmd, cond)

        # Remove empty entries and normalise whitespace
        cmds = [cmd_cond for cmd_cond in cmds if cmd_cond[0] != ""]
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            cmd = " ".join(cmd.split()).strip()
            cmds[index] = (cmd, cond)

        if self.Debug:
            print(f"cmds: {'*' * 80}")
            print(json.dumps(cmds, sort_keys=False, indent=4), flush=True)
        return cmds

    def RunCmds(self, cmds):
        errors = []
        msg_len = max((len(cmd) for cmd, _ in cmds), default=0) + 2
        for cmd, cond in cmds:
            if self.Debug or self.PrintCmdsOnly:
                print(f"cmd={cmd}")
            if self.PrintCmdsOnly:
                continue
            (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
            error_code = rc
            if rc and std_err:
                items = std_err.split()
                if "amdsmi_exception" in std_err:
                    for index, item in enumerate(items):
                        if item == "Error":
                            error_code = items[index + 4]
                else:
                    error_code = items[-1]
                    if "CRITICAL" in error_code:
                        error_code = "Bad loglevel"

            msg = f"{cmd:{msg_len}s}:"
            if "--file" in cmd:
                if not os.path.exists(self.tmp_filename):
                    errors.append(f"{msg} Failure: File {self.tmp_filename} does not exist")
                else:
                    with open(self.tmp_filename, "r") as fin:
                        std_out = fin.read()
                    if not len(std_out):
                        errors.append(f"{msg} Failure: File {self.tmp_filename} was empty")
                    os.chmod(self.tmp_filename, stat.S_IWRITE)
                    os.remove(self.tmp_filename)

            if rc and cond == self.PASS:
                msg += f" Failure: Received FAIL ({error_code}), expected PASS (0)"
                errors.append(msg)
            elif not rc and cond != self.PASS:
                msg += f" Failure: Received PASS (0), expected FAIL (!0)"
                errors.append(msg)
            else:
                expected = "PASS" if not rc else "FAIL"
                msg += f" Success: Received and Expected {expected} ({error_code})"

            self.common.print(f"{self.tab}{msg}")
            if self.Debug:
                print(f"{self.tab}rc={rc}")
                print(f"{self.tab}error_code={error_code}")
                print(f"{self.tab}std_out={std_out}")
                print(f"{self.tab}std_err={std_err}")

        if errors:
            msg = f"\n{self.tab}".join(errors)
            self.fail(f"Fail:\n{self.tab}{msg}")
