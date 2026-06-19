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
"""Shared base class for the CLI unit tests.

``TestCliBase`` provides the instance-level scaffolding every CLI test
needs -- ``__init__`` (command constants), ``FindArgs``, ``CreateCmds``
and ``RunCmds``.  Extracted from develop's monolithic ``TestAmdSmiCli``
so the per-feature CLI suites can share it: the per-command ``cli/test_*.py``
files each subclass ``TestCliBase`` and add their own test_* methods.
"""

import json
import os
import stat
import unittest

import common.common as common
import common.runcmd as runcmd


class TestCliBase(unittest.TestCase):
    """Base class for CLI functional tests.

    The base now provides a cached ``setUpClass`` that fetches the ``--json``
    baseline (metric/static/list/partition data, the derived ``gpus`` list and
    the ``sub_args`` dict) exactly once and shares it across all CLI test
    classes, so subclasses only need to add their own test_* methods.
    """

    TMP_FILENAME = "_tmp.log"
    TMP_FOLDER = "_tmp"

    # Scaffolding shared across every CLI test class.  setUpClass populates
    # these once, directly on ``TestCliBase``; subclasses then resolve them
    # natively through normal attribute inheritance -- no dynamic ``setattr``
    # for a type checker (or go-to-definition) to lose track of.  Declared here
    # so editors / type checkers can resolve ``self.util``, ``self.static_data``.
    common: "common.Common"
    util: "runcmd.Util"
    list_data: dict
    static_data: dict
    metric_data: dict
    partition_data: dict
    gpus: list
    sub_args: dict

    # Built lazily (not at import like common.py's pure, enum-derived parameter
    # lists) because the --json baseline needs a real GPU and root.  Still built
    # only once for the whole CLI suite, then inherited by every command class;
    # this flag guards the first-and-only initialization.
    _initialized = False

    @classmethod
    def setUpClass(cls):
        if TestCliBase._initialized:
            return
        TestCliBase.common = common.Common(common.verbose)
        TestCliBase.util = runcmd.Util("WARNING")
        # Print the per-device header (virtualization mode, asic and board
        # info) once, before any CLI test class runs, rather than per test.
        for i, _ in enumerate(TestCliBase.common.processors):
            TestCliBase.common.print_device_header(i)

        baseline = cls._build_baseline()
        TestCliBase.metric_data = baseline["metric_data"]
        TestCliBase.static_data = baseline["static_data"]
        TestCliBase.list_data = baseline["list_data"]
        TestCliBase.partition_data = baseline["partition_data"]
        TestCliBase.gpus = baseline["gpus"]
        TestCliBase.sub_args = baseline["sub_args"]
        TestCliBase._initialized = True

    @classmethod
    def _build_baseline(cls):
        baseline = {}

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
                baseline[f"{name}_data"] = json.loads(data)
            except (json.JSONDecodeError, TypeError) as e:
                # TODO(amdsmi_team): Known issue — several AI NIC and CPU commands can produce
                # malformed JSON/CSV/error output, causing parsing & other failures.
                # We need to log tickets on these issues.

                # Log warning but continue — malformed JSON output is a CLI bug,
                # not a test infrastructure failure; tests that depend on this
                # data will fail individually with a KeyError pointing to the
                # missing key, making the root cause clear.
                cls.common.print(f'\n\tERROR: Could not parse JSON from "{cmd}": {e}')
                baseline[f"{name}_data"] = {}

        gpus = ["all"]
        for entry in baseline["list_data"]:
            gpus.append(entry["gpu"])
            if entry["gpu"] == 0:
                # Only test bdf and uuid when gpu=0
                gpus.append(entry["bdf"])
                gpus.append(entry["uuid"])
        baseline["gpus"] = gpus

        # When parsing, expand each arg with array element
        baseline["sub_args"] = {
            "CLOCK": ["SYS", "DF", "DCEF", "SOC", "MEM", "VCLK0", "VCLK1", "DCLK0", "DCLK1", "ALL"],
            "PID": [123],
            "NAME": ["AMD"],
            "GPU": gpus,
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
        return baseline

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

        return

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
        lines = std_out.split("\n") if std_out else []

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
                            elif sub_arg == "MODE":  # arg --compute-partition-mem-alloc-mode
                                for mem_alloc_mode in ["CAPPING", "ALL"]:
                                    options.append(f"{items[item_index]} {mem_alloc_mode}")
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
                                pass
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

        # Calculate and substitute in dependent values
        # Removes cmds that are invalid
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            while self.openCurlyBrace in cmd:
                items = cmd.split()
                # Find gpu index
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
                        or not isinstance(power_type, dict)
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
                    clk_type = clk_type_name = limit_type = clk_limit_name = ""
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
                    if type(clk_type_limit_name) is dict:
                        value = clk_type_limit_name["value"]
                        cmd = cmd.replace(nameStr, f"{clk_type} {limit_type} {value}", 1)
                    else:
                        cmd = ""
                elif "clk_level" in nameStr:
                    clock = self.static_data["gpu_data"][gpu_index]["clock"]
                    value = -1
                    clk_type = clk_type_name = ""
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
                        if type(pcie_levels) is dict:
                            value = len(pcie_levels)
                            if value > 0:
                                value = 0
                    if clk_type != "PCIE" and value < 0:
                        clk_type_name = clock[clk_type_name]
                        if type(clk_type_name) is dict:
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
                    if type(soc_pstate) is dict:
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
                    if type(xgmi_plpd) is dict:
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

        # Pare down commands
        if self.ReduceCmds:
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
        if self.Debug:
            print(f"cmds: {'*' * 80}")
            print(json.dumps(cmds, sort_keys=False, indent=4), flush=True)
        return cmds

    def RunCmds(self, cmds):
        errors = []
        msg_len = 0
        for cmd, cond in cmds:
            num = len(cmd)
            if num > msg_len:
                msg_len = num
        msg_len += 2
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
                    # error code from amdsmi library exception
                    for index, item in enumerate(items):
                        if item == "Error":
                            error_code_str = items[index + 4]
                            error_code = error_code_str
                            # break
                else:
                    # error code from amd-smi CLI
                    error_code = items[-1]
                    # Check for parse error 'choice'
                    if "CRITICAL" in error_code:
                        error_code = "Bad loglevel"

            msg = f"{cmd:{msg_len}s}:"
            if "--file" in cmd:
                if not os.path.exists(self.tmp_filename):
                    _msg = f"{msg} Failure: File {self.tmp_filename} does not exist"
                    errors.append(_msg)
                else:
                    with open(self.tmp_filename, "r") as fin:
                        std_out = fin.read()
                    if not len(std_out):
                        _msg = f"{msg} Failure: File {self.tmp_filename} was empty"
                        errors.append(_msg)
                    os.chmod(self.tmp_filename, stat.S_IWRITE)
                    os.remove(self.tmp_filename)

            if rc and cond == self.PASS:
                msg += f" Failure: Received FAIL ({error_code}), expected PASS (0)"
                errors.append(msg)
            elif not rc and cond != self.PASS:
                msg += " Failure: Received PASS (0), expected FAIL (!0)"
                errors.append(msg)
            else:
                if not rc:
                    expected = "PASS"
                else:
                    expected = "FAIL"
                msg += f" Success: Received and Expected {expected} ({error_code})"

            self.common.print(f"{self.tab}{msg}")
            if self.Debug:
                print(f"{self.tab}rc={rc}")
                print(f"{self.tab}error_code={error_code}")
                print(f"{self.tab}std_out={std_out}")
                print(f"{self.tab}std_err={std_err}")
        if len(errors):
            msg = f"\n{self.tab}".join(errors)
            self.fail(f"Fail:\n{self.tab}{msg}")
        return
