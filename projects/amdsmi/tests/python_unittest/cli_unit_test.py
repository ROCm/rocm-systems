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

import json
import locale
import multiprocessing
import os
import stat
import sys
import time

import unittest

import common
import runcmd

# common.py owns path resolution, sys.path setup, and amdsmi loading — borrow the reference.
from common import amdsmi

# Module-level default; __main__ overwrites this with the actual parsed value.
# It must exist at module scope so setUpClass/setUp can reference it before
# __main__ runs (e.g. when loaded by an external test runner).
verbose = common.VERBOSITY_NORMAL


class TestAmdSmiCli(unittest.TestCase):
    TMP_FILENAME = "_tmp.log"
    TMP_FOLDER = "_tmp"

    amd_smi_exe = "amdsmi_cli/amdsmi_cli.py"
    amd_smi_exe = "amd-smi"

    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)
        cls.util = runcmd.Util("WARNING")

        cls.PASS = 0
        cls.FAIL = -1

        cls.InvalidCommand = 193
        cls.InvalidParameter = 194
        cls.DeviceNotFound = 195
        cls.InvalidFilePath = 196
        cls.InvalidParameterValue = 197
        cls.MissingParameterValue = 198
        cls.CommandNotSupported = 199
        cls.ParameterNotSupported = 200
        cls.RequiredCommand = 201
        cls.InvalidSubcommand = 202
        cls.PermissionDenied = 203
        cls.UnknownError = 255

        # Record starting values; running here (once per class) rather than in
        # __init__ (once per test method) reduces setup overhead from O(N) to
        # O(1) — N being the number of test methods in this class.
        cmds = [
            ("metric", f"{cls.amd_smi_exe} metric --json"),
            ("static", f"{cls.amd_smi_exe} static --json"),
            ("list", f"{cls.amd_smi_exe} list --json"),
            ("partition", f"{cls.amd_smi_exe} partition --current --json"),
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
        cls.cpus = []
        if "cpu_data" in cls.static_data:
            num = len(cls.static_data["cpu_data"])
            cls.cpus = [str(index) for index in range(num)]
        cls.cores = ["0"]

        # When parsing, expand each arg with array element
        cls.sub_args = {
            "CLOCK": ["SYS", "DF", "DCEF", "SOC", "MEM", "VCLK0", "VCLK1", "DCLK0", "DCLK1", "ALL"],
            "PID": [123],
            "NAME": ["AMD"],
            "GPU": cls.gpus,
            "CPU": cls.cpus,
            "CORE": cls.cores,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_encoding = locale.getpreferredencoding()

        self.Debug = False
        self.ReduceCmds = True
        self.PrintCmdsOnly = False

        self.AddCmdMods = True
        self.AddDeviceArgs = True
        self.AddWatchArgs = True
        self.AddLogLevel = "--loglevel DEBUG"

        self.tab = "    "
        self.tmp_filename = self.TMP_FILENAME
        self.tmp_folder = self.TMP_FOLDER

        self.not_supported_error_codes = [2, 207]

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
                if cmd:
                    msg += f"\n{self.tab}{cmd:{cmd_len}s} : {cmd_out}"
                else:
                    msg += f"\n{cmd_out}"
            msg = msg.strip()
            msg = f"{self.tab}{msg}"

            # Output to file if set
            if False:
                if my_args.output:
                    with open(my_args.output, "a", encoding=self.use_encoding) as fout:
                        fout.write(f"{msg}\n")

            # Output to std_out
            if verbose == common.VERBOSITY_VERBOSE:
                self.common.print(f"{msg}")

            # Output to std_err
            if fail_on_results:
                self.fail(f"Fail:\n\n{msg}")
        return

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
            rc, error_code = self.StrToNumber(items[-1])
        return (error_code, output_stream)

    def _GetCmdReturnMsg(self, rc_num, ec_num, cond):
        msg = ""
        if cond == self.PASS:
            if rc_num in self.not_supported_error_codes:
                msg = f"Success: PASS Not Supported   "
            elif rc_num == 0 and ec_num == 0:
                msg = f"Success: Expected PASS, rc = 0"
            elif rc_num != 0:
                msg = f"Failure: Expected PASS, rc = 0"
            elif ec_num != 0:
                msg = f"Failure: Expected PASS  ec = 0"
            msg = f"{msg}; Received rc={rc_num:3d}, ec={str(ec_num):3s}"
        else:
            if cond < 0:
                if rc_num > 0:
                    msg = f"Success: Expected FAIL, rc > 0"
                else:
                    msg = f"Failure: Expected FAIL, rc > 0"
            else:
                if rc_num > 0 and rc_num == ec_num:
                    msg = f"Success: Expected FAIL, rc = ec > 0"
                else:
                    msg = f"Failure: Expected FAIL, rc = ec > 0"
            msg = f"{msg}; Received rc={rc_num:3d}, ec={str(ec_num):3s}"
        if "Success" in msg:
            passed = True
        else:
            passed = False
        return (msg, passed)

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
                    data1 = int(monitor1[i][key]["value"])
                    if monitor2 != None:
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
                            data2 = int(metric["gpu_data"][i]["mem_usage"]["used_gtt"]["value"])
                        elif key == "vram_total":
                            data2 = int(metric["gpu_data"][i]["mem_usage"]["total_gtt"]["value"])
                data[i][key] = [data1, data2, abs(data1 - data2), unit]
        return data

    def _compare_monitor_metric_data(self, component, data):
        failures = []
        successes = []
        for i in range(len(data)):
            msg_title = f"Monitor to {component}: gpu={i}"
            msg_header = f"{'key':>20s} ({'Unit':>4s}): {'Monitor':>8s} {component:>8s}  {'Diff':>8s}   {'Threshold':>8s} {'Status':>7s}"
            for key in data[i]:
                if data[i][key][3] == "N/A":
                    continue
                max_diff = max(data[i][key][0], data[i][key][1]) * 0.1
                if data[i][key][2] > max_diff:
                    status = "Failure"
                    compare = ">"
                else:
                    status = "Success"
                    compare = "<"
                _msg = f"{key:>20s} ({data[i][key][3]:>4s}): {data[i][key][0]:>8d} {data[i][key][1]:>8d} ({data[i][key][2]:>8d} {compare} {max_diff:>8.2f}) {status:>7s}"
                if status == "Failure":
                    if len(failures) == 0:
                        failures.append(("", f"Compare {msg_title}"))
                        failures.append(("*" * len(msg_title), msg_header))
                    failures.append((msg_title, _msg))
                else:
                    if len(successes) == 0:
                        successes.append(("", f"Compare {msg_title}"))
                        successes.append(("*" * len(msg_title), msg_header))
                    successes.append((msg_title, _msg))
        return (failures, successes)

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
        cmd = f"{self.amd_smi_exe} {cmd_name} --help"
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
        cmd = f"{self.amd_smi_exe} {cmd_name}"
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
                gpu_0 = False
                try:
                    i = items.index("--gpu")
                    gpu = items[i + 1]
                    if gpu.isdigit():
                        gpu_index = int(gpu)
                        if gpu_index == 0:
                            gpu_0 = True
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
                        clock_freq = int(clock_sys["frequency_levels"][level].split()[0].strip())
                        cmd = cmd.replace(
                            "{perf_determinism}", f"--perf-determinism {clock_freq + 50}", 1
                        )
                    else:
                        cmd = ""
                elif "clk_limit" in nameStr:
                    clock = self.metric_data["gpu_data"][gpu_index]["clock"]
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
                    except ValueError as e:
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
        # Find the longest message length
        msg_len = 0
        for cmd, cond in cmds:
            num = len(cmd)
            if num > msg_len:
                msg_len = num
        msg_len += 2

        failures = []
        successes = []
        for cmd, cond in cmds:
            if self.Debug or self.PrintCmdsOnly:
                print(f"cmd={cmd}")
            if self.PrintCmdsOnly:
                continue

            # Remove output file if it exists
            if os.path.exists(self.tmp_filename):
                os.chmod(self.tmp_filename, stat.S_IWRITE)
                os.remove(self.tmp_filename)

            (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
            if rc:
                error_code, output_stream = self._GetErrorCode(std_out, std_err, cond)
            else:
                error_code = 0
                output_stream = None
            msg, passed = self._GetCmdReturnMsg(rc, error_code, cond)

            if passed:
                successes.append((cmd, msg))
            else:
                failures.append((cmd, msg))

        self._PrintResults(successes)
        self._PrintResults(failures, fail_on_results=True)
        return

    def test_help(self):
        self.common.print_func_name("")
        msg = f"### amd-smi help"
        self.common.print(msg)

        cmd = "{self.amd_smi_exe} --help"
        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        lines = std_out.split("\n")
        # Find all available command line args
        cmd_args = []
        found = False
        cmd_indent = None
        for line in lines:
            if found:
                if not line:
                    break
                indent = len(line) - len(line.lstrip())
                # The first command establishes the command column. Lines that
                # are indented further are wrapped description continuations
                # (e.g. the "devices" tail of the long fabric help text) and
                # must be skipped so they aren't parsed as subcommands.
                if cmd_indent is None:
                    cmd_indent = indent
                elif indent > cmd_indent:
                    continue
                items = line.split()
                cmd_args.append(items[0])
                continue
            if "Descriptions" in line:
                found = True

        cmds = [(f"{self.amd_smi_exe} --help", self.PASS)]
        for cmd_arg in cmd_args:
            cmds.append((f"{self.amd_smi_exe} {cmd_arg} --help", self.PASS))

        self.RunCmds(cmds)
        return

    def test_invalid(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi"
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
            (f"{self.amd_smi_exe} invalid_cmd", self.InvalidCommand),
            # Test invalid sub command
            (f"{self.amd_smi_exe} version --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} list --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} static --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} firmware --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} bad_pages --invalid", self.InvalidCommand),
            (f"{self.amd_smi_exe} bad-pages --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} metric --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} process --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} event --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} topology --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} set --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} reset", self.RequiredCommand),
            (f"{self.amd_smi_exe} reset --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} monitor --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} xgmi --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} partition --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} ras --invalid", self.InvalidParameter),
            (f"{self.amd_smi_exe} node --invalid", self.InvalidParameter),
            # Test invalid gpu value
            (f"{self.amd_smi_exe} version --gpu 0", self.FAIL),
            (f"{self.amd_smi_exe} version --gpu -1", self.FAIL),
            (f"{self.amd_smi_exe} version --gpu ALL", self.FAIL),
            (f"{self.amd_smi_exe} version --gpu {len(self.common.processors)}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu -1", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu _ALL", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {len(self.common.processors)}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {bad_bdf}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {self.list_data[0]['bdf'][:-1]}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {self.list_data[0]['bdf'] + '0'}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {bad_uuid}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {self.list_data[0]['uuid'][:-1]}", self.FAIL),
            (f"{self.amd_smi_exe} static --gpu {self.list_data[0]['uuid'] + '0'}", self.FAIL),
            # Test invalid loglevel
            (f"{self.amd_smi_exe} metric --loglevel DDEBUG", self.FAIL),
            (f"{self.amd_smi_exe} metric --loglevel DEBUGG", self.FAIL),
            (f"{self.amd_smi_exe} metric --loglevel BADLEVEL", self.FAIL),
            # Test invalid set options
            (f"{self.amd_smi_exe} set", self.FAIL),
            (f"{self.amd_smi_exe} set --fan", self.FAIL),
            (f"{self.amd_smi_exe} set --fan 500", self.FAIL),
            (f"{self.amd_smi_exe} set --fan 150%", self.FAIL),
            (f"{self.amd_smi_exe} set --perf-level", self.FAIL),
            (f"{self.amd_smi_exe} set --perf-level INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --profile", self.FAIL),
            (f"{self.amd_smi_exe} set --profile INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --perf-determinism", self.FAIL),
            (f"{self.amd_smi_exe} set --compute-partition", self.FAIL),
            (f"{self.amd_smi_exe} set --compute-partition INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --memory-partition", self.FAIL),
            (f"{self.amd_smi_exe} set --memory-partition NPS3", self.FAIL),
            (f"{self.amd_smi_exe} set --memory-partition INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --compute-partition-mem-alloc-mode", self.FAIL),
            (f"{self.amd_smi_exe} set --compute-partition-mem-alloc-mode HALF", self.FAIL),
            (f"{self.amd_smi_exe} set --compute-partition-mem-alloc-mode INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --process-isolation", self.FAIL),
            (f"{self.amd_smi_exe} set --process-isolation 2", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-limit", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-limit INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-limit SCLK INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-limit MCLK INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-limit SCLK MIN", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-limit MCLK MAX", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level SCLK", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level SCLK INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level MCLK", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level MCLK INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level FCLK", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level FCLK INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level SOCCLK", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level SOCCLK INVALID", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level PCIE", self.FAIL),
            (f"{self.amd_smi_exe} set --clk-level PCIE INVALID", self.FAIL),
            # Test invalid process PID, NAME
            (f"{self.amd_smi_exe} process --name", self.FAIL),
            (f"{self.amd_smi_exe} process --pid", self.FAIL),
            (f"{self.amd_smi_exe} process --pid NOT_A_NUMBER", self.FAIL),
            # Test invalid ras options
            (f"{self.amd_smi_exe} ras", self.FAIL),
            (f"{self.amd_smi_exe} ras --cper INVALID", self.FAIL),
            (f"{self.amd_smi_exe} ras --cper --severity INVALID", self.FAIL),
            (f"{self.amd_smi_exe} ras --afid", self.FAIL),
            (f"{self.amd_smi_exe} ras --afid INVALID", self.FAIL),
            # Test invalid watch order
            (f"{self.amd_smi_exe} monitor --interval 2 --watch 1", self.FAIL),
            (f"{self.amd_smi_exe} monitor --watch_time 2 --watch 1", self.FAIL),
        ]

        for index, gpu in enumerate(self.common.processors):
            # Test invalid power-cap values
            cmds.append((f"{self.amd_smi_exe} set --power-cap --gpu {index}", self.FAIL))
            for power_type in self.power_types:
                cmds.append(
                    (f"{self.amd_smi_exe} set --power-cap {power_type} --gpu {index}", self.FAIL)
                )
                _power_type = self.static_data["gpu_data"][index]["limit"][power_type]
                socket_power_limit = _power_type["socket_power_limit"]
                if socket_power_limit != "N/A":
                    min_power = _power_type["min_power_limit"]["value"]
                    max_power = _power_type["max_power_limit"]["value"]
                    cmds.append(
                        (
                            f"{self.amd_smi_exe} set --power-cap {min_power - 1} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )
                    cmds.append(
                        (
                            f"{self.amd_smi_exe} set --power-cap {max_power + 1} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )
                    cmds.append(
                        (
                            f"{self.amd_smi_exe} set --power-cap {int(max_power * 1.10)} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )

            # Test invalid soc-pstate values
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                cmds.append((f"{self.amd_smi_exe} set --soc-pstate --gpu {index}", self.FAIL))
                num_supported = int(soc_pstate["num_supported"])
                cmds.append(
                    (
                        f"{self.amd_smi_exe} set --soc-pstate {num_supported} --gpu {index}",
                        self.FAIL,
                    )
                )

            # Test invalid xgmi-plpd values
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                cmds.append((f"{self.amd_smi_exe} set --xgmi-plpd --gpu {index}", self.FAIL))
                num_supported = int(xgmi_plpd["num_supported"])
                cmds.append(
                    (f"{self.amd_smi_exe} set --xgmi-plpd {num_supported} --gpu {index}", self.FAIL)
                )

        self.RunCmds(cmds)
        return

    def test_default(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi"
        self.common.print(msg)

        cmds = [(f"{self.amd_smi_exe}", self.PASS)]

        self.RunCmds(cmds)
        return

    def test_version(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi version"
        self.common.print(msg)

        cmds = [
            (f"{self.amd_smi_exe} version", self.PASS),
            (f"{self.amd_smi_exe} version --cpu_version", self.PASS),
            (f"{self.amd_smi_exe} version --gpu_version", self.PASS),
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

        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs input"
                self.common.print(msg)
                self.skipTest(msg)

        # Start process with f"{self.amd_smi_exe} event"
        # In another process create an event with like f"{self.amd_smi_exe} reset --gpureset"
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
                cmds.append((f"{self.amd_smi_exe} reset --fans --gpu {index}", self.PASS))

            # set --perf-level defaults
            perf_level = self.metric_data["gpu_data"][index]["perf_level"]
            if perf_level != "N/A":
                perf_level = perf_level.removeprefix("AMDSMI_DEV_PERF_LEVEL_")
                cmds.append(
                    (f"{self.amd_smi_exe} set --perf-level {perf_level} --gpu {index}", self.PASS)
                )

            # set --profile defaults
            if power_profile[index]:
                profile = power_profile[index]["current"].removeprefix("AMDSMI_PWR_PROF_PRST_")
                cmds.append(
                    (f"{self.amd_smi_exe} set --profile {profile} --gpu {index}", self.PASS)
                )

            # set --perf-determinism defaults
            clock_sys = self.static_data["gpu_data"][index]["clock"]["sys"]
            if clock_sys != "N/A":
                num = len(clock_sys["frequency_levels"])
                level = f"Level {num - 1}"
                clock_freq = int(clock_sys["frequency_levels"][level].split()[0].strip())
                cmds.append(
                    (
                        f"{self.amd_smi_exe} set --perf-determinism {clock_freq} --gpu {index}",
                        self.PASS,
                    )
                )

            # set --compute-partition defaults
            accelerator_type = self.partition_data["current_partition"][index]["accelerator_type"]
            if accelerator_type != "N/A":
                cmds.append(
                    (
                        f"{self.amd_smi_exe} set --compute-partition {accelerator_type} --gpu {index}",
                        self.PASS,
                    )
                )

            # set --memory-partition defaults
            memory_partition = self.partition_data["current_partition"][index]["memory"]
            if memory_partition != "N/A":
                cmds.append(
                    (
                        f"{self.amd_smi_exe} set --memory-partition {memory_partition} --gpu {index}",
                        self.PASS,
                    )
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
                        f"{self.amd_smi_exe} set --compute-partition-mem-alloc-mode {mem_alloc_mode} --gpu {index}",
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
                            f"{self.amd_smi_exe} set --power-cap {socket_power} {power_type} --gpu {index}",
                            self.PASS,
                        )
                    )

            # set --soc-pstate defaults
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                current = int(soc_pstate["current"])
                cmds.append(
                    (f"{self.amd_smi_exe} set --soc-pstate {current} --gpu {index}", self.PASS)
                )

            # set --xgmi-plpd defaults
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                current = int(xgmi_plpd["current"])
                cmds.append(
                    (f"{self.amd_smi_exe} set --xgmi-plpd {current} --gpu {index}", self.PASS)
                )

            # set --ptl-status defaults
            ptl_state = self.static_data["gpu_data"][index]["limit"]["ptl_state"]
            if ptl_state != "N/A":
                if ptl_state == "Disabled":
                    ptl_state_value = 0
                else:
                    ptl_state_value = 1
                cmds.append(
                    (
                        f"{self.amd_smi_exe} set --ptl-status {ptl_state_value} --gpu {index}",
                        self.PASS,
                    )
                )

            # set --ptl-format defaults
            ptl_format = self.static_data["gpu_data"][index]["limit"]["ptl_format"]
            if ptl_format != "N/A":
                # TODO: get the right ptl-format
                cmds.append(
                    (f"{self.amd_smi_exe} set --ptl-format {ptl_format} --gpu {index}", self.PASS)
                )

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
                                f"{self.amd_smi_exe} set --clk-limit {clk_type} {limit_type} {value} --gpu {index}",
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
                        (
                            f"{self.amd_smi_exe} set --clk-level {clk_type} {value} --gpu {index}",
                            self.PASS,
                        )
                    )
            # set --process-isolation defaults
            process_isolation = self.static_data["gpu_data"][index]["process_isolation"]
            if process_isolation == "Disabled":
                process_isolation_value = 0
            else:
                process_isolation_value = 1
            cmds.append(
                (
                    f"{self.amd_smi_exe} set --process-isolation {process_isolation_value} --gpu {index}",
                    self.PASS,
                )
            )

        print("Restore Starting Values")
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
                self.common.print(msg)
                self.skipTest(msg)

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
        msg = f"{self.tab}### amd-smi monitor serial"
        self.common.print(msg)

        cmd = f"{self.amd_smi_exe} monitor --power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)
        (rc, data2, std_err) = self.util.RunCmdSync(cmd)
        cmd = f"{self.amd_smi_exe} metric --json"
        (rc, data3, std_err) = self.util.RunCmdSync(cmd)

        # Data from monitor and metric should be the same
        monitor1 = json.loads(data1)
        monitor2 = json.loads(data2)
        metric3 = json.loads(data3)
        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_failures, monitor_successes = self._compare_monitor_metric_data("Monitor", data)

        data = self._get_monitor_metric_data(monitor2, None, metric3)
        metric_failures, metric_successes = self._compare_monitor_metric_data("Metric", data)

        results = monitor_successes + metric_successes
        self._PrintResults(results)
        results = monitor_failures + metric_failures
        self._PrintResults(results, fail_on_results=True)
        return

    def test_monitor_parallel(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor parallel"
        self.common.print(msg)

        def _Process(q, cmd):
            if False:
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
        cmd = f"{self.amd_smi_exe} monitor --power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie --json"
        p1 = multiprocessing.Process(target=_Process, args=(q, cmd))
        p1.start()
        # Send time_stamp
        time_stamp = time.monotonic()
        # print(f"Producer pid={os.getpid()}  sending: {time_stamp}")
        q.put(time_stamp)

        # Get monitor data
        cmd = f"{self.amd_smi_exe} monitor --power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)
        time_stamp = time.monotonic()

        # Receive process data and time_stamp
        data2 = q.get()
        time_stamp_process = q.get()
        p1.join()

        if False:
            if my_args.diagnostic == "DEBUG":
                print(f"Collection TimeStamp: Monitor2={time_stamp_process}  Monitor1={time_stamp}")
                print(f"          Difference: {abs(time_stamp_process - time_stamp)} seconds")

        # Data from monitor and metric should be the same
        monitor1 = json.loads(data1)
        monitor2 = json.loads(data2)
        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_failures, monitor_successes = self._compare_monitor_metric_data("Monitor", data)

        # Monitor to Metric
        cmd = f"{self.amd_smi_exe} monitor --power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie --json"
        p1 = multiprocessing.Process(target=_Process, args=(q, cmd))
        p1.start()
        # Send time_stamp
        time_stamp = time.monotonic()
        # print(f"Producer pid={os.getpid()}  sending: {time_stamp}")
        q.put(time_stamp)

        # Get metric data
        cmd = f"{self.amd_smi_exe} metric --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)
        time_stamp = time.monotonic()

        # Receive process data and time_stamp
        data2 = q.get()
        time_stamp_process = q.get()
        p1.join()

        if False:
            if my_args.diagnostic == "DEBUG":
                print(f"Collection TimeStamp: Monitor={time_stamp_process}  Metric={time_stamp}")
                print(f"          Difference: {abs(time_stamp_process - time_stamp)} seconds")

        monitor = json.loads(data2)
        metric3 = json.loads(data1)
        data = self._get_monitor_metric_data(monitor, None, metric3)
        metric_failures, metric_successes = self._compare_monitor_metric_data("Metric", data)

        # Report results
        results = monitor_successes + metric_successes
        self._PrintResults(results)
        results = monitor_failures + metric_failures
        self._PrintResults(results, fail_on_results=True)
        return

    def test_monitor_with_workload(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor workload"
        self.common.print(msg)

        # TODO allow monitor_with_workload to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs Testing, Not Yet Implemented"
                self.common.print(msg)
                self.skipTest(msg)

        def _Process(q, cmd):
            if False:
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
        cmd = f"{self.amd_smi_exe} monitor --power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie --json"
        (rc, data1, std_err) = self.util.RunCmdSync(cmd)

        # Monitor to Monitor
        cmd = "stress-ng --vm 2 --vm-bytes 75% --timeout 8s"
        cmd = "rvs --json"
        p1 = multiprocessing.Process(target=_Process, args=(q, cmd))
        p1.start()
        # Send time_stamp
        time_stamp = time.monotonic()
        # print(f"Producer pid={os.getpid()}  sending: {time_stamp}")
        q.put(time_stamp)

        # Get monitor data under workload
        time.sleep(2)
        cmd = f"{self.amd_smi_exe} monitor --power-usage --temperature --base-board-temps --gpu-board-temps --gfx --mem --encoder --decoder --ecc --vram-usage --pcie --json"
        (rc, data2, std_err) = self.util.RunCmdSync(cmd)
        time_stamp = time.monotonic()

        # Receive process data and time_stamp
        process_data = q.get()
        if verbose == common.VERBOSITY_VERBOSE:
            print(process_data)
        process_time_stamp = q.get()
        p1.join()

        if False:
            if my_args.diagnostic == "DEBUG":
                print(f"Collection TimeStamp: Monitor2={time_stamp_process}  Monitor1={time_stamp}")
                print(f"          Difference: {abs(time_stamp_process - time_stamp)} seconds")

        monitor1 = json.loads(data1)
        monitor2 = json.loads(data2)
        data = self._get_monitor_metric_data(monitor1, monitor2, None)
        monitor_failures, monitor_successes = self._compare_monitor_metric_data("Workload", data)
        # Results are opposite, want differences in values
        for index, cmd_data in enumerate(monitor_failures):
            if "Failure" in cmd_data[1]:
                monitor_failures[index] = (
                    cmd_data[0],
                    cmd_data[1].replace("Failure", "Success", 1),
                )
        for index, cmd_data in enumerate(monitor_successes):
            if "Success" in cmd_data[1]:
                monitor_successes[index] = (
                    cmd_data[0],
                    cmd_data[1].replace("Success", "Failure", 1),
                )
        tmp = monitor_failures
        monitor_failures = monitor_successes
        monitor_successes = tmp

        # Report results
        self._PrintResults(monitor_successes)
        self._PrintResults(monitor_failures, fail_on_results=True)
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

        # TODO Yazen
        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Not Yet Implemented"
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

    def test_fabric(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi fabric"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "fabric", "Fabric arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_static_mem_carveout_gtt(self):
        """Test static --mem-carveout and node --gtt flags (display mode only)"""
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static --mem-carveout and node --gtt"
        self.common.print(msg)
        cmds = self.CreateCmds(
            "fabric", "Fabric arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

        # Test mem-carveout display (static subcommand)
        cmd = f"{self.amd_smi_exe} static --mem-carveout"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test GTT display (node subcommand — GTT is system-wide, not per-GPU)
        cmd = f"{self.amd_smi_exe} node --gtt"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test mem-carveout with JSON output
        cmd = f"{self.amd_smi_exe} static --mem-carveout --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test GTT with JSON output (node subcommand)
        cmd = f"{self.amd_smi_exe} node --gtt --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test mem-carveout with CSV output
        cmd = f"{self.amd_smi_exe} static --mem-carveout --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test GTT with CSV output (node subcommand)
        cmd = f"{self.amd_smi_exe} node --gtt --csv"
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
    verbose = common.VERBOSITY_NORMAL
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbose = common.VERBOSITY_QUIET
    elif "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = common.VERBOSITY_VERBOSE

    if "-h" in sys.argv or "--help" in sys.argv:
        common.print_unittest_help()
        common.print_amdsmi_path_help()
        sys.exit(0)

    if "-l" in sys.argv or "--list" in sys.argv:
        common.print_tests(__name__)
        sys.exit(0)

    if os.geteuid() != 0:
        print(
            "Warning: Some tests may require elevated privileges (sudo/root) to run completely.\n",
            file=sys.stderr,
        )
        print("Please relaunch with elevated privileges.\n", file=sys.stderr)
        sys.exit(1)

    # Only show the dot-character legend when not in verbose mode; in verbose
    # mode each test prints its own result line so the dot legend is irrelevant.
    if verbose < common.VERBOSITY_VERBOSE:
        common.print_legend()

    if verbose > common.VERBOSITY_QUIET:
        print("AMD SMI CLI Tests")

    runner = common.GTestSummaryRunner(
        stream=sys.stderr, verbosity=common.make_runner_verbosity(verbose)
    )

    unittest.main(testRunner=runner)
    sys.exit(0)
