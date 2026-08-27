#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared base class for the CLI unit tests.

``TestCliBase`` provides the instance-level scaffolding every CLI test
needs -- ``__init__`` (command constants), ``FindArgs``, ``CreateCmds``
and ``RunCmds``.  Extracted from develop's monolithic ``TestAmdSmiCli``
so the per-feature CLI suites can share it: the per-command ``cli/test_*.py``
files each subclass ``TestCliBase`` and add their own test_* methods.
"""

import collections
import json
import math
import os
import stat
import sys
import unittest

import common.common as common
import common.runcmd as runcmd
from common.common import amdsmi


def _load_cli_exceptions():
    """The CLI's own exception module, from whichever amd-smi is under test."""
    cli_dir = common.find_cli_dir(common.amdsmi_path, os.path.dirname(os.path.abspath(__file__)))
    if cli_dir and cli_dir not in sys.path:
        sys.path.append(cli_dir)
    # Tests name AmdSmiExitCode, so a miss has to fail here rather than as an
    # attribute error on None once a test is already running.
    try:
        import amdsmi_cli_exceptions
    except ImportError as exc:
        raise ImportError(
            f"amd-smi CLI not found under {cli_dir!r}; the CLI tests need it for exit codes"
        ) from exc

    return amdsmi_cli_exceptions


AmdSmiExitCode = _load_cli_exceptions().AmdSmiExitCode


def _build_exit_code_names():
    """Exit code -> name, e.g. ``NOT_SUPPORTED``."""
    names = {}
    # Folding a status to its low byte can collide, so the first member wins.
    for member in amdsmi.AmdSmiStatus:
        names.setdefault(abs(int(member.value)) & 0xFF, member.name)
    # CLI codes are exact, so they override any collision above.
    for member in AmdSmiExitCode:
        if member.value:
            names[member.value] = member.name
    return names


EXIT_CODE_NAMES = _build_exit_code_names()


def _describe_codes(codes):
    """Accepted exit codes, named, for a failure message."""
    named = ", ".join(f"{EXIT_CODE_NAMES.get(code, 'UNKNOWN')}({code})" for code in codes)
    return f"[{named}]"


# One command's verdict: the line RunCmds prints, and whether it counts as a
# test failure -- which is not the same as the command failing, since a command
# declared to fail counts as a pass when it does.
Verdict = collections.namedtuple("Verdict", "message counts_as_failure")

# --clk-level placeholder -> the clock name the CLI expects. PCIE is sized from
# a different source than the rest; see _clk_level_count().
CLK_LEVEL_TYPES = {
    "{clk_level_sclk}": "SCLK",
    "{clk_level_mclk}": "MCLK",
    "{clk_level_fclk}": "FCLK",
    "{clk_level_socclk}": "SOCCLK",
    "{clk_level_pcie}": "PCIE",
}

# --clk-limit placeholder -> (clock name the CLI expects, which bound to set).
# The metric key and the value key both follow from these, so they cannot drift
# apart the way separately written entries did.
CLK_LIMIT_TARGETS = {
    "{clk_limit_sclk_max}": ("SCLK", "MAX"),
    "{clk_limit_sclk_min}": ("SCLK", "MIN"),
    "{clk_limit_mclk_max}": ("MCLK", "MAX"),
    "{clk_limit_mclk_min}": ("MCLK", "MIN"),
    "{clk_limit_fclk_max}": ("FCLK", "MAX"),
    "{clk_limit_fclk_min}": ("FCLK", "MIN"),
}

# Clock name -> the ``metric --json`` key reporting its min_clk/max_clk. SCLK is
# the graphics clock (the CLI maps it to AmdSmiClkType.GFX), not the SOC clock.
CLK_LIMIT_METRIC_KEYS = {"SCLK": "gfx_0", "MCLK": "mem_0", "FCLK": "fclk_0"}

# --clk-limit probe values are rounded up to a multiple of this. Cosmetic only:
# the driver accepts any MHz within range.
CLK_LIMIT_MHZ_STEP = 10


class TestCliBase(unittest.TestCase):
    """Base class for CLI functional tests.

    The base now provides a cached ``setUpClass`` that fetches the ``--json``
    baseline (metric/static/list/partition data, the derived ``gpus`` list and
    the ``sub_args`` dict) exactly once and shares it across all CLI test
    classes, so subclasses only need to add their own test_* methods.
    """

    TMP_FILENAME = "_tmp.log"
    TMP_FOLDER = "_tmp"

    # TODO(amdsmi_team): drop these once the CLI tests supports automated
    #                     input
    SWEEP_EXCLUDED_ARGS = {"set": {"--compute-partition": "waits for confirmation"}}

    # TODO(amdsmi_team): User input is not a problem, we just need to test above ^
    PROMPT_ANSWERS = {"set": {"--fan": "y\n", "--memory-partition": "y\n"}}

    # Commands whose sweep also accepts NOT_SUPPORTED. A write may legitimately
    # refuse a feature the ASIC lacks, but a read answering NOT_SUPPORTED is a
    # regression the sweep should catch. Add "reset" when that suite is enabled.
    NOT_SUPPORTED_TOLERANT_COMMANDS = {"set"}

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
    clk_freq: list
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
        TestCliBase.clk_freq = baseline["clk_freq"]
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

        # When parsing, expand each arg with its value(s).
        # CPU/Core are given default values because their sub-arguments require a value.
        # Otherwise, tests will populate the args incorrectly on CPU/APU-capable systems.
        baseline["sub_args"] = {
            "CLOCK": ["SYS", "DF", "DCEF", "SOC", "MEM", "VCLK0", "VCLK1", "DCLK0", "DCLK1", "ALL"],
            "PID": [123],
            "NAME": ["AMD"],
            "GPU": gpus,
            "CPU": ["all", "0"],
            "CORE": ["all", "0"],
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
        baseline["clk_freq"] = cls._read_clk_freq()
        return baseline

    @classmethod
    def _read_clk_freq(cls):
        """Per-GPU DPM clock tables, read from the library rather than the CLI.

        Neither CLI command can say how many levels are settable, and combining them
        does not close the gap either:

        - ``amd-smi static --clock`` lists every row of the table as ``Level N``,
          including the non-settable deep-sleep row, and does not mark which one it
          is. A three-entry list is therefore either three settable levels, or two
          plus an ``S`` row, and the two cases are indistinguishable.
        - ``amd-smi metric --clock`` does report a per-clock ``deep_sleep``, but it
          discards the library value and recomputes it as (current clk < min clk).
          That is a runtime state which flips as the GPU idles, so it cannot identify
          the row static left ambiguous -- the same device answers differently
          depending on how busy it was when the command ran.

        ``amdsmi_get_clock_info`` returns ``clk_deep_sleep`` before the CLI recomputes
        it, and it is populated only when the table carries the ``S`` row, which is
        what decides the settable level indices.

        Returns:
            list: one dict per GPU, keyed by the CLI's clock names; a value is None
            when the device does not expose that clock.
        """
        clk_types = {
            "SCLK": amdsmi.AmdSmiClkType.SYS,
            "MCLK": amdsmi.AmdSmiClkType.MEM,
            "FCLK": amdsmi.AmdSmiClkType.DF,
            "SOCCLK": amdsmi.AmdSmiClkType.SOC,
        }
        per_gpu = []
        # Common.__init__ shuts the library down once it has read its own data.
        cls.common.amdsmi_smart_init()
        try:
            for processor in amdsmi.amdsmi_get_processor_handles():
                entry = {}
                for name, clk_type in clk_types.items():
                    try:
                        freq = amdsmi.amdsmi_get_clk_freq(processor, clk_type)
                        info = amdsmi.amdsmi_get_clock_info(processor, clk_type)
                    except amdsmi.AmdSmiLibraryException:
                        entry[name] = None
                        continue
                    freq["has_deep_sleep"] = info["clk_deep_sleep"] != "N/A"
                    entry[name] = freq
                per_gpu.append(entry)
        finally:
            amdsmi.amdsmi_shut_down()
        return per_gpu

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Debug = False
        self.ReduceCmds = True
        self.PrintCmdsOnly = False

        self.AddCmdMods = True
        self.AddDeviceArgs = True
        self.AddWatchArgs = True
        self.AddLogLevel = "--loglevel DEBUG"

        self.PASS = amdsmi.AmdSmiStatus.SUCCESS
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
            "CUSTOM",
            "VIDEO",
            "POWER_SAVING",
            "COMPUTE",
            "VR",
            "3D_FULL_SCREEN",
            "BOOTUP_DEFAULT",
        ]
        self.compute_partition_modes = ["SPX", "DPX", "TPX", "QPX", "CPX"]
        self.memory_partition_modes = ["NPS1", "NPS2", "NPS4", "NPS8"]
        self.power_types = ["ppt0", "ppt1"]
        self.ptl_formats = ["I8", "F16", "BF16", "F32", "F64", "F8", "VECTOR"]
        self.clk_limits = ["SCLK", "MCLK", "FCLK"]
        self.limit_types = ["MAX", "MIN"]
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
                                # Raise the ceiling before the floor: a MIN above the
                                # current MAX is refused, so MIN-first cannot widen a
                                # range upwards.
                                options.append(f"{items[item_index]} {{clk_limit_sclk_max}}")
                                options.append(f"{items[item_index]} {{clk_limit_sclk_min}}")
                                options.append(f"{items[item_index]} {{clk_limit_mclk_max}}")
                                options.append(f"{items[item_index]} {{clk_limit_mclk_min}}")
                                options.append(f"{items[item_index]} {{clk_limit_fclk_max}}")
                                options.append(f"{items[item_index]} {{clk_limit_fclk_min}}")
                            elif (
                                sub_arg == "STATUS" and "process" in items[item_index]
                            ):  # arg --process-isolation
                                options.append(f"{items[item_index]} 0")
                                options.append(f"{items[item_index]} 1")
                            else:
                                self.common.print(
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
        ok = [self.PASS]
        if cmd_name in self.NOT_SUPPORTED_TOLERANT_COMMANDS:
            ok.append(amdsmi.AmdSmiStatus.NOT_SUPPORTED)
        for list1_arg in list1_args:
            if list1_arg != "pass":
                cmds.append((f"{cmd} {list1_arg} {self.AddLogLevel}", ok))
                if not list1_arg:
                    cmds.append((f"{cmd} --file {self.tmp_filename} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{json}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{json_file}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{json_file_append}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{json_file_overwrite}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{csv}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{csv_file}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{csv_file_append}} {self.AddLogLevel}", ok))
                    cmds.append((f"{cmd} {{csv_file_overwrite}} {self.AddLogLevel}", ok))
            else:
                list1_arg = ""
            for list2_arg in list2_args:
                if list2_arg != "pass":
                    cmds.append((f"{cmd} {list1_arg} {list2_arg} {self.AddLogLevel}", ok))
                else:
                    list2_arg = ""
                for list3_arg in list3_args:
                    if list3_arg != "pass":
                        cmds.append(
                            (f"{cmd} {list1_arg} {list2_arg} {list3_arg} {self.AddLogLevel}", ok)
                        )
                    else:
                        list3_arg = ""
                    for list4_arg in list4_args:
                        if list4_arg != "pass":
                            cmds.append(
                                (
                                    f"{cmd} {list1_arg} {list2_arg} {list3_arg} {list4_arg} {self.AddLogLevel}",
                                    ok,
                                )
                            )

        # Calculate and substitute in dependent values
        # Removes cmds that are invalid
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            while self.openCurlyBrace in cmd:
                items = cmd.split()
                # Find gpu index. No --gpu, "all" or a bdf all target every device.
                explicit_gpu = False
                try:
                    i = items.index("--gpu")
                    gpu = items[i + 1]
                    if gpu.isdigit():
                        gpu_index = int(gpu)
                        explicit_gpu = True
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
                    # Stop at the match; a later iteration would overwrite it with "N/A".
                    for power_type in self.power_types:
                        if power_type in cmd:
                            power_type = self.static_data["gpu_data"][gpu_index]["limit"][
                                power_type
                            ]
                            break
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
                        # A current cap of 0 means the CLI refuses every write to this
                        # sensor, whatever value is asked for.
                        current_cap = power_type["socket_power_limit"]
                        if isinstance(current_cap, dict) and current_cap["value"] == 0:
                            cond = [*ok, AmdSmiExitCode.INVALID_PARAMETER_VALUE]
                        # The CLI clamps its lower bound to 1, so a reported min of 0
                        # is refused rather than being the smallest settable cap.
                        min_power = max(power_type["min_power_limit"]["value"], 1)
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
                    clk_type, bound = CLK_LIMIT_TARGETS[nameStr]
                    value = self._clk_limit_probe_value(
                        self.metric_data["gpu_data"][gpu_index]["clock"], clk_type, bound
                    )
                    if value is None:
                        cmd = ""
                    else:
                        cmd = cmd.replace(nameStr, f"{clk_type} {bound} {value}", 1)
                elif "clk_level" in nameStr:
                    clk_type = CLK_LEVEL_TYPES[nameStr]
                    count = self._clk_level_count(clk_type, gpu_index, explicit_gpu)
                    if clk_type != "PCIE":
                        # A DPM table's length is a runtime value, not a fixed property:
                        # the same device reports fewer levels when the domain is idle.
                        # The CLI re-reads it as the command runs, so a mask sized here
                        # can name a level it no longer accepts.
                        cond = [*ok, AmdSmiExitCode.INVALID_PARAMETER_VALUE]
                    if count:
                        levels = " ".join(str(level) for level in range(count))
                        cmd = cmd.replace(nameStr, f"{clk_type} {levels}", 1)
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

        # Remove commands the sweep must not run
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            if cmd and self._lookup(self.SWEEP_EXCLUDED_ARGS, cmd) is not None:
                cmds[index] = ("", cond)

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

    @staticmethod
    def _lookup(table, cmd):
        """Value for *cmd* in a ``{command: {sub-arg: value}}`` table, else None.

        Matches on the words the command contains rather than where they sit, so
        an option that moves (``set --gpu 0 --fan 50``) is still found.
        """
        words = cmd.split()
        for command, sub_args in table.items():
            if command not in words:
                continue
            for sub_arg, value in sub_args.items():
                if sub_arg in words:
                    return value
        return None

    def _prompt_answer_for(self, cmd):
        """Reply to pipe to a command whose parser prompts, else None."""
        return self._lookup(self.PROMPT_ANSWERS, cmd)

    def _clk_level_count(self, clk_type, gpu_index, explicit_gpu):
        """How many levels a ``--clk-level`` mask for *clk_type* may name.

        Args:
            clk_type: clock name as the CLI spells it, e.g. ``SOCCLK``.
            gpu_index: device the command reads its values from.
            explicit_gpu: whether the command named that device, as opposed to
                defaulting to it while actually running on every device.

        Returns:
            int: the highest settable level plus one, or 0 when the device does
            not expose the clock -- in which case the caller drops the command.
        """
        if clk_type == "PCIE":
            pcie_levels = self.static_data["gpu_data"][gpu_index]["bus"]["pcie_levels"]
            return len(pcie_levels) if isinstance(pcie_levels, dict) else 0

        # Without an explicit --gpu the command runs on every device, so the mask
        # has to be one the smallest table also accepts.
        tables = [self.clk_freq[gpu_index]] if explicit_gpu else self.clk_freq
        per_gpu = []
        for table in tables:
            freq = table.get(clk_type)
            if freq and freq["num_supported"]:
                # The deep-sleep row occupies a table slot but is not settable.
                per_gpu.append(freq["num_supported"] - int(freq["has_deep_sleep"]))
        return min(per_gpu) if per_gpu else 0

    @staticmethod
    def _clk_limit_value(clocks, clk_type, bound):
        """The clock's currently reported *bound*, or None when it is not readable.

        Args:
            clocks: the ``clock`` object from ``metric --json`` for one device.
            clk_type: clock name as the CLI spells it, e.g. ``SCLK``.
            bound: ``"MIN"`` or ``"MAX"``.

        Returns:
            int | None: the reported value, or None when the device does not
            report it.
        """
        entry = clocks.get(CLK_LIMIT_METRIC_KEYS[clk_type])
        if not isinstance(entry, dict):
            return None
        value = entry["min_clk" if bound == "MIN" else "max_clk"]
        return value["value"] if isinstance(value, dict) else None

    @classmethod
    def _clk_limit_probe_value(cls, clocks, clk_type, bound):
        """A value inside the clock's reported range, or None when there is none.

        Replaying the reported bound is a no-op -- the CLI answers "already set"
        -- so a sweep built from it never shows that a limit can move.

        Args:
            clocks: the ``clock`` object from ``metric --json`` for one device.
            clk_type: clock name as the CLI spells it, e.g. ``SCLK``.
            bound: ``"MIN"`` or ``"MAX"``.

        Returns:
            int | None: the value to send, or None when the clock is unreadable
            or its range is too narrow to step into.
        """
        low = cls._clk_limit_value(clocks, clk_type, "MIN")
        high = cls._clk_limit_value(clocks, clk_type, "MAX")
        if low is None or high is None:
            return None

        # A quarter in from each end leaves MIN below MAX whichever is written first.
        quarter = (high - low) // 4
        if quarter <= 0:
            return None
        target = low + quarter if bound == "MIN" else high - quarter

        # Rounding up can carry a narrow range past its own ceiling. It cannot fall
        # below the floor: quarter is at least 1, so target already exceeds it.
        target = math.ceil(target / CLK_LIMIT_MHZ_STEP) * CLK_LIMIT_MHZ_STEP
        return min(target, high)

    def _accepted_codes(self, cond):
        """Exit codes *cond* names, as a tuple; a bare code becomes a 1-tuple.

        Args:
            cond: one exit code, or a list/tuple of the codes that are acceptable.

        Returns:
            tuple: the accepted codes, in the order they were declared.
        """
        if isinstance(cond, (list, tuple)):
            return tuple(cond)
        return (cond,)

    def _grade(self, cond, rc, file_error, detail):
        """Judge one command result against the outcome it declared.

        Args:
            cond: the declared outcome -- an exit code, a list of acceptable
                codes, or ``self.FAIL`` for "any non-zero code will do".
            rc: the exit code the command actually returned.
            file_error: complaint about the ``--file`` output, or None.
            detail: the ``rc=N NAME`` fragment every verdict line ends with.

        Returns:
            Verdict: the message to print and whether it counts as a failure.
        """
        # The one condition naming no specific code, so it cannot go through
        # the accepted-code comparison below.
        if cond == self.FAIL:
            if not rc:
                return Verdict(" Failure: Received PASS (0), expected FAIL (!0)", True)
            return Verdict(f" Success: Received and Expected FAIL ({detail})", False)

        accept = self._accepted_codes(cond)
        if rc not in accept:
            received = "PASS" if not rc else "FAIL"
            return Verdict(
                f" Failure: Received {received} ({detail}), expected {_describe_codes(accept)}",
                True,
            )

        tolerates_pass = self.PASS in accept
        if file_error is not None and tolerates_pass:
            return Verdict(f" Failure: {file_error}", True)

        if not rc:
            if accept == (self.PASS,):
                return Verdict(f" Success: Received and Expected PASS ({detail})", False)
            return Verdict(f" Success: Received PASS ({detail})", False)
        # A non-zero code accepted alongside PASS is leniency, not a declared outcome.
        if tolerates_pass:
            prefix = "Unsupported" if rc == amdsmi.AmdSmiStatus.NOT_SUPPORTED else "Success"
            return Verdict(f" {prefix}: Received and Allowed FAIL ({detail})", False)
        return Verdict(f" Success: Received and Expected FAIL ({detail})", False)

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
            # Remove any stale output file so amd-smi does not block on its
            # interactive "file exists, overwrite?" prompt (e.g. a leftover
            # from a previously interrupted run).
            if "--file" in cmd and os.path.exists(self.tmp_filename):
                os.chmod(self.tmp_filename, stat.S_IWRITE)
                os.remove(self.tmp_filename)
            (rc, std_out, std_err) = self.util.RunCmdSync(cmd, msg_in=self._prompt_answer_for(cmd))
            # Name the code from the same tables the CLI maps it with. Scraping
            # stderr yielded its last word instead ("...without setting value"),
            # and most failures write nothing to stderr at all.
            error_code = rc
            reason = EXIT_CODE_NAMES.get(rc, "") if rc else ""
            detail = f"rc={error_code}" + (f" {reason}" if reason else "")

            msg = f"{cmd:{msg_len}s}:"
            file_error = None
            if "--file" in cmd:
                if not os.path.exists(self.tmp_filename):
                    file_error = f"File {self.tmp_filename} does not exist"
                else:
                    with open(self.tmp_filename, "r") as fin:
                        std_out = fin.read()
                    if not len(std_out):
                        file_error = f"File {self.tmp_filename} was empty"
                    os.chmod(self.tmp_filename, stat.S_IWRITE)
                    os.remove(self.tmp_filename)

            verdict = self._grade(cond, rc, file_error, detail)
            msg += verdict.message
            if verdict.counts_as_failure:
                errors.append(msg)

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
