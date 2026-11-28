#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Driver-free unit tests for the ``amd-smi set`` CLI sweep on multi-GPU hosts.

A ``set`` command without ``--gpu`` runs on every GPU, and the CLI folds the
GPUs' results into one exit code. On a multi-GPU host the sweep therefore also
runs each such command once per GPU, grades each GPU against its own data, and
checks the all-GPU code against the aggregate of the per-GPU codes.

The per-GPU values below are what CI runners reported, and each host's
``baseline`` is the set sweep those runners ran before the per-GPU runs existed.
"""

import collections
import itertools
import sys
import unittest
from unittest import mock

from common.common import amdsmi

# cli.base appends the installed CLI dir to sys.path and imports amdsmi_cli_exceptions
# from it to load its exit codes. Restore both sys.path and the sys.modules entry for
# amdsmi_cli_exceptions once it has, so later test modules import CLI modules the same
# way they would without this module in the run. cli.base keeps its own reference.
_sys_path = list(sys.path)
_CLI_EXCEPTIONS = "amdsmi_cli_exceptions"
_had_cli_exceptions = _CLI_EXCEPTIONS in sys.modules
_cli_exceptions = sys.modules.get(_CLI_EXCEPTIONS)
try:
    # Import modules, not classes: a TestCase class in this module's namespace
    # would be collected by discovery and run the real set sweep.
    from cli import base as cli_base
    from cli import test_set as cli_test_set

    # The CLI's own aggregation, from the install cli.base takes its exit codes from.
    AmdSmiErrorCollector = cli_base._load_cli_exceptions().AmdSmiErrorCollector
finally:
    sys.path[:] = _sys_path
    if _had_cli_exceptions:
        sys.modules[_CLI_EXCEPTIONS] = _cli_exceptions
    else:
        sys.modules.pop(_CLI_EXCEPTIONS, None)

PASS = amdsmi.AmdSmiStatus.SUCCESS
NOT_SUPPORTED = amdsmi.AmdSmiStatus.NOT_SUPPORTED
INVAL = amdsmi.AmdSmiStatus.INVAL
INVALID_VALUE = cli_base.AmdSmiExitCode.INVALID_PARAMETER_VALUE
MIXED = cli_base.AmdSmiExitCode.MIXED_DEVICE_ERRORS
OK = [PASS, NOT_SUPPORTED]
NA = "N/A"
LOGLEVEL = "--loglevel DEBUG"
PTL_FORMATS = ["I8", "F16", "BF16", "F32", "F64", "F8", "VECTOR"]
PERF_LEVELS = [
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
PROFILES = ["CUSTOM", "VIDEO", "POWER_SAVING", "COMPUTE", "BOOTUP_DEFAULT"]

# What FindArgs returns for `amd-smi set --help` on these hosts, in help order.
SET_ARGS = (
    ["--fan 50%", "--fan 50"]
    + [f"--perf-level {level}" for level in PERF_LEVELS]
    + [
        f"--profile {{profile_{profile}}}"
        for profile in [
            "CUSTOM",
            "VIDEO",
            "POWER_SAVING",
            "COMPUTE",
            "VR",
            "3D_FULL_SCREEN",
            "BOOTUP_DEFAULT",
        ]
    ]
    + ["{perf_determinism}"]
    + [f"--compute-partition {mode}" for mode in ["SPX", "DPX", "TPX", "QPX", "CPX"]]
    + [f"--memory-partition {mode}" for mode in ["NPS1", "NPS2", "NPS4", "NPS8"]]
    + ["--compute-partition-mem-alloc-mode CAPPING", "--compute-partition-mem-alloc-mode ALL"]
    + [
        f"--power-cap {{{value}_power}} {sensor}"
        for sensor in ["ppt0", "ppt1"]
        for value in ["min", "avg", "max"]
    ]
    + ["--soc-pstate {soc_pstate}", "--xgmi-plpd {xgmi_plpd}"]
    + [f"--clk-level {{clk_level_{clk}}}" for clk in ["sclk", "mclk", "fclk", "socclk", "pcie"]]
    + ["--ptl-status 0", "--ptl-status 1"]
    + [f"--ptl-format {a},{b}" for a in PTL_FORMATS for b in PTL_FORMATS if a != b]
    + [
        f"--clk-limit {{clk_limit_{clk}_{bound}}}"
        for clk in ["sclk", "mclk", "fclk"]
        for bound in ["max", "min"]
    ]
    + ["--process-isolation 0", "--process-isolation 1"]
)
MODIFIER_ARGS = [
    "{json}",
    "{json_file}",
    "{json_file_append}",
    "{json_file_overwrite}",
    "{csv}",
    "{csv_file}",
    "{csv_file_append}",
    "{csv_file_overwrite}",
    "--file _tmp.log",
    "--file _tmp.log --overwrite",
    "--file _tmp.log --append",
]
# MODIFIER_ARGS as they read once resolved.
MODIFIERS = [
    "--json",
    "--json --file _tmp.log",
    "--json --file _tmp.log --append",
    "--json --file _tmp.log --overwrite",
    "--csv",
    "--csv --file _tmp.log",
    "--csv --file _tmp.log --append",
    "--csv --file _tmp.log --overwrite",
    "--file _tmp.log",
    "--file _tmp.log --overwrite",
    "--file _tmp.log --append",
]


def _watts(value):
    return {"value": value, "unit": "W"}


def _mhz(value):
    return {"value": value, "unit": "MHz"}


def _cap(low, high, current):
    return {
        "min_power_limit": _watts(low),
        "max_power_limit": _watts(high),
        "socket_power_limit": _watts(current),
    }


# What `amd-smi static` reports for a power-cap sensor the GPU does not support.
NO_CAP = {"min_power_limit": NA, "max_power_limit": NA, "socket_power_limit": NA}


def _dpm(levels, deep_sleep=False):
    return {"num_supported": levels, "has_deep_sleep": deep_sleep}


def _gpu(
    cap=NO_CAP,
    sclk_levels=None,
    pcie_levels=0,
    xgmi_plpd=NA,
    profiles=None,
    gfx=None,
    mem=None,
    fclk=None,
    dpm=None,
):
    """One GPU's baseline: the static and metric fields the set sweep reads, and its DPM tables.

    Args:
        cap: the ppt0 power-cap limits; ppt1 is unsupported on every host here.
        sclk_levels: the static SCLK table's frequencies (the sweep reads the last).
        pcie_levels: how many PCIe levels static reports (0: unreadable).
        gfx, mem, fclk: metric (min, max) clock in MHz, or None when unreadable.
        dpm: settable-level info per clock, as TestCliBase._read_clk_freq returns it.
    """
    frequency_levels = NA
    if sclk_levels is not None:
        frequency_levels = {
            "frequency_levels": {
                f"Level {index}": _mhz(value) for index, value in enumerate(sclk_levels)
            }
        }
    static = {
        "limit": {"ppt0": cap, "ppt1": NO_CAP, "ptl_state": NA, "ptl_format": NA},
        "clock": {"sys": frequency_levels},
        # Only the number of PCIe levels matters to the sweep.
        "bus": {
            "pcie_levels": (
                {f"Level {index}": NA for index in range(pcie_levels)} if pcie_levels else NA
            )
        },
        "soc_pstate": NA,
        "xgmi_plpd": xgmi_plpd,
        "process_isolation": "Disabled",
    }
    if profiles:
        static["profile"] = {"current": "BOOTUP_DEFAULT", "available_profiles": profiles}
    clock = {}
    for key, bounds in (("gfx_0", gfx), ("mem_0", mem), ("fclk_0", fclk)):
        clock[key] = NA
        if bounds is not None:
            clock[key] = {"min_clk": _mhz(bounds[0]), "max_clk": _mhz(bounds[1])}
    return {"static": static, "metric": {"clock": clock, "perf_level": NA}, "dpm": dpm}


def _with_cap(gpu, cap):
    """*gpu* with another ppt0 cap (a derived fixture, not one CI reported)."""
    limit = dict(gpu["static"]["limit"], ppt0=cap)
    return dict(gpu, static=dict(gpu["static"], limit=limit))


# Per-GPU data CI runners reported. The gfx/mem/fclk ranges are the metric clock
# limits when the baseline was read; dpm is the settable-level info read then.
XGMI_PLPD = {"num_supported": 2, "current_id": 1}
GFX908 = _gpu(
    cap=_cap(0, 290, 290),
    sclk_levels=(300, 1502),
    pcie_levels=1,
    xgmi_plpd=XGMI_PLPD,
    profiles=PROFILES,
    gfx=(300, 1502),
    mem=(1200, 1200),
    fclk=(1402, 1402),
    dpm={"SCLK": _dpm(16), "MCLK": _dpm(1), "FCLK": _dpm(2, deep_sleep=True), "SOCCLK": _dpm(1)},
)
GFX1030 = _gpu(
    cap=_cap(255, 272, 272),
    sclk_levels=(500, 2575),
    pcie_levels=2,
    profiles=PROFILES,
    gfx=(500, 2575),
    mem=(96, 1000),
    fclk=(500, 1941),
    dpm={"SCLK": _dpm(3), "MCLK": _dpm(4), "FCLK": _dpm(3), "SOCCLK": _dpm(3)},
)
GFX90A = _gpu(
    cap=_cap(0, 300, 300),
    sclk_levels=(500, 1700),
    pcie_levels=24,
    xgmi_plpd=XGMI_PLPD,
    gfx=(500, 500),
    mem=(400, 1600),
    dpm={"SCLK": _dpm(2), "MCLK": _dpm(4), "FCLK": None, "SOCCLK": _dpm(5)},
)
GFX1100 = _gpu(
    cap=_cap(272, 350, 303),
    sclk_levels=(500, 2371),
    pcie_levels=3,
    profiles=PROFILES,
    gfx=(500, 2371),
    mem=(96, 1249),
    fclk=(601, 2301),
    dpm={
        "SCLK": _dpm(3, deep_sleep=True),
        "MCLK": _dpm(4),
        "FCLK": _dpm(8),
        "SOCCLK": _dpm(3, deep_sleep=True),
    },
)
GFX1201 = _gpu(
    cap=_cap(212, 340, 304),
    sclk_levels=(500, 2400),
    pcie_levels=3,
    profiles=PROFILES,
    gfx=(500, 2400),
    mem=(96, 1258),
    fclk=(313, 2400),
    dpm={"SCLK": _dpm(3, deep_sleep=True), "MCLK": _dpm(6), "FCLK": _dpm(3), "SOCCLK": _dpm(2)},
)
GFX1151 = _gpu(
    sclk_levels=(600, 600),
    fclk=(400, 2000),
    dpm={"SCLK": _dpm(3), "MCLK": _dpm(3), "FCLK": _dpm(8), "SOCCLK": _dpm(8)},
)


def _set(args):
    return f"amd-smi set {args} {LOGLEVEL}"


def _fan_cmds(bdf, uuid, gpu_count):
    cmds = []
    for value in ("50%", "50"):
        fan = f"--fan {value}"
        cmds += [_set(fan), _set(f"{fan} --gpu all"), _set(f"{fan} --gpu 0")]
        cmds += [_set(f"{fan} --gpu 0 {modifier}") for modifier in MODIFIERS]
        cmds += [_set(f"{fan} --gpu {bdf}"), _set(f"{fan} --gpu {uuid}")]
        cmds += [_set(f"{fan} --gpu {gpu}") for gpu in range(1, gpu_count)]
    return cmds


def _baseline(bdf, uuid, gpu_count, profiles, determinism, caps, xgmi, clk_levels, clk_limits):
    """The set sweep CI ran on a host, in order (develop before the per-GPU runs)."""
    args = [f"--perf-level {level}" for level in PERF_LEVELS]
    args += [f"--profile {profile}" for profile in profiles]
    args += [f"--perf-determinism {determinism}"]
    args += [f"--memory-partition {mode}" for mode in ["NPS1", "NPS2", "NPS4", "NPS8"]]
    args += ["--compute-partition-mem-alloc-mode CAPPING", "--compute-partition-mem-alloc-mode ALL"]
    args += [f"--power-cap {watts} ppt0" for watts in caps]
    args += [f"--xgmi-plpd {policy}" for policy in xgmi]
    args += [
        f"--clk-level {clk} {' '.join(str(level) for level in range(count))}"
        for clk, count in clk_levels
    ]
    args += ["--ptl-status 0", "--ptl-status 1"]
    args += [f"--ptl-format {a},{b}" for a in PTL_FORMATS for b in PTL_FORMATS if a != b]
    args += [f"--clk-limit {limit}" for limit in clk_limits]
    args += ["--process-isolation 0", "--process-isolation 1"]
    return _fan_cmds(bdf, uuid, gpu_count) + [_set(arg) for arg in args]


Host = collections.namedtuple("Host", "gpus bdf uuid baseline")


def _host(gpus, bdf, uuid, **sweep):
    return Host(gpus, bdf, uuid, _baseline(bdf, uuid, len(gpus), **sweep))


GFX90A_UUID = "0000740f-0000-1000-8000-000000000000"
GFX908_GFX1030 = _host(
    [GFX908, GFX1030],
    "0000:43:00.0",
    "0000738c-0000-1000-8000-000000000000",
    profiles=PROFILES,
    determinism=1552,
    caps=[1, 145, 290],
    xgmi=[0],
    clk_levels=[("SCLK", 3), ("MCLK", 1), ("FCLK", 1), ("SOCCLK", 1), ("PCIE", 1)],
    clk_limits=["SCLK MAX 1210", "SCLK MIN 600"],
)
GFX90A_X2_GFX1100 = _host(
    [GFX90A, GFX90A, GFX1100],
    "0000:23:00.0",
    GFX90A_UUID,
    profiles=[],
    determinism=1750,
    caps=[1, 150, 300],
    xgmi=[0],
    clk_levels=[("SCLK", 2), ("MCLK", 4), ("FCLK", 8), ("SOCCLK", 2), ("PCIE", 24)],
    clk_limits=["MCLK MAX 1300", "MCLK MIN 700"],
)
GFX90A_GFX1201 = _host(
    [GFX90A, GFX1201],
    "0000:43:00.0",
    GFX90A_UUID,
    profiles=[],
    determinism=1750,
    caps=[1, 150, 300],
    xgmi=[0],
    clk_levels=[("SCLK", 2), ("MCLK", 4), ("FCLK", 3), ("SOCCLK", 2), ("PCIE", 24)],
    clk_limits=["MCLK MAX 1300", "MCLK MIN 700"],
)
# Same model: the 2x gfx90a + gfx1100 host's gfx90a GPUs. CI swept these values on
# a single-gfx90a host; the second GPU only adds its fan --gpu 1 commands.
GFX90A_X2 = _host(
    [GFX90A, GFX90A],
    "0000:23:00.0",
    GFX90A_UUID,
    profiles=[],
    determinism=1750,
    caps=[1, 150, 300],
    xgmi=[0],
    clk_levels=[("SCLK", 2), ("MCLK", 4), ("SOCCLK", 5), ("PCIE", 24)],
    clk_limits=["MCLK MAX 1300", "MCLK MIN 700"],
)
GFX1151_APU = _host(
    [GFX1151],
    "0000:c5:00.0",
    "00001586-0000-1000-8000-000000000000",
    profiles=[],
    determinism=650,
    caps=[],
    xgmi=[],
    clk_levels=[("SCLK", 3), ("MCLK", 3), ("FCLK", 8), ("SOCCLK", 8)],
    clk_limits=["FCLK MAX 1600", "FCLK MIN 800"],
)
MIXED_HOSTS = {
    "gfx908 + gfx1030": GFX908_GFX1030,
    "2x gfx90a + gfx1100": GFX90A_X2_GFX1100,
    "gfx90a + gfx1201": GFX90A_GFX1201,
}
MULTI_GPU_HOSTS = dict(MIXED_HOSTS, **{"2x gfx90a": GFX90A_X2})
ALL_HOSTS = dict(MULTI_GPU_HOSTS, **{"gfx1151 APU": GFX1151_APU})


class _FakeCommon:
    def __init__(self, count):
        self.processors = list(range(count))
        self.lines = []

    def print(self, msg="", *_args, **_kwargs):
        self.lines.append(msg)

    def cleaner_shader_resets_gpu(self):
        return False


def _make_test(host, other_devices=()):
    test = cli_test_set.TestSet("test_command")
    test.common = _FakeCommon(len(host.gpus))
    test.static_data = {"gpu_data": [gpu["static"] for gpu in host.gpus]}
    test.metric_data = {"gpu_data": [gpu["metric"] for gpu in host.gpus]}
    test.clk_freq = [gpu["dpm"] for gpu in host.gpus]
    test.partition_data = {
        "current_partition": [{"memory": NA, "accelerator_type": NA} for _ in host.gpus]
    }
    device_args = ["--gpu all", "--gpu 0", f"--gpu {host.bdf}", f"--gpu {host.uuid}"]
    device_args += [f"--gpu {gpu}" for gpu in range(1, len(host.gpus))]
    device_args += list(other_devices)
    args = {
        "Set Arguments:": SET_ARGS,
        "Device Arguments:": device_args,
        "Command Modifiers:": MODIFIER_ARGS,
        "": ["pass"],
    }
    test.FindArgs = lambda _cmd, section: list(args[section])
    return test


def _sweep(host, other_devices=()):
    return _make_test(host, other_devices).CreateCmds(
        "set", "Set Arguments:", "Device Arguments:", "Command Modifiers:", ""
    )


def _baseline_cond(cmd):
    """The codes the baseline sweep accepted for *cmd*."""
    words = cmd.split()
    if "--memory-partition" in words:
        return [*OK, INVAL]
    if "--clk-limit" in words or ("--clk-level" in words and "PCIE" not in words):
        return [*OK, INVALID_VALUE]
    return OK


class TestBaselineSweep(unittest.TestCase):
    def test_sweep_without_per_gpu_runs_is_the_ci_baseline(self):
        no_per_gpu = mock.patch.object(
            cli_base.TestCliBase, "PER_GPU_SWEEP_COMMANDS", frozenset(), create=True
        )
        for name, host in ALL_HOSTS.items():
            with self.subTest(host=name), no_per_gpu:
                self.assertEqual(
                    _sweep(host), [(cmd, _baseline_cond(cmd)) for cmd in host.baseline]
                )


def _on_gpu(cmd, gpu):
    """*cmd* aimed at one GPU, written the way the sweep writes its own --gpu commands."""
    words = cmd.replace(" --gpu all ", " ").split()
    at = words.index("--loglevel")
    return " ".join(words[:at] + ["--gpu", str(gpu)] + words[at:])


def _targets_all(cmd):
    words = cmd.split()
    return "--gpu" not in words or " --gpu all " in cmd


def _for_other_device(cmd):
    return bool({"--cpu", "--core", "--nic"}.intersection(cmd.split()))


def _run(test, cmds, codes):
    """RunCmds over *cmds*, each answering with its exit code in *codes*.

    Returns:
        tuple: (printed lines, failure message or None).
    """
    test.util = mock.Mock()
    test.util.RunCmdSync.side_effect = lambda cmd, msg_in=None: (codes[cmd], "", "")
    test.common.lines.clear()
    try:
        test.RunCmds(cmds)
    except AssertionError as exc:
        return test.common.lines, str(exc)
    return test.common.lines, None


def _group(cmds, cmd):
    """*cmd*'s sweep entry and its per-GPU runs, in sweep order."""
    members = set(dict(cmds)[cmd].members)
    return [(text, cond) for text, cond in cmds if text == cmd or text in members]


class TestPerGpuSweepCoverage(unittest.TestCase):
    def test_single_gpu_sweep_is_unchanged(self):
        host = GFX1151_APU
        self.assertEqual(_sweep(host), [(cmd, _baseline_cond(cmd)) for cmd in host.baseline])

    def test_every_baseline_command_still_runs_exactly_once(self):
        for name, host in MULTI_GPU_HOSTS.items():
            with self.subTest(host=name):
                texts = [cmd for cmd, _ in _sweep(host)]
                self.assertEqual(len(texts), len(set(texts)))
                self.assertEqual(sorted(set(host.baseline) - set(texts)), [])

    def test_single_gpu_commands_keep_their_baseline_codes(self):
        for name, host in MULTI_GPU_HOSTS.items():
            conds = dict(_sweep(host))
            for cmd in host.baseline:
                if not _targets_all(cmd):
                    with self.subTest(host=name, cmd=cmd):
                        self.assertEqual(conds[cmd], _baseline_cond(cmd))

    def test_all_gpu_commands_are_followed_by_a_run_on_each_gpu(self):
        for name, host in MULTI_GPU_HOSTS.items():
            cmds = _sweep(host)
            texts = [cmd for cmd, _ in cmds]
            conds = dict(cmds)
            for cmd in host.baseline:
                if not _targets_all(cmd):
                    continue
                with self.subTest(host=name, cmd=cmd):
                    members = tuple(_on_gpu(cmd, gpu) for gpu in range(len(host.gpus)))
                    self.assertEqual(conds[cmd].members, members)
                    at = texts.index(cmd)
                    # A run queued earlier is reused; the others follow at once.
                    later = [member for member in members if texts.index(member) > at]
                    self.assertEqual(texts[at + 1 : at + 1 + len(later)], later)

    def test_compute_partition_stays_excluded(self):
        # FindArgs offers it, but it waits for a confirmation the sweep cannot
        # give: it must not run, not even as a per-GPU or own-value run.
        self.assertIn("--compute-partition SPX", SET_ARGS)
        for name, host in ALL_HOSTS.items():
            with self.subTest(host=name):
                test = _make_test(host)
                cmds = test.CreateCmds(
                    "set", "Set Arguments:", "Device Arguments:", "Command Modifiers:", ""
                )
                self.assertEqual([cmd for cmd, _ in cmds if " --compute-partition " in cmd], [])
                excluded = "Excluded (waits for confirmation): amd-smi set --compute-partition"
                self.assertIn(excluded, [line.strip() for line in test.common.lines])

    def test_commands_for_other_devices_stay_ungrouped(self):
        other_devices = ["--cpu all", "--cpu 0", "--core 0", "--nic 0"]
        no_per_gpu = mock.patch.object(
            cli_base.TestCliBase, "PER_GPU_SWEEP_COMMANDS", frozenset(), create=True
        )
        for name, host in MULTI_GPU_HOSTS.items():
            with self.subTest(host=name):
                cmds = _sweep(host, other_devices)
                with no_per_gpu:
                    unexpanded = _sweep(host, other_devices)
                other = [(cmd, cond) for cmd, cond in cmds if _for_other_device(cmd)]
                self.assertTrue(other)
                # The same commands with the same codes as without per-GPU runs:
                # no GroupCheck, no --gpu run and no own-value run of them.
                self.assertEqual(
                    other, [entry for entry in unexpanded if _for_other_device(entry[0])]
                )
                for cmd, cond in other:
                    self.assertNotIn("--gpu", cmd.split(), cmd)
                    self.assertEqual(cond, _baseline_cond(cmd), cmd)
                # They leave the GPU sweep as it was.
                self.assertEqual(
                    [entry for entry in cmds if not _for_other_device(entry[0])], _sweep(host)
                )


class TestPerGpuExpectations(unittest.TestCase):
    def _conds(self, host):
        return dict(_sweep(host))

    def test_power_cap_outside_a_gpus_range_expects_invalid_value(self):
        conds = self._conds(GFX908_GFX1030)
        for watts in (1, 145, 290):
            cmd = _set(f"--power-cap {watts} ppt0")
            self.assertEqual(conds[_on_gpu(cmd, 1)], [INVALID_VALUE], watts)
        conds = self._conds(GFX90A_X2_GFX1100)
        for watts in (1, 150):
            self.assertEqual(conds[_on_gpu(_set(f"--power-cap {watts} ppt0"), 2)], [INVALID_VALUE])
        conds = self._conds(GFX90A_GFX1201)
        for watts in (1, 150):
            self.assertEqual(conds[_on_gpu(_set(f"--power-cap {watts} ppt0"), 1)], [INVALID_VALUE])

    def test_power_cap_at_the_current_cap_expects_pass(self):
        self.assertEqual(
            self._conds(GFX908_GFX1030)[_on_gpu(_set("--power-cap 290 ppt0"), 0)], [PASS]
        )
        conds = self._conds(GFX90A_X2_GFX1100)
        for gpu in (0, 1):
            self.assertEqual(conds[_on_gpu(_set("--power-cap 300 ppt0"), gpu)], [PASS])

    def test_power_cap_inside_a_gpus_range_allows_pass_or_not_supported(self):
        conds = self._conds(GFX908_GFX1030)
        for watts in (1, 145):
            self.assertEqual(conds[_on_gpu(_set(f"--power-cap {watts} ppt0"), 0)], OK)
        self.assertEqual(
            self._conds(GFX90A_X2_GFX1100)[_on_gpu(_set("--power-cap 300 ppt0"), 2)], OK
        )
        self.assertEqual(self._conds(GFX90A_GFX1201)[_on_gpu(_set("--power-cap 300 ppt0"), 1)], OK)

    def test_power_cap_boundaries_follow_the_cli(self):
        test = _make_test(GFX90A_GFX1201)
        cases = {
            (1, 211): [INVALID_VALUE],
            (1, 212): OK,
            (1, 304): [PASS],
            (1, 340): OK,
            (1, 341): [INVALID_VALUE],
            # The CLI clamps a reported minimum of 0 W up to 1 W.
            (0, 0): [INVALID_VALUE],
            (0, 1): OK,
        }
        for (gpu, watts), expected in cases.items():
            words = _on_gpu(_set(f"--power-cap {watts} ppt0"), gpu).split()
            self.assertEqual(test._power_cap_cond(words, gpu), expected, (gpu, watts))

    def test_power_cap_on_a_gpu_without_a_cap_expects_not_supported(self):
        # Derived: the gfx1030 GPU reporting no ppt0 cap.
        host = _host(
            [GFX908, _with_cap(GFX1030, NO_CAP)],
            GFX908_GFX1030.bdf,
            GFX908_GFX1030.uuid,
            profiles=PROFILES,
            determinism=1552,
            caps=[1, 145, 290],
            xgmi=[0],
            clk_levels=[("SCLK", 3), ("MCLK", 1), ("FCLK", 1), ("SOCCLK", 1), ("PCIE", 1)],
            clk_limits=["SCLK MAX 1210", "SCLK MIN 600"],
        )
        conds = self._conds(host)
        for watts in (1, 145, 290):
            self.assertEqual(conds[_on_gpu(_set(f"--power-cap {watts} ppt0"), 1)], [NOT_SUPPORTED])

    def test_power_cap_with_a_zero_current_cap_expects_invalid_value(self):
        # Derived: the gfx1030 GPU with its current cap at 0, which the CLI refuses
        # to change even for a value inside the range.
        host = _host(
            [GFX908, _with_cap(GFX1030, _cap(255, 272, 0))],
            GFX908_GFX1030.bdf,
            GFX908_GFX1030.uuid,
            profiles=PROFILES,
            determinism=1552,
            caps=[1, 145, 290],
            xgmi=[0],
            clk_levels=[("SCLK", 3), ("MCLK", 1), ("FCLK", 1), ("SOCCLK", 1), ("PCIE", 1)],
            clk_limits=["SCLK MAX 1210", "SCLK MIN 600"],
        )
        conds = self._conds(host)
        for watts in (1, 145, 290, 255, 263, 272):
            with self.subTest(watts=watts):
                self.assertEqual(
                    conds[_on_gpu(_set(f"--power-cap {watts} ppt0"), 1)], [INVALID_VALUE]
                )

    def test_pcie_levels_beyond_a_gpus_count_expect_invalid_value(self):
        pcie = _set(f"--clk-level PCIE {' '.join(str(level) for level in range(24))}")
        conds = self._conds(GFX90A_X2_GFX1100)
        self.assertEqual(conds[_on_gpu(pcie, 0)], OK)
        self.assertEqual(conds[_on_gpu(pcie, 1)], OK)
        self.assertEqual(conds[_on_gpu(pcie, 2)], [INVALID_VALUE])
        self.assertEqual(self._conds(GFX90A_GFX1201)[_on_gpu(pcie, 1)], [INVALID_VALUE])

    def test_pcie_level_boundary_and_unreadable_levels(self):
        test = _make_test(GFX90A_X2_GFX1100)
        self.assertEqual(test._pcie_level_cond(["0", "1", "2"], 2, OK), OK)
        self.assertEqual(test._pcie_level_cond(["3"], 2, OK), [INVALID_VALUE])
        self.assertEqual(test._pcie_level_cond(["0", "--gpu", "2"], 2, OK), OK)
        # No PCIe level list: the CLI fails reading bandwidth before it validates.
        self.assertEqual(_make_test(GFX1151_APU)._pcie_level_cond(["0"], 0, OK), OK)

    def test_other_commands_keep_their_codes_on_each_gpu(self):
        conds = self._conds(GFX90A_GFX1201)
        for cmd in (
            _set("--clk-level FCLK 0 1 2"),
            _set("--clk-level SCLK 0 1"),
            _set("--memory-partition NPS1"),
            _set("--perf-level AUTO"),
            _set("--ptl-format I8,F16"),
            _set("--clk-limit MCLK MAX 1300"),
        ):
            for gpu in (0, 1):
                with self.subTest(cmd=cmd, gpu=gpu):
                    self.assertEqual(conds[_on_gpu(cmd, gpu)], _baseline_cond(cmd))


class TestOwnValueRuns(unittest.TestCase):
    def _own_values(self, host):
        cmds = _sweep(host)
        members = set()
        for _, cond in cmds:
            members.update(getattr(cond, "members", ()))
        return {cmd: cond for cmd, cond in cmds if cmd not in host.baseline and cmd not in members}

    def test_mixed_hosts_run_each_gpus_own_values(self):
        expected = OWN_VALUE_RUNS
        for name, host in MIXED_HOSTS.items():
            with self.subTest(host=name):
                self.assertEqual(self._own_values(host), expected[name])

    def test_same_model_host_adds_no_own_value_runs(self):
        self.assertEqual(self._own_values(GFX90A_X2), {})

    def test_own_clock_limits_keep_max_before_min(self):
        texts = [cmd for cmd, _ in _sweep(GFX908_GFX1030)]
        for clk in ("SCLK", "MCLK", "FCLK"):
            prefix = f"amd-smi set --clk-limit {clk} "
            gpu1 = [cmd.split() for cmd in texts if cmd.startswith(prefix) and " --gpu 1 " in cmd]
            bounds = [words[4] for words in gpu1]
            self.assertIn("MIN", bounds, clk)
            expected = ["MAX"] * bounds.count("MAX") + ["MIN"] * bounds.count("MIN")
            self.assertEqual(bounds, expected, clk)


class TestAggregateExitCode(unittest.TestCase):
    def test_matches_the_cli_collector(self):
        for count in (1, 2, 3):
            for codes in itertools.product([0, 2, 198, 204], repeat=count):
                collector = AmdSmiErrorCollector()
                for code in codes:
                    # A GPU exits 204 only when its own failures differ.
                    for recorded in [2, 198] if code == MIXED else [code] if code else []:
                        collector.record(recorded)
                with self.subTest(codes=codes):
                    self.assertEqual(
                        cli_base._aggregate_exit_code(codes), collector.resolve_exit_code()
                    )

    def test_reachable_aggregates_match_every_combination(self):
        codes = [PASS, NOT_SUPPORTED, INVALID_VALUE, MIXED]
        subsets = [
            list(subset)
            for size in range(len(codes) + 1)
            for subset in itertools.combinations(codes, size)
        ]
        for count in (1, 2, 3, 4):
            mismatches = []
            for per_gpu in itertools.product(subsets, repeat=count):
                combinations = itertools.product(*per_gpu)
                expected = sorted({cli_base._aggregate_exit_code(c) for c in combinations})
                actual = cli_base._reachable_aggregates(per_gpu)
                if actual != expected:
                    mismatches.append((per_gpu, actual, expected))
            self.assertEqual(mismatches[:3], [], f"{count} GPUs")

    def test_reachable_aggregates_scale_to_many_gpus(self):
        # 3**64 combinations: enumerating them would never finish.
        self.assertEqual(
            cli_base._reachable_aggregates([[*OK, INVALID_VALUE]] * 64),
            [PASS, NOT_SUPPORTED, INVALID_VALUE, MIXED],
        )
        self.assertEqual(
            cli_base._reachable_aggregates([OK] * 63 + [[INVALID_VALUE]]), [INVALID_VALUE, MIXED]
        )

    def test_only_runtime_dependent_commands_accept_any_allowed_aggregate(self):
        conds = dict(_sweep(GFX908_GFX1030))
        exact = {
            _set("--power-cap 1 ppt0"): True,
            _set("--clk-level PCIE 0"): True,
            _set("--perf-level AUTO"): True,
            _set("--ptl-format I8,F16"): True,
            _set("--clk-level SCLK 0 1 2"): False,
            _set("--clk-level MCLK 0"): False,
            _set("--clk-limit SCLK MAX 1210"): False,
            _set("--memory-partition NPS1"): False,
        }
        for cmd, expected in exact.items():
            self.assertEqual(conds[cmd].exact, expected, cmd)


class TestGroupGrading(unittest.TestCase):
    def _grade(self, host, cmd, all_gpu_rc, per_gpu_rcs):
        test = _make_test(host)
        cmds = _group(_sweep(host), cmd)
        codes = {cmd: all_gpu_rc}
        codes.update({_on_gpu(cmd, gpu): rc for gpu, rc in enumerate(per_gpu_rcs)})
        return _run(test, cmds, codes)

    def test_all_gpu_code_must_equal_the_observed_aggregate(self):
        cmd = _set("--power-cap 1 ppt0")
        _, failure = self._grade(GFX908_GFX1030, cmd, INVALID_VALUE, [PASS, INVALID_VALUE])
        self.assertIsNone(failure)
        # MIXED_DEVICE_ERRORS is a possible aggregate here, but not the observed one.
        _, failure = self._grade(GFX908_GFX1030, cmd, MIXED, [PASS, INVALID_VALUE])
        self.assertIn(cmd, failure)

    def test_differing_gpu_failures_aggregate_to_mixed_device_errors(self):
        cmd = _set(f"--clk-level PCIE {' '.join(str(level) for level in range(24))}")
        rcs = [NOT_SUPPORTED, NOT_SUPPORTED, INVALID_VALUE]
        _, failure = self._grade(GFX90A_X2_GFX1100, cmd, MIXED, rcs)
        self.assertIsNone(failure)
        _, failure = self._grade(GFX90A_X2_GFX1100, cmd, INVALID_VALUE, rcs)
        self.assertIn(cmd, failure)

    def test_runtime_dependent_commands_accept_any_allowed_aggregate(self):
        cases = [
            (GFX90A_GFX1201, _set("--clk-level FCLK 0 1 2"), MIXED, [NOT_SUPPORTED] * 2),
            (GFX908_GFX1030, _set("--clk-limit SCLK MAX 1210"), INVALID_VALUE, [PASS] * 2),
            (GFX908_GFX1030, _set("--memory-partition NPS1"), MIXED, [NOT_SUPPORTED] * 2),
        ]
        for host, cmd, all_gpu_rc, per_gpu_rcs in cases:
            with self.subTest(cmd=cmd):
                _, failure = self._grade(host, cmd, all_gpu_rc, per_gpu_rcs)
                self.assertIsNone(failure)
        # An aggregate no combination of per-GPU codes can produce still fails.
        cmd = _set("--clk-level FCLK 0 1 2")
        _, failure = self._grade(GFX90A_GFX1201, cmd, INVAL, [NOT_SUPPORTED] * 2)
        self.assertIn(cmd, failure)

    def test_a_gpu_outside_its_codes_fails_even_when_the_aggregate_matches(self):
        cmd = _set("--power-cap 1 ppt0")
        _, failure = self._grade(GFX908_GFX1030, cmd, PASS, [PASS, PASS])
        self.assertIn(_on_gpu(cmd, 1), failure)
        self.assertNotIn(f"{cmd} ", failure.replace(_on_gpu(cmd, 1), ""))

    def test_grouped_line_follows_its_per_gpu_runs_and_names_their_codes(self):
        cmd = _set("--power-cap 1 ppt0")
        lines, _ = self._grade(GFX908_GFX1030, cmd, INVALID_VALUE, [PASS, INVALID_VALUE])
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].strip().startswith(_on_gpu(cmd, 0)))
        self.assertTrue(lines[1].strip().startswith(_on_gpu(cmd, 1)))
        self.assertTrue(lines[2].strip().startswith(cmd))
        self.assertIn("[aggregate of per-GPU runs: GPU 0 rc=0, GPU 1 rc=198]", lines[2])

    def test_ungrouped_lines_keep_their_format(self):
        test = _make_test(GFX1151_APU)
        cmd = _set("--perf-level DETERMINISM")
        lines, failure = _run(test, [(cmd, OK)], {cmd: NOT_SUPPORTED})
        self.assertIsNone(failure)
        expected = f"    {cmd:{len(cmd) + 2}s}: Unsupported: Received and Allowed FAIL"
        self.assertEqual(lines, [f"{expected} (rc=2 NOT_SUPPORTED)"])

    def test_a_group_whose_per_gpu_runs_did_not_run_fails(self):
        test = _make_test(GFX908_GFX1030)
        cmd = _set("--perf-level AUTO")
        check = dict(_sweep(GFX908_GFX1030))[cmd]
        _, failure = _run(test, [(cmd, check)], {cmd: PASS})
        self.assertIn("not every per-GPU run of it ran", failure)


def _own(gpu, args, cond):
    return _on_gpu(_set(args), gpu), cond


CLK_OK = [*OK, INVALID_VALUE]
OWN_VALUE_RUNS = {
    # gfx1030 (GPU 1) next to gfx908 (GPU 0)
    "gfx908 + gfx1030": dict(
        [
            _own(1, "--perf-determinism 2625", OK),
            _own(1, "--power-cap 255 ppt0", OK),
            _own(1, "--power-cap 263 ppt0", OK),
            _own(1, "--power-cap 272 ppt0", [PASS]),
            _own(0, "--clk-level SCLK 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", CLK_OK),
            _own(1, "--clk-level MCLK 0 1 2 3", CLK_OK),
            _own(1, "--clk-level FCLK 0 1 2", CLK_OK),
            _own(1, "--clk-level SOCCLK 0 1 2", CLK_OK),
            _own(1, "--clk-level PCIE 0 1", OK),
            _own(1, "--clk-limit SCLK MAX 2060", CLK_OK),
            _own(1, "--clk-limit SCLK MIN 1020", CLK_OK),
            _own(1, "--clk-limit MCLK MAX 780", CLK_OK),
            _own(1, "--clk-limit MCLK MIN 330", CLK_OK),
            _own(1, "--clk-limit FCLK MAX 1590", CLK_OK),
            _own(1, "--clk-limit FCLK MIN 860", CLK_OK),
        ]
    ),
    # gfx1100 (GPU 2) next to two gfx90a
    "2x gfx90a + gfx1100": dict(
        [
            _own(2, "--perf-determinism 2421", OK),
            _own(2, "--power-cap 272 ppt0", OK),
            _own(2, "--power-cap 311 ppt0", OK),
            _own(2, "--power-cap 350 ppt0", OK),
            _own(0, "--clk-level SOCCLK 0 1 2 3 4", CLK_OK),
            _own(1, "--clk-level SOCCLK 0 1 2 3 4", CLK_OK),
            _own(2, "--clk-level PCIE 0 1 2", OK),
            _own(2, "--clk-limit SCLK MAX 1910", CLK_OK),
            _own(2, "--clk-limit SCLK MIN 970", CLK_OK),
            _own(2, "--clk-limit MCLK MAX 970", CLK_OK),
            _own(2, "--clk-limit MCLK MIN 390", CLK_OK),
            _own(2, "--clk-limit FCLK MAX 1880", CLK_OK),
            _own(2, "--clk-limit FCLK MIN 1030", CLK_OK),
        ]
    ),
    # gfx1201 (GPU 1) next to gfx90a (GPU 0)
    "gfx90a + gfx1201": dict(
        [
            _own(1, "--perf-determinism 2450", OK),
            _own(1, "--power-cap 212 ppt0", OK),
            _own(1, "--power-cap 276 ppt0", OK),
            _own(1, "--power-cap 340 ppt0", OK),
            _own(0, "--clk-level SOCCLK 0 1 2 3 4", CLK_OK),
            _own(1, "--clk-level MCLK 0 1 2 3 4 5", CLK_OK),
            _own(1, "--clk-level PCIE 0 1 2", OK),
            _own(1, "--clk-limit SCLK MAX 1930", CLK_OK),
            _own(1, "--clk-limit SCLK MIN 980", CLK_OK),
            _own(1, "--clk-limit MCLK MAX 970", CLK_OK),
            _own(1, "--clk-limit MCLK MIN 390", CLK_OK),
            _own(1, "--clk-limit FCLK MAX 1880", CLK_OK),
            _own(1, "--clk-limit FCLK MIN 840", CLK_OK),
        ]
    ),
}
