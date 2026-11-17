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


def _make_test(host):
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
    args = {
        "Set Arguments:": SET_ARGS,
        "Device Arguments:": device_args,
        "Command Modifiers:": MODIFIER_ARGS,
        "": ["pass"],
    }
    test.FindArgs = lambda _cmd, section: list(args[section])
    return test


def _sweep(host):
    return _make_test(host).CreateCmds(
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
