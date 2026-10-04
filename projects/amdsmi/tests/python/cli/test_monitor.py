#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: monitor command and its agreement with metric."""

import json
import os
import signal
import subprocess
import time
import unittest
from collections import namedtuple

import common.common as common

from cli.base import TestCliBase

# Live counters. Two samples of these legitimately differ, so only their
# presence and units are cross-checked, never their values.
VOLATILE_KEYS = (
    "power_usage",
    "hotspot_temperature",
    "memory_temperature",
    "gfx_clk",
    "gfx",
    "mem",
)
# Memory accounting. Both commands read it from the same driver call.
STABLE_KEYS = ("vram_used", "vram_free", "vram_total")

# Monitor column -> path into a metric gpu_data entry. The vram keys are
# resolved separately because metric reports both the VRAM and GTT pools.
METRIC_PATHS = {
    "power_usage": ("power", "socket_power"),
    "hotspot_temperature": ("temperature", "hotspot"),
    "memory_temperature": ("temperature", "mem"),
    "gfx_clk": ("clock", "gfx_0", "clk"),
    "gfx": ("usage", "gfx_activity"),
    "mem": ("usage", "umc_activity"),
}

Field = namedtuple("Field", "unit value")
Result = namedtuple("Result", "gpu key passed message")

NA = Field("N/A", None)


def _field(container, *path):
    """Resolve a {value, unit} node, yielding NA if any level is absent or "N/A"."""
    node = container
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return NA
        node = node[key]
    if not isinstance(node, dict) or "value" not in node:
        return NA
    return Field(node.get("unit", ""), node["value"])


class TestMonitor(TestCliBase):
    # Only the columns the comparisons read; test_command covers the rest.
    monitor_args = "--power-usage --temperature --gfx --mem --vram-usage"
    tolerance = 0.1
    # Counters a compute/memory workload is expected to move.
    workload_keys = ("power_usage", "gfx", "mem")
    # Config file, extra options, and the time the workload needs to ramp up.
    rvs_configs = (("iet_single.conf", "", 5.0), ("mem.conf", "--numTimes 3", 1.0))

    def test_command(self):
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

    def _run_json(self, cmd, time_out=30.0):
        """Run a CLI command, require a zero exit, and return its parsed JSON."""
        rc, std_out, std_err = self.util.RunCmdSync(cmd, time_out=time_out)
        self.assertEqual(rc, 0, f"{cmd}\n{self.tab}rc={rc} std_err={std_err}")
        self.assertTrue(std_out, f"{cmd}\n{self.tab}produced no output")
        return json.loads(std_out)

    def _compare(self, gpu, key, left, right, component):
        """Compare one field. Volatile counters are checked for shape, not value."""
        if left.unit == "N/A" and right.unit == "N/A":
            return None
        if (left.unit == "N/A") != (right.unit == "N/A"):
            return Result(gpu, key, False, f"reported by only one of Monitor/{component}")
        if left.unit != right.unit:
            return Result(gpu, key, False, f"unit {left.unit} != {right.unit}")
        if key in VOLATILE_KEYS:
            return Result(gpu, key, True, f"present in both ({left.unit})")
        diff = abs(left.value - right.value)
        max_diff = max(1.0, max(abs(left.value), abs(right.value)) * self.tolerance)
        return Result(
            gpu,
            key,
            diff <= max_diff,
            f"{left.value:g} vs {right.value:g} {left.unit}: diff {diff:g} max {max_diff:g}",
        )

    def _compare_monitor_runs(self, monitor1, monitor2):
        self.assertEqual(len(monitor1), len(monitor2), "monitor row count changed between runs")
        results = []
        for gpu, (row1, row2) in enumerate(zip(monitor1, monitor2)):
            for key in VOLATILE_KEYS + STABLE_KEYS:
                result = self._compare(gpu, key, _field(row1, key), _field(row2, key), "Monitor")
                if result is not None:
                    results.append(result)
        return results

    def _metric_vram_pool(self, monitor_row, metric_row):
        """Metric reports both pools; pick the one whose total is what monitor chose."""
        total = _field(monitor_row, "vram_total")
        for pool in ("vram", "gtt"):
            if _field(metric_row, "mem_usage", f"total_{pool}").value == total.value:
                return pool
        return None

    def _compare_monitor_to_metric(self, monitor, metric):
        gpu_data = metric["gpu_data"]
        self.assertEqual(
            len(monitor),
            len(gpu_data),
            f"monitor returned {len(monitor)} rows, metric returned {len(gpu_data)}",
        )
        results = []
        for gpu, (row, metric_row) in enumerate(zip(monitor, gpu_data)):
            for key, path in METRIC_PATHS.items():
                result = self._compare(
                    gpu, key, _field(row, key), _field(metric_row, *path), "Metric"
                )
                if result is not None:
                    results.append(result)
            pool = self._metric_vram_pool(row, metric_row)
            if pool is None:
                continue
            for key in STABLE_KEYS:
                metric_field = _field(metric_row, "mem_usage", f"{key[5:]}_{pool}")
                result = self._compare(gpu, key, _field(row, key), metric_field, "Metric")
                if result is not None:
                    results.append(result)
        return results

    def _compare_under_load(self, baseline, loaded):
        """A workload must move the responsive counters; other fields are ignored."""
        self.assertEqual(len(baseline), len(loaded), "monitor row count changed under load")
        results = []
        for gpu, (base_row, load_row) in enumerate(zip(baseline, loaded)):
            for key in self.workload_keys:
                base = _field(base_row, key)
                load = _field(load_row, key)
                if base.unit == "N/A" or load.unit == "N/A":
                    continue
                diff = abs(load.value - base.value)
                min_diff = max(1.0, abs(base.value) * self.tolerance)
                results.append(
                    Result(
                        gpu,
                        key,
                        diff >= min_diff,
                        f"{base.value:g} -> {load.value:g} {base.unit}: "
                        f"diff {diff:g} min {min_diff:g}",
                    )
                )
        return results

    def _report_results(self, component, results):
        """Print every comparison, then fail listing only the ones that did not hold."""
        self.assertTrue(results, f"Monitor to {component}: no comparable fields")
        width = max(len(result.key) for result in results)
        lines = [
            f"{self.tab}gpu={result.gpu} {result.key:>{width}s}: "
            f"{'ok' if result.passed else 'FAIL'} {result.message}"
            for result in results
        ]
        if common.verbose == common.VERBOSITY_VERBOSE:
            self.common.print(f"{self.tab}Monitor to {component}\n" + "\n".join(lines))

        failures = [line for line, result in zip(lines, results) if not result.passed]
        if failures:
            self.fail(f"Monitor to {component}:\n" + "\n".join(failures))
        return

    @staticmethod
    def _start_workload(cmd):
        """Own process group, so the whole workload process tree can be killed."""
        return subprocess.Popen(
            cmd.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    @staticmethod
    def _stop_workload(proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
        return

    def test_monitor_repeatability(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor repeatability"
        self.common.print(msg)

        cmd = f"amd-smi monitor {self.monitor_args} --json"
        monitor1 = self._run_json(cmd)
        monitor2 = self._run_json(cmd)
        self._report_results("Monitor", self._compare_monitor_runs(monitor1, monitor2))
        return

    def test_monitor_metric(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor metric"
        self.common.print(msg)

        monitor = self._run_json(f"amd-smi monitor {self.monitor_args} --json")
        metric = self._run_json("amd-smi metric --json")
        self._report_results("Metric", self._compare_monitor_to_metric(monitor, metric))
        return

    @unittest.skip("needs a reliable workload generator")
    def test_monitor_with_workload(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor with workload"
        self.common.print(msg)

        rocm_root = os.getenv("ROCM_HOME") or os.getenv("ROCM_PATH") or "/opt/rocm"
        rvs_exe = os.path.join(rocm_root, "bin", "rvs")
        conf_dir = os.path.join(rocm_root, "share", "rocm-validation-suite", "conf")

        rc, _, _ = self.util.RunCmdSync(f"{rvs_exe} --version")
        if rc != 0:
            self.skipTest(f"workload generator {rvs_exe} not found")
        missing = [
            name
            for name, _, _ in self.rvs_configs
            if not os.path.exists(os.path.join(conf_dir, name))
        ]
        if missing:
            self.skipTest(f"rvs configs not found in {conf_dir}: {', '.join(missing)}")

        cmd_monitor = f"amd-smi monitor {self.monitor_args} --json"
        baseline = self._run_json(cmd_monitor)

        workloads = [
            self._start_workload(
                f"{rvs_exe} --config {os.path.join(conf_dir, name)} {options} --json --quiet"
            )
            for name, options, _ in self.rvs_configs
        ]
        try:
            time.sleep(max(ramp for _, _, ramp in self.rvs_configs))
            loaded = self._run_json(cmd_monitor)
        finally:
            for proc in workloads:
                self._stop_workload(proc)

        self._report_results("Workload", self._compare_under_load(baseline, loaded))
        return
