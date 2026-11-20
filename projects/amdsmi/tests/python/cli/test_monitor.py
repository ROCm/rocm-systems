#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: monitor command."""

import csv
import io
import json
import numbers

from cli.base import TestCliBase
from common.common import amdsmi

_IDLE_PROCESS = "No running processes detected"

# Table columns ahead of the memory ones; those come last and are named after
# the memory type (VRAM_* on dGPUs, GTT_* on some APUs).
_DEFAULT_COLUMNS = [
    "GPU",
    "XCP",
    "POWER",
    "GPU_T",
    "MEM_T",
    "GFX_CLK",
    "GFX%",
    "MEM%",
    "ENC%",
    "DEC%",
]
_DEFAULT_MEM_COLUMNS = 1
_DEFAULT_FIELDS = [
    "power_usage",
    "hotspot_temperature",
    "memory_temperature",
    "gfx_clk",
    "gfx",
    "mem",
    "encoder",
    "decoder",
    "vram_used",
    "vram_total",
]

_ARGS = "-p -t -u -m -n -d -v -q"
_ARGS_COLUMNS = [
    "GPU",
    "XCP",
    "POWER",
    "PWR_CAP",
    "GPU_T",
    "MEM_T",
    "GFX_CLK",
    "GFX%",
    "MEM%",
    "MEM_CLOCK",
    "ENC%",
    "DEC%",
    "VCLOCK",
    "DCLOCK",
]
_ARGS_MEM_COLUMNS = 4
_ARGS_FIELDS = [
    "power_usage",
    "max_power",
    "hotspot_temperature",
    "memory_temperature",
    "gfx_clk",
    "gfx",
    "mem",
    "mem_clock",
    "encoder",
    "decoder",
    "vclock",
    "dclock",
    "vram_used",
    "vram_free",
    "vram_total",
    "vram_percent",
]


def _call(fn, *args):
    try:
        return fn(*args)
    except amdsmi.AmdSmiLibraryException:
        return None


def _has_value(value):
    if isinstance(value, list):
        return any(_has_value(item) for item in value)
    return value is not None and value != "N/A"


def _library_fields(handle, xcp):
    """Monitor fields the library can back on *handle*, from every source monitor reads.

    A field may only show N/A when this says the library has nothing for it.
    *xcp* is the partition row monitor reads per-partition stats from.
    """
    metrics = _call(amdsmi.amdsmi_get_gpu_metrics_info, handle) or {}
    vcn_busy = metrics.get("xcp_stats.vcn_busy")
    if isinstance(vcn_busy, list) and isinstance(xcp, int) and xcp < len(vcn_busy):
        vcn_busy = vcn_busy[xcp]
    else:
        vcn_busy = None

    def metric(*keys):
        return any(_has_value(metrics.get(key)) for key in keys)

    def dpm(clk_type):
        freq = _call(amdsmi.amdsmi_get_clk_freq, handle, clk_type)
        return bool(freq) and freq["current"] < len(freq["frequency"])

    def temp(sensor):
        reading = _call(
            amdsmi.amdsmi_get_temp_metric, handle, sensor, amdsmi.AmdSmiTemperatureMetric.CURRENT
        )
        return _has_value(reading)

    cap = _call(amdsmi.amdsmi_get_power_cap_info, handle, 0) or {}
    memory = any(
        _call(amdsmi.amdsmi_get_gpu_memory_total, handle, mem_type) is not None
        for mem_type in (amdsmi.AmdSmiMemoryType.VRAM, amdsmi.AmdSmiMemoryType.GTT)
    )
    return {
        "power_usage": metric("current_socket_power", "average_socket_power"),
        "max_power": _has_value(cap.get("power_cap")),
        "hotspot_temperature": metric("temperature_hotspot", "apu_metrics.temperature_gfx")
        or temp(amdsmi.AmdSmiTemperatureType.HOTSPOT),
        "memory_temperature": metric("temperature_mem") or temp(amdsmi.AmdSmiTemperatureType.VRAM),
        "gfx_clk": metric(
            "current_gfxclk", "apu_metrics.current_gfxclk", "apu_metrics.average_gfxclk_frequency"
        )
        or dpm(amdsmi.AmdSmiClkType.GFX),
        "gfx": metric("average_gfx_activity"),
        "mem": metric("average_umc_activity"),
        "mem_clock": metric(
            "current_uclk", "apu_metrics.current_uclk", "apu_metrics.average_uclk_frequency"
        )
        or dpm(amdsmi.AmdSmiClkType.MEM),
        # The CLI does not read encoder activity yet.
        "encoder": False,
        "decoder": metric("vcn_activity") or _has_value(vcn_busy),
        "vclock": metric("current_vclk0", "apu_metrics.average_vclk_frequency"),
        "dclock": metric("current_dclk0"),
        "vram_used": memory,
        "vram_free": memory,
        "vram_total": memory,
        "vram_percent": memory,
    }


class TestMonitor(TestCliBase):
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

    def test_monitor_without_workload(self):
        """Idle ``amd-smi monitor``: table, JSON and CSV shape, and no unexpected N/A."""
        self.common.print_func_name("")
        self._check_monitor("", _DEFAULT_COLUMNS, _DEFAULT_MEM_COLUMNS, _DEFAULT_FIELDS)

    def test_monitor_without_workload_with_arguments(self):
        """Idle ``amd-smi monitor`` with explicit metric flags and the process table."""
        self.common.print_func_name("")
        self._check_monitor(_ARGS, _ARGS_COLUMNS, _ARGS_MEM_COLUMNS, _ARGS_FIELDS, processes=True)

    def _run(self, cmd):
        (rc, out, err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"'{cmd}' failed with rc={rc}: {err}")
        self.assertTrue(out, f"'{cmd}' printed nothing")
        return out

    def _check_monitor(self, args, columns, mem_columns, fields, processes=False):
        gpu_ids = [str(entry["gpu"]) for entry in self.list_data]
        self.assertTrue(gpu_ids, "amd-smi list reported no GPUs")

        entries = self._check_json(f"amd-smi monitor {args} --json", gpu_ids, fields, processes)
        self._check_csv(f"amd-smi monitor {args} --csv", entries, fields, processes)
        self._check_table(f"amd-smi monitor {args}", gpu_ids, columns, mem_columns, processes)

        unexpected = []
        self.common.amdsmi_smart_init()
        try:
            # Handle order is the CLI's GPU numbering.
            handles = amdsmi.amdsmi_get_processor_handles()
            for entry in entries:
                readable = _library_fields(handles[entry["gpu"]], entry["xcp"])
                for field in fields:
                    if entry[field] == "N/A" and readable[field]:
                        unexpected.append(f"gpu {entry['gpu']} {field}")
        finally:
            amdsmi.amdsmi_shut_down()
        self.assertFalse(
            unexpected, f"N/A where the library reports a value: {', '.join(unexpected)}"
        )

    def _check_json(self, cmd, gpu_ids, fields, processes):
        entries = json.loads(self._run(cmd))
        self.assertIsInstance(entries, list, cmd)
        self.assertEqual([str(entry["gpu"]) for entry in entries], gpu_ids, cmd)

        expected = ["gpu", "xcp"] + (["process_list"] if processes else []) + fields
        for entry in entries:
            self.assertEqual(list(entry), expected, f"{cmd}: gpu {entry['gpu']} keys")
            for field in fields:
                value = entry[field]
                if value == "N/A":
                    continue
                self.assertIsInstance(value, dict, f"{cmd}: gpu {entry['gpu']} {field}={value!r}")
                self.assertIsInstance(value["value"], numbers.Number, f"{cmd}: {field}")
                self.assertGreaterEqual(value["value"], 0, f"{cmd}: {field}")
                self.assertTrue(value["unit"], f"{cmd}: {field} has no unit")
            if processes:
                self._check_processes(cmd, entry["process_list"])
        return entries

    def _check_processes(self, cmd, process_list):
        self.assertTrue(process_list, f"{cmd}: empty process_list")
        for row in process_list:
            info = row["process_info"]
            if info == _IDLE_PROCESS:
                continue
            self.assertIsInstance(info, dict, f"{cmd}: process_info={info!r}")
            for key in ("name", "pid", "mem_usage"):
                self.assertIn(key, info, f"{cmd}: process_info")

    def _check_csv(self, cmd, entries, fields, processes):
        # --process prints its own table after a blank line.
        tables = [table for table in self._run(cmd).split("\n\n") if table.strip()]
        self.assertEqual(len(tables), 2 if processes else 1, f"{cmd}: tables")

        reader = csv.DictReader(io.StringIO(tables[0]))
        rows = list(reader)
        self.assertEqual(reader.fieldnames, ["gpu", "xcp"] + fields, f"{cmd}: header")
        self.assertEqual([row["gpu"] for row in rows], [str(e["gpu"]) for e in entries], cmd)
        for row, entry in zip(rows, entries):
            for field in fields:
                cell = row[field]
                # Readings move between runs; whether one exists does not.
                self.assertEqual(
                    cell == "N/A", entry[field] == "N/A", f"{cmd}: gpu {row['gpu']} {field}"
                )
                if cell != "N/A":
                    float(cell)

        if processes:
            reader = csv.DictReader(io.StringIO(tables[1]))
            self.assertEqual(reader.fieldnames[0], "gpu", f"{cmd}: process header")
            self.assertTrue(list(reader), f"{cmd}: no process rows")

    def _check_table(self, cmd, gpu_ids, columns, mem_columns, processes):
        lines = [line for line in self._run(cmd).splitlines() if line.strip()]
        header = lines[0].split()
        self.assertEqual(header[: len(columns)], columns, f"{cmd}: header")
        self.assertEqual(len(header), len(columns) + mem_columns, f"{cmd}: header {header}")
        self.assertEqual(
            [line.split()[0] for line in lines[1 : 1 + len(gpu_ids)]], gpu_ids, f"{cmd}: rows"
        )

        if processes:
            self.assertIn("PROCESS INFO:", lines, f"{cmd}: no process table")
            start = lines.index("PROCESS INFO:")
            self.assertEqual(lines[start + 1].split()[:3], ["GPU", "NAME", "PID"], cmd)
            self.assertGreater(len(lines), start + 2, f"{cmd}: no process rows")
