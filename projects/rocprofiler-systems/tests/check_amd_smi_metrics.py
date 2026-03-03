#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Check available AMD SMI metrics on the current system and output a
GpuMetricAvailability struct per GPU. The struct fields match the
rocprofsys::amd_smi::settings C++ struct and can be used to filter
validation rules in tests.

Uses 'amd-smi monitor --json' for the bulk of detection, then
'amd-smi metric -x' for XGMI support (not exposed by monitor).

Usage:
    # Query live GPU(s)
    python3 check-amd-smi-metrics.py

    # JSON output
    python3 check-amd-smi-metrics.py --json
"""

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass


# Mapping from amd-smi monitor JSON keys to settings struct fields.
#
# monitor key            -> settings field
# ─────────────────────────────────────────
# gfx, mem               -> busy
# hotspot_temperature,
#   memory_temperature    -> temp
# power_usage            -> power
# vram_used, vram_total  -> mem_usage
# decoder                -> vcn_activity   (VCN decode engine)
# encoder                -> jpeg_activity  (JPEG / encode engine)
# pcie_bw                -> pcie
# (from amd-smi metric)  -> xgmi
_MONITOR_FIELD_MAP: dict[str, list[str]] = {
    "busy": ["gfx", "mem"],
    "temp": ["hotspot_temperature", "memory_temperature"],
    "power": ["power_usage"],
    "mem_usage": ["vram_used", "vram_total"],
    "vcn_activity": ["decoder"],
    "jpeg_activity": ["encoder"],
    "pcie": ["pcie_bw"],
}


@dataclass
class GpuMetricAvailability:
    """Mirrors rocprofsys::amd_smi::settings — one instance per GPU."""

    gpu_id: int = 0
    busy: bool = False
    temp: bool = False
    power: bool = False
    mem_usage: bool = False
    vcn_activity: bool = False
    jpeg_activity: bool = False
    xgmi: bool = False
    pcie: bool = False
    # sdma_usage is not reported by amd-smi CLI; it is a compile-time
    # feature (AMD_SMI_SDMA_SUPPORTED) and cannot be detected here.

    def to_metrics_string(self) -> str:
        """Return comma-separated string for ROCPROFSYS_AMD_SMI_METRICS."""
        categories = []
        for name in (
            "busy",
            "temp",
            "power",
            "mem_usage",
            "vcn_activity",
            "jpeg_activity",
            "xgmi",
            "pcie",
        ):
            if getattr(self, name):
                categories.append(name)
        return ", ".join(categories)


def _is_available(value) -> bool:
    """Return True if a monitor JSON value represents an available metric."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != "N/A"
    if isinstance(value, dict):
        # e.g. {"value": 25, "unit": "W"} — present means available
        return True
    # numeric
    return True


def parse_monitor_json(data: list[dict]) -> list[GpuMetricAvailability]:
    """Parse 'amd-smi monitor --json' output into per-GPU availability structs."""
    gpus: list[GpuMetricAvailability] = []

    for entry in data:
        gpu = GpuMetricAvailability(gpu_id=entry.get("gpu", 0))

        for field, keys in _MONITOR_FIELD_MAP.items():
            available = any(_is_available(entry.get(k)) for k in keys)
            setattr(gpu, field, available)

        gpus.append(gpu)

    return gpus


def _parse_xgmi_metric(text: str) -> dict[int, bool]:
    """Parse 'amd-smi metric -x' output. Returns {gpu_id: xgmi_available}."""
    result: dict[int, bool] = {}
    current_gpu = None

    for line in text.splitlines():
        gpu_match = re.match(r"^GPU:\s*(\d+)", line.strip())
        if gpu_match:
            current_gpu = int(gpu_match.group(1))
            continue
        if current_gpu is not None:
            m = re.match(r"^\s+XGMI_ERR:\s*(.+)$", line)
            if m:
                result[current_gpu] = m.group(1).strip() != "N/A"
                current_gpu = None

    return result


def _run_command(cmd: list[str]) -> str:
    """Run a command and return stdout, or exit on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(
                f"{' '.join(cmd)} failed (rc={result.returncode}):",
                file=sys.stderr,
            )
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        return result.stdout
    except FileNotFoundError:
        print("Error: 'amd-smi' not found in PATH", file=sys.stderr)
        sys.exit(1)


def run_amd_smi_monitor() -> list[dict]:
    """Run 'amd-smi monitor --json -w 1 -i 1' and return parsed JSON."""
    stdout = _run_command(["amd-smi", "monitor", "-w", "1", "-i", "1", "--json"])

    # Strip the leading info line ("'CTRL' + 'C' to stop watching output:")
    json_start = stdout.find("[")
    if json_start == -1:
        print("No JSON array found in amd-smi monitor output", file=sys.stderr)
        sys.exit(1)

    return json.loads(stdout[json_start:])


def run_amd_smi_xgmi() -> dict[int, bool]:
    """Run 'amd-smi metric -x' and return {gpu_id: xgmi_available}."""
    stdout = _run_command(["amd-smi", "metric", "-x"])
    return _parse_xgmi_metric(stdout)


def get_available_metrics() -> list[GpuMetricAvailability]:
    """Detect available metrics on all GPUs. Importable entry point."""
    data = run_amd_smi_monitor()
    gpus = parse_monitor_json(data)

    xgmi_map = run_amd_smi_xgmi()
    for gpu in gpus:
        gpu.xgmi = xgmi_map.get(gpu.gpu_id, False)

    return gpus


def get_available_metrics_set() -> set[str]:
    """Return the union of available metric names across all GPUs."""
    gpus = get_available_metrics()
    metrics: set[str] = set()
    for gpu in gpus:
        for name in (
            "busy", "temp", "power", "mem_usage",
            "vcn_activity", "jpeg_activity", "xgmi", "pcie",
        ):
            if getattr(gpu, name):
                metrics.add(name)
    return metrics


def get_available_metrics_per_gpu() -> dict[int, set[str]]:
    """Return a dict mapping GPU ID to its available metric names."""
    gpus = get_available_metrics()
    result: dict[int, set[str]] = {}
    for gpu in gpus:
        metrics: set[str] = set()
        for name in (
            "busy", "temp", "power", "mem_usage",
            "vcn_activity", "jpeg_activity", "xgmi", "pcie",
        ):
            if getattr(gpu, name):
                metrics.add(name)
        result[gpu.gpu_id] = metrics
    return result


def main():
    json_output = "--json" in sys.argv

    gpus = get_available_metrics()

    if not gpus:
        print("No GPUs found in amd-smi output", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps([asdict(g) for g in gpus], indent=2))
    else:
        for gpu in gpus:
            print(f"GPU {gpu.gpu_id}:")
            print(f"  busy:          {gpu.busy}")
            print(f"  temp:          {gpu.temp}")
            print(f"  power:         {gpu.power}")
            print(f"  mem_usage:     {gpu.mem_usage}")
            print(f"  vcn_activity:  {gpu.vcn_activity}")
            print(f"  jpeg_activity: {gpu.jpeg_activity}")
            print(f"  xgmi:          {gpu.xgmi}")
            print(f"  pcie:          {gpu.pcie}")
            print(
                f'  -> ROCPROFSYS_AMD_SMI_METRICS="{gpu.to_metrics_string()}"'
            )
            print()


if __name__ == "__main__":
    main()
