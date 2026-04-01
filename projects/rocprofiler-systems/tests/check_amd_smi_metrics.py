#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Check available AMD SMI metrics on the current system and output a
GpuMetricAvailability struct per GPU. The struct fields mirror the
PMC collector's initialize_supported_metrics() logic and can be used
to filter validation rules in tests.

Uses 'amd-smi metric --json' per category (-u, -t, -p, -P) for
fine-grained sub-metric detection matching the C++ PMC collector.
Uses 'amd-smi monitor --json' for mem_usage (separate API in C++).
Uses 'amd-smi metric -x' for XGMI (only CLI source for XGMI presence).

Usage:
    # Query live GPU(s)
    python3 check_amd_smi_metrics.py

    # JSON output
    python3 check_amd_smi_metrics.py --json
"""

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class GpuMetricAvailability:
    """Per-GPU metric availability. Fine-grained fields mirror the C++ PMC
    collector's initialize_supported_metrics() in device.hpp. Coarse fields
    are derived for backward compatibility with ROCPROFSYS_AMD_SMI_METRICS."""

    gpu_id: int = 0

    # Fine-grained activity (from amd-smi metric -u --json)
    gfx_activity: bool = False  # average_gfx_activity
    umc_activity: bool = False  # average_umc_activity
    mm_activity: bool = False  # average_mm_activity

    # Fine-grained temperature (from amd-smi metric -t --json)
    hotspot_temperature: bool = False  # temperature_hotspot
    edge_temperature: bool = False  # temperature_edge

    # Fine-grained power (from amd-smi metric -p --json)
    current_socket_power: bool = False  # current_socket_power

    # Instinct per-XCP VCN/JPEG (from amd-smi metric -u --json)
    vcn_busy: bool = False  # xcp_stats[].vcn_busy[]
    jpeg_busy: bool = False  # xcp_stats[].jpeg_busy[]

    # Coarse fields (derived from fine-grained, kept for backward compat)
    busy: bool = False  # gfx_activity or umc_activity or mm_activity
    temp: bool = False  # hotspot_temperature or edge_temperature
    power: bool = False  # current_socket_power
    mem_usage: bool = False  # from amd-smi monitor (separate API in C++)
    vcn_activity: bool = False  # device-level Radeon (mutually exclusive w/ vcn_busy)
    jpeg_activity: bool = False  # device-level Radeon (mutually exclusive w/ jpeg_busy)
    xgmi: bool = False  # from amd-smi metric -x (XGMI_ERR heuristic)
    pcie: bool = False  # from amd-smi metric -P --json
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
    """Return True if a JSON value represents an available metric.
    Handles: None, "N/A", {"value": N, "unit": "..."}, and plain numerics."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != "N/A"
    if isinstance(value, dict):
        # e.g. {"value": 25, "unit": "W"} - present means available
        return True
    # numeric (int, float)
    return True


# ---------------------------------------------------------------------------
# amd-smi command runners
# ---------------------------------------------------------------------------


def _run_command(cmd: list[str], exit_on_failure: bool = True) -> str | None:
    """Run a command and return stdout.

    If exit_on_failure is True (default), exits on error.
    If False, returns None on error (for graceful fallback).
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            if exit_on_failure:
                print(
                    f"{' '.join(cmd)} failed (rc={result.returncode}):",
                    file=sys.stderr,
                )
                print(result.stderr, file=sys.stderr)
                sys.exit(1)
            return None
        return result.stdout
    except FileNotFoundError:
        if exit_on_failure:
            print("Error: 'amd-smi' not found in PATH", file=sys.stderr)
            sys.exit(1)
        return None
    except subprocess.TimeoutExpired:
        if exit_on_failure:
            print(f"Timeout running: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(1)
        return None


def _run_amd_smi_metric_json(flags: list[str]) -> list[dict]:
    """Run 'amd-smi metric <flags> --json' and return gpu_data list.

    Returns [] on any failure (graceful fallback).
    """
    cmd = ["amd-smi", "metric"] + flags + ["--json"]
    stdout = _run_command(cmd, exit_on_failure=False)
    if stdout is None:
        return []

    try:
        data = json.loads(stdout)
        return data.get("gpu_data", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def run_amd_smi_monitor() -> list[dict]:
    """Run 'amd-smi monitor --json -w 1 -i 1' and return parsed JSON.

    Used only for mem_usage detection (C++ uses separate get_memory_usage() API).
    Returns [] on failure (graceful fallback).
    """
    stdout = _run_command(
        ["amd-smi", "monitor", "-w", "1", "-i", "1", "--json"], exit_on_failure=False
    )
    if stdout is None:
        return []

    # Strip the leading info line ("'CTRL' + 'C' to stop watching output:")
    json_start = stdout.find("[")
    if json_start == -1:
        print("No JSON array found in amd-smi monitor output", file=sys.stderr)
        return []

    try:
        return json.loads(stdout[json_start:])
    except json.JSONDecodeError as exc:
        print(f"Failed to parse amd-smi monitor JSON: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Per-category detection from amd-smi metric --json
# ---------------------------------------------------------------------------


def _find_gpu_entry(gpu_data: list[dict], gpu_id: int) -> dict | None:
    """Find the entry matching gpu_id in gpu_data list."""
    for entry in gpu_data:
        if entry.get("gpu") == gpu_id:
            return entry
    return None


def detect_usage_metrics(gpus: list[GpuMetricAvailability]) -> None:
    """Detect activity metrics from 'amd-smi metric -u --json'.

    Populates: gfx_activity, umc_activity, mm_activity,
               vcn_activity, jpeg_activity, vcn_busy, jpeg_busy.

    Mirrors C++ initialize_supported_metrics() in device.hpp:293-354.
    """
    gpu_data = _run_amd_smi_metric_json(["-u"])
    if not gpu_data:
        return

    for gpu in gpus:
        entry = _find_gpu_entry(gpu_data, gpu.gpu_id)
        if entry is None:
            continue

        usage = entry.get("usage")
        if not isinstance(usage, dict):
            # GPU with "usage": "N/A" - no metrics available
            continue

        gpu.gfx_activity = _is_available(usage.get("gfx_activity"))
        gpu.umc_activity = _is_available(usage.get("umc_activity"))
        gpu.mm_activity = _is_available(usage.get("mm_activity"))

        # Per-XCP VCN/JPEG busy (Instinct) - check all XCP slots
        vcn_busy_data = usage.get("vcn_busy", {})
        if isinstance(vcn_busy_data, dict):
            gpu.vcn_busy = any(
                _is_available(v)
                for xcp_vals in vcn_busy_data.values()
                if isinstance(xcp_vals, list)
                for v in xcp_vals
            )

        jpeg_busy_data = usage.get("jpeg_busy", {})
        if isinstance(jpeg_busy_data, dict):
            gpu.jpeg_busy = any(
                _is_available(v)
                for xcp_vals in jpeg_busy_data.values()
                if isinstance(xcp_vals, list)
                for v in xcp_vals
            )

        # Device-level VCN/JPEG activity (Radeon)
        # C++ mutual exclusion: only set if per-XCP busy is NOT available
        vcn_activity_data = usage.get("vcn_activity", [])
        if isinstance(vcn_activity_data, list) and not gpu.vcn_busy:
            gpu.vcn_activity = any(_is_available(v) for v in vcn_activity_data)

        jpeg_activity_data = usage.get("jpeg_activity", [])
        if isinstance(jpeg_activity_data, list) and not gpu.jpeg_busy:
            gpu.jpeg_activity = any(_is_available(v) for v in jpeg_activity_data)


def detect_temperature_metrics(gpus: list[GpuMetricAvailability]) -> None:
    """Detect temperature metrics from 'amd-smi metric -t --json'.

    Populates: hotspot_temperature, edge_temperature.
    """
    gpu_data = _run_amd_smi_metric_json(["-t"])
    if not gpu_data:
        return

    for gpu in gpus:
        entry = _find_gpu_entry(gpu_data, gpu.gpu_id)
        if entry is None:
            continue

        temp = entry.get("temperature")
        if not isinstance(temp, dict):
            continue

        gpu.hotspot_temperature = _is_available(temp.get("hotspot"))
        gpu.edge_temperature = _is_available(temp.get("edge"))


def detect_power_metrics(gpus: list[GpuMetricAvailability]) -> None:
    """Detect power metrics from 'amd-smi metric -p --json'.

    Populates: current_socket_power.
    """
    gpu_data = _run_amd_smi_metric_json(["-p"])
    if not gpu_data:
        return

    for gpu in gpus:
        entry = _find_gpu_entry(gpu_data, gpu.gpu_id)
        if entry is None:
            continue

        power = entry.get("power")
        if not isinstance(power, dict):
            continue

        # amd-smi may expose current or average socket power under different keys.
        # The C++ collector treats them independently (device.hpp:308-311);
        # either one being valid causes device_power rows to be emitted.
        gpu.current_socket_power = _is_available(
            power.get("socket_power")
        ) or _is_available(power.get("average_socket_power"))


def detect_pcie_metrics(gpus: list[GpuMetricAvailability]) -> None:
    """Detect PCIe metrics from 'amd-smi metric -P --json'.

    Populates: pcie.
    Mirrors C++ OR logic in device.hpp:363-367.
    """
    gpu_data = _run_amd_smi_metric_json(["-P"])
    if not gpu_data:
        return

    for gpu in gpus:
        entry = _find_gpu_entry(gpu_data, gpu.gpu_id)
        if entry is None:
            continue

        pcie = entry.get("pcie")
        if not isinstance(pcie, dict):
            continue

        # Mirrors C++ OR logic at device.hpp:363-367
        gpu.pcie = (
            _is_available(pcie.get("width"))
            or _is_available(pcie.get("speed"))
            or _is_available(pcie.get("bandwidth"))
            or _is_available(pcie.get("current_bandwidth_sent"))
            or _is_available(pcie.get("current_bandwidth_received"))
        )


# ---------------------------------------------------------------------------
# XGMI detection (kept from original - text parsing of amd-smi metric -x)
# ---------------------------------------------------------------------------


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


def run_amd_smi_xgmi() -> dict[int, bool]:
    """Run 'amd-smi metric -x' and return {gpu_id: xgmi_available}."""
    stdout = _run_command(["amd-smi", "metric", "-x"], exit_on_failure=False)
    if stdout is None:
        return {}
    return _parse_xgmi_metric(stdout)


# ---------------------------------------------------------------------------
# Monitor-based detection (kept for mem_usage only)
# ---------------------------------------------------------------------------

# Only mem_usage uses monitor - it maps to a separate C++ API (get_memory_usage)
_MONITOR_FIELD_MAP: dict[str, list[str]] = {
    "mem_usage": ["vram_used", "vram_total"],
}


def parse_monitor_json(data: list[dict]) -> list[GpuMetricAvailability]:
    """Parse 'amd-smi monitor --json' output into per-GPU availability structs.

    Only populates mem_usage. Other fields come from amd-smi metric --json.
    """
    gpus: list[GpuMetricAvailability] = []

    for entry in data:
        gpu = GpuMetricAvailability(gpu_id=entry.get("gpu", 0))

        for field, keys in _MONITOR_FIELD_MAP.items():
            available = any(_is_available(entry.get(k)) for k in keys)
            setattr(gpu, field, available)

        gpus.append(gpu)

    return gpus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Names emitted by get_available_metrics_set / get_available_metrics_per_gpu
_FINE_GRAINED_NAMES = (
    "gfx_activity",
    "umc_activity",
    "mm_activity",
    "hotspot_temperature",
    "edge_temperature",
    "current_socket_power",
    "vcn_busy",
    "jpeg_busy",
)

# Note: vcn_activity and jpeg_activity are handled separately in
# collect_metric_names() because they can be sourced from two C++ paths
# (device-level Radeon vs per-XCP MI300).
_COARSE_NAMES = (
    "busy",
    "temp",
    "power",
    "mem_usage",
    "xgmi",
    "pcie",
)


def _build_gpu_list_from_metric(gpu_data: list[dict]) -> list[GpuMetricAvailability]:
    """Build GPU list from amd-smi metric output (fallback when monitor fails)."""
    return [
        GpuMetricAvailability(gpu_id=entry.get("gpu", i))
        for i, entry in enumerate(gpu_data)
    ]


def get_available_metrics() -> list[GpuMetricAvailability]:
    """Detect available metrics on all GPUs. Importable entry point."""
    # 1. Get GPU list + mem_usage from monitor
    monitor_data = run_amd_smi_monitor()
    gpus = parse_monitor_json(monitor_data)

    # Fallback: if monitor failed, build GPU list from metric -u output
    if not gpus:
        metric_data = _run_amd_smi_metric_json(["-u"])
        gpus = _build_gpu_list_from_metric(metric_data)

    # 2. Fine-grained detection via amd-smi metric (each gracefully falls back)
    detect_usage_metrics(gpus)
    detect_temperature_metrics(gpus)
    detect_power_metrics(gpus)
    detect_pcie_metrics(gpus)

    # 3. XGMI from existing text parser
    xgmi_map = run_amd_smi_xgmi()
    for gpu in gpus:
        gpu.xgmi = xgmi_map.get(gpu.gpu_id, False)

    # 4. Derive coarse fields from fine-grained
    for gpu in gpus:
        gpu.busy = gpu.gfx_activity or gpu.umc_activity or gpu.mm_activity
        gpu.temp = gpu.hotspot_temperature or gpu.edge_temperature
        gpu.power = gpu.current_socket_power

    return gpus


def collect_metric_names(gpu: GpuMetricAvailability) -> set[str]:
    """Collect all available metric names (fine-grained + coarse) for a GPU."""
    metrics: set[str] = set()

    for name in _FINE_GRAINED_NAMES:
        if getattr(gpu, name):
            metrics.add(name)

    for name in _COARSE_NAMES:
        if getattr(gpu, name):
            metrics.add(name)

    # vcn_activity and jpeg_activity are kept separate from vcn_busy/jpeg_busy.
    # The Radeon (device-level) and Instinct (per-XCP) paths use different data
    # sources and workloads may not produce nonzero values on both paths.
    if gpu.vcn_activity:
        metrics.add("vcn_activity")
    if gpu.jpeg_activity:
        metrics.add("jpeg_activity")

    return metrics


def get_available_metrics_set() -> set[str]:
    """Return the union of available metric names across all GPUs."""
    gpus = get_available_metrics()
    metrics: set[str] = set()
    for gpu in gpus:
        metrics |= collect_metric_names(gpu)
    return metrics


def get_available_metrics_per_gpu() -> dict[int, set[str]]:
    """Return a dict mapping GPU ID to its available metric names."""
    gpus = get_available_metrics()
    return {gpu.gpu_id: collect_metric_names(gpu) for gpu in gpus}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
            print(f"  gfx_activity:         {gpu.gfx_activity}")
            print(f"  umc_activity:         {gpu.umc_activity}")
            print(f"  mm_activity:          {gpu.mm_activity}")
            print(f"  hotspot_temperature:  {gpu.hotspot_temperature}")
            print(f"  edge_temperature:     {gpu.edge_temperature}")
            print(f"  current_socket_power: {gpu.current_socket_power}")
            print(f"  vcn_activity:         {gpu.vcn_activity}")
            print(f"  jpeg_activity:        {gpu.jpeg_activity}")
            print(f"  vcn_busy:             {gpu.vcn_busy}")
            print(f"  jpeg_busy:            {gpu.jpeg_busy}")
            print(f"  mem_usage:            {gpu.mem_usage}")
            print(f"  xgmi:                 {gpu.xgmi}")
            print(f"  pcie:                 {gpu.pcie}")
            print(f"  ---")
            print(f"  busy (derived):       {gpu.busy}")
            print(f"  temp (derived):       {gpu.temp}")
            print(f"  power (derived):      {gpu.power}")
            print(f'  -> ROCPROFSYS_AMD_SMI_METRICS="{gpu.to_metrics_string()}"')
            print()


if __name__ == "__main__":
    main()
