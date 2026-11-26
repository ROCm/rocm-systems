# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Description of the machine a snapshot was taken on (CPU, memory, OS, GPUs, ROCm, suite commit)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROBE_TIMEOUT_S = 20.0
SUITE_ROOT = Path(__file__).resolve().parents[2]


def _run(argv: list[str], cwd: Path | None = None) -> str | None:
    """stdout of argv, or None when it is missing, fails or times out (warned on stderr)."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"bench-scenario: warning: '{argv[0]}' unavailable: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(
            f"bench-scenario: warning: '{' '.join(argv)}' exited {proc.returncode}",
            file=sys.stderr,
        )
        return None
    return proc.stdout


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def cpu_info() -> dict[str, Any]:
    models, sockets = [], set()
    for line in _read("/proc/cpuinfo").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            models.append(value.strip())
        elif key.strip() == "physical id":
            sockets.add(value.strip())
    return {
        "model": models[0] if models else None,
        "sockets": len(sockets) or None,
        "threads": os.cpu_count(),
    }


def memory_total_kb() -> int | None:
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def os_name() -> str | None:
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _amd_smi_json(*args: str) -> Any | None:
    out = _run(["amd-smi", *args, "--json"])
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return None


def gpu_info() -> list[dict[str, Any]] | None:
    """One entry per GPU from `amd-smi static`; None when amd-smi is not usable."""
    data = _amd_smi_json("static")
    if not isinstance(data, dict) or "gpu_data" not in data:
        return None
    return [
        {
            "name": g.get("asic", {}).get("market_name"),
            "arch": g.get("asic", {}).get("target_graphics_version"),
            "compute_units": g.get("asic", {}).get("num_compute_units"),
            "vram_mb": g.get("vram", {}).get("size", {}).get("value"),
            "driver": g.get("driver", {}).get("version"),
        }
        for g in data["gpu_data"]
    ]


def rocm_version() -> str | None:
    data = _amd_smi_json("version")
    return data[0].get("rocm_version") if isinstance(data, list) and data else None


def suite_commit() -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], SUITE_ROOT)
    status = _run(["git", "status", "--porcelain", "--", "."], SUITE_ROOT)
    return {
        "commit": head.strip() if head else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def collect() -> dict[str, Any]:
    """Everything that can make two snapshots differ for reasons other than the tool."""
    return {
        "hostname": platform.node(),
        "os": os_name(),
        "kernel": platform.release(),
        "cpu": cpu_info(),
        "memory_total_kb": memory_total_kb(),
        "gpus": gpu_info(),
        "rocm_version": rocm_version(),
        "benchmark_suite": suite_commit(),
    }
