#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
AI-powered performance analysis for GPU traces.

This module analyzes rocpd database files and provides human-readable insights,
bottleneck identification, and optimization recommendations.
"""

import argparse
import os
import re
import shlex
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from importlib.metadata import version as _pkg_version
    _ROCPD_VERSION = _pkg_version("rocpd")
except Exception:
    _ROCPD_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)

from .importer import RocpdImportData, execute_statement
from . import output_config

__all__ = [
    "compute_time_breakdown",
    "identify_hotspots",
    "analyze_memory_copies",
    "analyze_hardware_counters",
    "generate_recommendations",
    "format_analysis_output",
    "analyze_performance",
    "add_args",
    "execute",
    "main",
]


def compute_time_breakdown(connection: RocpdImportData) -> Dict[str, Any]:
    """
    Calculate time distribution across kernel execution, memory copies, and overhead.

    Args:
        connection: RocpdImportData database connection

    Returns:
        Dictionary with time breakdown metrics including percentages
    """
    query = """
    WITH kernel_time AS (
        SELECT COALESCE(SUM(duration), 0) as total_kernel_time
        FROM kernels
    ),
    memcpy_time AS (
        SELECT COALESCE(SUM(duration), 0) as total_memcpy_time
        FROM memory_copies
    ),
    total_time AS (
        SELECT MAX(end) - MIN(start) as total_runtime
        FROM (
            SELECT start, end FROM kernels
            UNION ALL
            SELECT start, end FROM memory_copies
        )
    )
    SELECT
        k.total_kernel_time,
        m.total_memcpy_time,
        COALESCE(t.total_runtime, 0) as total_runtime,
        CASE WHEN COALESCE(t.total_runtime, 0) > 0
             THEN (k.total_kernel_time * 100.0 / t.total_runtime)
             ELSE 0 END as kernel_percent,
        CASE WHEN COALESCE(t.total_runtime, 0) > 0
             THEN (m.total_memcpy_time * 100.0 / t.total_runtime)
             ELSE 0 END as memcpy_percent,
        CASE WHEN COALESCE(t.total_runtime, 0) > 0
             THEN ((t.total_runtime - k.total_kernel_time - m.total_memcpy_time) * 100.0 / t.total_runtime)
             ELSE 0 END as overhead_percent
    FROM kernel_time k, memcpy_time m, total_time t
    """

    try:
        result = execute_statement(connection, query).fetchone()

        if not result:
            return {
                "total_kernel_time": 0,
                "total_memcpy_time": 0,
                "total_runtime": 0,
                "kernel_percent": 0,
                "memcpy_percent": 0,
                "overhead_percent": 0,
            }

        return {
            "total_kernel_time": result[0] or 0,
            "total_memcpy_time": result[1] or 0,
            "total_runtime": result[2] or 0,
            "kernel_percent": result[3] or 0,
            "memcpy_percent": result[4] or 0,
            "overhead_percent": max(0.0, result[5] or 0),
        }

    except Exception as e:
        print(f"Warning: Could not compute time breakdown: {e}", file=sys.stderr)
        return {
            "error": str(e),
            "total_kernel_time": 0,
            "total_memcpy_time": 0,
            "total_runtime": 0,
            "kernel_percent": 0,
            "memcpy_percent": 0,
            "overhead_percent": 0,
        }


def identify_hotspots(
    connection: RocpdImportData, top_n: int = 10, min_duration: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Identify top N kernels by total execution time.

    Args:
        connection: RocpdImportData database connection
        top_n: Number of top kernels to return
        min_duration: Minimum duration threshold in nanoseconds

    Returns:
        List of dictionaries containing kernel statistics
    """
    # Build query with string formatting to avoid parameter binding issues
    # Use f-strings for both min_duration and top_n to avoid SQLite datatype issues
    if min_duration > 0:
        query = f"""
        SELECT
            name,
            COUNT(*) as calls,
            SUM(duration) as total_duration,
            AVG(duration) as avg_duration,
            MIN(duration) as min_duration,
            MAX(duration) as max_duration,
            (SUM(duration) * 100.0 / NULLIF((SELECT SUM(duration) FROM kernels), 0)) as percent_of_total
        FROM kernels
        WHERE duration >= {int(min_duration)}
        GROUP BY name
        ORDER BY total_duration DESC
        LIMIT {int(top_n)}
        """
    else:
        query = f"""
        SELECT
            name,
            COUNT(*) as calls,
            SUM(duration) as total_duration,
            AVG(duration) as avg_duration,
            MIN(duration) as min_duration,
            MAX(duration) as max_duration,
            (SUM(duration) * 100.0 / NULLIF((SELECT SUM(duration) FROM kernels), 0)) as percent_of_total
        FROM kernels
        GROUP BY name
        ORDER BY total_duration DESC
        LIMIT {int(top_n)}
        """

    try:
        # No parameters needed with string formatting
        results = execute_statement(connection, query, ()).fetchall()

        hotspots = []
        for row in results:
            hotspots.append(
                {
                    "name": row[0],
                    "calls": row[1],
                    "total_duration": row[2],
                    "avg_duration": row[3],
                    "min_duration": row[4],
                    "max_duration": row[5],
                    "percent_of_total": row[6] or 0,
                }
            )

        return hotspots

    except Exception as e:
        print(f"Warning: Could not identify hotspots: {e}", file=sys.stderr)
        return []


def analyze_memory_copies(connection: RocpdImportData) -> Dict[str, Dict[str, Any]]:
    """
    Analyze memory copy operations by direction and calculate bandwidth.

    Args:
        connection: RocpdImportData database connection

    Returns:
        Dictionary keyed by direction containing memory copy statistics
    """
    query = """
    SELECT
        CASE
            WHEN category LIKE '%HostToDevice%' OR category LIKE '%H2D%' THEN 'Host-to-Device'
            WHEN category LIKE '%DeviceToHost%' OR category LIKE '%D2H%' THEN 'Device-to-Host'
            WHEN category LIKE '%DeviceToDevice%' OR category LIKE '%D2D%' THEN 'Device-to-Device'
            ELSE category
        END as direction,
        COUNT(*) as count,
        COALESCE(SUM(CAST(size AS INTEGER)), 0) as total_bytes,
        SUM(end - start) as total_duration,
        COALESCE(AVG(CAST(size AS INTEGER)), 0.0) as avg_bytes,
        AVG(end - start) as avg_duration,
        CASE WHEN SUM(end - start) > 0
             THEN (COALESCE(SUM(CAST(size AS INTEGER)), 0) * 1.0e9 / SUM(end - start))
             ELSE 0.0
        END as bandwidth_bytes_per_sec
    FROM memory_copies
    WHERE category IS NOT NULL
    GROUP BY direction
    ORDER BY total_duration DESC
    """

    try:
        results = execute_statement(connection, query).fetchall()

        analysis = {}
        for row in results:
            direction = row[0]
            analysis[direction] = {
                "count": row[1],
                "total_bytes": row[2],
                "total_duration": row[3],
                "avg_bytes": row[4],
                "avg_duration": row[5],
                "bandwidth_bytes_per_sec": row[6],
            }

        return analysis

    except Exception as e:
        print(f"Warning: Could not analyze memory copies: {e}", file=sys.stderr)
        return {}


def analyze_hardware_counters(connection: RocpdImportData) -> Dict[str, Any]:
    """
    Analyze hardware performance counters (Tier 2 analysis).

    Args:
        connection: RocpdImportData database connection

    Returns:
        Dictionary containing hardware counter analysis:
        - has_counters: bool indicating if counter data exists
        - counters: dict of counter statistics by name
        - metrics: derived metrics (occupancy, utilization, etc.)
        - per_kernel: counter analysis by kernel name
    """
    try:
        # Check if pmc_events table exists by trying to query it
        check_query = "SELECT COUNT(*) FROM pmc_events LIMIT 1"
        result = execute_statement(connection, check_query, ()).fetchone()
        if not result or result[0] == 0:
            return {
                "has_counters": False,
                "reason": "pmc_events table exists but contains no data",
            }

        # Get available counters
        counters_query = """
        SELECT
            counter_name,
            COUNT(*) as sample_count,
            AVG(counter_value) as avg_value,
            MIN(counter_value) as min_value,
            MAX(counter_value) as max_value,
            SUM(counter_value) as total_value
        FROM pmc_events
        GROUP BY counter_name
        ORDER BY counter_name
        """

        counter_results = execute_statement(connection, counters_query, ()).fetchall()

        counters = {}
        for row in counter_results:
            counters[row[0]] = {
                "sample_count": row[1],
                "avg_value": row[2],
                "min_value": row[3],
                "max_value": row[4],
                "total_value": row[5],
            }

        # Get per-kernel counter analysis
        per_kernel_query = """
        SELECT
            name,
            counter_name,
            COUNT(DISTINCT dispatch_id) as dispatch_count,
            AVG(counter_value) as avg_value,
            MIN(counter_value) as min_value,
            MAX(counter_value) as max_value
        FROM pmc_events
        GROUP BY name, counter_name
        ORDER BY name, counter_name
        LIMIT 5000
        """

        kernel_results = execute_statement(connection, per_kernel_query, ()).fetchall()

        per_kernel = {}
        for row in kernel_results:
            kernel_name = row[0]
            counter_name = row[1]

            if kernel_name not in per_kernel:
                per_kernel[kernel_name] = {}

            per_kernel[kernel_name][counter_name] = {
                "dispatch_count": row[2],
                "avg_value": row[3],
                "min_value": row[4],
                "max_value": row[5],
            }

        # Calculate derived metrics
        metrics = {}

        # GPU Utilization (GRBM_GUI_ACTIVE / GRBM_COUNT)
        if "GRBM_GUI_ACTIVE" in counters and "GRBM_COUNT" in counters:
            grbm_count = counters["GRBM_COUNT"]["avg_value"]
            grbm_active = counters["GRBM_GUI_ACTIVE"]["avg_value"]
            if grbm_count > 0:
                metrics["gpu_utilization_percent"] = (grbm_active / grbm_count) * 100

        # Average wave occupancy
        if "SQ_WAVES" in counters:
            metrics["avg_waves"] = counters["SQ_WAVES"]["avg_value"]
            metrics["max_waves"] = counters["SQ_WAVES"]["max_value"]
            metrics["min_waves"] = counters["SQ_WAVES"]["min_value"]

        return {
            "has_counters": True,
            "counters": counters,
            "metrics": metrics,
            "per_kernel": per_kernel,
        }

    except Exception as e:
        print(f"Warning: Could not analyze hardware counters: {e}", file=sys.stderr)
        return {"has_counters": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Collection-context detection
# ---------------------------------------------------------------------------

# Flags that --sys-trace subsumes.  Any flag in this set is redundant when
# kernel + memory-copy trace data is already present in the database.
_SYS_TRACE_IMPLIED: frozenset = frozenset({
    "--sys-trace",
    "--hip-trace",
    "--hip-api-trace",
    "--hsa-trace",
    "--kernel-trace",
    "--memory-copy-trace",
    "--marker-trace",
    "--roctx-trace",
})

# Args that only specify output location — not considered "new data collection"
_OUTPUT_ONLY_ARGS: frozenset = frozenset({"-d", "-o", "--output-directory", "--output-file"})

# Hardware counter collection limits for rocprofv3 --pmc.
#
# AMD GPUs limit how many performance counters from the SAME hardware block can
# be collected simultaneously in a single kernel dispatch pass.  The "block name"
# is the prefix before the first "_" in a counter name:
#
#   SQ_WAVES        → block "SQ"    (shader / wavefront counters)
#   GRBM_COUNT      → block "GRBM"  (GPU register bus manager)
#   TCP_*           → block "TCP"   (L1 vector cache)
#   TCC_*           → block "TCC"   (L2 cache)
#
# IMPORTANT: FETCH_SIZE and WRITE_SIZE are DERIVED metrics, not raw hardware counters.
# Internally rocprofv3 expands them to TCC hardware block counters:
#   FETCH_SIZE → TCC_BUBBLE + TCC_EA0_RDREQ + GRBM_GUI_ACTIVE  (TCC block)
#   WRITE_SIZE → TCC_EA0_WRREQ + TCC_EA0_WRREQ_64B              (TCC block)
# Combined they require ~4 TCC hardware counter slots (across 32 TCC instances on MI300X).
# They MUST be isolated in their own pass whenever SQ counters are also requested.
#
# Exceeding a block's per-pass limit causes rocprofv3 to abort with error code 38:
#   "Request exceeds the capabilities of the hardware to collect"
#
# Actual limits vary by GPU generation (MI100/MI200/MI300X) and block type.
# The values below are conservative safe defaults; some blocks (e.g. SQ on
# gfx942/MI300X) support up to 8 counters per pass in practice.
_PMC_BLOCK_LIMIT_DEFAULT: int = 4
_PMC_BLOCK_LIMITS: Dict[str, int] = {
    "SQ":    4,   # shader/wave; gfx942 supports up to 8 — use 4 as safe default
    "GRBM":  4,   # GPU register bus manager
    "TCP":   4,   # L1 vector cache
    "TCC":   4,   # L2 cache
    "TA":    4,   # texture addressing
    "TD":    4,   # texture data
}

# FETCH_SIZE and WRITE_SIZE are derived metrics that each expand to multiple TCC
# hardware counters (FETCH_SIZE → 3 counters, WRITE_SIZE → 2 counters; combined 5
# exceed the TCC per-pass limit). Each must be in its own dedicated pass, isolated
# from all other counters — including each other.
_TCC_DERIVED_COUNTERS: frozenset = frozenset({"FETCH_SIZE", "WRITE_SIZE"})


def _pmc_block(counter: str) -> str:
    """Return the hardware block name for a counter (prefix before first '_')."""
    return counter.split("_")[0]


def _pmc_block_limit(block: str) -> int:
    """Return the per-pass counter limit for the given hardware block."""
    return _PMC_BLOCK_LIMITS.get(block, _PMC_BLOCK_LIMIT_DEFAULT)


def _split_pmc_into_passes(
    counters: List[str],
    base_flags: List[str],
    base_args: List[Dict[str, Any]],
    output_dir: str,
    output_prefix: str,
    description: str,
    app_placeholder: str = "./app",
) -> List[Dict[str, Any]]:
    """
    Split a counter list into the minimum number of rocprofv3 commands so that
    no hardware block exceeds its per-pass collection limit.

    Strategy:
    - FETCH_SIZE and WRITE_SIZE are TCC-derived metrics that expand internally to
      multiple TCC hardware counters (FETCH_SIZE→3 TCC counters, WRITE_SIZE→2).
      Together they exceed the TCC block per-pass limit, so each derived counter
      MUST be in its own dedicated pass, isolated from all other counters.
    - For all other counters: group by hardware block (prefix before '_'),
      passes needed = max(ceil(block_count / block_limit)), distribute evenly.

    Returns a list of command dicts. Single-element when one pass suffices.
    """
    from collections import defaultdict

    if not counters:
        return []

    # Each TCC-derived counter must be in its own dedicated pass.
    derived = [c for c in counters if c in _TCC_DERIVED_COUNTERS]
    regular = [c for c in counters if c not in _TCC_DERIVED_COUNTERS]

    if derived and (len(derived) > 1 or regular):
        # Multiple derived counters can't share a pass (combined TCC hw counter count
        # exceeds the block limit). Each derived counter gets its own dedicated pass;
        # regular counters are handled together as a separate group.
        all_cmds = []
        if regular:
            all_cmds.extend(_split_pmc_into_passes(
                regular, base_flags, base_args, output_dir, output_prefix, description, app_placeholder
            ))
        for dc in derived:
            # Single derived counter: build its command directly (no recursion).
            pmc_str = dc
            flags_str = " ".join(base_flags)
            non_pmc = [a for a in base_args if a.get("name") not in ("--pmc",)]
            args = list(non_pmc) + [
                {"name": "--pmc", "value": pmc_str},
                {"name": "-d", "value": output_dir},
                {"name": "-o", "value": output_prefix},
            ]
            all_cmds.append({
                "tool": "rocprofv3",
                "description": description,
                "flags": list(base_flags),
                "args": args,
                "full_command": (
                    f"rocprofv3 {flags_str} --pmc {pmc_str}"
                    f" -d {output_dir} -o {output_prefix} -- {app_placeholder}"
                ).strip(),
            })
        n = len(all_cmds)
        if n > 1:
            for idx, cmd in enumerate(all_cmds):
                out_name = f"{output_prefix}_pass{idx + 1}"
                pmc_val = next((a["value"] for a in cmd["args"] if a["name"] == "--pmc"), "")
                flags_str = " ".join(base_flags)
                cmd["description"] = f"{description} (pass {idx + 1}/{n})"
                for arg in cmd["args"]:
                    if arg["name"] == "-o":
                        arg["value"] = out_name
                cmd["full_command"] = (
                    f"rocprofv3 {flags_str} --pmc {pmc_val}"
                    f" -d {output_dir} -o {out_name} -- {app_placeholder}"
                ).strip()
        return all_cmds

    # Standard path: group by block and distribute round-robin.
    block_groups: Dict[str, List[str]] = defaultdict(list)
    for c in counters:
        block_groups[_pmc_block(c)].append(c)

    if not block_groups:
        return []

    n_passes = max(
        (len(cs) + _pmc_block_limit(blk) - 1) // max(_pmc_block_limit(blk), 1)
        for blk, cs in block_groups.items()
    )

    pass_counters: List[List[str]] = [[] for _ in range(n_passes)]
    for blk, cs in block_groups.items():
        limit = _pmc_block_limit(blk)
        for pass_idx in range(n_passes):
            chunk = cs[pass_idx * limit: (pass_idx + 1) * limit]
            pass_counters[pass_idx].extend(chunk)

    pass_counters = [p for p in pass_counters if p]
    n = len(pass_counters)

    commands = []
    for idx, pctrs in enumerate(pass_counters):
        suffix = f" (pass {idx + 1}/{n})" if n > 1 else ""
        out_name = f"{output_prefix}_pass{idx + 1}" if n > 1 else output_prefix
        pmc_str = " ".join(pctrs)
        flags_str = " ".join(base_flags)
        non_pmc_args = [a for a in base_args if a.get("name") not in ("--pmc",)]
        args = list(non_pmc_args) + [
            {"name": "--pmc", "value": pmc_str},
            {"name": "-d", "value": output_dir},
            {"name": "-o", "value": out_name},
        ]
        full_cmd = (
            f"rocprofv3 {flags_str} --pmc {pmc_str}"
            f" -d {output_dir} -o {out_name} -- {app_placeholder}"
        ).strip()
        commands.append({
            "tool": "rocprofv3",
            "description": f"{description}{suffix}",
            "flags": list(base_flags),
            "args": args,
            "full_command": full_cmd,
        })
    return commands


def _detect_already_collected(connection: RocpdImportData) -> frozenset:
    """
    Inspect the database to infer which rocprofv3 flags were used during
    the original profiling run.

    Returns a frozenset of flag strings that are already covered by the
    existing trace, so recommendations can avoid suggesting redundant
    re-collection steps.

    Detection heuristics:
    - ``kernels`` rows    → ``--kernel-trace`` (or ``--sys-trace``) was used
    - ``regions`` rows    → ``--hip-trace`` / ``--hsa-trace`` API spans were
      captured (HIP/HSA API region data, a proxy for sys-trace level coverage)
    - ``memory_copies`` rows → ``--memory-copy-trace`` was used
    - kernels + regions together → full ``--sys-trace`` implied; all flags
      in ``_SYS_TRACE_IMPLIED`` are marked as already collected
    - ``pmc_events`` rows → per-counter names stored as ``"pmc:<NAME>"``
      (e.g. ``"pmc:GRBM_COUNT"``) so ``_filter_rec_commands`` can strip
      already-collected counters from ``--pmc`` recommendation commands
    """
    has_kernels = False
    has_api_regions = False   # 'regions' view = HIP/HSA API spans → hip/hsa-trace
    has_memcpy = False

    checks = (
        ("kernels",       "kernels"),
        ("regions",       "api_regions"),
        ("memory_copies", "memcpy"),
    )
    for table, key in checks:
        try:
            row = execute_statement(
                connection, f"SELECT COUNT(*) FROM {table} LIMIT 1", ()
            ).fetchone()
            if row and row[0] > 0:
                if key == "kernels":
                    has_kernels = True
                elif key == "api_regions":
                    has_api_regions = True
                else:
                    has_memcpy = True
        except Exception:
            pass  # table may not exist; expected for Tier 1-only traces

    covered: set = set()
    if has_kernels:
        covered.add("--kernel-trace")
    if has_memcpy:
        covered.add("--memory-copy-trace")

    # If kernel-dispatch AND API-region data both exist, the user ran
    # --sys-trace (or --hip-trace/--hsa-trace alongside --kernel-trace),
    # which implies every flag in _SYS_TRACE_IMPLIED.
    if has_kernels and has_api_regions:
        covered.update(_SYS_TRACE_IMPLIED)

    # Detect which hardware counters are already present in pmc_events.
    # Stored as "pmc:<COUNTER_NAME>" to avoid collisions with flag strings.
    try:
        rows = execute_statement(
            connection, "SELECT DISTINCT counter_name FROM pmc_events", ()
        ).fetchall()
        for row in rows:
            if row and row[0]:
                covered.add(f"pmc:{row[0]}")
    except Exception:
        pass  # pmc_events table absent; expected for Tier 1-only traces

    return frozenset(covered)


def _filter_rec_commands(
    commands: List[Dict[str, Any]],
    already_collected: frozenset,
) -> List[Dict[str, Any]]:
    """
    Remove or trim recommendation commands whose flags are entirely covered
    by the data already present in the database.

    Rules:
    - A flag in ``already_collected`` is stripped from ``flags`` and from
      ``full_command``.
    - ``--pmc`` counter names are checked against ``"pmc:<NAME>"`` entries in
      ``already_collected`` (populated by ``_detect_already_collected``).
      Already-collected counters are removed from the ``--pmc`` arg value; if
      all counters in a ``--pmc`` arg are already present the arg (and flag)
      are dropped entirely.
    - If after stripping, a rocprofv3 command has no remaining flags AND
      its args contain only output-path or scope-filter entries (-d / -o /
      --kernel-names / etc.), the command adds no new data and is dropped.
    - ``rocprof-sys --trace`` alone is equivalent to ``rocprofv3 --sys-trace``
      (same HIP/HSA API data, just in Perfetto format instead of rocpd format)
      and is dropped when sys-trace data is already present.  ``rocprof-sys``
      commands that carry *additional* flags beyond ``--trace`` (e.g.
      ``--trace-gpu-memory``, ``--call-stack-sampling``) are always kept
      because they collect data that rocprofv3 cannot.
    - ``rocprof-compute`` commands are always kept — they perform a deep
      hardware counter analysis that neither rocprofv3 nor rocprof-sys covers.
    - A short note is appended to ``description`` when flags/counters are
      stripped so the user knows why the command looks different from the docs.
    """
    if not already_collected:
        return commands

    has_sys_trace = "--sys-trace" in already_collected

    # Args that are scope filters or output-only — they don't represent new
    # data collection on their own.
    _NON_DATA_ARGS = _OUTPUT_ONLY_ARGS | frozenset({
        "--kernel-names", "--include-names", "--exclude-names",
    })

    filtered = []
    for cmd in commands:
        tool = cmd.get("tool", "")
        flags = cmd.get("flags", [])
        args = cmd.get("args", [])

        # ── rocprof-sys ──────────────────────────────────────────────────────
        if tool == "rocprof-sys" and has_sys_trace:
            # --trace alone ≈ rocprofv3 --sys-trace; drop if it adds nothing new
            extra_flags = [f for f in flags if f != "--trace"]
            meaningful_args = [
                a for a in args if a.get("name", "") not in _OUTPUT_ONLY_ARGS
            ]
            if not extra_flags and not meaningful_args:
                continue  # equivalent to already-collected sys-trace data
            # Has meaningful extra flags (e.g. --trace-gpu-memory) → keep as-is
            filtered.append(cmd)
            continue

        # ── rocprof-compute ──────────────────────────────────────────────────
        if tool == "rocprof-compute":
            filtered.append(cmd)  # always keep — deep hardware counter analysis
            continue

        # ── rocprofv3 ────────────────────────────────────────────────────────
        redundant = [f for f in flags if f in already_collected]
        new_flags = [f for f in flags if f not in already_collected]

        # Process --pmc arg: strip counters already present in pmc_events.
        # pmc_counters / new_pmc / removed_pmc are kept in outer scope so the
        # full_command rebuild below can reference them.
        new_args: list = list(args)
        pmc_counters: list = []
        new_pmc: list = []
        removed_pmc: list = []
        pmc_idx = next(
            (i for i, a in enumerate(new_args) if a.get("name") == "--pmc"), -1
        )
        if pmc_idx >= 0:
            pmc_val = new_args[pmc_idx].get("value") or ""
            pmc_counters = pmc_val.split()
            new_pmc = [c for c in pmc_counters if f"pmc:{c}" not in already_collected]
            removed_pmc = [c for c in pmc_counters if f"pmc:{c}" in already_collected]
            if removed_pmc:
                if new_pmc:
                    new_args[pmc_idx] = {"name": "--pmc", "value": " ".join(new_pmc)}
                else:
                    # All counters already collected — drop arg and flag entirely
                    new_args.pop(pmc_idx)
                    new_flags = [f for f in new_flags if f != "--pmc"]

        nothing_changed = not redundant and not removed_pmc
        if nothing_changed:
            filtered.append(cmd)
            continue

        # Meaningful args: anything that isn't an output path or a scope filter.
        # --kernel-names scopes collection but doesn't collect new data itself.
        meaningful_args = [
            a for a in new_args if a.get("name", "") not in _NON_DATA_ARGS
        ]
        if not new_flags and not meaningful_args:
            continue  # nothing new to collect — drop the command entirely

        # Build updated full_command
        new_full_cmd = cmd.get("full_command", "")
        for f in redundant:
            new_full_cmd = new_full_cmd.replace(f" {f}", "")
        if removed_pmc:
            old_pmc_block = "--pmc " + " ".join(pmc_counters)
            if new_pmc:
                new_full_cmd = new_full_cmd.replace(
                    old_pmc_block, "--pmc " + " ".join(new_pmc)
                )
            else:
                new_full_cmd = new_full_cmd.replace(" " + old_pmc_block, "")
                new_full_cmd = new_full_cmd.replace(old_pmc_block, "")
        new_full_cmd = re.sub(r" +", " ", new_full_cmd).strip()

        new_cmd = dict(cmd)
        new_cmd["flags"] = new_flags
        new_cmd["args"] = new_args
        new_cmd["full_command"] = new_full_cmd

        note_parts = []
        if redundant:
            note_parts.append(f"flags: {' '.join(sorted(redundant))}")
        if removed_pmc:
            note_parts.append(f"PMC counters: {' '.join(sorted(removed_pmc))}")
        new_cmd["description"] = (
            new_cmd.get("description", "")
            + f" (Already collected in this run: {'; '.join(note_parts)})"
        )
        filtered.append(new_cmd)

    return filtered


def generate_recommendations(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    already_collected: Optional[frozenset] = None,
) -> List[Dict[str, Any]]:
    """
    Generate performance recommendations based on analysis results.

    Args:
        time_breakdown: Time distribution metrics
        hotspots: Top kernel hotspots
        memory_analysis: Memory copy analysis
        hardware_counters: Hardware counter analysis (Tier 2)
        already_collected: Frozenset of rocprofv3 flags already present in the
            database (from ``_detect_already_collected``).  Commands that only
            repeat already-collected flags are stripped or dropped so the user
            is not told to re-run something they already did.

    Returns:
        List of recommendation dictionaries with priority, issue, and suggestions
    """
    already_collected = already_collected or frozenset()
    recommendations = []

    # Tier 2: Hardware counter-based recommendations
    if hardware_counters and hardware_counters.get("has_counters"):
        metrics = hardware_counters.get("metrics", {})

        # Low wave occupancy
        avg_waves = metrics.get("avg_waves", 0)
        if avg_waves > 0 and avg_waves < 16:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Low Occupancy",
                    "issue": f"Low wave occupancy detected: average {avg_waves:.1f} waves per SIMD",
                    "suggestion": "Increase kernel occupancy to improve GPU utilization",
                    "actions": [
                        "Increase block/workgroup size to launch more waves per CU",
                        "Reduce register usage per thread (check with --save-temps or rocm-llvm-mc)",
                        "Reduce shared memory (LDS) usage per workgroup",
                        "Check for resource limitations preventing more waves with rocprof-compute",
                    ],
                    "estimated_impact": "10-30% throughput improvement depending on occupancy gap",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Collect wave occupancy and cycle counters per kernel dispatch",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "--pmc", "value": "SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY"},
                                {"name": "-d", "value": "./occupancy_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace --pmc SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY -d ./occupancy_output -o profile -- ./app",
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "Deep-dive occupancy analysis: theoretical vs achieved waves per CU",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {"name": "--block", "value": "SQ"},
                            ],
                            "full_command": "rocprof-compute profile --block SQ -- ./app",
                        },
                    ],
                }
            )

        # Low GPU utilization
        gpu_util = metrics.get("gpu_utilization_percent", 0)
        if gpu_util > 0 and gpu_util < 70:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "GPU Utilization",
                    "issue": f"GPU utilization is only {gpu_util:.1f}% (target: >70%)",
                    "suggestion": "Reduce GPU idle time by overlapping work and eliminating synchronization gaps",
                    "actions": [
                        "Launch independent kernels concurrently using hipStreams",
                        "Increase kernel grid size to fill all CUs when problem size allows",
                        "Reduce hipDeviceSynchronize() and hipStreamSynchronize() call frequency",
                        "Overlap host-device transfers with compute using async streams",
                    ],
                    "estimated_impact": f"Up to {100 - gpu_util:.0f}% reduction in idle time",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Collect GPU active vs total cycle counters to confirm utilization",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "--pmc", "value": "GRBM_GUI_ACTIVE GRBM_COUNT"},
                                {"name": "-d", "value": "./utilization_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace --pmc GRBM_GUI_ACTIVE GRBM_COUNT -d ./utilization_output -o profile -- ./app",
                        },
                        {
                            "tool": "rocprof-sys",
                            "description": "System-level timeline: identify host/GPU idle gaps and synchronization stalls",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- ./app",
                        },
                    ],
                }
            )

    # Tier 1: Trace-level recommendations

    # Rule 1: High memory copy overhead
    memcpy_percent = time_breakdown.get("memcpy_percent", 0)
    if memcpy_percent > 20:
        recommendations.append(
            {
                "priority": "HIGH",
                "category": "Memory Transfer",
                "issue": f"Memory copies consume {memcpy_percent:.1f}% of execution time",
                "suggestion": "Reduce host-device transfer overhead by batching and overlapping transfers",
                "actions": [
                    "Batch multiple small hipMemcpy calls into one large transfer",
                    "Allocate pinned host memory with hipHostMalloc for faster PCIe transfers",
                    "Use hipMemcpyAsync with streams to overlap transfers with kernel execution",
                    "Minimize round-trips: keep data on GPU between consecutive kernels",
                ],
                "estimated_impact": "15-30% reduction in total runtime when transfers dominate",
                "commands": [
                    {
                        "tool": "rocprofv3",
                        "description": "Trace HIP and HSA memory copy operations with timing",
                        "flags": ["--sys-trace", "--hsa-trace"],
                        "args": [
                            {"name": "-d", "value": "./memcpy_output"},
                            {"name": "-o", "value": "profile"},
                        ],
                        "full_command": "rocprofv3 --sys-trace --hsa-trace -d ./memcpy_output -o profile -- ./app",
                    },
                    {
                        "tool": "rocprof-sys",
                        "description": "Detailed memory transfer timeline with PCIe bandwidth and overlap analysis",
                        "flags": [],
                        "args": [
                            {"name": "--trace-gpu-memory", "value": None},
                        ],
                        "full_command": "rocprof-sys --trace-gpu-memory -- ./app",
                    },
                ],
            }
        )

    # Rule 2: High API overhead
    overhead_percent = time_breakdown.get("overhead_percent", 0)
    if overhead_percent > 15:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "category": "API Overhead",
                "issue": f"API and launch overhead is {overhead_percent:.1f}% of total time",
                "suggestion": "Reduce the number of API calls and kernel launches",
                "actions": [
                    "Fuse multiple small kernels into fewer larger launches",
                    "Replace repeated hipMalloc/hipFree with a pre-allocated memory pool",
                    "Batch hipMemcpy calls; use hipMemcpyAsync where possible",
                    "Minimize hipDeviceSynchronize() — synchronize at stream level instead",
                ],
                "estimated_impact": "5-15% reduction when overhead exceeds 15%",
                "commands": [
                    {
                        "tool": "rocprofv3",
                        "description": "Trace all HIP runtime API calls to identify highest-frequency calls",
                        "flags": ["--hip-api-trace", "--hsa-trace"],
                        "args": [
                            {"name": "-d", "value": "./api_output"},
                            {"name": "-o", "value": "profile"},
                        ],
                        "full_command": "rocprofv3 --hip-api-trace --hsa-trace -d ./api_output -o profile -- ./app",
                    },
                    {
                        "tool": "rocprof-sys",
                        "description": "System-level API call frequency and per-call latency breakdown",
                        "flags": ["--trace"],
                        "args": [],
                        "full_command": "rocprof-sys --trace -- ./app",
                    },
                ],
            }
        )

    # Rule 3: Single kernel dominates
    if hotspots and len(hotspots) > 0:
        top_kernel = hotspots[0]
        percent = top_kernel.get("percent_of_total", 0)
        if percent > 50:
            kernel_name = top_kernel.get("name", "unknown")
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Compute Bottleneck",
                    "issue": f"Kernel '{kernel_name}' consumes {percent:.1f}% of GPU time",
                    "suggestion": "Profile this kernel with hardware counters to identify its specific bottleneck",
                    "actions": [
                        "Collect hardware counters to classify compute vs memory bound",
                        "Check memory access patterns for coalescing issues",
                        "Analyze instruction mix: VALU, MFMA, load/store ratios",
                        "Tune occupancy: balance registers, LDS, and block size",
                    ],
                    "estimated_impact": "Highly dependent on bottleneck type; 20-50% improvement possible",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": f"Collect GPU hardware counters scoped to the dominant kernel",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "--pmc", "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"},
                                {"name": "--kernel-names", "value": kernel_name},  # display only; full_command uses shlex.quote
                                {"name": "-d", "value": "./kernel_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": (
                                f'rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES'
                                f' --kernel-names {shlex.quote(kernel_name)}'
                                f' -d ./kernel_output -o profile -- ./app'
                            ),
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "Roofline model, instruction mix, and memory bottleneck analysis for this kernel",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {"name": "--kernel", "value": kernel_name},  # display only; full_command uses shlex.quote
                            ],
                            "full_command": f'rocprof-compute profile --kernel {shlex.quote(kernel_name)} -- ./app',
                        },
                    ],
                }
            )

    # Rule 4: Many small kernels
    if hotspots:
        total_calls = sum(k.get("calls", 0) for k in hotspots)
        if total_calls > 1000:
            avg_duration = (
                time_breakdown.get("total_kernel_time", 0) / total_calls
                if total_calls > 0
                else 0
            )
            if avg_duration < 10000:  # Less than 10 microseconds
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "Launch Overhead",
                        "issue": f"Many small kernels detected: {total_calls} launches, avg {avg_duration/1000:.1f} μs each",
                        "suggestion": "Fuse kernels or batch work to amortize per-launch overhead (~5-10 μs each)",
                        "actions": [
                            "Combine sequential element-wise kernels (e.g., add + multiply) into a single fused kernel",
                            "Increase problem size per launch to push avg duration above 50 μs",
                            "Use persistent kernels for iterative workloads to eliminate repeated launches",
                        ],
                        "estimated_impact": "Eliminates up to 50% of launch overhead for fine-grained workloads",
                        "commands": [
                            {
                                "tool": "rocprofv3",
                                "description": "Capture full kernel dispatch timeline to visualize launch frequency and gaps",
                                "flags": ["--sys-trace"],
                                "args": [
                                    {"name": "-d", "value": "./launch_output"},
                                    {"name": "-o", "value": "profile"},
                                ],
                                "full_command": "rocprofv3 --sys-trace -d ./launch_output -o profile -- ./app",
                            },
                            {
                                "tool": "rocprof-sys",
                                "description": "Visualize kernel launch timeline and inter-launch gaps in a Perfetto trace",
                                "flags": ["--trace"],
                                "args": [],
                                "full_command": "rocprof-sys --trace -- ./app",
                            },
                        ],
                    }
                )

    # Rule 5: Low memory bandwidth
    for direction, stats in memory_analysis.items():
        bandwidth_gbps = stats.get("bandwidth_bytes_per_sec", 0) / 1e9
        if bandwidth_gbps > 0 and bandwidth_gbps < 10:
            avg_bytes = stats.get("avg_bytes", 0)
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Memory Bandwidth",
                    "issue": f"{direction} copies achieving only {bandwidth_gbps:.2f} GB/s (avg transfer: {avg_bytes/1024:.1f} KB)",
                    "suggestion": "Increase transfer size per operation to reach PCIe or HBM saturation bandwidth",
                    "actions": [
                        f"Consolidate many {avg_bytes/1024:.1f} KB transfers into fewer large transfers (>1 MB each)",
                        "Use hipHostMalloc with hipHostMallocPinned flag to enable DMA engine transfers",
                        "Consider hipMemcpyAsync with stream to overlap with compute",
                        "For multi-GPU: evaluate hipMemcpyPeer for direct device-to-device transfers",
                    ],
                    "estimated_impact": "2-10x bandwidth improvement by eliminating small-transfer PCIe overhead",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Trace memory copy operations with size and timing data",
                            "flags": ["--hsa-trace"],
                            "args": [
                                {"name": "-d", "value": "./bandwidth_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": "rocprofv3 --hsa-trace -d ./bandwidth_output -o profile -- ./app",
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "HBM bandwidth utilization analysis for memory-bound kernels",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {"name": "--block", "value": "TD"},
                            ],
                            "full_command": "rocprof-compute profile --block TD -- ./app",
                        },
                    ],
                }
            )

    # Rule 6: Default if no issues found
    if not recommendations:
        recommendations.append(
            {
                "priority": "INFO",
                "category": "Performance",
                "issue": "No obvious performance issues detected at this analysis tier",
                "suggestion": "Collect deeper profiling data to find optimization opportunities",
                "actions": [
                    "Collect hardware counters to check GPU utilization and occupancy",
                    "Enable PC sampling for instruction-level hotspot analysis",
                    "Profile with rocprof-compute for roofline model and bottleneck classification",
                ],
                "estimated_impact": "Depends on findings from deeper analysis",
                "commands": [
                    {
                        "tool": "rocprofv3",
                        "description": "Collect standard hardware performance counters for Tier 2 analysis",
                        "flags": ["--sys-trace"],
                        "args": [
                            {"name": "--pmc", "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"},
                            {"name": "-d", "value": "./counters_output"},
                            {"name": "-o", "value": "profile"},
                        ],
                        "full_command": "rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -d ./counters_output -o profile -- ./app",
                    },
                    {
                        "tool": "rocprof-sys",
                        "description": "Full system trace for comprehensive performance timeline",
                        "flags": ["--trace"],
                        "args": [],
                        "full_command": "rocprof-sys --trace -- ./app",
                    },
                    {
                        "tool": "rocprof-compute",
                        "description": "Complete hardware counter sweep for roofline model and bottleneck classification",
                        "flags": [],
                        "args": [
                            {"name": "profile", "value": None},
                        ],
                        "full_command": "rocprof-compute profile -- ./app",
                    },
                ],
            }
        )

    # Strip or drop commands whose flags are already covered by the original run
    if already_collected:
        for rec in recommendations:
            rec["commands"] = _filter_rec_commands(rec.get("commands", []), already_collected)

    return recommendations


def _format_as_json(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
) -> str:
    """
    Serialize analysis results to JSON conforming to schema v0.1.0.

    The output document contains a top-level ``schema_version`` field that
    consumers MUST check before parsing.  See
    ``rocpd/ai_analysis/docs/analysis-output.schema.json`` for the
    normative schema and ``SCHEMA_CHANGELOG.md`` for migration guidance.
    """
    import json as _json

    breakdown = time_breakdown or {}
    hw = hardware_counters or {}
    total_runtime_ns = int(breakdown.get("total_runtime", 0))
    kernel_time_ns = int(breakdown.get("total_kernel_time", 0))
    memcpy_time_ns = int(breakdown.get("total_memcpy_time", 0))
    kernel_pct = float(breakdown.get("kernel_percent", 0))
    memcpy_pct = float(breakdown.get("memcpy_percent", 0))
    overhead_pct = float(breakdown.get("overhead_percent", 0))
    # Derive api_overhead_ns from the percentage; clamp negative values to 0
    api_overhead_ns = max(0, int(total_runtime_ns * overhead_pct / 100.0))
    idle_time_ns = max(0, total_runtime_ns - kernel_time_ns - memcpy_time_ns - api_overhead_ns)
    idle_pct = float(idle_time_ns / total_runtime_ns * 100.0) if total_runtime_ns > 0 else 0.0

    # --- metadata ---
    has_counters = bool(hw.get("has_counters", False))
    doc: Dict[str, Any] = {
        "schema_version": "0.1.0",
        "metadata": {
            "rocpd_version": _ROCPD_VERSION,
            "analysis_version": "0.1.0",  # schema version, not module version
            "database_file": database_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_duration_ms": 0,
            "custom_prompt": None,
        },
        # --- profiling_info ---
        "profiling_info": {
            "total_duration_ns": total_runtime_ns,
            "profiling_mode": "sys_trace_with_counters" if has_counters else "sys_trace_only",
            "analysis_tier": 2 if has_counters else 1,
            "gpus": [],
        },
        # --- summary ---
        "summary": _build_summary(breakdown, hotspots, has_counters),
        # --- execution_breakdown ---
        "execution_breakdown": {
            "total_runtime_ns": total_runtime_ns,
            "kernel_time_ns": kernel_time_ns,
            "kernel_time_pct": round(kernel_pct, 2),
            "memcpy_time_ns": memcpy_time_ns,
            "memcpy_time_pct": round(memcpy_pct, 2),
            "api_overhead_ns": api_overhead_ns,
            "api_overhead_pct": round(overhead_pct, 2),
            "idle_time_ns": idle_time_ns,
            "idle_time_pct": round(idle_pct, 2),
        },
        # --- hotspots ---
        "hotspots": [
            {
                "rank": i + 1,
                "name": k.get("name", "unknown"),
                "calls": int(k.get("calls", 0)),
                "total_duration_ns": int(k.get("total_duration", 0)),
                "avg_duration_ns": float(k.get("avg_duration", 0)),
                "min_duration_ns": int(k.get("min_duration", 0)),
                "max_duration_ns": int(k.get("max_duration", 0)),
                "pct_of_total": round(float(k.get("percent_of_total", 0)), 2),
            }
            for i, k in enumerate(hotspots or [])
        ],
        # --- memory_analysis ---
        "memory_analysis": {
            direction: {
                "count": int(s.get("count", 0)),
                "total_bytes": int(s.get("total_bytes", 0)),
                "total_duration_ns": int(s.get("total_duration", 0)),
                "avg_bytes": float(s.get("avg_bytes", 0)),
                "avg_duration_ns": float(s.get("avg_duration", 0)),
                "bandwidth_gbps": round(
                    float(s.get("bandwidth_bytes_per_sec", 0)) / 1e9, 4
                ),
            }
            for direction, s in (memory_analysis or {}).items()
        },
        # --- hardware_counters ---
        "hardware_counters": _build_hw_counters_json(hw),
        # --- recommendations ---
        "recommendations": _build_recommendations_json(recommendations or []),
        # --- warnings ---
        "warnings": _build_warnings_json(has_counters),
        "errors": [],
        "llm_enhanced_explanation": None,
    }

    return _json.dumps(doc, indent=2)


def _build_summary(
    breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    has_counters: bool,
) -> Dict[str, Any]:
    """Derive the summary section from analysis data."""
    memcpy_pct = float(breakdown.get("memcpy_percent", 0))
    kernel_pct = float(breakdown.get("kernel_percent", 0))
    overhead_pct = float(breakdown.get("overhead_percent", 0))

    # Simple bottleneck classification
    if memcpy_pct > 30:
        bottleneck = "memory_transfer"
        confidence = 0.85
    elif memcpy_pct > 20:
        bottleneck = "memory_transfer"
        confidence = 0.70
    elif overhead_pct > 25:
        bottleneck = "latency"
        confidence = 0.75
    elif kernel_pct > 70 and has_counters:
        bottleneck = "compute"
        confidence = 0.80
    elif kernel_pct > 70:
        bottleneck = "compute"
        confidence = 0.60
    else:
        bottleneck = "mixed"
        confidence = 0.50

    top_kernel = hotspots[0].get("name", "N/A") if hotspots else "N/A"
    key_findings = [
        f"Kernel execution: {kernel_pct:.1f}% of total runtime",
        f"Memory copy overhead: {memcpy_pct:.1f}% of total runtime",
        f"Top kernel: {top_kernel}",
    ]
    if has_counters:
        key_findings.append("Hardware counter data available (Tier 2 analysis)")
    else:
        key_findings.append("No hardware counters — Tier 1 trace analysis only")

    return {
        "overall_assessment": (
            f"Workload is {bottleneck.replace('_', ' ')}-bound "
            f"with {len(hotspots)} unique kernels analyzed. "
            f"Kernel time: {kernel_pct:.1f}%, memory copies: {memcpy_pct:.1f}%."
        ),
        "primary_bottleneck": bottleneck,
        "confidence": round(confidence, 2),
        "key_findings": key_findings,
    }


def _build_hw_counters_json(hw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert hardware_counters internal dict to schema-compliant form."""
    has_counters = bool(hw.get("has_counters", False))
    if not has_counters:
        return {"has_counters": False, "metrics": None, "counters": None}

    raw_metrics = hw.get("metrics", {}) or {}
    metrics: Dict[str, Any] = {
        "gpu_utilization_pct": raw_metrics.get("gpu_utilization_percent"),
        "avg_waves": raw_metrics.get("avg_waves"),
        "max_waves": raw_metrics.get("max_waves"),
        "min_waves": raw_metrics.get("min_waves"),
    }

    raw_counters = hw.get("counters", {}) or {}
    counters = {
        name: {
            "sample_count": int(s.get("sample_count", 0)),
            "avg_value": float(s.get("avg_value", 0)),
            "min_value": float(s.get("min_value", 0)),
            "max_value": float(s.get("max_value", 0)),
            "total_value": float(s.get("total_value", 0)),
        }
        for name, s in raw_counters.items()
    }

    return {"has_counters": True, "metrics": metrics, "counters": counters}


# Stable IDs for known recommendation categories.
_CATEGORY_IDS = {
    "Low Occupancy": "ROCPD-OCCUPANCY-001",
    "GPU Utilization": "ROCPD-UTILIZATION-001",
    "Memory Transfer": "ROCPD-MEMCPY-001",
    "API Overhead": "ROCPD-API-001",
    "Compute Bottleneck": "ROCPD-COMPUTE-001",
    "Launch Overhead": "ROCPD-LAUNCH-001",
    "Memory Bandwidth": "ROCPD-MEMBW-001",
    "Performance": "ROCPD-INFO-001",
}


def _build_recommendations_json(
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map internal recommendation dicts to the schema v0.1.0 format."""
    out = []
    seen_ids: Dict[str, int] = {}
    for rec in recommendations:
        category = rec.get("category", "General")
        base_id = _CATEGORY_IDS.get(category, f"ROCPD-{category.upper().replace(' ', '-')[:12]}-001")
        count = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = count
        rec_id = base_id if count == 1 else f"{base_id[:-3]}{count:03d}"

        out.append({
            "id": rec_id,
            "priority": rec.get("priority", "INFO"),
            "category": category,
            "issue": rec.get("issue", ""),
            "suggestion": rec.get("suggestion", ""),
            "actions": rec.get("actions", []),
            "estimated_impact": rec.get("estimated_impact", ""),
            "commands": rec.get("commands", []),
        })
    return out


def _build_warnings_json(has_counters: bool) -> List[Dict[str, Any]]:
    """Build the warnings list based on analysis context."""
    if not has_counters:
        return [
            {
                "severity": "warning",
                "message": (
                    "No hardware counters collected. Analysis limited to "
                    "Tier 1 (trace data only)."
                ),
                "recommendation": (
                    "Collect counters with: "
                    "rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app"
                ),
            }
        ]
    return []


def _format_as_markdown(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
) -> str:
    """Format analysis results as Markdown."""
    breakdown = time_breakdown or {}
    hw = hardware_counters or {}
    has_counters = bool(hw.get("has_counters", False))

    total_runtime_ms = breakdown.get("total_runtime", 0) / 1e6
    kernel_pct = breakdown.get("kernel_percent", 0)
    memcpy_pct = breakdown.get("memcpy_percent", 0)
    overhead_pct = breakdown.get("overhead_percent", 0)
    kernel_ms = breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_ms = breakdown.get("total_memcpy_time", 0) / 1e6

    lines = []
    lines.append("# ROCpd AI Performance Analysis")
    lines.append("")
    if database_path:
        lines.append(f"**Database:** `{database_path}`")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tier = 2 if has_counters else 1
    lines.append(f"**Analysis Tier:** {tier} ({'Hardware Counters' if has_counters else 'Trace Only'})")
    lines.append("")

    lines.append("## Time Breakdown")
    lines.append("")
    lines.append("| Category | Time (ms) | Percentage |")
    lines.append("|----------|-----------|------------|")
    lines.append(f"| Kernel Execution | {kernel_ms:,.2f} | {kernel_pct:.1f}% |")
    lines.append(f"| Memory Copies | {memcpy_ms:,.2f} | {memcpy_pct:.1f}% |")
    overhead_ms = max(0.0, total_runtime_ms - kernel_ms - memcpy_ms) if total_runtime_ms > 0 else 0
    lines.append(f"| API Overhead | {overhead_ms:,.2f} | {overhead_pct:.1f}% |")
    lines.append(f"| **Total** | **{total_runtime_ms:,.2f}** | **100%** |")
    lines.append("")

    if hotspots:
        lines.append("## Top Kernel Hotspots")
        lines.append("")
        lines.append("| Rank | Kernel | Calls | Total (ms) | Avg (μs) | % Total |")
        lines.append("|------|--------|-------|------------|----------|---------|")
        for i, k in enumerate(hotspots, 1):
            name = k.get("name", "unknown")
            if len(name) > 40:
                name = name[:37] + "..."
            lines.append(
                f"| {i} | `{name}` | {k.get('calls', 0)} "
                f"| {k.get('total_duration', 0)/1e6:,.2f} "
                f"| {k.get('avg_duration', 0)/1e3:,.1f} "
                f"| {k.get('percent_of_total', 0):.1f}% |"
            )
        lines.append("")

    if memory_analysis:
        lines.append("## Memory Copy Analysis")
        lines.append("")
        lines.append("| Direction | Count | Total Size | Duration (ms) | Bandwidth (GB/s) |")
        lines.append("|-----------|-------|------------|---------------|-----------------|")
        for direction, s in memory_analysis.items():
            tb = s.get("total_bytes", 0)
            if tb >= 1e9:
                size_str = f"{tb/1e9:.1f} GB"
            elif tb >= 1e6:
                size_str = f"{tb/1e6:.1f} MB"
            elif tb >= 1e3:
                size_str = f"{tb/1e3:.1f} KB"
            else:
                size_str = f"{tb:.0f} B"
            bw = s.get("bandwidth_bytes_per_sec", 0) / 1e9
            lines.append(
                f"| {direction} | {s.get('count', 0)} | {size_str} "
                f"| {s.get('total_duration', 0)/1e6:,.2f} | {bw:.2f} |"
            )
        lines.append("")

    if has_counters:
        metrics = hw.get("metrics", {}) or {}
        lines.append("## Hardware Counters (Tier 2)")
        lines.append("")
        if "gpu_utilization_percent" in metrics:
            lines.append(f"- **GPU Utilization:** {metrics['gpu_utilization_percent']:.1f}%")
        if "avg_waves" in metrics:
            lines.append(f"- **Avg Wave Occupancy:** {metrics['avg_waves']:.1f} waves")
            lines.append(f"- **Max Wave Occupancy:** {metrics.get('max_waves', 0):.1f} waves")
        lines.append("")

    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "🔵"}
        for rec in recommendations:
            p = rec.get("priority", "INFO")
            emoji = priority_emoji.get(p, "•")
            lines.append(f"### {emoji} [{p}] {rec.get('category', '')}")
            lines.append("")
            lines.append(f"**Issue:** {rec.get('issue', '')}")
            lines.append("")
            lines.append(f"**Suggestion:** {rec.get('suggestion', '')}")
            actions = rec.get("actions", [])
            if actions:
                lines.append("")
                for action in actions:
                    lines.append(f"{action}")
            estimated_impact = rec.get("estimated_impact", "")
            if estimated_impact:
                lines.append("")
                lines.append(f"**Estimated Impact:** {estimated_impact}")
            commands = rec.get("commands", [])
            if commands:
                lines.append("")
                lines.append("**Recommended Commands:**")
                lines.append("")
                for cmd in commands:
                    tool = cmd.get("tool", "")
                    desc = cmd.get("description", "")
                    full_command = cmd.get("full_command", "")
                    flags = cmd.get("flags", [])
                    args = cmd.get("args", [])
                    lines.append(f"*{tool}* — {desc}")
                    if flags:
                        lines.append(f"- Flags: `{' '.join(flags)}`")
                    if args:
                        arg_strs = []
                        for a in args:
                            name = a.get("name", "")
                            value = a.get("value")
                            arg_strs.append(
                                f"{name} {value}" if value is not None else name
                            )
                        lines.append(f"- Args: `{' '.join(arg_strs)}`")
                    if full_command:
                        lines.append(f"```bash\n{full_command}\n```")
                    lines.append("")
            lines.append("")

    lines.append("---")
    lines.append(f"*Generated by rocpd analyze • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# WebView output format
# ─────────────────────────────────────────────────────────────────────────────

def _format_as_webview(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
) -> str:
    """
    Generate a self-contained interactive HTML report.

    The file has no external dependencies — all CSS and JS are inlined so it
    opens correctly from any local path or file-share without a web server.
    """
    import html as _html
    import json as _json

    def _h(v: Any) -> str:
        """HTML-escape a value for safe text embedding."""
        return _html.escape(str(v), quote=True)

    def _fmt_ns(ns: Any) -> str:
        if ns is None:
            return "—"
        ns = float(ns)
        if ns < 1_000:
            return f"{ns:.0f} ns"
        if ns < 1_000_000:
            return f"{ns / 1_000:.1f} µs"
        if ns < 1_000_000_000:
            return f"{ns / 1_000_000:.1f} ms"
        return f"{ns / 1_000_000_000:.2f} s"

    def _fmt_bytes(b: Any) -> str:
        if not b:
            return "—"
        b = float(b)
        if b < 1_024:
            return f"{b:.0f} B"
        if b < 1_048_576:
            return f"{b / 1_024:.1f} KB"
        if b < 1_073_741_824:
            return f"{b / 1_048_576:.1f} MB"
        return f"{b / 1_073_741_824:.2f} GB"

    def _svg_gauge(pct: float, color: str, label: str, value_str: str) -> str:
        """SVG donut gauge — semicircle (180°) style."""
        r = 36
        cx = cy = 44
        full = 3.14159265 * r  # half circumference (180°)
        dash = full * max(0.0, min(1.0, pct / 100.0))
        offset = full  # start at the left (270° → top, then we rotate 90° via transform)
        return (
            f'<div class="gauge-box">'
            f'<svg viewBox="0 0 88 50" width="130" height="74">'
            # track arc
            f'<path d="M {cx - r},{cy} A {r},{r} 0 0 1 {cx + r},{cy}"'
            f' fill="none" stroke="var(--bg3)" stroke-width="9" stroke-linecap="round"/>'
            # filled arc (clipped at cy so only top half shows)
            f'<path d="M {cx - r},{cy} A {r},{r} 0 0 1 {cx + r},{cy}"'
            f' fill="none" stroke="{_h(color)}" stroke-width="9" stroke-linecap="round"'
            f' stroke-dasharray="{dash:.2f} {full:.2f}"'
            f' stroke-dashoffset="0"/>'
            # value text
            f'<text x="{cx}" y="{cy - 4}" text-anchor="middle"'
            f' font-size="13" font-weight="700" fill="var(--text)">{_h(value_str)}</text>'
            f'<text x="{cx}" y="{cy + 10}" text-anchor="middle"'
            f' font-size="7.5" fill="var(--dim)">{_h(label.upper())}</text>'
            f'</svg>'
            f'</div>'
        )

    # ── derived values ──────────────────────────────────────────────────────
    breakdown = time_breakdown or {}
    hw        = hardware_counters or {}
    has_counters   = bool(hw.get("has_counters", False))
    total_ns       = float(breakdown.get("total_runtime", 0))
    total_ms       = total_ns / 1e6
    kernel_pct     = float(breakdown.get("kernel_percent", 0))
    memcpy_pct     = float(breakdown.get("memcpy_percent", 0))
    overhead_pct   = float(breakdown.get("overhead_percent", 0))
    kernel_ms      = breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_ms      = breakdown.get("total_memcpy_time", 0) / 1e6
    overhead_ms    = max(0.0, total_ms * overhead_pct / 100.0)
    idle_pct       = max(0.0, 100.0 - kernel_pct - memcpy_pct - overhead_pct)
    idle_ms        = max(0.0, total_ms - kernel_ms - memcpy_ms - overhead_ms)
    tier           = 2 if has_counters else 1
    analysis_date  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary        = _build_summary(breakdown, hotspots or [], has_counters)
    bottleneck     = summary.get("primary_bottleneck", "unknown")
    confidence     = int(summary.get("confidence", 0) * 100)
    assessment     = summary.get("overall_assessment", "")
    key_findings   = summary.get("key_findings", [])
    metrics        = hw.get("metrics", {}) or {}
    gpu_util       = metrics.get("gpu_utilization_pct") or metrics.get("gpu_utilization_percent")
    avg_waves      = metrics.get("avg_waves")
    max_waves      = metrics.get("max_waves")

    BN_COLOR = {
        "compute": "#5599ee", "memory_transfer": "#ff8c00",
        "latency": "#cc44cc", "mixed": "#9999bb", "unknown": "#666677",
    }
    bn_color = BN_COLOR.get(bottleneck, "#888899")

    PRIORITY = {
        "HIGH":   ("#e84040", "#2a0808"),
        "MEDIUM": ("#f08432", "#2a1600"),
        "LOW":    ("#caa828", "#241e08"),
        "INFO":   ("#4d8ef2", "#081428"),
    }
    PRIORITY_ICON = {"HIGH": "&#128308;", "MEDIUM": "&#128992;", "LOW": "&#128993;", "INFO": "&#8505;"}

    # ── recommendations HTML ────────────────────────────────────────────────
    recs_parts = []
    for ri, rec in enumerate(recommendations or []):
        p    = rec.get("priority", "INFO")
        cat  = rec.get("category", "")
        fg, bg_rec = PRIORITY.get(p, ("#888", "#1a1a2a"))
        picon = PRIORITY_ICON.get(p, "&#8505;")
        actions_li = "".join(
            f"<li>{_h(a)}</li>" for a in rec.get("actions", [])
        )
        actions_html = (
            f'<ol class="r-actions">{actions_li}</ol>' if actions_li else ""
        )
        impact = rec.get("estimated_impact", "")
        impact_html = (
            f'<p class="r-impact">&#9889; Expected impact: {_h(impact)}</p>'
            if impact else ""
        )
        cmds_parts = []
        for ci, cmd in enumerate(rec.get("commands", [])):
            fc   = cmd.get("full_command", "")
            tool = cmd.get("tool", "")
            desc = cmd.get("description", "")
            if not fc:
                continue
            cid = f"c{ri}_{ci}"
            cmds_parts.append(
                f'<div class="cmd-blk">'
                f'<span class="tool-tag">{_h(tool)}</span>'
                f'<span class="cmd-desc">{_h(desc)}</span>'
                f'<div class="cmd-row" id="{cid}">'
                f'<code>{_h(fc)}</code>'
                f'<button class="cp-btn" onclick="cpCmd(\'{cid}\')">Copy</button>'
                f'</div></div>'
            )
        cmds_html = "".join(cmds_parts)
        issue_txt = rec.get("issue", "")
        suggest   = rec.get("suggestion", "")
        recs_parts.append(
            f'<div class="r-card" style="border-left-color:{fg}" data-p="{_h(p)}">'
            f'<div class="r-hdr" onclick="toggleR(this)">'
            f'<span class="r-priority-icon">{picon}</span>'
            f'<span class="r-badge" style="background:{fg};color:#fff">{_h(p)}</span>'
            f'<span class="r-cat">{_h(cat)}</span>'
            f'<span class="r-chev">&#9660;</span>'
            f'</div>'
            f'<div class="r-body">'
            f'<p class="r-issue"><strong>Issue:</strong> {_h(issue_txt)}</p>'
            f'<p class="r-suggest"><strong>What to do:</strong> {_h(suggest)}</p>'
            f'{actions_html}{impact_html}{cmds_html}'
            f'</div></div>'
        )
    recs_html = (
        "".join(recs_parts)
        or '<p class="dim">No recommendations — workload looks well-optimized.</p>'
    )

    # ── hotspots table ──────────────────────────────────────────────────────
    hotspot_rows = []
    for i, k in enumerate(hotspots or []):
        pct  = float(k.get("percent_of_total", 0))
        bar  = min(100.0, pct)
        name = k.get("name", "unknown")
        hot  = ' class="hot-row"' if pct >= 20 else ""
        hotspot_rows.append(
            f'<tr{hot}>'
            f'<td>{i + 1}</td>'
            f'<td class="kname" title="{_h(name)}"><code>{_h(name)}</code></td>'
            f'<td data-v="{k.get("calls",0)}">{int(k.get("calls",0)):,}</td>'
            f'<td data-v="{k.get("total_duration",0)}">{_fmt_ns(k.get("total_duration",0))}</td>'
            f'<td data-v="{k.get("avg_duration",0)}">{_fmt_ns(k.get("avg_duration",0))}</td>'
            f'<td data-v="{k.get("min_duration",0)}">{_fmt_ns(k.get("min_duration",0))}</td>'
            f'<td data-v="{pct}">'
            f'<div class="pbar"><div class="pfill" style="width:{bar:.1f}%"></div>'
            f'<span>{pct:.1f}%</span></div>'
            f'</td></tr>'
        )
    hotspots_html = ""
    if hotspot_rows:
        hotspots_html = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128293;</span>'
            '<h2>Top Kernel Hotspots</h2>'
            '</div>'
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable sortable" id="hs-tbl">'
            '<thead><tr>'
            '<th data-tip=\'Rank by total execution time — 1 is the hottest kernel.\'>#</th>'
            '<th data-tip=\'Demangled GPU kernel function name dispatched to the GPU. Rows highlighted in red consume &gt;20% of total runtime.\'>Kernel Name</th>'
            '<th data-tip=\'Number of times this kernel was dispatched. Very high call counts with low avg time suggest kernel launch overhead dominates useful work.\'>Calls &#8645;</th>'
            '<th data-tip=\'Sum of all dispatch durations for this kernel — the primary metric for identifying hotspots. Longer total time = bigger optimization target.\'>Total Time &#8645;</th>'
            '<th data-tip=\'Mean duration per single dispatch. Values below 10 &micro;s suggest kernel launch overhead may dominate the actual computation.\'>Avg Time &#8645;</th>'
            '<th data-tip=\'Fastest observed single dispatch. Useful for spotting variance — a large gap between min and avg suggests irregular execution (cache effects, branch divergence).\'>Min Time &#8645;</th>'
            '<th data-tip=\'Percentage of total profiling window time consumed by this kernel. Kernels above 20% are highlighted and are the highest-priority optimization targets.\'>% Total &#8645;</th>'
            '</tr></thead>'
            '<tbody>' + "".join(hotspot_rows) + '</tbody>'
            '</table></div></div></section>'
        )

    # ── memory analysis table ───────────────────────────────────────────────
    _MEM_DIR_TIPS = {
        "Host-to-Device": (
            "<strong>Host-to-Device (H2D)</strong>"
            "CPU &rarr; GPU transfer over PCIe. Used to upload inputs, weights, or parameters before kernel execution. "
            "<em>PCIe 4.0 x16 peak: ~32 GB/s. Minimize by reusing GPU allocations across iterations.</em>"
        ),
        "Device-to-Host": (
            "<strong>Device-to-Host (D2H)</strong>"
            "GPU &rarr; CPU transfer over PCIe. Used to read results back after kernel execution. "
            "<em>Minimize these — prefer keeping results on GPU across multiple kernels. Use async memcpy to overlap with compute.</em>"
        ),
        "Device-to-Device": (
            "<strong>Device-to-Device (D2D)</strong>"
            "GPU &rarr; GPU on the same device, using HBM bandwidth directly (not PCIe). Very fast — can approach peak HBM bandwidth. "
            "<em>Use for in-GPU data reorganization. MI300X HBM peak: ~5.3 TB/s.</em>"
        ),
        "Peer-to-Peer": (
            "<strong>Peer-to-Peer (P2P)</strong>"
            "GPU &rarr; different GPU transfer. Speed depends on interconnect: Infinity Fabric is fast (&sim;900 GB/s on MI300X); PCIe is slower (~32 GB/s). "
            "<em>Enable peer access with hipDeviceEnablePeerAccess for direct transfers.</em>"
        ),
    }
    mem_rows = []
    for direction, s in (memory_analysis or {}).items():
        tb  = s.get("total_bytes", 0)
        bw  = s.get("bandwidth_bytes_per_sec", 0) / 1e9
        dir_tip = _MEM_DIR_TIPS.get(
            direction,
            f"<strong>{_h(direction)}</strong>Memory transfer direction between host and device."
        )
        mem_rows.append(
            f'<tr>'
            f'<td data-tip=\'{dir_tip}\'>{_h(direction)}</td>'
            f'<td>{int(s.get("count", 0)):,}</td>'
            f'<td>{_fmt_bytes(tb)}</td>'
            f'<td>{_fmt_ns(s.get("total_duration", 0))}</td>'
            f'<td>{_fmt_bytes(s.get("avg_bytes", 0))}</td>'
            f'<td>{bw:.2f} GB/s</td>'
            f'</tr>'
        )
    mem_html = ""
    if mem_rows:
        mem_html = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128190;</span>'
            '<h2>Memory Transfer Analysis</h2>'
            '</div>'
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable">'
            '<thead><tr>'
            '<th data-tip=\'Transfer direction. Hover each row to learn what each direction means.\'>Direction</th>'
            '<th data-tip=\'Number of individual copy operations in this direction. Many small transfers are inefficient — batch them when possible.\'>Count</th>'
            '<th data-tip=\'Total data volume transferred in this direction across all operations.\'>Total Bytes</th>'
            '<th data-tip=\'Total wall-clock time spent on copies in this direction.\'>Total Time</th>'
            '<th data-tip=\'Average bytes per copy operation. Transfers below 1 MB are typically inefficient due to PCIe transaction overhead — batch small transfers.\'>Avg Size</th>'
            '<th data-tip=\'Achieved transfer bandwidth. PCIe 4.0 x16 theoretical peak is ~32 GB/s. Low bandwidth usually means many small transfers, not PCIe saturation.\'>Bandwidth</th>'
            '</tr></thead>'
            '<tbody>' + "".join(mem_rows) + '</tbody>'
            '</table></div></div></section>'
        )

    # ── hardware counters ───────────────────────────────────────────────────
    gauges_html = ""
    if gpu_util is not None:
        _gpu_u = float(gpu_util)
        gc = "#44dd66" if _gpu_u >= 70 else "#ff8800"
        hint = (
            '<p class="g-hint warn">&#9888; Low — increase parallelism</p>'
            if _gpu_u < 70
            else '<p class="g-hint ok">&#10003; Good utilization</p>'
        )
        _gpu_ok = _gpu_u >= 70
        _gpu_status = (
            '<span class="tok">Good — GPU is well-utilized.</span>'
            if _gpu_ok else
            '<span class="twarn">Low — reduce synchronization barriers, increase batch size, or launch larger kernels.</span>'
        )
        _gpu_tip = (
            f"<strong>GPU Utilization ({_gpu_u:.1f}%)</strong>"
            f"Percentage of GPU clock cycles where the hardware was actively processing work. "
            f"Derived from hardware counters: <code>GRBM_GUI_ACTIVE &divide; GRBM_COUNT</code>.<br>"
            f"<em>Target: &ge;70%. Below 70% means the GPU is frequently idle.</em><br>"
            f"{_gpu_status}"
        )
        gauges_html += (
            f'<div class="gauge-wrap" data-tip=\'{_gpu_tip}\'>'
            f'{_svg_gauge(_gpu_u, gc, "GPU Utilization", f"{_gpu_u:.1f}%")}'
            f'{hint}</div>'
        )
    if avg_waves is not None:
        _aw = float(avg_waves)
        wc = "#44dd66" if _aw >= 16 else "#ff8800"
        # Normalize waves to 0-100% assuming 64 waves/SIMD as 100%
        wpct = min(100.0, _aw / 64.0 * 100.0)
        whint = (
            '<p class="g-hint warn">&#9888; Low occupancy — check registers/LDS</p>'
            if _aw < 16
            else '<p class="g-hint ok">&#10003; Adequate occupancy</p>'
        )
        wave_str = f"{_aw:.0f}"
        if max_waves is not None:
            wave_str += f" / {float(max_waves):.0f}"
        _wave_ok = _aw >= 16
        _wave_status = (
            '<span class="tok">Good — adequate wavefront occupancy for latency hiding.</span>'
            if _wave_ok else
            '<span class="twarn">Low — reduce register usage or LDS allocation per wavefront to increase occupancy and hide memory latency.</span>'
        )
        _wave_tip = (
            f"<strong>Wave Occupancy (avg {_aw:.0f} waves)</strong>"
            f"Average number of wavefronts (64 threads each) simultaneously in-flight per compute unit. "
            f"Collected via the <code>SQ_WAVES</code> hardware counter. "
            f"Higher occupancy lets the GPU hide memory latency by switching to another wavefront while one waits for data.<br>"
            f"<em>Target: &ge;16 waves. Max practical: 64 waves/SIMD unit. "
            f"Low occupancy usually means each wavefront uses too many registers or too much LDS.</em><br>"
            f"{_wave_status}"
        )
        gauges_html += (
            f'<div class="gauge-wrap" data-tip=\'{_wave_tip}\'>'
            f'{_svg_gauge(wpct, wc, "Avg Waves", wave_str)}'
            f'{whint}</div>'
        )
    raw_counters = hw.get("counters", {}) or {}
    ctr_rows = "".join(
        f'<tr class="ctr-row" data-ctr="{_h(n)}"><td><code>{_h(n)}</code></td>'
        f'<td>{int(v.get("sample_count", 0)):,}</td>'
        f'<td>{float(v.get("avg_value", 0)):.2f}</td>'
        f'<td>{float(v.get("min_value", 0)):.2f}</td>'
        f'<td>{float(v.get("max_value", 0)):.2f}</td>'
        f'<td>{float(v.get("total_value", 0)):,.0f}</td></tr>'
        for n, v in raw_counters.items()
    )
    ctr_table = (
        '<table class="dtable" style="margin-top:1rem">'
        '<thead><tr><th>Counter</th><th>Samples</th>'
        '<th>Avg</th><th>Min</th><th>Max</th><th>Total</th></tr></thead>'
        '<tbody>' + ctr_rows + '</tbody></table>'
    ) if ctr_rows else ""

    hw_inner = (
        f'<div class="gauges">{gauges_html}</div>{ctr_table}'
        if has_counters
        else (
            '<p class="dim">No hardware counter data — Tier 1 (trace-only) analysis.</p>'
            '<p class="hint" style="margin-top:.5rem">Collect counters with:</p>'
            '<div class="cmd-row" id="hw-hint">'
            '<code>rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app</code>'
            '<button class="cp-btn" onclick="cpCmd(\'hw-hint\')">Copy</button>'
            '</div>'
        )
    )

    # ── key findings list ───────────────────────────────────────────────────
    findings_li = "".join(f"<li>{_h(f)}</li>" for f in key_findings)
    findings_html = f'<ul class="findings">{findings_li}</ul>' if findings_li else ""

    # ── embed full JSON (sanitized for HTML context) ────────────────────────
    json_str = _format_as_json(
        time_breakdown, hotspots, memory_analysis,
        recommendations, hardware_counters, database_path,
    )
    json_embedded = json_str.replace("</script>", r"<\/script>").replace("<!--", r"<\!--")

    # ══════════════════════════════════════════════════════════════════════
    # HTML template
    # All CSS { } must be doubled inside f-strings.
    # JS template literals (`${}`) avoided; no external resources.
    # ══════════════════════════════════════════════════════════════════════
    db_meta = (
        f'<div>Database: <code>{_h(database_path)}</code></div>'
        if database_path else ""
    )
    tier_label = "Hardware Counters (Tier 2)" if has_counters else "Trace Only (Tier 1)"
    bn_display = bottleneck.replace("_", " ").title()

    # ── Pre-computed tooltip strings (single-quote delimited in HTML attrs) ──
    _TIP_KERNEL = (
        "<strong>Kernel Execution</strong>"
        "Time actively running GPU compute kernels. Higher is better — means more "
        "useful work is being done on the GPU silicon. "
        "<em>If this is low (&lt;40%), look for excessive GPU idle time or API launch overhead.</em>"
    )
    _TIP_MEMCPY = (
        "<strong>Memory Copies</strong>"
        "Time transferring data between CPU (host) and GPU (device) over the PCIe bus. "
        "High values (&gt;20%) indicate a PCIe bandwidth bottleneck. "
        "<em>Minimize by batching transfers, using pinned (page-locked) memory, "
        "or overlapping copies with kernel execution via async streams.</em>"
    )
    _TIP_OVERHEAD = (
        "<strong>API &amp; Launch Overhead</strong>"
        "Time in HIP/HSA runtime calls: kernel launch latency, "
        "synchronization barriers, and runtime bookkeeping. "
        "High values (&gt;15%) suggest too many small kernel dispatches or excessive "
        "CPU&ndash;GPU synchronization points. "
        "<em>Batch work into fewer larger kernels and minimize hipDeviceSynchronize calls.</em>"
    )
    _TIP_IDLE = (
        "<strong>GPU Idle</strong>"
        "Time when the GPU had no work to execute — pipeline bubbles between kernel launches. "
        "High idle time means the CPU is not submitting work fast enough, "
        "or there are long synchronization stalls waiting on host results. "
        "<em>Use asynchronous launches, CUDA/HIP streams, and reduce host processing "
        "between dispatches.</em>"
    )
    _BN_TIPS = {
        "compute": (
            "<strong>Compute Bottleneck</strong>"
            "GPU arithmetic units (VALU/MFMA) are the limiting factor. "
            "The workload is doing more FLOPs than the memory system can supply data for, "
            "meaning arithmetic throughput is the ceiling. "
            "<em>Optimize: use MFMA (matrix FMA) instructions, reduce register pressure, "
            "increase thread-level parallelism.</em>"
        ),
        "memory_transfer": (
            "<strong>Memory Transfer Bottleneck</strong>"
            "PCIe data transfers between CPU and GPU dominate execution time. "
            "The application is spending more time moving data than computing. "
            "<em>Optimize: keep data resident on GPU across multiple kernels, "
            "use pinned host memory, overlap transfers with computation via async streams.</em>"
        ),
        "memory_bandwidth": (
            "<strong>Memory Bandwidth Bottleneck</strong>"
            "HBM (High Bandwidth Memory) bandwidth is the limiting factor. "
            "Kernels are reading/writing more data than HBM can deliver per clock. "
            "<em>Optimize: improve data reuse via tiling, exploit L1/L2 cache locality, "
            "use LDS (shared memory) to reduce HBM traffic.</em>"
        ),
        "latency": (
            "<strong>Latency Bottleneck</strong>"
            "Many small, short-lived kernels where launch overhead dominates actual computation. "
            "GPU spends more time being launched than running. "
            "<em>Optimize: fuse multiple small kernels into one, increase work per dispatch, "
            "or use persistent kernel patterns.</em>"
        ),
        "mixed": (
            "<strong>Mixed Bottleneck</strong>"
            "Multiple performance limiters are present simultaneously. "
            "No single dominant bottleneck was identified. "
            "<em>Address the highest-priority recommendation first, re-profile, "
            "then iterate.</em>"
        ),
        "unknown": (
            "<strong>Bottleneck Unknown</strong>"
            "Analysis could not determine a clear primary bottleneck from available data. "
            "<em>Collect hardware counters for deeper analysis: "
            "rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app</em>"
        ),
    }
    _tip_bn = _BN_TIPS.get(bottleneck, _BN_TIPS["unknown"])
    _tip_tier = (
        "<strong>Analysis Tier 2 — Hardware Counters</strong>"
        "Profiling data includes hardware performance counters collected via "
        "<code>rocprofv3 --pmc</code>. Enables GPU utilization, wave occupancy, "
        "and per-kernel counter breakdowns in addition to timing data."
        if has_counters else
        "<strong>Analysis Tier 1 — Trace Only</strong>"
        "Profiling data contains timing information only (no hardware counters). "
        "For deeper GPU-level insights, re-profile with: "
        "<em>rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app</em>"
    )

    # ── Pre-compute badge / KPI status values ───────────────────────────────
    n_high   = sum(1 for r in (recommendations or []) if r.get("priority") == "HIGH")
    n_medium = sum(1 for r in (recommendations or []) if r.get("priority") == "MEDIUM")
    n_low    = sum(1 for r in (recommendations or []) if r.get("priority") == "LOW")
    n_info   = sum(1 for r in (recommendations or []) if r.get("priority") == "INFO")

    # kernel utilization KPI health class
    if kernel_pct >= 60:
        _kpi_kernel_cls = "kpi-ok";   _kpi_kernel_lbl = "Good"
    elif kernel_pct >= 30:
        _kpi_kernel_cls = "kpi-warn"; _kpi_kernel_lbl = "Moderate"
    else:
        _kpi_kernel_cls = "kpi-crit"; _kpi_kernel_lbl = "Low"

    _BN_ICON = {
        "compute": "&#128293;", "memory_transfer": "&#128230;",
        "memory_bandwidth": "&#128190;", "latency": "&#9889;",
        "mixed": "&#128256;", "unknown": "&#10067;",
    }
    _bn_icon = _BN_ICON.get(bottleneck, "&#10067;")

    _badge_parts = []
    if n_high:   _badge_parts.append(f'<span class="hbadge hbadge-crit">&#9679; {n_high} Critical</span>')
    if n_medium: _badge_parts.append(f'<span class="hbadge hbadge-warn">&#9679; {n_medium} Warning</span>')
    if n_low:    _badge_parts.append(f'<span class="hbadge hbadge-ok">&#9679; {n_low} Low</span>')
    if n_info:   _badge_parts.append(f'<span class="hbadge hbadge-info">&#9679; {n_info} Info</span>')
    header_badges_html = " ".join(_badge_parts)

    _recs_badge_html = ""
    if n_high:   _recs_badge_html += f'<span class="shdr-badge sbadge-crit">{n_high} Critical</span> '
    if n_medium: _recs_badge_html += f'<span class="shdr-badge sbadge-warn">{n_medium} Warning</span>'

    _tier_icon = "&#128300;" if has_counters else "&#128225;"
    _tier_status_lbl = "HW Counters" if has_counters else "Trace Only"
    _hw_badge_html = (
        f'<span class="shdr-badge sbadge-info">Tier 2</span>'
        if has_counters else
        f'<span class="shdr-badge sbadge-info">Tier 1</span>'
    )

    _db_pill_html = ""
    if database_path:
        _db_label = database_path[-45:] if len(database_path) > 45 else database_path
        _db_pill_html = (
            f'<div class="hpill">'
            f'<span class="hpill-label">DB:</span>'
            f'<span class="hpill-value" title="{_h(database_path)}">{_h(_db_label)}</span>'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROCpd AI Analysis &#8212; {_h(database_path or "GPU Performance Report")}</title>
<style>
/* ── Reset + Variables ─────────────────────────────────────────────── */
:root {{
  --bg:#0d0d14; --bg2:#14141f; --bg3:#1c1c2c; --bg4:#242438;
  --bdr:#2c2c48; --bdr2:#3a3a58;
  --text:#e0e3f2; --sub:#a8aace; --dim:#6868a0;
  --amd:#e01a22;
  --blue:#4d8ef2; --green:#3acc66; --orange:#f08432;
  --purple:#9866cc; --teal:#28bca8; --yellow:#caa828;
  --c-ok:#3acc66;   --c-ok-bg:rgba(58,204,102,.13);
  --c-warn:#f08432; --c-warn-bg:rgba(240,132,50,.13);
  --c-crit:#e84040; --c-crit-bg:rgba(232,64,64,.13);
  --c-info:#4d8ef2; --c-info-bg:rgba(77,142,242,.13);
  --r:10px; --r-sm:6px;
  --font:-apple-system,"Segoe UI",system-ui,Ubuntu,sans-serif;
  --mono:"JetBrains Mono","Cascadia Code","Fira Code",ui-monospace,monospace;
  --shadow:0 4px 18px rgba(0,0,0,.42);
  --shadow-lg:0 8px 36px rgba(0,0,0,.55);
  --trans:all 0.22s cubic-bezier(0.4,0,0.2,1);
}}
[data-theme="light"] {{
  --bg:#f2f2f8; --bg2:#ffffff; --bg3:#eaeaf2; --bg4:#dddde8;
  --bdr:#c8c8dc; --bdr2:#b4b4cc;
  --text:#181828; --sub:#444468; --dim:#6868a0;
  --c-ok-bg:rgba(58,204,102,.10); --c-warn-bg:rgba(240,132,50,.10);
  --c-crit-bg:rgba(232,64,64,.10); --c-info-bg:rgba(77,142,242,.10);
  --shadow:0 2px 12px rgba(0,0,0,.10);
  --shadow-lg:0 4px 20px rgba(0,0,0,.14);
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior:smooth; }}
body {{ font-family:var(--font); background:var(--bg); color:var(--text);
       line-height:1.65; font-size:15px; min-height:100vh;
       transition:background .25s,color .25s; }}
a {{ color:var(--blue); }}
code {{ font-family:var(--mono); font-size:.87em; }}
/* ── Header ──────────────────────────────────────────────────────── */
.hdr {{
  background:linear-gradient(135deg,#080810 0%,#120e1c 100%);
  border-bottom:3px solid var(--amd); padding:.9rem 0;
  position:sticky; top:0; z-index:100;
  box-shadow:0 2px 16px rgba(0,0,0,.55);
}}
[data-theme="light"] .hdr {{ background:linear-gradient(135deg,#1a0a10 0%,#280e18 100%); }}
.hdr-inner {{ max-width:1140px; margin:0 auto; padding:0 1.25rem;
              display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }}
.hdr-brand {{ display:flex; align-items:baseline; gap:.6rem; }}
.logo {{ font-size:1.6rem; font-weight:900; color:var(--amd);
         letter-spacing:-.04em; line-height:1; }}
.logo em {{ color:#f0f0ff; font-style:normal; }}
.hdr-subtitle {{ font-size:.88rem; color:rgba(255,255,255,.55); font-weight:500; }}
.hdr-badges {{ display:flex; gap:.4rem; flex-wrap:wrap; margin-left:auto; }}
.hbadge {{ font-size:.7rem; font-weight:800; padding:.2em .65em;
           border-radius:100px; letter-spacing:.04em;
           display:inline-flex; align-items:center; gap:.25em; }}
.hbadge-crit {{ background:var(--c-crit-bg); color:var(--c-crit); border:1px solid rgba(232,64,64,.4); }}
.hbadge-warn {{ background:var(--c-warn-bg); color:var(--c-warn); border:1px solid rgba(240,132,50,.4); }}
.hbadge-ok   {{ background:var(--c-ok-bg);   color:var(--c-ok);   border:1px solid rgba(58,204,102,.4); }}
.hbadge-info {{ background:var(--c-info-bg);  color:var(--c-info); border:1px solid rgba(77,142,242,.4); }}
.hdr-controls {{ display:flex; gap:.5rem; align-items:center; }}
.hdr-btn {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
            color:rgba(255,255,255,.7); border-radius:var(--r-sm);
            padding:.3em .75em; font-size:.79rem; cursor:pointer;
            font-family:var(--font); transition:var(--trans);
            display:flex; align-items:center; gap:.3em; }}
.hdr-btn:hover {{ background:rgba(255,255,255,.14); color:#fff; }}
.hdr-pills {{ max-width:1140px; margin:.55rem auto 0; padding:.5rem 1.25rem 0;
              display:flex; gap:.45rem; flex-wrap:wrap;
              border-top:1px solid rgba(255,255,255,.07); }}
.hpill {{ font-size:.72rem; background:rgba(255,255,255,.06);
          border:1px solid rgba(255,255,255,.1); border-radius:5px;
          padding:.12em .55em; display:flex; align-items:center; gap:.3em; }}
.hpill-label {{ color:rgba(255,255,255,.4); }}
.hpill-value {{ font-family:var(--mono); font-weight:600; color:rgba(255,255,255,.75); }}
/* ── Layout ──────────────────────────────────────────────────────── */
.wrap {{ max-width:1140px; margin:0 auto; padding:1.5rem 1.25rem 5rem; }}
/* ── Section Card ────────────────────────────────────────────────── */
.scard {{ background:var(--bg2); border:1px solid var(--bdr); border-radius:var(--r);
          margin-bottom:1.5rem; box-shadow:var(--shadow); overflow:hidden;
          animation:fadeInUp .35s ease both; }}
.scard:nth-child(1) {{ animation-delay:.04s; }}
.scard:nth-child(2) {{ animation-delay:.08s; }}
.scard:nth-child(3) {{ animation-delay:.12s; }}
.scard:nth-child(4) {{ animation-delay:.16s; }}
.scard:nth-child(5) {{ animation-delay:.20s; }}
.scard:nth-child(6) {{ animation-delay:.24s; }}
.shdr {{ display:flex; align-items:center; gap:.6rem;
         padding:.85rem 1.4rem; border-bottom:1px solid var(--bdr);
         background:var(--bg3); }}
.shdr-icon {{ font-size:1.1rem; flex-shrink:0; }}
.shdr h2 {{ font-size:.97rem; font-weight:700; letter-spacing:.02em; flex:1; color:var(--text); }}
.shdr-badge {{ font-size:.69rem; font-weight:800; padding:.15em .55em;
               border-radius:100px; letter-spacing:.04em; flex-shrink:0; }}
.sbadge-crit {{ background:var(--c-crit-bg); color:var(--c-crit); }}
.sbadge-warn {{ background:var(--c-warn-bg); color:var(--c-warn); }}
.sbadge-ok   {{ background:var(--c-ok-bg);   color:var(--c-ok); }}
.sbadge-info {{ background:var(--c-info-bg);  color:var(--c-info); }}
.sbody {{ padding:1.25rem 1.4rem; }}
.dim {{ color:var(--dim); }}
.hint {{ font-size:.85rem; color:var(--dim); }}
/* ── Assessment / Quote ──────────────────────────────────────────── */
.assess {{ font-style:italic; color:var(--sub); font-size:.92rem; line-height:1.7;
           padding:.7rem 1rem; border-left:3px solid var(--blue);
           background:var(--c-info-bg); border-radius:0 var(--r-sm) var(--r-sm) 0;
           margin-bottom:1.25rem; }}
/* ── KPI Grid ────────────────────────────────────────────────────── */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr));
             gap:1rem; margin-bottom:1.25rem; }}
.kpi {{ border:1px solid var(--bdr); border-radius:var(--r); padding:1rem;
        position:relative; overflow:hidden; transition:var(--trans); cursor:help; }}
.kpi:hover {{ transform:translateY(-2px); box-shadow:var(--shadow); }}
.kpi::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
.kpi-ok   {{ background:var(--c-ok-bg); }}   .kpi-ok::before   {{ background:var(--c-ok); }}
.kpi-warn {{ background:var(--c-warn-bg); }}  .kpi-warn::before {{ background:var(--c-warn); }}
.kpi-crit {{ background:var(--c-crit-bg); }}  .kpi-crit::before {{ background:var(--c-crit); }}
.kpi-info {{ background:var(--c-info-bg); }}  .kpi-info::before {{ background:var(--c-info); }}
.kpi-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:.4rem; }}
.kpi-icon {{ font-size:1.25rem; }}
.kpi-status {{ font-size:.68rem; font-weight:800; padding:.14em .5em; border-radius:100px; }}
.kpi-ok   .kpi-status {{ background:rgba(58,204,102,.2);  color:var(--c-ok); }}
.kpi-warn .kpi-status {{ background:rgba(240,132,50,.2);  color:var(--c-warn); }}
.kpi-crit .kpi-status {{ background:rgba(232,64,64,.2);   color:var(--c-crit); }}
.kpi-info .kpi-status {{ background:rgba(77,142,242,.2);  color:var(--c-info); }}
.kpi-label {{ font-size:.69rem; text-transform:uppercase; letter-spacing:.1em; color:var(--dim); margin-bottom:.2rem; }}
.kpi-value {{ font-size:1.55rem; font-weight:800; line-height:1.1; font-family:var(--mono); margin-bottom:.15rem; }}
.kpi-ok   .kpi-value {{ color:var(--c-ok); }}
.kpi-warn .kpi-value {{ color:var(--c-warn); }}
.kpi-crit .kpi-value {{ color:var(--c-crit); }}
.kpi-info .kpi-value {{ color:var(--c-info); }}
.kpi-sub {{ font-size:.77rem; color:var(--dim); }}
/* ── Key Findings ────────────────────────────────────────────────── */
.findings {{ list-style:none; margin-top:.85rem; border-top:1px solid var(--bdr); padding-top:.75rem; }}
.findings li {{ font-size:.87rem; color:var(--sub); padding:.28rem 0 .28rem 1.3rem;
                position:relative; border-bottom:1px solid rgba(44,44,72,.3); }}
.findings li:last-child {{ border-bottom:none; }}
.findings li::before {{ content:'→'; position:absolute; left:0; color:var(--blue); font-weight:700; }}
/* ── Breakdown ───────────────────────────────────────────────────── */
.stacked {{ height:34px; display:flex; border-radius:8px; overflow:hidden;
            box-shadow:0 0 0 1px var(--bdr); margin:1rem 0 .85rem; }}
.seg {{ height:100%; transition:opacity .18s; cursor:help; }}
.seg:hover {{ opacity:.75; }}
.legend {{ display:flex; flex-wrap:wrap; gap:.55rem; margin-bottom:1rem; }}
.leg {{ display:flex; align-items:center; gap:.4rem; font-size:.8rem; color:var(--sub); }}
.dot {{ width:10px; height:10px; border-radius:3px; flex-shrink:0; }}
.brows {{ display:flex; flex-direction:column; gap:.55rem; }}
.brow {{ display:grid; grid-template-columns:155px 1fr 165px;
         align-items:center; gap:.75rem; font-size:.87rem; cursor:help; }}
.brow:hover .bval {{ color:var(--text); }}
.blabel {{ color:var(--sub); font-weight:500; }}
.btrack {{ background:var(--bg3); border-radius:4px; height:20px;
           overflow:hidden; border:1px solid var(--bdr); }}
.bfill {{ height:100%; border-radius:4px; }}
.bval {{ text-align:right; font-family:var(--mono); font-size:.81rem; color:var(--dim); }}
.bpct {{ color:var(--sub); font-weight:600; }}
/* ── Recommendations ─────────────────────────────────────────────── */
.r-card {{ border-left:4px solid; border-radius:0 var(--r) var(--r) 0;
           background:var(--bg3); margin-bottom:.6rem; overflow:hidden;
           transition:background .15s; }}
.r-card:hover {{ background:var(--bg4); }}
.r-hdr {{ display:flex; align-items:center; gap:.55rem; padding:.8rem 1rem;
          cursor:pointer; user-select:none; }}
.r-priority-icon {{ font-size:.9rem; flex-shrink:0; }}
.r-badge {{ padding:.14em .55em; border-radius:4px; font-size:.69rem;
            font-weight:800; letter-spacing:.06em; flex-shrink:0; }}
.r-cat {{ font-weight:600; font-size:.9rem; flex:1; color:var(--text); }}
.r-chev {{ color:var(--dim); font-size:.7rem; transition:transform .2s; flex-shrink:0; }}
.r-card.open .r-chev {{ transform:rotate(180deg); }}
.r-body {{ display:none; padding:.85rem 1rem 1rem; border-top:1px solid var(--bdr); }}
.r-card.open .r-body {{ display:block; }}
.r-issue {{ margin-bottom:.5rem; font-size:.9rem; }}
.r-suggest {{ font-size:.9rem; margin-bottom:.5rem; }}
.r-actions {{ padding-left:1.5rem; margin:.5rem 0; color:var(--sub); font-size:.87rem; }}
.r-actions li {{ margin-bottom:.22rem; }}
.r-impact {{ color:var(--c-ok); font-size:.84rem; margin-top:.65rem;
             padding:.35rem .65rem; background:var(--c-ok-bg);
             border-radius:var(--r-sm); display:inline-block; }}
.cmd-blk {{ margin-top:.85rem; }}
.tool-tag {{ color:var(--blue); font-weight:700; font-size:.83rem; }}
.cmd-desc {{ color:var(--dim); font-size:.81rem; margin-left:.4rem; }}
.cmd-row {{ display:flex; align-items:center; justify-content:space-between;
            gap:.5rem; background:var(--bg); border:1px solid var(--bdr);
            border-radius:var(--r-sm); padding:.55rem .85rem; margin-top:.35rem;
            overflow-x:auto; }}
.cmd-row code {{ color:#a0e870; white-space:nowrap; font-size:.84rem; }}
.cp-btn {{ flex-shrink:0; background:var(--bg3); border:1px solid var(--bdr);
           color:var(--dim); padding:.2em .6em; border-radius:4px;
           cursor:pointer; font-size:.73rem; font-family:var(--font); transition:var(--trans); }}
.cp-btn:hover {{ color:var(--text); border-color:var(--blue); }}
/* ── Tables ──────────────────────────────────────────────────────── */
.tbl-wrap {{ overflow-x:auto; }}
.dtable {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
.dtable th {{ background:var(--bg3); color:var(--dim); font-weight:700; font-size:.76rem;
              text-transform:uppercase; letter-spacing:.06em;
              text-align:left; padding:.55rem .75rem; border-bottom:2px solid var(--bdr);
              cursor:pointer; user-select:none; white-space:nowrap; transition:color .15s; }}
.dtable th:hover {{ color:var(--text); }}
.dtable td {{ padding:.5rem .75rem; border-bottom:1px solid rgba(44,44,72,.4); vertical-align:middle; }}
.dtable tr:last-child td {{ border-bottom:none; }}
.dtable tr:hover td {{ background:rgba(255,255,255,.02); }}
.hot-row td {{ background:rgba(224,26,34,.08) !important; }}
.hot-row td:last-child {{ font-weight:700; }}
.kname {{ max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.pbar {{ display:flex; align-items:center; gap:.4rem; }}
.pfill {{ height:10px; background:var(--blue); border-radius:3px; min-width:2px; }}
.pbar span {{ font-size:.78rem; white-space:nowrap; color:var(--dim); }}
/* ── Hardware Gauges ─────────────────────────────────────────────── */
.gauges {{ display:flex; flex-wrap:wrap; gap:1.5rem; margin-bottom:1rem; }}
.gauge-wrap {{ text-align:center; padding:.8rem 1.1rem; background:var(--bg3);
               border:1px solid var(--bdr); border-radius:var(--r); min-width:145px;
               transition:var(--trans); }}
.gauge-wrap:hover {{ border-color:var(--bdr2); box-shadow:var(--shadow); transform:translateY(-1px); }}
.g-hint {{ font-size:.78rem; margin-top:.3rem; font-weight:600; }}
.g-hint.warn {{ color:var(--c-warn); }}
.g-hint.ok   {{ color:var(--c-ok); }}
.gauge-box {{ display:flex; justify-content:center; }}
/* ── Floating Tooltip ────────────────────────────────────────────── */
#tt {{
  position:fixed; z-index:9999; pointer-events:none; max-width:320px;
  padding:.7rem 1rem; background:#0e0e1c; border:1px solid #3a3a5c;
  border-radius:10px; box-shadow:0 10px 40px rgba(0,0,0,.7);
  font-size:.8rem; line-height:1.65; color:#dde0f2;
  opacity:0; transition:opacity .12s; white-space:normal;
}}
#tt.show {{ opacity:1; }}
#tt strong {{ color:var(--blue); display:block; margin-bottom:.2rem; font-size:.85rem; }}
#tt code {{ font-size:.78rem; background:rgba(255,255,255,.08); padding:.05em .3em; border-radius:3px; }}
#tt em {{ color:var(--dim); font-size:.77rem; display:block; margin-top:.3rem; }}
#tt .tok  {{ color:var(--c-ok); font-weight:600; }}
#tt .twarn {{ color:var(--c-warn); font-weight:600; }}
[data-tip] {{ cursor:help; }}
/* ── FAB ─────────────────────────────────────────────────────────── */
.fab {{ position:fixed; bottom:1.5rem; right:1.5rem; width:46px; height:46px;
        border-radius:50%; background:linear-gradient(135deg,var(--blue) 0%,var(--purple) 100%);
        color:#fff; border:none; font-size:1.25rem; box-shadow:var(--shadow-lg);
        cursor:pointer; display:flex; align-items:center; justify-content:center;
        transition:var(--trans); z-index:200; opacity:0; pointer-events:none; }}
.fab.visible {{ opacity:1; pointer-events:all; }}
.fab:hover {{ transform:scale(1.1) translateY(-2px); }}
/* ── Footer ──────────────────────────────────────────────────────── */
footer {{ border-top:1px solid var(--bdr); padding:1.25rem 1.25rem; max-width:1140px; margin:0 auto; }}
footer p {{ color:var(--dim); font-size:.77rem; text-align:center; }}
/* ── Animations ──────────────────────────────────────────────────── */
@keyframes fadeInUp {{
  from {{ opacity:0; transform:translateY(14px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width:7px; height:7px; }}
::-webkit-scrollbar-track {{ background:var(--bg); }}
::-webkit-scrollbar-thumb {{ background:var(--bdr2); border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:var(--dim); }}
/* ── Mobile ──────────────────────────────────────────────────────── */
@media (max-width:640px) {{
  .brow {{ grid-template-columns:120px 1fr auto; }}
  .kpi-value {{ font-size:1.3rem; }}
  .hdr-subtitle {{ display:none; }}
}}
</style>
</head>
<body>

<div id="tt"></div>

<!-- ── Header ────────────────────────────────────────────────────── -->
<header class="hdr">
  <div class="hdr-inner">
    <div class="hdr-brand">
      <span class="logo">ROC<em>pd</em></span>
      <span class="hdr-subtitle">AI Performance Analysis</span>
    </div>
    <div class="hdr-badges">{header_badges_html}</div>
    <div class="hdr-controls">
      <button class="hdr-btn" id="theme-btn" onclick="toggleTheme()">&#9728; Light</button>
    </div>
  </div>
  <div class="hdr-pills">
    <div class="hpill"><span class="hpill-label">Runtime:</span><span class="hpill-value">{total_ms:,.2f} ms</span></div>
    <div class="hpill"><span class="hpill-label">Kernels:</span><span class="hpill-value">{len(hotspots or [])}</span></div>
    <div class="hpill"><span class="hpill-label">Tier:</span><span class="hpill-value">{_h(tier_label)}</span></div>
    <div class="hpill"><span class="hpill-label">Generated:</span><span class="hpill-value">{analysis_date}</span></div>
    {_db_pill_html}
  </div>
</header>

<div class="wrap">

<!-- ── Overview ──────────────────────────────────────────────────── -->
<section class="scard">
  <div class="shdr">
    <span class="shdr-icon">&#128202;</span>
    <h2>Overview</h2>
    <span class="shdr-badge sbadge-info">Tier {tier}</span>
  </div>
  <div class="sbody">
    <p class="assess">{_h(assessment)}</p>
    <div class="kpi-grid">
      <div class="kpi kpi-info" data-tip='{_tip_bn}'>
        <div class="kpi-head"><span class="kpi-icon">{_bn_icon}</span><span class="kpi-status">Bottleneck</span></div>
        <div class="kpi-label">Primary Bottleneck</div>
        <div class="kpi-value" style="color:{bn_color}">{_h(bn_display)}</div>
        <div class="kpi-sub">Confidence: {confidence}%</div>
      </div>
      <div class="kpi kpi-info" data-tip='<strong>Total Runtime</strong>Wall-clock duration of the profiled application from first observed event to last. Includes kernel execution, memory copies, API calls, and GPU idle time.'>
        <div class="kpi-head"><span class="kpi-icon">&#9201;</span><span class="kpi-status">Duration</span></div>
        <div class="kpi-label">Total Runtime</div>
        <div class="kpi-value">{total_ms:,.2f}</div>
        <div class="kpi-sub">milliseconds &bull; {len(hotspots or [])} kernels</div>
      </div>
      <div class="kpi {_kpi_kernel_cls}" data-tip='{_TIP_KERNEL}'>
        <div class="kpi-head"><span class="kpi-icon">&#128187;</span><span class="kpi-status">{_kpi_kernel_lbl}</span></div>
        <div class="kpi-label">Kernel Execution</div>
        <div class="kpi-value">{kernel_pct:.1f}%</div>
        <div class="kpi-sub">{kernel_ms:,.2f} ms active compute</div>
      </div>
      <div class="kpi kpi-info" data-tip='{_tip_tier}'>
        <div class="kpi-head"><span class="kpi-icon">{_tier_icon}</span><span class="kpi-status">{_tier_status_lbl}</span></div>
        <div class="kpi-label">Analysis Tier</div>
        <div class="kpi-value">{tier}</div>
        <div class="kpi-sub">{'Hardware counters available' if has_counters else 'Trace-level only'}</div>
      </div>
    </div>
    {findings_html}
  </div>
</section>

<!-- ── Execution Breakdown ────────────────────────────────────────── -->
<section class="scard">
  <div class="shdr">
    <span class="shdr-icon">&#9200;</span>
    <h2>Execution Breakdown</h2>
  </div>
  <div class="sbody">
    <div class="stacked">
      <div class="seg" data-tip='{_TIP_KERNEL}' style="width:{kernel_pct:.2f}%;background:linear-gradient(90deg,#4d8ef2,#3a7de0)"></div>
      <div class="seg" data-tip='{_TIP_MEMCPY}' style="width:{memcpy_pct:.2f}%;background:linear-gradient(90deg,#f08432,#d86c20)"></div>
      <div class="seg" data-tip='{_TIP_OVERHEAD}' style="width:{overhead_pct:.2f}%;background:linear-gradient(90deg,#9866cc,#7a4db0)"></div>
      <div class="seg" data-tip='{_TIP_IDLE}' style="width:{idle_pct:.2f}%;background:linear-gradient(90deg,#2c2c48,#222236)"></div>
    </div>
    <div class="legend">
      <div class="leg"><div class="dot" style="background:#4d8ef2"></div>Kernel &nbsp;<strong style="color:#4d8ef2">{kernel_pct:.1f}%</strong></div>
      <div class="leg"><div class="dot" style="background:#f08432"></div>Memory Copies &nbsp;<strong style="color:#f08432">{memcpy_pct:.1f}%</strong></div>
      <div class="leg"><div class="dot" style="background:#9866cc"></div>API Overhead &nbsp;<strong style="color:#9866cc">{overhead_pct:.1f}%</strong></div>
      <div class="leg"><div class="dot" style="background:#2c2c48;border:1px solid #3a3a55"></div>GPU Idle &nbsp;<strong style="color:var(--dim)">{idle_pct:.1f}%</strong></div>
    </div>
    <div class="brows">
      <div class="brow" data-tip='{_TIP_KERNEL}'>
        <div class="blabel">Kernel Execution</div>
        <div class="btrack"><div class="bfill" style="width:{kernel_pct:.2f}%;background:linear-gradient(90deg,#4d8ef2,#3a7de0)"></div></div>
        <div class="bval"><span class="bpct">{kernel_pct:.1f}%</span>&ensp;{kernel_ms:,.2f} ms</div>
      </div>
      <div class="brow" data-tip='{_TIP_MEMCPY}'>
        <div class="blabel">Memory Copies</div>
        <div class="btrack"><div class="bfill" style="width:{memcpy_pct:.2f}%;background:linear-gradient(90deg,#f08432,#d86c20)"></div></div>
        <div class="bval"><span class="bpct">{memcpy_pct:.1f}%</span>&ensp;{memcpy_ms:,.2f} ms</div>
      </div>
      <div class="brow" data-tip='{_TIP_OVERHEAD}'>
        <div class="blabel">API Overhead</div>
        <div class="btrack"><div class="bfill" style="width:{overhead_pct:.2f}%;background:linear-gradient(90deg,#9866cc,#7a4db0)"></div></div>
        <div class="bval"><span class="bpct">{overhead_pct:.1f}%</span>&ensp;{overhead_ms:,.2f} ms</div>
      </div>
      <div class="brow" data-tip='{_TIP_IDLE}'>
        <div class="blabel">GPU Idle</div>
        <div class="btrack"><div class="bfill" style="width:{idle_pct:.2f}%;background:#2c2c48"></div></div>
        <div class="bval"><span class="bpct">{idle_pct:.1f}%</span>&ensp;{idle_ms:,.2f} ms</div>
      </div>
    </div>
  </div>
</section>

<!-- ── Recommendations ────────────────────────────────────────────── -->
<section class="scard">
  <div class="shdr">
    <span class="shdr-icon">&#128161;</span>
    <h2>Optimization Recommendations</h2>
    {_recs_badge_html}
  </div>
  <div class="sbody">
    {recs_html}
  </div>
</section>

{hotspots_html}
{mem_html}

<!-- ── Hardware Counters ──────────────────────────────────────────── -->
<section class="scard">
  <div class="shdr">
    <span class="shdr-icon">&#128300;</span>
    <h2>Hardware Counters</h2>
    {_hw_badge_html}
  </div>
  <div class="sbody">
    {hw_inner}
  </div>
</section>

</div><!-- /wrap -->

<footer>
  <p>Generated by <strong>rocpd analyze</strong> &mdash; AMD ROCm GPU Performance Analysis &bull; {analysis_date}</p>
</footer>

<!-- scroll-to-top FAB -->
<button class="fab" id="fab-top" title="Back to top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">&#8679;</button>

<script>
var ANALYSIS = {json_embedded};

/* ── Theme toggle ── */
var htmlEl = document.documentElement;
var themeBtn = document.getElementById('theme-btn');
var _saved = localStorage.getItem('rocpd-theme') || 'dark';
if (_saved === 'light') {{ htmlEl.setAttribute('data-theme','light'); themeBtn.innerHTML = '&#127769; Dark'; }}
function toggleTheme() {{
  var isLight = htmlEl.getAttribute('data-theme') === 'light';
  htmlEl.setAttribute('data-theme', isLight ? 'dark' : 'light');
  themeBtn.innerHTML = isLight ? '&#9728; Light' : '&#127769; Dark';
  localStorage.setItem('rocpd-theme', isLight ? 'dark' : 'light');
}}

/* ── Scroll-to-top FAB ── */
var fabEl = document.getElementById('fab-top');
window.addEventListener('scroll', function() {{
  if (window.scrollY > 250) {{ fabEl.classList.add('visible'); }}
  else {{ fabEl.classList.remove('visible'); }}
}});

/* ── Recommendation toggle ── */
function toggleR(hdr) {{
  hdr.closest('.r-card').classList.toggle('open');
}}
document.querySelectorAll('.r-card[data-p="HIGH"]').forEach(function(c) {{
  c.classList.add('open');
}});

/* ── Copy command ── */
function cpCmd(id) {{
  var el = document.getElementById(id);
  var txt = el.querySelector('code').textContent;
  if (navigator.clipboard) {{
    navigator.clipboard.writeText(txt).then(function() {{
      var btn = el.querySelector('.cp-btn');
      var orig = btn.textContent;
      btn.textContent = '\u2713 Copied!';
      btn.style.color = 'var(--c-ok)';
      setTimeout(function() {{ btn.textContent = orig; btn.style.color = ''; }}, 1600);
    }});
  }}
}}

/* ── Sortable tables ── */
document.querySelectorAll('.sortable thead th').forEach(function(th) {{
  th.addEventListener('click', function() {{
    var tbl   = th.closest('table');
    var tbody = tbl.querySelector('tbody');
    var col   = Array.prototype.indexOf.call(th.parentElement.children, th);
    var dir   = th.dataset.dir === '1' ? -1 : 1;
    tbl.querySelectorAll('thead th').forEach(function(t) {{
      delete t.dataset.dir;
      t.textContent = t.textContent.replace(/ [\u25b2\u25bc]$/, '');
    }});
    th.dataset.dir = String(dir);
    th.textContent += dir === 1 ? ' \u25b2' : ' \u25bc';
    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
    rows.sort(function(a, b) {{
      var av = a.cells[col].dataset.v || a.cells[col].textContent.trim();
      var bv = b.cells[col].dataset.v || b.cells[col].textContent.trim();
      var an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dir;
      return av < bv ? -dir : av > bv ? dir : 0;
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }});
}});

/* ── Floating tooltip ── */
var ttEl = document.getElementById('tt');
function showTip(e, html_content) {{
  ttEl.innerHTML = html_content; ttEl.classList.add('show'); moveTip(e);
}}
function moveTip(e) {{
  var x = e.clientX + 16, y = e.clientY - 12;
  var w = ttEl.offsetWidth || 320;
  if (x + w + 10 > window.innerWidth) {{ x = e.clientX - w - 14; }}
  if (y + ttEl.offsetHeight + 10 > window.innerHeight) {{ y = e.clientY - ttEl.offsetHeight - 10; }}
  ttEl.style.left = x + 'px'; ttEl.style.top = y + 'px';
}}
function hideTip() {{ ttEl.classList.remove('show'); }}
document.querySelectorAll('[data-tip]').forEach(function(el) {{
  el.addEventListener('mouseenter', function(e) {{ showTip(e, el.dataset.tip); }});
  el.addEventListener('mousemove',  moveTip);
  el.addEventListener('mouseleave', hideTip);
}});

/* ── AMD GPU hardware counter definitions ── */
var COUNTER_TIPS = {{
  'GRBM_COUNT': '<strong>GRBM_COUNT</strong>Total GPU clock cycles elapsed during the profiling window. Acts as the time denominator for all utilization metrics.<em>Usage: GPU Utilization = GRBM_GUI_ACTIVE &divide; GRBM_COUNT &times; 100%</em>',
  'GRBM_GUI_ACTIVE': '<strong>GRBM_GUI_ACTIVE</strong>Clock cycles where the GPU Command Processor had active work queued. Numerator for GPU utilization — higher relative to GRBM_COUNT means better GPU occupancy.<em>Target: &ge;70% of GRBM_COUNT for a well-utilized GPU.</em>',
  'SQ_WAVES': '<strong>SQ_WAVES</strong>Total number of wavefronts (groups of 64 threads) launched across all compute units during the profiling window. Each wavefront is one SIMD execution unit.<em>High counts = good parallelism. Used to compute wave occupancy (avg simultaneous waves).</em>',
  'SQ_WAVE_CYCLES': '<strong>SQ_WAVE_CYCLES</strong>Total clock cycles consumed across all wavefronts. Divide by SQ_WAVES to get average cycles per wavefront — a proxy for per-kernel execution time.<em>Compare with GRBM_COUNT to estimate how busy the compute units were vs total time.</em>',
  'SQ_INSTS_VALU': '<strong>SQ_INSTS_VALU</strong>Vector ALU instructions executed — floating-point and integer arithmetic (add, mul, fma, transcendental). This is the primary compute workload.<em>High VALU counts relative to VMEM reads indicate a compute-bound kernel (good use of GPU).</em>',
  'SQ_INSTS_SALU': '<strong>SQ_INSTS_SALU</strong>Scalar ALU instructions — operations applied identically to all 64 threads in a wavefront (address calculation, control flow, predication).<em>Very high SALU relative to VALU may indicate excessive branching or non-uniform control flow.</em>',
  'SQ_INSTS_VMEM_RD': '<strong>SQ_INSTS_VMEM_RD</strong>Vector memory read instructions (global/local memory loads). Each instruction may trigger multiple cache line fetches depending on access patterns.<em>High counts relative to VALU confirm a memory-bound workload. Improve data locality or increase compute intensity.</em>',
  'SQ_INSTS_VMEM_WR': '<strong>SQ_INSTS_VMEM_WR</strong>Vector memory write instructions (global/local memory stores).<em>High write traffic alongside high reads can saturate HBM bandwidth. Consider write-combining or reducing redundant stores.</em>',
  'SQ_INSTS_LDS': '<strong>SQ_INSTS_LDS</strong>Local Data Share (LDS / shared memory) instructions. LDS is fast on-chip memory shared within a workgroup — much faster than HBM.<em>High LDS usage is generally good (data reuse within workgroup). Watch for LDS bank conflicts which serialize access.</em>',
  'SQ_INSTS_SMEM': '<strong>SQ_INSTS_SMEM</strong>Scalar memory instructions — loads from constant/uniform memory accessed by all threads in a wavefront identically.<em>Used for kernel arguments, constant buffers. Low latency due to scalar cache.</em>',
  'FETCH_SIZE': '<strong>FETCH_SIZE</strong>Total kilobytes fetched from the L2 cache to the compute units (read bandwidth from L2 to L1/VGPR).<em>Compare against theoretical L2 bandwidth to assess cache pressure. High values with low VALU suggest memory-bound kernel.</em>',
  'WRITE_SIZE': '<strong>WRITE_SIZE</strong>Total kilobytes written back to the L2 cache from compute units.<em>High write traffic alongside FETCH_SIZE indicates significant memory bandwidth demand. Check if writes can be reduced or deferred.</em>',
  'TCP_TOTAL_READ_REQ': '<strong>TCP_TOTAL_READ_REQ</strong>Texture Cache Processor (TCP / L1 vector data cache) total read requests issued by compute units.<em>Used to compute L1 cache hit rate when combined with TCP miss counters.</em>',
  'TCP_TOTAL_CACHE_ACCESSES': '<strong>TCP_TOTAL_CACHE_ACCESSES</strong>Total accesses to the L1 vector (TCP) cache.<em>Combine with miss counters to compute L1 hit rate. Low hit rate means working set exceeds L1 capacity.</em>',
  'TCC_EA_RDREQ_COUNT_sum': '<strong>TCC L2 Read Requests</strong>L2 cache (TCC) read requests forwarded to the memory system (HBM). High values confirm HBM bandwidth is being heavily utilized.<em>If GPU is memory-bound and this is high, improve data reuse or reduce working set size.</em>',
  'TCC_EA_WRREQ_COUNT_sum': '<strong>TCC L2 Write Requests</strong>L2 cache write requests to HBM. Combine with read requests for total HBM bandwidth demand.<em>High write counts may indicate unnecessary stores or lack of write combining.</em>',
  'TCC_HIT_sum': '<strong>TCC L2 Cache Hits</strong>Number of requests satisfied by the L2 cache without going to HBM.<em>Higher is better. Low L2 hit rate means working set exceeds L2 capacity — consider tiling or blocking.</em>',
  'TCC_MISS_sum': '<strong>TCC L2 Cache Misses</strong>Number of requests that missed L2 and had to fetch from HBM.<em>Each miss adds significant latency (~300-400 cycles on MI300X). Reduce misses via better data locality.</em>',
  'TA_TA_BUSY': '<strong>TA_TA_BUSY</strong>Texture Addresser busy cycles — measures how actively the texture/address unit is computing memory addresses for vector loads.<em>High TA_BUSY alongside low VALU suggests address calculation is a bottleneck.</em>',
  'SQ_ACTIVE_INST_VALU': '<strong>SQ_ACTIVE_INST_VALU</strong>Cycles where VALU instructions were actively executing (not stalled). A measure of effective compute throughput.<em>Compare with SQ_WAVES * cycles to estimate VALU utilization efficiency.</em>',
}};

/* Apply COUNTER_TIPS to counter table rows */
document.querySelectorAll('.ctr-row').forEach(function(tr) {{
  var name = tr.dataset.ctr;
  var tip = COUNTER_TIPS[name] ||
    ('<strong>' + name + '</strong>Hardware performance counter. ' +
     'Values are raw HW event counts for the profiling window. ' +
     '<em>Consult AMD CDNA ISA documentation or rocprofv3 counter reference for full semantics.</em>');
  tr.addEventListener('mouseenter', function(e) {{ showTip(e, tip); }});
  tr.addEventListener('mousemove',  moveTip);
  tr.addEventListener('mouseleave', hideTip);
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tier 0 format helpers
# ---------------------------------------------------------------------------

def _tier0_recommendations_text(recommendations: List[Dict[str, Any]], width: int = 80) -> List[str]:
    """Render Tier 0 recommendations as text lines (same format as Tier 1/2)."""
    lines = []
    for rec in recommendations:
        pri        = rec.get("priority", "INFO")
        cat        = rec.get("category", "")
        issue      = rec.get("issue", "")
        suggestion = rec.get("suggestion", "")
        impact     = rec.get("estimated_impact", "")
        actions    = rec.get("actions", [])
        commands   = rec.get("commands", [])

        lines.append(f"[{pri}] {cat}")
        lines.append("─" * width)
        lines.append(f"  Issue: {issue}")
        lines.append("")
        if suggestion:
            lines.append(f"  Suggestion: {suggestion}")
            for action in actions:
                lines.append(f"    {action}")
            lines.append("")
        if impact:
            lines.append(f"  Estimated Impact: {impact}")
            lines.append("")
        if commands:
            lines.append(f"  Recommended Commands:")
            for cmd in commands:
                tool         = cmd.get("tool", "")
                desc         = cmd.get("description", "")
                full_command = cmd.get("full_command", "")
                flags        = cmd.get("flags", [])
                args         = cmd.get("args", [])
                lines.append(f"    [{tool}] {desc}")
                if flags:
                    lines.append(f"      Flags: {' '.join(flags)}")
                if args:
                    arg_strs = []
                    for a in args:
                        name  = a.get("name", "")
                        value = a.get("value")
                        arg_strs.append(f"{name} {value}" if value is not None else name)
                    lines.append(f"      Args:  {' '.join(arg_strs)}")
                if full_command:
                    lines.append(f"      $ {full_command}")
            lines.append("")
        lines.append("")
    return lines


def _format_tier0_text(tier0_result: Any) -> str:
    """Format Tier 0 source-only analysis as plain text."""
    width = 80
    lines = []
    lines.append("=" * width)
    lines.append("ROCPD AI PROFILING PLAN (TIER 0: SOURCE CODE ANALYSIS)".center(width))
    lines.append("=" * width)
    lines.append(f"Source Directory: {tier0_result.source_dir}")
    lines.append(f"Analysis Date:    {tier0_result.analysis_timestamp}")
    lines.append(f"Programming Model: {tier0_result.programming_model}")
    lines.append(f"Files Scanned:    {tier0_result.files_scanned}  "
                 f"(skipped: {tier0_result.files_skipped})")
    lines.append("")

    # Kernels
    lines.append("━" * width)
    lines.append("DETECTED GPU KERNELS".center(width))
    lines.append("━" * width)
    lines.append(f"  Total kernels found: {tier0_result.kernel_count}")
    if tier0_result.detected_kernels:
        for k in tier0_result.detected_kernels[:20]:
            lines.append(f"  • {k['name']}  ({k.get('launch_type','')})  "
                         f"{k.get('file','').split('/')[-1]}:{k.get('line','')}")
        if len(tier0_result.detected_kernels) > 20:
            lines.append(f"  ... and {len(tier0_result.detected_kernels) - 20} more")
    else:
        lines.append("  No GPU kernels detected in source.")
    lines.append("")

    # Patterns by severity
    lines.append("━" * width)
    lines.append("DETECTED PATTERNS".center(width))
    lines.append("━" * width)
    if tier0_result.detected_patterns:
        for p in tier0_result.detected_patterns:
            sev = p.get("severity", "info").upper()
            cat = p.get("category", "")
            desc = p.get("description", "")
            count = p.get("count", 0)
            lines.append(f"  [{sev}] {cat} — {desc} (×{count})")
    else:
        lines.append("  No significant patterns detected.")
    lines.append("")

    # Risk areas
    if tier0_result.risk_areas:
        lines.append("━" * width)
        lines.append("RISK AREAS".center(width))
        lines.append("━" * width)
        for risk in tier0_result.risk_areas:
            lines.append(f"  ⚠  {risk}")
        lines.append("")

    # ROCTx
    if tier0_result.already_instrumented:
        lines.append(f"  ✓ ROCTx markers detected ({tier0_result.roctx_marker_count} markers)")
        lines.append("")

    # Recommended counters
    if tier0_result.suggested_counters:
        lines.append("━" * width)
        lines.append("SUGGESTED HARDWARE COUNTERS".center(width))
        lines.append("━" * width)
        lines.append("  " + "  ".join(tier0_result.suggested_counters))
        lines.append("")

    # Recommendations
    lines.append("━" * width)
    lines.append("PROFILING RECOMMENDATIONS".center(width))
    lines.append("━" * width)
    lines.append("")
    lines.extend(_tier0_recommendations_text(tier0_result.recommendations, width))

    # Suggested first command
    if tier0_result.suggested_first_command:
        lines.append("━" * width)
        lines.append("START HERE — SUGGESTED FIRST COMMAND".center(width))
        lines.append("━" * width)
        lines.append("")
        lines.append(f"  $ {tier0_result.suggested_first_command}")
        lines.append("")

    # LLM explanation
    if tier0_result.llm_explanation:
        lines.append("━" * width)
        lines.append("AI-ENHANCED INSIGHTS".center(width))
        lines.append("━" * width)
        lines.append("")
        lines.append(tier0_result.llm_explanation)
        lines.append("")

    lines.append("=" * width)
    lines.append("Analysis complete.".center(width))
    lines.append("=" * width)

    return "\n".join(lines)


def _tier0_to_dict(tier0_result: Any) -> Dict[str, Any]:
    """Convert SourceAnalysisResult to a JSON-serializable dict for the tier0 field."""
    return {
        "source_dir": tier0_result.source_dir,
        "analysis_timestamp": tier0_result.analysis_timestamp,
        "programming_model": tier0_result.programming_model,
        "files_scanned": tier0_result.files_scanned,
        "files_skipped": tier0_result.files_skipped,
        "kernel_count": tier0_result.kernel_count,
        "detected_kernels": tier0_result.detected_kernels,
        "detected_patterns": tier0_result.detected_patterns,
        "risk_areas": tier0_result.risk_areas,
        "already_instrumented": tier0_result.already_instrumented,
        "roctx_marker_count": tier0_result.roctx_marker_count,
        "recommendations": _build_recommendations_json(tier0_result.recommendations),
        "suggested_counters": tier0_result.suggested_counters,
        "suggested_first_command": tier0_result.suggested_first_command,
        "llm_explanation": tier0_result.llm_explanation,
    }


def _format_tier0_json(tier0_result: Any) -> str:
    """Format Tier 0 source-only analysis as schema v0.2.0 JSON."""
    import json as _json
    doc: Dict[str, Any] = {
        "schema_version": "0.2.0",
        "metadata": {
            "rocpd_version": _ROCPD_VERSION,
            "analysis_version": "0.2.0",  # schema version, not module version
            "database_file": None,
            "analysis_timestamp": tier0_result.analysis_timestamp,
            "analysis_duration_ms": 0,
            "custom_prompt": None,
        },
        "profiling_info": {
            "total_duration_ns": 0,
            "profiling_mode": "source_only",
            "analysis_tier": 0,
            "gpus": [],
        },
        "summary": {
            "overall_assessment": (
                f"Static analysis of {tier0_result.files_scanned} source files found "
                f"{tier0_result.kernel_count} GPU kernels. "
                f"Programming model: {tier0_result.programming_model}. "
                f"See recommendations for next profiling steps."
            ),
            "primary_bottleneck": "unknown",
            "confidence": 0.0,
            "key_findings": tier0_result.risk_areas,
        },
        "tier0": _tier0_to_dict(tier0_result),
        "execution_breakdown": None,
        "hotspots": [],
        "memory_analysis": {},
        "hardware_counters": {"has_counters": False, "metrics": None, "counters": None},
        "recommendations": _build_recommendations_json(tier0_result.recommendations),
        "warnings": [],
        "errors": [],
        "llm_enhanced_explanation": tier0_result.llm_explanation,
    }
    return _json.dumps(doc, indent=2)


def _format_tier0_markdown(tier0_result: Any) -> str:
    """Format Tier 0 source-only analysis as Markdown."""
    lines = []
    lines.append("# ROCpd AI Profiling Plan — Tier 0: Source Code Analysis")
    lines.append("")
    lines.append(f"**Source Directory:** `{tier0_result.source_dir}`")
    lines.append(f"**Analysis Date:** {tier0_result.analysis_timestamp}")
    lines.append(f"**Programming Model:** {tier0_result.programming_model}")
    lines.append(f"**Analysis Tier:** 0 (Source Code Analysis)")
    lines.append("")

    lines.append("## Detected Kernels")
    lines.append("")
    lines.append(f"**Total GPU kernels found:** {tier0_result.kernel_count}")
    lines.append("")
    if tier0_result.detected_kernels:
        lines.append("| Kernel | Launch Type | File | Line |")
        lines.append("|--------|-------------|------|------|")
        for k in tier0_result.detected_kernels[:20]:
            fname = k.get("file", "").split("/")[-1]
            lines.append(f"| `{k['name']}` | {k.get('launch_type','')} | {fname} | {k.get('line','')} |")
        if len(tier0_result.detected_kernels) > 20:
            lines.append(f"\n*... and {len(tier0_result.detected_kernels) - 20} more kernels*")
    else:
        lines.append("*No GPU kernels detected in source.*")
    lines.append("")

    lines.append("## Detected Patterns")
    lines.append("")
    if tier0_result.detected_patterns:
        lines.append("| Severity | Category | Description | Count |")
        lines.append("|----------|----------|-------------|-------|")
        for p in tier0_result.detected_patterns:
            sev = p.get("severity", "info")
            lines.append(
                f"| **{sev.upper()}** | {p.get('category','')} | {p.get('description','')} | {p.get('count',0)} |"
            )
    else:
        lines.append("*No significant patterns detected.*")
    lines.append("")

    if tier0_result.risk_areas:
        lines.append("## Risk Areas")
        lines.append("")
        for risk in tier0_result.risk_areas:
            lines.append(f"- ⚠ {risk}")
        lines.append("")

    if tier0_result.suggested_counters:
        lines.append("## Suggested Hardware Counters")
        lines.append("")
        lines.append("```")
        lines.append(" ".join(tier0_result.suggested_counters))
        lines.append("```")
        lines.append("")

    lines.append("## Profiling Recommendations")
    lines.append("")
    priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "🔵"}
    for rec in tier0_result.recommendations:
        pri  = rec.get("priority", "INFO")
        cat  = rec.get("category", "")
        emoji = priority_emoji.get(pri, "•")
        lines.append(f"### {emoji} [{pri}] {cat}")
        lines.append("")
        lines.append(f"**Issue:** {rec.get('issue', '')}")
        lines.append("")
        lines.append(f"**Suggestion:** {rec.get('suggestion', '')}")
        actions = rec.get("actions", [])
        if actions:
            lines.append("")
            for action in actions:
                lines.append(f"{action}")
        impact = rec.get("estimated_impact", "")
        if impact:
            lines.append("")
            lines.append(f"**Estimated Impact:** {impact}")
        commands = rec.get("commands", [])
        if commands:
            lines.append("")
            lines.append("**Recommended Commands:**")
            lines.append("")
            for cmd in commands:
                tool         = cmd.get("tool", "")
                desc         = cmd.get("description", "")
                full_command = cmd.get("full_command", "")
                flags        = cmd.get("flags", [])
                args         = cmd.get("args", [])
                lines.append(f"*{tool}* — {desc}")
                if flags:
                    lines.append(f"- Flags: `{' '.join(flags)}`")
                if args:
                    arg_strs = []
                    for a in args:
                        name  = a.get("name", "")
                        value = a.get("value")
                        arg_strs.append(f"{name} {value}" if value is not None else name)
                    lines.append(f"- Args: `{' '.join(arg_strs)}`")
                if full_command:
                    lines.append(f"```bash\n{full_command}\n```")
                lines.append("")
        lines.append("")

    if tier0_result.suggested_first_command:
        lines.append("## Start Here — Suggested First Command")
        lines.append("")
        lines.append("```bash")
        lines.append(tier0_result.suggested_first_command)
        lines.append("```")
        lines.append("")

    if tier0_result.llm_explanation:
        lines.append("## AI-Enhanced Insights")
        lines.append("")
        lines.append(tier0_result.llm_explanation)
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by rocpd analyze (Tier 0) \u2022 {tier0_result.analysis_timestamp}*")
    return "\n".join(lines)


def _format_tier0_webview(tier0_result: Any) -> str:
    """Generate a self-contained AMD-themed HTML Tier 0 report (identical design system as Tier 1/2)."""
    import html as _html
    import json as _json

    def _h(v: Any) -> str:
        return _html.escape(str(v), quote=True)

    SEV_FG = {"high": "#e84040", "medium": "#f08432", "low": "#caa828", "info": "#4d8ef2"}
    SEV_BG = {
        "high":   "rgba(232,64,64,.13)",
        "medium": "rgba(240,132,50,.13)",
        "low":    "rgba(202,168,40,.13)",
        "info":   "rgba(77,142,242,.13)",
    }
    PRIORITY = {
        "HIGH":   ("#e84040", "#2a0808"),
        "MEDIUM": ("#f08432", "#2a1600"),
        "LOW":    ("#caa828", "#241e08"),
        "INFO":   ("#4d8ef2", "#081428"),
    }
    PRIORITY_ICON = {"HIGH": "&#128308;", "MEDIUM": "&#128992;", "LOW": "&#128993;", "INFO": "&#8505;"}

    analysis_date = tier0_result.analysis_timestamp
    src_dir       = str(tier0_result.source_dir)
    src_display   = src_dir[-45:] if len(src_dir) > 45 else src_dir

    # ── Counts ──────────────────────────────────────────────────────────────
    recs = tier0_result.recommendations or []
    n_high   = sum(1 for r in recs if r.get("priority") == "HIGH")
    n_medium = sum(1 for r in recs if r.get("priority") == "MEDIUM")
    n_low    = sum(1 for r in recs if r.get("priority") == "LOW")
    n_info   = sum(1 for r in recs if r.get("priority") == "INFO")

    _badge_parts = []
    if n_high:   _badge_parts.append(f'<span class="hbadge hbadge-crit">&#9679; {n_high} Critical</span>')
    if n_medium: _badge_parts.append(f'<span class="hbadge hbadge-warn">&#9679; {n_medium} Warning</span>')
    if n_low:    _badge_parts.append(f'<span class="hbadge hbadge-ok">&#9679; {n_low} Low</span>')
    if n_info:   _badge_parts.append(f'<span class="hbadge hbadge-info">&#9679; {n_info} Info</span>')
    header_badges_html = " ".join(_badge_parts)

    _recs_badge_html = ""
    if n_high:   _recs_badge_html += f'<span class="shdr-badge sbadge-crit">{n_high} Critical</span> '
    if n_medium: _recs_badge_html += f'<span class="shdr-badge sbadge-warn">{n_medium} Warning</span>'

    # ── Recommendations HTML (same .r-card format as Tier 1/2) ──────────────
    recs_parts = []
    for ri, rec in enumerate(recs):
        p    = rec.get("priority", "INFO")
        cat  = rec.get("category", "")
        fg, _ = PRIORITY.get(p, ("#888", "#1a1a2a"))
        picon = PRIORITY_ICON.get(p, "&#8505;")
        actions_li   = "".join(f"<li>{_h(a)}</li>" for a in rec.get("actions", []))
        actions_html = f'<ol class="r-actions">{actions_li}</ol>' if actions_li else ""
        impact       = rec.get("estimated_impact", "")
        impact_html  = (
            f'<p class="r-impact">&#9889; Expected impact: {_h(impact)}</p>'
            if impact else ""
        )
        cmds_parts = []
        for ci, cmd in enumerate(rec.get("commands", [])):
            fc   = cmd.get("full_command", "")
            tool = cmd.get("tool", "")
            desc = cmd.get("description", "")
            if not fc:
                continue
            cid = f"c{ri}_{ci}"
            cmds_parts.append(
                f'<div class="cmd-blk">'
                f'<span class="tool-tag">{_h(tool)}</span>'
                f'<span class="cmd-desc">{_h(desc)}</span>'
                f'<div class="cmd-row" id="{cid}">'
                f'<code>{_h(fc)}</code>'
                f'<button class="cp-btn" onclick="cpCmd(\'{cid}\')">Copy</button>'
                f'</div></div>'
            )
        cmds_html  = "".join(cmds_parts)
        issue_txt  = rec.get("issue", "")
        suggest    = rec.get("suggestion", "")
        recs_parts.append(
            f'<div class="r-card" style="border-left-color:{fg}" data-p="{_h(p)}">'
            f'<div class="r-hdr" onclick="toggleR(this)">'
            f'<span class="r-priority-icon">{picon}</span>'
            f'<span class="r-badge" style="background:{fg};color:#fff">{_h(p)}</span>'
            f'<span class="r-cat">{_h(cat)}</span>'
            f'<span class="r-chev">&#9660;</span>'
            f'</div>'
            f'<div class="r-body">'
            f'<p class="r-issue"><strong>Issue:</strong> {_h(issue_txt)}</p>'
            f'<p class="r-suggest"><strong>What to do:</strong> {_h(suggest)}</p>'
            f'{actions_html}{impact_html}{cmds_html}'
            f'</div></div>'
        )
    recs_html = (
        "".join(recs_parts)
        or '<p class="dim">No recommendations — workload looks well-optimized.</p>'
    )

    # ── Kernels table ────────────────────────────────────────────────────────
    kernel_rows = []
    for i, k in enumerate(tier0_result.detected_kernels[:50]):
        fname = _h(k.get("file", "").split("/")[-1])
        kernel_rows.append(
            f'<tr>'
            f'<td>{i + 1}</td>'
            f'<td class="kname" title="{_h(k.get("name",""))}"><code>{_h(k.get("name",""))}</code></td>'
            f'<td>{_h(k.get("launch_type",""))}</td>'
            f'<td>{fname}</td>'
            f'<td data-v="{k.get("line",0)}">{_h(str(k.get("line","")))}</td>'
            f'</tr>'
        )
    if kernel_rows:
        kernels_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128187;</span>'
            '<h2>Detected GPU Kernels</h2>'
            f'<span class="shdr-badge sbadge-info">{tier0_result.kernel_count} found</span>'
            '</div>'
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable sortable">'
            '<thead><tr>'
            '<th data-tip=\'Rank by order found in source.\'>#</th>'
            '<th data-tip=\'GPU kernel function name detected in source code. For HIP/CUDA: __global__ functions.\'>Kernel Name</th>'
            '<th data-tip=\'How the kernel is launched: __global__ for HIP/CUDA, kernel for OpenCL.\'>Launch Type</th>'
            '<th data-tip=\'Source file where the kernel is defined (basename only).\'>File</th>'
            '<th data-tip=\'Line number of the kernel definition in the source file.\'>Line &#8645;</th>'
            '</tr></thead>'
            '<tbody>' + "".join(kernel_rows) + '</tbody>'
            '</table></div></div></section>'
        )
    else:
        kernels_section = (
            '<section class="scard">'
            '<div class="shdr"><span class="shdr-icon">&#128187;</span>'
            '<h2>Detected GPU Kernels</h2></div>'
            '<div class="sbody"><p class="dim">No GPU kernels detected in the source directory.</p></div>'
            '</section>'
        )

    # ── Patterns table ───────────────────────────────────────────────────────
    pattern_rows = []
    for pat in tier0_result.detected_patterns:
        sev  = pat.get("severity", "info").lower()
        sfg  = SEV_FG.get(sev, "#6b7280")
        sbg  = SEV_BG.get(sev, "rgba(107,114,128,.13)")
        pattern_rows.append(
            f'<tr>'
            f'<td><span style="display:inline-block;padding:.14em .55em;border-radius:4px;'
            f'font-size:.69rem;font-weight:800;letter-spacing:.06em;'
            f'background:{sbg};color:{sfg}">{_h(sev.upper())}</span></td>'
            f'<td>{_h(pat.get("category",""))}</td>'
            f'<td>{_h(pat.get("description",""))}</td>'
            f'<td data-v="{pat.get("count",0)}">{pat.get("count",0)}</td>'
            f'</tr>'
        )
    if pattern_rows:
        patterns_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128202;</span>'
            '<h2>Detected Performance Patterns</h2>'
            f'<span class="shdr-badge sbadge-warn">{len(tier0_result.detected_patterns)} found</span>'
            '</div>'
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable sortable">'
            '<thead><tr>'
            '<th data-tip=\'Issue severity. HIGH = likely significant performance impact. MEDIUM = moderate. LOW = minor.\'>Severity</th>'
            '<th data-tip=\'Category of the anti-pattern detected in source code (memory, compute, synchronization, etc.).\'>Category</th>'
            '<th data-tip=\'Description of the specific pattern found and its likely performance impact.\'>Description</th>'
            '<th data-tip=\'Number of occurrences of this pattern across all scanned source files.\'>Count &#8645;</th>'
            '</tr></thead>'
            '<tbody>' + "".join(pattern_rows) + '</tbody>'
            '</table></div></div></section>'
        )
    else:
        patterns_section = ""

    # ── Risk areas ───────────────────────────────────────────────────────────
    risk_li = "".join(f'<li>{_h(r)}</li>' for r in tier0_result.risk_areas)
    risk_section = ""
    if risk_li:
        risk_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#9888;</span>'
            '<h2>Risk Areas</h2>'
            f'<span class="shdr-badge sbadge-warn">{len(tier0_result.risk_areas)}</span>'
            '</div>'
            '<div class="sbody">'
            f'<ul class="findings">{risk_li}</ul>'
            '</div></section>'
        )

    # ── Suggested counters ───────────────────────────────────────────────────
    ctr_badges = " ".join(
        f'<code style="background:rgba(77,142,242,.15);color:#4d8ef2;'
        f'padding:.14em .55em;border-radius:4px;font-size:.83rem;margin:.18rem .1rem;'
        f'display:inline-block">{_h(c)}</code>'
        for c in tier0_result.suggested_counters
    )
    counters_section = ""
    if tier0_result.suggested_counters:
        collect_cmd = (
            "rocprofv3 --sys-trace --pmc "
            + " ".join(tier0_result.suggested_counters)
            + " -- ./your_app"
        )
        counters_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128300;</span>'
            '<h2>Suggested Hardware Counters</h2>'
            f'<span class="shdr-badge sbadge-info">{len(tier0_result.suggested_counters)} counters</span>'
            '</div>'
            '<div class="sbody">'
            '<p style="margin-bottom:.85rem;color:var(--sub);font-size:.9rem">'
            'Collect these counters to enable Tier 2 (hardware-level) analysis:</p>'
            f'<p style="margin-bottom:1rem;line-height:1.9">{ctr_badges}</p>'
            f'<div class="cmd-row" id="cmd-ctr">'
            f'<code>{_h(collect_cmd)}</code>'
            f'<button class="cp-btn" onclick="cpCmd(\'cmd-ctr\')">Copy</button>'
            '</div>'
            '</div></section>'
        )

    # ── Start Here ───────────────────────────────────────────────────────────
    start_here_section = ""
    if tier0_result.suggested_first_command:
        fc = tier0_result.suggested_first_command
        start_here_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#9654;</span>'
            '<h2>Start Here</h2>'
            '<span class="shdr-badge sbadge-info">Recommended First Step</span>'
            '</div>'
            '<div class="sbody">'
            '<p style="margin-bottom:.85rem;color:var(--sub);font-size:.9rem">'
            'Run this command to collect profiling data for Tier 1/2 analysis:</p>'
            f'<div class="cmd-row" id="cmd-start">'
            f'<code>{_h(fc)}</code>'
            f'<button class="cp-btn" onclick="cpCmd(\'cmd-start\')">Copy</button>'
            '</div>'
            '</div></section>'
        )

    # ── LLM section ──────────────────────────────────────────────────────────
    llm_section = ""
    if tier0_result.llm_explanation:
        llm_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#129302;</span>'
            '<h2>AI-Enhanced Insights</h2>'
            '<span class="shdr-badge sbadge-info">LLM</span>'
            '</div>'
            '<div class="sbody">'
            f'<pre style="white-space:pre-wrap;line-height:1.6;'
            f'color:var(--sub);font-size:.9rem">{_h(tier0_result.llm_explanation)}</pre>'
            '</div></section>'
        )

    # ── KPI grid ─────────────────────────────────────────────────────────────
    n_risks        = len(tier0_result.risk_areas)
    risk_kpi_cls   = "kpi-warn" if n_risks > 0 else "kpi-ok"
    risk_kpi_label = "Needs Attention" if n_risks > 0 else "None Found"
    model_upper    = _h(tier0_result.programming_model.upper())
    assessment_txt = (
        f"Static source analysis of {tier0_result.files_scanned} file(s) found "
        f"{tier0_result.kernel_count} GPU kernel(s). "
        f"Programming model: {tier0_result.programming_model}. "
        "See recommendations below for the suggested profiling workflow."
    )
    n_patterns = len(tier0_result.detected_patterns)

    payload = _json.dumps(_tier0_to_dict(tier0_result))
    payload = payload.replace("</script>", r"<\/script>").replace("<!--", r"<\!--")

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROCpd AI Profiling Plan &#8212; Tier 0 Source Analysis</title>
<style>
/* ── Reset + Variables ─────────────────────────────────────────────── */
:root {{
  --bg:#0d0d14; --bg2:#14141f; --bg3:#1c1c2c; --bg4:#242438;
  --bdr:#2c2c48; --bdr2:#3a3a58;
  --text:#e0e3f2; --sub:#a8aace; --dim:#6868a0;
  --amd:#e01a22;
  --blue:#4d8ef2; --green:#3acc66; --orange:#f08432;
  --purple:#9866cc; --teal:#28bca8; --yellow:#caa828;
  --c-ok:#3acc66;   --c-ok-bg:rgba(58,204,102,.13);
  --c-warn:#f08432; --c-warn-bg:rgba(240,132,50,.13);
  --c-crit:#e84040; --c-crit-bg:rgba(232,64,64,.13);
  --c-info:#4d8ef2; --c-info-bg:rgba(77,142,242,.13);
  --r:10px; --r-sm:6px;
  --font:-apple-system,"Segoe UI",system-ui,Ubuntu,sans-serif;
  --mono:"JetBrains Mono","Cascadia Code","Fira Code",ui-monospace,monospace;
  --shadow:0 4px 18px rgba(0,0,0,.42);
  --shadow-lg:0 8px 36px rgba(0,0,0,.55);
  --trans:all 0.22s cubic-bezier(0.4,0,0.2,1);
}}
[data-theme="light"] {{
  --bg:#f2f2f8; --bg2:#ffffff; --bg3:#eaeaf2; --bg4:#dddde8;
  --bdr:#c8c8dc; --bdr2:#b4b4cc;
  --text:#181828; --sub:#444468; --dim:#6868a0;
  --c-ok-bg:rgba(58,204,102,.10); --c-warn-bg:rgba(240,132,50,.10);
  --c-crit-bg:rgba(232,64,64,.10); --c-info-bg:rgba(77,142,242,.10);
  --shadow:0 2px 12px rgba(0,0,0,.10);
  --shadow-lg:0 4px 20px rgba(0,0,0,.14);
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior:smooth; }}
body {{ font-family:var(--font); background:var(--bg); color:var(--text);
       line-height:1.65; font-size:15px; min-height:100vh;
       transition:background .25s,color .25s; }}
a {{ color:var(--blue); }}
code {{ font-family:var(--mono); font-size:.87em; }}
/* ── Header ──────────────────────────────────────────────────────── */
.hdr {{
  background:linear-gradient(135deg,#080810 0%,#120e1c 100%);
  border-bottom:3px solid var(--amd); padding:.9rem 0;
  position:sticky; top:0; z-index:100;
  box-shadow:0 2px 16px rgba(0,0,0,.55);
}}
[data-theme="light"] .hdr {{ background:linear-gradient(135deg,#1a0a10 0%,#280e18 100%); }}
.hdr-inner {{ max-width:1140px; margin:0 auto; padding:0 1.25rem;
              display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }}
.hdr-brand {{ display:flex; align-items:baseline; gap:.6rem; }}
.logo {{ font-size:1.6rem; font-weight:900; color:var(--amd);
         letter-spacing:-.04em; line-height:1; }}
.logo em {{ color:#f0f0ff; font-style:normal; }}
.hdr-subtitle {{ font-size:.88rem; color:rgba(255,255,255,.55); font-weight:500; }}
.hdr-badges {{ display:flex; gap:.4rem; flex-wrap:wrap; margin-left:auto; }}
.hbadge {{ font-size:.7rem; font-weight:800; padding:.2em .65em;
           border-radius:100px; letter-spacing:.04em;
           display:inline-flex; align-items:center; gap:.25em; }}
.hbadge-crit {{ background:var(--c-crit-bg); color:var(--c-crit); border:1px solid rgba(232,64,64,.4); }}
.hbadge-warn {{ background:var(--c-warn-bg); color:var(--c-warn); border:1px solid rgba(240,132,50,.4); }}
.hbadge-ok   {{ background:var(--c-ok-bg);   color:var(--c-ok);   border:1px solid rgba(58,204,102,.4); }}
.hbadge-info {{ background:var(--c-info-bg);  color:var(--c-info); border:1px solid rgba(77,142,242,.4); }}
.hdr-controls {{ display:flex; gap:.5rem; align-items:center; }}
.hdr-btn {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
            color:rgba(255,255,255,.7); border-radius:var(--r-sm);
            padding:.3em .75em; font-size:.79rem; cursor:pointer;
            font-family:var(--font); transition:var(--trans);
            display:flex; align-items:center; gap:.3em; }}
.hdr-btn:hover {{ background:rgba(255,255,255,.14); color:#fff; }}
.hdr-pills {{ max-width:1140px; margin:.55rem auto 0; padding:.5rem 1.25rem 0;
              display:flex; gap:.45rem; flex-wrap:wrap;
              border-top:1px solid rgba(255,255,255,.07); }}
.hpill {{ font-size:.72rem; background:rgba(255,255,255,.06);
          border:1px solid rgba(255,255,255,.1); border-radius:5px;
          padding:.12em .55em; display:flex; align-items:center; gap:.3em; }}
.hpill-label {{ color:rgba(255,255,255,.4); }}
.hpill-value {{ font-family:var(--mono); font-weight:600; color:rgba(255,255,255,.75); }}
/* ── Layout ──────────────────────────────────────────────────────── */
.wrap {{ max-width:1140px; margin:0 auto; padding:1.5rem 1.25rem 5rem; }}
/* ── Section Card ────────────────────────────────────────────────── */
.scard {{ background:var(--bg2); border:1px solid var(--bdr); border-radius:var(--r);
          margin-bottom:1.5rem; box-shadow:var(--shadow); overflow:hidden;
          animation:fadeInUp .35s ease both; }}
.scard:nth-child(1) {{ animation-delay:.04s; }}
.scard:nth-child(2) {{ animation-delay:.08s; }}
.scard:nth-child(3) {{ animation-delay:.12s; }}
.scard:nth-child(4) {{ animation-delay:.16s; }}
.scard:nth-child(5) {{ animation-delay:.20s; }}
.scard:nth-child(6) {{ animation-delay:.24s; }}
.scard:nth-child(7) {{ animation-delay:.28s; }}
.scard:nth-child(8) {{ animation-delay:.32s; }}
.shdr {{ display:flex; align-items:center; gap:.6rem;
         padding:.85rem 1.4rem; border-bottom:1px solid var(--bdr);
         background:var(--bg3); }}
.shdr-icon {{ font-size:1.1rem; flex-shrink:0; }}
.shdr h2 {{ font-size:.97rem; font-weight:700; letter-spacing:.02em; flex:1; color:var(--text); }}
.shdr-badge {{ font-size:.69rem; font-weight:800; padding:.15em .55em;
               border-radius:100px; letter-spacing:.04em; flex-shrink:0; }}
.sbadge-crit {{ background:var(--c-crit-bg); color:var(--c-crit); }}
.sbadge-warn {{ background:var(--c-warn-bg); color:var(--c-warn); }}
.sbadge-ok   {{ background:var(--c-ok-bg);   color:var(--c-ok); }}
.sbadge-info {{ background:var(--c-info-bg);  color:var(--c-info); }}
.sbody {{ padding:1.25rem 1.4rem; }}
.dim {{ color:var(--dim); }}
.hint {{ font-size:.85rem; color:var(--dim); }}
/* ── Assessment / Quote ──────────────────────────────────────────── */
.assess {{ font-style:italic; color:var(--sub); font-size:.92rem; line-height:1.7;
           padding:.7rem 1rem; border-left:3px solid var(--blue);
           background:var(--c-info-bg); border-radius:0 var(--r-sm) var(--r-sm) 0;
           margin-bottom:1.25rem; }}
/* ── KPI Grid ────────────────────────────────────────────────────── */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr));
             gap:1rem; margin-bottom:1.25rem; }}
.kpi {{ border:1px solid var(--bdr); border-radius:var(--r); padding:1rem;
        position:relative; overflow:hidden; transition:var(--trans); cursor:help; }}
.kpi:hover {{ transform:translateY(-2px); box-shadow:var(--shadow); }}
.kpi::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
.kpi-ok   {{ background:var(--c-ok-bg); }}   .kpi-ok::before   {{ background:var(--c-ok); }}
.kpi-warn {{ background:var(--c-warn-bg); }}  .kpi-warn::before {{ background:var(--c-warn); }}
.kpi-crit {{ background:var(--c-crit-bg); }}  .kpi-crit::before {{ background:var(--c-crit); }}
.kpi-info {{ background:var(--c-info-bg); }}  .kpi-info::before {{ background:var(--c-info); }}
.kpi-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:.4rem; }}
.kpi-icon {{ font-size:1.25rem; }}
.kpi-status {{ font-size:.68rem; font-weight:800; padding:.14em .5em; border-radius:100px; }}
.kpi-ok   .kpi-status {{ background:rgba(58,204,102,.2);  color:var(--c-ok); }}
.kpi-warn .kpi-status {{ background:rgba(240,132,50,.2);  color:var(--c-warn); }}
.kpi-crit .kpi-status {{ background:rgba(232,64,64,.2);   color:var(--c-crit); }}
.kpi-info .kpi-status {{ background:rgba(77,142,242,.2);  color:var(--c-info); }}
.kpi-label {{ font-size:.69rem; text-transform:uppercase; letter-spacing:.1em; color:var(--dim); margin-bottom:.2rem; }}
.kpi-value {{ font-size:1.55rem; font-weight:800; line-height:1.1; font-family:var(--mono); margin-bottom:.15rem; }}
.kpi-ok   .kpi-value {{ color:var(--c-ok); }}
.kpi-warn .kpi-value {{ color:var(--c-warn); }}
.kpi-crit .kpi-value {{ color:var(--c-crit); }}
.kpi-info .kpi-value {{ color:var(--c-info); }}
.kpi-sub {{ font-size:.77rem; color:var(--dim); }}
/* ── Key Findings / Risk list ────────────────────────────────────── */
.findings {{ list-style:none; margin-top:.85rem; border-top:1px solid var(--bdr); padding-top:.75rem; }}
.findings li {{ font-size:.87rem; color:var(--sub); padding:.28rem 0 .28rem 1.3rem;
                position:relative; border-bottom:1px solid rgba(44,44,72,.3); }}
.findings li:last-child {{ border-bottom:none; }}
.findings li::before {{ content:'⚠'; position:absolute; left:0; color:var(--c-warn); font-weight:700; }}
/* ── Recommendations ─────────────────────────────────────────────── */
.r-card {{ border-left:4px solid; border-radius:0 var(--r) var(--r) 0;
           background:var(--bg3); margin-bottom:.6rem; overflow:hidden;
           transition:background .15s; }}
.r-card:hover {{ background:var(--bg4); }}
.r-hdr {{ display:flex; align-items:center; gap:.55rem; padding:.8rem 1rem;
          cursor:pointer; user-select:none; }}
.r-priority-icon {{ font-size:.9rem; flex-shrink:0; }}
.r-badge {{ padding:.14em .55em; border-radius:4px; font-size:.69rem;
            font-weight:800; letter-spacing:.06em; flex-shrink:0; }}
.r-cat {{ font-weight:600; font-size:.9rem; flex:1; color:var(--text); }}
.r-chev {{ color:var(--dim); font-size:.7rem; transition:transform .2s; flex-shrink:0; }}
.r-card.open .r-chev {{ transform:rotate(180deg); }}
.r-body {{ display:none; padding:.85rem 1rem 1rem; border-top:1px solid var(--bdr); }}
.r-card.open .r-body {{ display:block; }}
.r-issue {{ margin-bottom:.5rem; font-size:.9rem; }}
.r-suggest {{ font-size:.9rem; margin-bottom:.5rem; }}
.r-actions {{ padding-left:1.5rem; margin:.5rem 0; color:var(--sub); font-size:.87rem; }}
.r-actions li {{ margin-bottom:.22rem; }}
.r-impact {{ color:var(--c-ok); font-size:.84rem; margin-top:.65rem;
             padding:.35rem .65rem; background:var(--c-ok-bg);
             border-radius:var(--r-sm); display:inline-block; }}
.cmd-blk {{ margin-top:.85rem; }}
.tool-tag {{ color:var(--blue); font-weight:700; font-size:.83rem; }}
.cmd-desc {{ color:var(--dim); font-size:.81rem; margin-left:.4rem; }}
.cmd-row {{ display:flex; align-items:center; justify-content:space-between;
            gap:.5rem; background:var(--bg); border:1px solid var(--bdr);
            border-radius:var(--r-sm); padding:.55rem .85rem; margin-top:.35rem;
            overflow-x:auto; }}
.cmd-row code {{ color:#a0e870; white-space:nowrap; font-size:.84rem; }}
.cp-btn {{ flex-shrink:0; background:var(--bg3); border:1px solid var(--bdr);
           color:var(--dim); padding:.2em .6em; border-radius:4px;
           cursor:pointer; font-size:.73rem; font-family:var(--font); transition:var(--trans); }}
.cp-btn:hover {{ color:var(--text); border-color:var(--blue); }}
/* ── Tables ──────────────────────────────────────────────────────── */
.tbl-wrap {{ overflow-x:auto; }}
.dtable {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
.dtable th {{ background:var(--bg3); color:var(--dim); font-weight:700; font-size:.76rem;
              text-transform:uppercase; letter-spacing:.06em;
              text-align:left; padding:.55rem .75rem; border-bottom:2px solid var(--bdr);
              cursor:pointer; user-select:none; white-space:nowrap; transition:color .15s; }}
.dtable th:hover {{ color:var(--text); }}
.dtable td {{ padding:.5rem .75rem; border-bottom:1px solid rgba(44,44,72,.4); vertical-align:middle; }}
.dtable tr:last-child td {{ border-bottom:none; }}
.dtable tr:hover td {{ background:rgba(255,255,255,.02); }}
.kname {{ max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
/* ── Floating Tooltip ────────────────────────────────────────────── */
#tt {{
  position:fixed; z-index:9999; pointer-events:none; max-width:320px;
  padding:.7rem 1rem; background:#0e0e1c; border:1px solid #3a3a5c;
  border-radius:10px; box-shadow:0 10px 40px rgba(0,0,0,.7);
  font-size:.8rem; line-height:1.65; color:#dde0f2;
  opacity:0; transition:opacity .12s; white-space:normal;
}}
#tt.show {{ opacity:1; }}
#tt strong {{ color:var(--blue); display:block; margin-bottom:.2rem; font-size:.85rem; }}
#tt code {{ font-size:.78rem; background:rgba(255,255,255,.08); padding:.05em .3em; border-radius:3px; }}
#tt em {{ color:var(--dim); font-size:.77rem; display:block; margin-top:.3rem; }}
[data-tip] {{ cursor:help; }}
/* ── FAB ─────────────────────────────────────────────────────────── */
.fab {{ position:fixed; bottom:1.5rem; right:1.5rem; width:46px; height:46px;
        border-radius:50%; background:linear-gradient(135deg,var(--blue) 0%,var(--purple) 100%);
        color:#fff; border:none; font-size:1.25rem; box-shadow:var(--shadow-lg);
        cursor:pointer; display:flex; align-items:center; justify-content:center;
        transition:var(--trans); z-index:200; opacity:0; pointer-events:none; }}
.fab.visible {{ opacity:1; pointer-events:all; }}
.fab:hover {{ transform:scale(1.1) translateY(-2px); }}
/* ── Footer ──────────────────────────────────────────────────────── */
footer {{ border-top:1px solid var(--bdr); padding:1.25rem 1.25rem; max-width:1140px; margin:0 auto; }}
footer p {{ color:var(--dim); font-size:.77rem; text-align:center; }}
/* ── Animations ──────────────────────────────────────────────────── */
@keyframes fadeInUp {{
  from {{ opacity:0; transform:translateY(14px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width:7px; height:7px; }}
::-webkit-scrollbar-track {{ background:var(--bg); }}
::-webkit-scrollbar-thumb {{ background:var(--bdr2); border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:var(--dim); }}
/* ── Mobile ──────────────────────────────────────────────────────── */
@media (max-width:640px) {{
  .kpi-value {{ font-size:1.3rem; }}
  .hdr-subtitle {{ display:none; }}
}}
</style>
</head>
<body>

<div id="tt"></div>

<!-- ── Header ────────────────────────────────────────────────────── -->
<header class="hdr">
  <div class="hdr-inner">
    <div class="hdr-brand">
      <span class="logo">ROC<em>pd</em></span>
      <span class="hdr-subtitle">AI Profiling Plan</span>
    </div>
    <div class="hdr-badges">{header_badges_html}</div>
    <div class="hdr-controls">
      <button class="hdr-btn" id="theme-btn" onclick="toggleTheme()">&#9728; Light</button>
    </div>
  </div>
  <div class="hdr-pills">
    <div class="hpill"><span class="hpill-label">Source:</span><span class="hpill-value" title="{_h(src_dir)}">{_h(src_display)}</span></div>
    <div class="hpill"><span class="hpill-label">Kernels:</span><span class="hpill-value">{tier0_result.kernel_count}</span></div>
    <div class="hpill"><span class="hpill-label">Tier:</span><span class="hpill-value">0 (Source)</span></div>
    <div class="hpill"><span class="hpill-label">Generated:</span><span class="hpill-value">{_h(analysis_date)}</span></div>
    <div class="hpill"><span class="hpill-label">Model:</span><span class="hpill-value">{_h(tier0_result.programming_model)}</span></div>
  </div>
</header>

<div class="wrap">

<!-- ── Overview ──────────────────────────────────────────────────── -->
<section class="scard">
  <div class="shdr">
    <span class="shdr-icon">&#128202;</span>
    <h2>Overview</h2>
    <span class="shdr-badge sbadge-info">Tier 0</span>
  </div>
  <div class="sbody">
    <p class="assess">{_h(assessment_txt)}</p>
    <div class="kpi-grid">
      <div class="kpi kpi-info" data-tip='<strong>GPU Kernels Detected</strong>Number of GPU kernel functions found in the source directory by static analysis. Each __global__ (HIP/CUDA) or kernel (OpenCL) function is counted.'>
        <div class="kpi-head"><span class="kpi-icon">&#128187;</span><span class="kpi-status">Detected</span></div>
        <div class="kpi-label">GPU Kernels</div>
        <div class="kpi-value">{tier0_result.kernel_count}</div>
        <div class="kpi-sub">{tier0_result.files_scanned} file(s) scanned</div>
      </div>
      <div class="kpi kpi-info" data-tip='<strong>Programming Model</strong>GPU programming model detected in source files. HIP is AMD&#39;s primary GPU programming interface, compatible with CUDA syntax.'>
        <div class="kpi-head"><span class="kpi-icon">&#129520;</span><span class="kpi-status">Model</span></div>
        <div class="kpi-label">Programming Model</div>
        <div class="kpi-value" style="font-size:1.2rem">{model_upper}</div>
        <div class="kpi-sub">{tier0_result.files_scanned} files &bull; {tier0_result.files_skipped} skipped</div>
      </div>
      <div class="kpi kpi-info" data-tip='<strong>Performance Patterns</strong>Anti-patterns and potential bottlenecks detected by static source analysis. Patterns are classified by severity (HIGH, MEDIUM, LOW).'>
        <div class="kpi-head"><span class="kpi-icon">&#128202;</span><span class="kpi-status">Found</span></div>
        <div class="kpi-label">Patterns Detected</div>
        <div class="kpi-value">{n_patterns}</div>
        <div class="kpi-sub">potential issues identified</div>
      </div>
      <div class="kpi {risk_kpi_cls}" data-tip='<strong>Risk Areas</strong>High-level risk categories identified in the source code that may cause performance issues at runtime. Run profiling to confirm and quantify each risk.'>
        <div class="kpi-head"><span class="kpi-icon">&#9888;</span><span class="kpi-status">{risk_kpi_label}</span></div>
        <div class="kpi-label">Risk Areas</div>
        <div class="kpi-value">{n_risks}</div>
        <div class="kpi-sub">{"requires profiling to confirm" if n_risks > 0 else "no obvious risk areas"}</div>
      </div>
    </div>
  </div>
</section>

<!-- ── Recommendations ────────────────────────────────────────────── -->
<section class="scard">
  <div class="shdr">
    <span class="shdr-icon">&#128161;</span>
    <h2>Profiling Recommendations</h2>
    {_recs_badge_html}
  </div>
  <div class="sbody">
    {recs_html}
  </div>
</section>

{kernels_section}
{patterns_section}
{risk_section}
{counters_section}
{start_here_section}
{llm_section}

</div><!-- /wrap -->

<footer>
  <p>Generated by <strong>rocpd analyze</strong> (Tier 0) &mdash; AMD ROCm GPU Performance Analysis &bull; {_h(analysis_date)}</p>
</footer>

<!-- scroll-to-top FAB -->
<button class="fab" id="fab-top" title="Back to top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">&#8679;</button>

<script>
var TIER0 = {payload};

/* ── Theme toggle ── */
var htmlEl = document.documentElement;
var themeBtn = document.getElementById('theme-btn');
var _saved = localStorage.getItem('rocpd-theme') || 'dark';
if (_saved === 'light') {{ htmlEl.setAttribute('data-theme','light'); themeBtn.innerHTML = '&#127769; Dark'; }}
function toggleTheme() {{
  var isLight = htmlEl.getAttribute('data-theme') === 'light';
  htmlEl.setAttribute('data-theme', isLight ? 'dark' : 'light');
  themeBtn.innerHTML = isLight ? '&#9728; Light' : '&#127769; Dark';
  localStorage.setItem('rocpd-theme', isLight ? 'dark' : 'light');
}}

/* ── Scroll-to-top FAB ── */
var fabEl = document.getElementById('fab-top');
window.addEventListener('scroll', function() {{
  if (window.scrollY > 250) {{ fabEl.classList.add('visible'); }}
  else {{ fabEl.classList.remove('visible'); }}
}});

/* ── Recommendation toggle ── */
function toggleR(hdr) {{
  hdr.closest('.r-card').classList.toggle('open');
}}
document.querySelectorAll('.r-card[data-p="HIGH"]').forEach(function(c) {{
  c.classList.add('open');
}});

/* ── Copy command ── */
function cpCmd(id) {{
  var el = document.getElementById(id);
  var txt = el.querySelector('code').textContent;
  if (navigator.clipboard) {{
    navigator.clipboard.writeText(txt).then(function() {{
      var btn = el.querySelector('.cp-btn');
      var orig = btn.textContent;
      btn.textContent = '\u2713 Copied!';
      btn.style.color = 'var(--c-ok)';
      setTimeout(function() {{ btn.textContent = orig; btn.style.color = ''; }}, 1600);
    }});
  }}
}}

/* ── Sortable tables ── */
document.querySelectorAll('.sortable thead th').forEach(function(th) {{
  th.addEventListener('click', function() {{
    var tbl   = th.closest('table');
    var tbody = tbl.querySelector('tbody');
    var col   = Array.prototype.indexOf.call(th.parentElement.children, th);
    var dir   = th.dataset.dir === '1' ? -1 : 1;
    tbl.querySelectorAll('thead th').forEach(function(t) {{
      delete t.dataset.dir;
      t.textContent = t.textContent.replace(/ [\u25b2\u25bc]$/, '');
    }});
    th.dataset.dir = String(dir);
    th.textContent += dir === 1 ? ' \u25b2' : ' \u25bc';
    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
    rows.sort(function(a, b) {{
      var av = a.cells[col].dataset.v || a.cells[col].textContent.trim();
      var bv = b.cells[col].dataset.v || b.cells[col].textContent.trim();
      var an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dir;
      return av < bv ? -dir : av > bv ? dir : 0;
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }});
}});

/* ── Floating tooltip ── */
var ttEl = document.getElementById('tt');
function showTip(e, html_content) {{
  ttEl.innerHTML = html_content; ttEl.classList.add('show'); moveTip(e);
}}
function moveTip(e) {{
  var x = e.clientX + 16, y = e.clientY - 12;
  var w = ttEl.offsetWidth || 320;
  if (x + w + 10 > window.innerWidth) {{ x = e.clientX - w - 14; }}
  if (y + ttEl.offsetHeight + 10 > window.innerHeight) {{ y = e.clientY - ttEl.offsetHeight - 10; }}
  ttEl.style.left = x + 'px'; ttEl.style.top = y + 'px';
}}
function hideTip() {{ ttEl.classList.remove('show'); }}
document.querySelectorAll('[data-tip]').forEach(function(el) {{
  el.addEventListener('mouseenter', function(e) {{ showTip(e, el.dataset.tip); }});
  el.addEventListener('mousemove',  moveTip);
  el.addEventListener('mouseleave', hideTip);
}});
</script>
</body>
</html>"""


def format_analysis_output(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    output_format: str = "text",
    tier0_result: Optional[Any] = None,
    source_only: bool = False,
) -> str:
    """
    Format analysis results for display.

    Args:
        time_breakdown: Time distribution metrics
        hotspots: Top kernel hotspots
        memory_analysis: Memory copy analysis
        recommendations: Performance recommendations
        database_path: Path to analyzed database
        output_format: Output format (text, json, markdown, webview)
        tier0_result: Optional Tier 0 source analysis result
        source_only: True when no database was provided (Tier 0 only)

    Returns:
        Formatted string output
    """
    # Source-only mode: dispatch entirely to Tier 0 formatters
    if source_only and tier0_result is not None:
        if output_format == "json":
            return _format_tier0_json(tier0_result)
        if output_format == "markdown":
            return _format_tier0_markdown(tier0_result)
        if output_format == "webview":
            return _format_tier0_webview(tier0_result)
        return _format_tier0_text(tier0_result)

    if output_format == "json":
        output = _format_as_json(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )
        # Combined mode: embed tier0 into JSON document
        if tier0_result is not None:
            import json as _json
            try:
                doc = _json.loads(output)
                doc["tier0"] = _tier0_to_dict(tier0_result)
                output = _json.dumps(doc, indent=2)
            except Exception:
                pass  # Tier0 embedding into combined JSON is non-fatal; return Tier1/2 output unchanged
        return output

    if output_format == "markdown":
        output = _format_as_markdown(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )
        if tier0_result is not None:
            output += "\n\n---\n\n## Tier 0: Source Code Analysis\n\n"
            output += _format_tier0_markdown(tier0_result)
        return output

    if output_format == "webview":
        return _format_as_webview(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )

    # Default: text
    lines = []
    width = 80

    # Header
    lines.append("=" * width)
    lines.append("ROCPD AI PERFORMANCE ANALYSIS".center(width))
    lines.append("=" * width)
    if database_path:
        lines.append(f"Database: {database_path}")
    lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_runtime_ms = time_breakdown.get("total_runtime", 0) / 1e6
    lines.append(f"Total Runtime: {total_runtime_ms:,.2f} ms")
    lines.append("")

    # Time Breakdown
    lines.append("━" * width)
    lines.append("TIME BREAKDOWN".center(width))
    lines.append("━" * width)
    lines.append("")

    def make_bar(percent: float, bar_width: int = 30) -> str:
        """Create a visual percentage bar."""
        filled = int(percent / 100.0 * bar_width)
        return "█" * filled

    kernel_pct = time_breakdown.get("kernel_percent", 0)
    memcpy_pct = time_breakdown.get("memcpy_percent", 0)
    overhead_pct = time_breakdown.get("overhead_percent", 0)

    kernel_time_ms = time_breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_time_ms = time_breakdown.get("total_memcpy_time", 0) / 1e6
    overhead_time_ms = max(0.0, total_runtime_ms - kernel_time_ms - memcpy_time_ms) if total_runtime_ms > 0 else 0

    lines.append(
        f"  Kernel Execution:  {kernel_time_ms:10,.2f} ms  ({kernel_pct:5.1f}%)  {make_bar(kernel_pct)}"
    )
    lines.append(
        f"  Memory Copies:     {memcpy_time_ms:10,.2f} ms  ({memcpy_pct:5.1f}%)  {make_bar(memcpy_pct)}"
    )
    lines.append(
        f"  API Overhead:      {overhead_time_ms:10,.2f} ms  ({overhead_pct:5.1f}%)  {make_bar(overhead_pct)}"
    )
    lines.append("")

    # Hotspots
    if hotspots:
        lines.append("━" * width)
        lines.append("HOTSPOTS".center(width))
        lines.append("━" * width)
        lines.append("")
        lines.append(f"Top {len(hotspots)} Kernels by Duration:")
        lines.append("")

        # Table header
        lines.append(
            f" #  {'Kernel Name':<30}  {'Calls':>6}  {'Total (ms)':>10}  {'Avg (μs)':>9}  {'% Total':>7}"
        )
        lines.append("─" * width)

        # Table rows
        for i, kernel in enumerate(hotspots, 1):
            name = kernel.get("name", "unknown")
            if len(name) > 30:
                name = name[:27] + "..."

            calls = kernel.get("calls", 0)
            total_ms = kernel.get("total_duration", 0) / 1e6
            avg_us = kernel.get("avg_duration", 0) / 1e3
            percent = kernel.get("percent_of_total", 0)

            lines.append(
                f"{i:2}  {name:<30}  {calls:6}  {total_ms:10,.2f}  {avg_us:9,.1f}  {percent:6.1f}%"
            )

        lines.append("")

    # Memory Analysis
    if memory_analysis:
        lines.append("━" * width)
        lines.append("MEMORY COPY ANALYSIS".center(width))
        lines.append("━" * width)
        lines.append("")

        # Table header
        lines.append(
            f"{'Direction':<20}  {'Count':>6}  {'Total Size':>12}  {'Duration':>10}  {'Bandwidth':>10}"
        )
        lines.append("─" * width)

        # Table rows
        for direction, stats in memory_analysis.items():
            count = stats.get("count", 0)
            total_bytes = stats.get("total_bytes", 0)
            duration_ms = stats.get("total_duration", 0) / 1e6
            bandwidth_gbps = stats.get("bandwidth_bytes_per_sec", 0) / 1e9

            # Format size
            if total_bytes >= 1e9:
                size_str = f"{total_bytes/1e9:.1f} GB"
            elif total_bytes >= 1e6:
                size_str = f"{total_bytes/1e6:.1f} MB"
            elif total_bytes >= 1e3:
                size_str = f"{total_bytes/1e3:.1f} KB"
            else:
                size_str = f"{total_bytes:.0f} B"

            lines.append(
                f"{direction:<20}  {count:6}  {size_str:>12}  {duration_ms:9,.2f} ms  {bandwidth_gbps:8.2f} GB/s"
            )

        lines.append("")

    # Hardware Counters (Tier 2)
    if hardware_counters and hardware_counters.get("has_counters"):
        lines.append("━" * width)
        lines.append("HARDWARE COUNTERS (Tier 2)".center(width))
        lines.append("━" * width)
        lines.append("")

        metrics = hardware_counters.get("metrics", {})
        counters = hardware_counters.get("counters", {})

        # Display derived metrics
        if metrics:
            lines.append("Derived Metrics:")
            lines.append("")

            if "gpu_utilization_percent" in metrics:
                util_pct = metrics["gpu_utilization_percent"]
                lines.append(f"  GPU Utilization:        {util_pct:6.1f}%  {make_bar(util_pct)}")

            if "avg_waves" in metrics:
                avg_waves = metrics["avg_waves"]
                max_waves = metrics.get("max_waves", 0)
                lines.append(f"  Avg Wave Occupancy:     {avg_waves:6.1f} waves")
                lines.append(f"  Max Wave Occupancy:     {max_waves:6.1f} waves")

            lines.append("")

        # Display raw counters
        if counters:
            lines.append("Collected Counters:")
            lines.append("")
            lines.append(f"{'Counter Name':<25}  {'Avg Value':>15}  {'Min Value':>15}  {'Max Value':>15}")
            lines.append("─" * width)

            for counter_name, stats in counters.items():
                avg = stats.get("avg_value", 0)
                min_val = stats.get("min_value", 0)
                max_val = stats.get("max_value", 0)

                lines.append(
                    f"{counter_name:<25}  {avg:15,.1f}  {min_val:15,.1f}  {max_val:15,.1f}"
                )

            lines.append("")

    # Recommendations
    lines.append("━" * width)
    lines.append("RECOMMENDATIONS".center(width))
    lines.append("━" * width)
    lines.append("")

    for rec in recommendations:
        priority = rec.get("priority", "INFO")
        category = rec.get("category", "")
        issue = rec.get("issue", "")
        suggestion = rec.get("suggestion", "")
        actions = rec.get("actions", [])
        commands = rec.get("commands", [])
        estimated_impact = rec.get("estimated_impact", "")

        lines.append(f"[{priority}] {category}")
        lines.append("─" * width)
        lines.append(f"  Issue: {issue}")
        lines.append("")
        if suggestion:
            lines.append(f"  Suggestion: {suggestion}")
            if actions:
                for action in actions:
                    lines.append(f"    {action}")
            lines.append("")
        if estimated_impact:
            lines.append(f"  Estimated Impact: {estimated_impact}")
            lines.append("")
        if commands:
            lines.append(f"  Recommended Commands:")
            for cmd in commands:
                tool = cmd.get("tool", "")
                desc = cmd.get("description", "")
                full_command = cmd.get("full_command", "")
                flags = cmd.get("flags", [])
                args = cmd.get("args", [])
                lines.append(f"    [{tool}] {desc}")
                if flags:
                    lines.append(f"      Flags: {' '.join(flags)}")
                if args:
                    arg_strs = []
                    for a in args:
                        name = a.get("name", "")
                        value = a.get("value")
                        arg_strs.append(f"{name} {value}" if value is not None else name)
                    lines.append(f"      Args:  {' '.join(arg_strs)}")
                if full_command:
                    lines.append(f"      $ {full_command}")
            lines.append("")
        lines.append("")

    # Footer
    lines.append("=" * width)
    lines.append("Analysis complete.".center(width))
    lines.append("=" * width)

    return "\n".join(lines)


def analyze_source_code(
    source_dir: str,
    prompt: Optional[str] = None,
    llm: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = False,
) -> Any:
    """
    Run Tier 0 static source code analysis.

    Args:
        source_dir: Path to source directory
        prompt: Optional user question to guide analysis
        llm: LLM provider ("anthropic", "openai")
        llm_api_key: API key for LLM provider
        llm_model: Override LLM model name
        verbose: Enable verbose logging

    Returns:
        SourceAnalysisResult from ai_analysis.api
    """
    from pathlib import Path as _Path
    from .ai_analysis.source_analyzer import SourceAnalyzer
    from .ai_analysis.api import _plan_to_source_result

    _src_path = _Path(source_dir)
    if not _src_path.exists() or not _src_path.is_dir():
        from .ai_analysis.exceptions import SourceDirectoryNotFoundError
        raise SourceDirectoryNotFoundError(
            f"Source directory not found or not a directory: {source_dir}"
        )

    if verbose:
        print(f"[Tier0] Scanning source directory: {source_dir}")

    scanner = SourceAnalyzer(_src_path, verbose=verbose)
    plan = scanner.analyze()

    if verbose:
        print(f"[Tier0] Scanned {plan.files_scanned} files, "
              f"{plan.kernel_count} kernels, model: {plan.programming_model}")

    # Convert ProfilingPlan → SourceAnalysisResult dataclass
    result = _plan_to_source_result(plan)

    if llm:
        _prev = os.environ.get("ROCPD_LLM_MODEL")
        try:
            from .ai_analysis.llm_analyzer import LLMAnalyzer

            if llm_model:
                os.environ["ROCPD_LLM_MODEL"] = llm_model
            try:
                analyzer = LLMAnalyzer(provider=llm, api_key=llm_api_key, verbose=verbose)
                from .ai_analysis.llm_analyzer import AnalysisContext as _AnalysisContext
                _llm_ctx = _AnalysisContext(tier=0, custom_prompt=prompt)
                result.llm_explanation = analyzer.analyze_source_with_llm(
                    result, custom_prompt=prompt, context=_llm_ctx
                )
            finally:
                if llm_model:
                    if _prev is None:
                        os.environ.pop("ROCPD_LLM_MODEL", None)
                    else:
                        os.environ["ROCPD_LLM_MODEL"] = _prev
        except Exception as e:
            print(f"⚠️  Tier 0 LLM enhancement failed: {e}", file=sys.stderr)

    return result


def analyze_performance(
    connection: Optional[RocpdImportData],
    prompt: Optional[str] = None,
    top_kernels: int = 10,
    min_duration: float = 0.0,
    output_format: str = "text",
    database_path: str = "",
    llm: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = False,
    source_dir: Optional[str] = None,
    _collect_result: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """
    Main analysis orchestrator that runs all analyses and formats output.

    Args:
        connection: RocpdImportData database connection
        prompt: Optional custom analysis prompt
        top_kernels: Number of top kernels to analyze
        min_duration: Minimum kernel duration threshold
        output_format: Output format (text, json, markdown)
        database_path: Path to database file
        llm: LLM provider (anthropic or openai)
        llm_api_key: API key for LLM provider
        verbose: Enable verbose logging
        **kwargs: Additional arguments

    Returns:
        Formatted analysis output string
    """
    # ------------------------------------------------------------------
    # Tier 0 — static source code analysis (optional)
    # ------------------------------------------------------------------
    tier0_result = None
    if source_dir:
        tier0_result = analyze_source_code(
            source_dir=source_dir,
            prompt=prompt,
            llm=llm,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Tier 1/2 — database analysis (only when a connection is provided)
    # ------------------------------------------------------------------
    source_only = (connection is None)
    if not source_only:
        time_breakdown = compute_time_breakdown(connection)
        hotspots = identify_hotspots(connection, top_n=top_kernels, min_duration=min_duration)
        memory_analysis = analyze_memory_copies(connection)
        hardware_counters = analyze_hardware_counters(connection)  # Tier 2
        already_collected = _detect_already_collected(connection)
        # Generate recommendations (redundant re-collection commands are filtered out)
        recommendations = generate_recommendations(
            time_breakdown, hotspots, memory_analysis, hardware_counters,
            already_collected=already_collected,
        )
    else:
        time_breakdown = {}
        hotspots = []
        memory_analysis = {}
        hardware_counters = {}
        already_collected = frozenset()
        recommendations = tier0_result.recommendations if tier0_result else []

    # Format output
    output = format_analysis_output(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=recommendations,
        hardware_counters=hardware_counters,
        database_path=database_path,
        output_format=output_format,
        tier0_result=tier0_result,
        source_only=source_only,
    )

    # Expose structured results to caller (used by interactive mode)
    if _collect_result is not None:
        _collect_result["recommendations"] = recommendations
        _collect_result["tier0_result"]    = tier0_result
        _collect_result["database_path"]   = database_path

    # LLM enhancement (if enabled) — only for Tier 1/2; Tier 0 LLM runs in analyze_source_code()
    if llm and not source_only:
        # Initialize before try so the finally block can always reference these names safely.
        _prev_model_env = os.environ.get("ROCPD_LLM_MODEL")
        try:
            if verbose:
                print(f"[LLM] Enabling {llm} enhancement...")

            from .ai_analysis.llm_analyzer import LLMAnalyzer

            # If caller provided --llm-model, set it in the environment so
            # LLMAnalyzer._call_anthropic/_call_openai can pick it up.
            # We restore the original value afterwards.
            if llm_model:
                os.environ["ROCPD_LLM_MODEL"] = llm_model

            # Initialize LLM analyzer
            analyzer = LLMAnalyzer(
                provider=llm,
                api_key=llm_api_key,
                verbose=verbose,
            )

            # Prepare data for LLM
            analysis_data = {
                "gpu": {"name": "AMD GPU", "arch": "unknown"},  # TODO: Extract from DB
                "execution_breakdown": {
                    "kernel_time_pct": time_breakdown.get("kernel_percent", 0),
                    "memcpy_time_pct": time_breakdown.get("memcpy_percent", 0),
                    "api_overhead_pct": time_breakdown.get("overhead_percent", 0),
                },
                "kernels": [
                    {
                        "name": h.get("name", "unknown"),
                        "dispatch_count": h.get("calls", 0),
                        "pct_total_time": h.get("percent_of_total", 0),
                        "avg_duration_ns": h.get("avg_duration", 0),
                    }
                    for h in hotspots[:5]  # Top 5 kernels
                ],
                "memory_ops": {
                    direction: {
                        "count": data.get("count", 0),
                        "total_bytes": data.get("total_bytes", 0),
                        "bandwidth_gbps": data.get("bandwidth_bytes_per_sec", 0) / 1e9,
                    }
                    for direction, data in memory_analysis.items()
                },
                "has_counters": bool(hardware_counters),
                "has_pc_sampling": False,
            }

            # Build analysis context for guide filtering
            from .ai_analysis.llm_analyzer import AnalysisContext as _AnalysisContext
            _has_ctr = bool(hardware_counters and hardware_counters.get("has_counters"))
            _summary = _build_summary(time_breakdown, hotspots, _has_ctr)
            _llm_ctx = _AnalysisContext(
                tier=2 if _has_ctr else 1,
                has_counters=_has_ctr,
                bottleneck_type=_summary.get("primary_bottleneck"),
                gpu_arch=None,  # reserved for future per-GPU filtering
                custom_prompt=prompt,
            )

            # Get LLM enhancement
            llm_explanation = analyzer.analyze_with_llm(
                analysis_data=analysis_data,
                custom_prompt=prompt,
                context=_llm_ctx,
            )

            # Append LLM explanation to output
            if output_format == "text":
                output += "\n\n" + "=" * 80 + "\n"
                output += "AI-ENHANCED EXPLANATION (powered by {})".format(llm.upper()).center(80) + "\n"
                output += "=" * 80 + "\n\n"
                output += llm_explanation
                output += "\n\n" + "=" * 80 + "\n"
            elif output_format == "json":
                # Parse JSON and add LLM explanation
                import json
                try:
                    output_dict = json.loads(output)
                    output_dict["llm_enhanced_explanation"] = llm_explanation
                    output = json.dumps(output_dict, indent=2)
                except (json.JSONDecodeError, ValueError, KeyError) as _je:
                    print(
                        f"Warning: Could not embed LLM explanation in JSON output: {_je}",
                        file=sys.stderr,
                    )

            if verbose:
                print(f"[LLM] Enhancement complete")

        except Exception as e:
            # Always show LLM failures on console (even without --verbose)
            import sys
            error_msg = f"⚠️  LLM enhancement failed: {e}"
            print(error_msg, file=sys.stderr)

            # Also add to output file
            warning_msg = f"\n\n{error_msg}\n(Analysis completed with local results only)\n"
            if output_format == "text":
                output += warning_msg

            # Show full traceback only in verbose mode
            if verbose:
                import traceback
                traceback.print_exc()

        finally:
            # Restore the ROCPD_LLM_MODEL env var to its previous state
            if llm_model:
                if _prev_model_env is None:
                    os.environ.pop("ROCPD_LLM_MODEL", None)
                else:
                    os.environ["ROCPD_LLM_MODEL"] = _prev_model_env

    return output


def _is_code_change_rec(rec: Dict[str, Any]) -> bool:
    """Return True if this recommendation suggests source-code modifications."""
    CODE_CHANGE_KEYWORDS = (
        "replace ", "convert ", "add ", "insert ", "remove ", "delete ",
        "change ", "modify ", "update ", "use hip", "hipstream", "hipmemcpy",
        "hiplaunchkernel", "block size", "blockdim", "thread block",
        "merge kernel", "fuse kernel", "combine kernel", "async",
        "hipstreamcreate", "batch ", "coalesce", "stride", "unroll",
        "pragma ", "#pragma", "__launch_bounds__", "wave32", "wave64",
    )
    for action in rec.get("actions", []):
        al = action.lower()
        if any(kw in al for kw in CODE_CHANGE_KEYWORDS):
            return True
    return False


def _call_llm_for_code(
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    prompt: str,
) -> str:
    """Call Anthropic or OpenAI to generate code-change suggestions."""
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("No Anthropic API key. Set ANTHROPIC_API_KEY or pass --llm-api-key.")
        use_model = model or os.environ.get("ROCPD_LLM_MODEL", "claude-sonnet-4-20250514")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=use_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif provider in ("openai", "gpt"):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("No OpenAI API key. Set OPENAI_API_KEY or pass --llm-api-key.")
        use_model = model or os.environ.get("ROCPD_LLM_MODEL", "gpt-4-turbo-preview")
        client = openai.OpenAI(api_key=key)
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
        except Exception:
            resp = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")


def _apply_code_change_interactive(
    rec: Dict[str, Any],
    source_dir: str,
    llm_provider: Optional[str],
    llm_api_key: Optional[str],
    llm_model: Optional[str],
    colors: Dict[str, str],
) -> None:
    """Walk the user through applying a code-change recommendation."""
    _os = os  # alias to keep existing _os.path.* calls working
    import glob as _glob
    import difflib
    import shutil

    C   = colors["C"]
    G   = colors["G"]
    Y   = colors["Y"]
    R   = colors["R"]
    DIM = colors["DIM"]
    N   = colors["N"]

    cat        = rec.get("category", "")
    issue      = rec.get("issue", "")
    suggestion = rec.get("suggestion", "")
    actions    = rec.get("actions", [])
    impact     = rec.get("estimated_impact", "")

    # ── Show recommendation details ──────────────────────────────────────────
    print(f"\n{C}{'─' * 80}{N}")
    print(f"{C}  Code Change Recommendation: {cat}{N}")
    print(f"{C}{'─' * 80}{N}")
    print(f"\n  {Y}Issue:{N}      {issue}")
    print(f"  {Y}Suggestion:{N} {suggestion}")
    if actions:
        print(f"\n  {Y}Required Changes:{N}")
        for i, action in enumerate(actions, 1):
            print(f"    {i}. {action}")
    if impact:
        print(f"\n  {Y}Estimated Impact:{N} {impact}")
    print()

    if not source_dir:
        print(f"  {DIM}Tip: run with --source-dir <path> to enable AI code editing.{N}\n")
        return

    # ── Find GPU source files ────────────────────────────────────────────────
    source_files: List[str] = []
    for ext in ("*.hip", "*.cpp", "*.cu", "*.cuh", "*.h"):
        source_files.extend(_glob.glob(_os.path.join(source_dir, "**", ext), recursive=True))
    source_files = [f for f in source_files if _os.path.isfile(f)]

    if not source_files:
        print(f"  {DIM}No GPU source files found in {source_dir}/{N}\n")
        return

    # ── Auto-detect LLM provider from environment if not explicitly set ─────
    if not llm_provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            llm_provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            llm_provider = "openai"

    # ── No LLM configured: show manual steps and offer $EDITOR ──────────────
    if not llm_provider:
        print(f"  {DIM}To enable AI code editing, set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in your"
              f" environment, or pass --llm anthropic to rocpd analyze.{N}")
        print(f"\n  {Y}Manual steps:{N}")
        for i, action in enumerate(actions, 1):
            print(f"    {i}. {action}")
        editor = _os.environ.get("EDITOR", "")
        if editor and source_files:
            try:
                ans = input(f"\n  Open source files in {editor}? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans in ("y", "yes"):
                import subprocess
                subprocess.run([editor] + source_files[:3])
        print()
        return

    # ── Ask user before invoking LLM ────────────────────────────────────────
    try:
        ans = input(
            f"  {Y}Would you like the AI to apply this change to your source code? [y/N]: {N}"
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans not in ("y", "yes"):
        print()
        return

    # ── Read source files ────────────────────────────────────────────────────
    MAX_FILES     = 5
    MAX_FILE_SIZE = 50_000  # bytes per file

    print(f"\n  {DIM}Reading source files...{N}")
    file_contents: Dict[str, str] = {}
    for fpath in source_files[:MAX_FILES]:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                file_contents[fpath] = fh.read(MAX_FILE_SIZE)
        except OSError:
            pass

    if not file_contents:
        print(f"  {R}Could not read source files.{N}\n")
        return

    # ── Build LLM prompt ─────────────────────────────────────────────────────
    files_text = "\n\n".join(
        f"=== {_os.path.relpath(fp, source_dir)} ===\n{content}"
        for fp, content in file_contents.items()
    )
    changes_text = "\n".join(f"- {a}" for a in actions)

    llm_prompt = (
        "You are a GPU performance optimization expert. The following GPU source files "
        "have a performance issue that needs to be fixed.\n\n"
        f"ISSUE: {issue}\n"
        f"SUGGESTION: {suggestion}\n"
        f"REQUIRED CHANGES:\n{changes_text}\n\n"
        f"SOURCE FILES:\n{files_text}\n\n"
        "OUTPUT INSTRUCTIONS:\n"
        "For each file that needs modification, output EXACTLY this format:\n"
        "MODIFY_FILE: <relative_filename>\n"
        "<<<ORIGINAL\n"
        "<exact original code section to replace — copy verbatim from the source>\n"
        "ORIGINAL\n"
        "<<<REPLACEMENT\n"
        "<new replacement code>\n"
        "REPLACEMENT\n\n"
        "Only output sections that need to change. Be precise — the ORIGINAL block must "
        "match exactly what appears in the file (used for find-and-replace). "
        "If no changes are needed, output: NO_CHANGES_NEEDED"
    )

    print(f"  {DIM}Calling {llm_provider} for code change suggestions...{N}")

    try:
        llm_response = _call_llm_for_code(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            prompt=llm_prompt,
        )
    except Exception as exc:
        print(f"  {R}LLM error: {exc}{N}\n")
        return

    if "NO_CHANGES_NEEDED" in llm_response:
        print(f"  {G}AI analysis: no code changes are needed for this issue.{N}\n")
        return

    # ── Parse MODIFY_FILE blocks ─────────────────────────────────────────────
    patches: List[tuple] = []
    pattern = re.compile(
        r"MODIFY_FILE:\s*(\S+)\s*<<<ORIGINAL\n(.*?)ORIGINAL\s*<<<REPLACEMENT\n(.*?)REPLACEMENT",
        re.DOTALL,
    )
    for m in pattern.finditer(llm_response):
        rel_path    = m.group(1).strip()
        original    = m.group(2).strip()
        replacement = m.group(3).strip()
        abs_path    = _os.path.join(source_dir, rel_path)
        # Guard against path traversal (e.g. rel_path = "../../etc/passwd")
        _resolved     = _os.path.realpath(abs_path)
        _src_resolved = _os.path.realpath(source_dir)
        if not _resolved.startswith(_src_resolved + _os.sep) and _resolved != _src_resolved:
            continue  # reject: path escapes source_dir
        if _os.path.isfile(abs_path) and abs_path in file_contents:
            patches.append((abs_path, rel_path, original, replacement))

    if not patches:
        print(f"  {Y}AI did not produce actionable code changes.{N}")
        print(f"  {DIM}Raw AI response (first 20 lines):{N}")
        for line in llm_response.splitlines()[:20]:
            print(f"    {DIM}{line}{N}")
        print()
        return

    # ── Show unified diff ────────────────────────────────────────────────────
    print(f"\n{C}{'─' * 80}{N}")
    print(f"{C}  Proposed changes:{N}")
    print(f"{C}{'─' * 80}{N}")

    valid_patches: List[tuple] = []
    for abs_path, rel_path, original, replacement in patches:
        orig_content = file_contents[abs_path]
        if original not in orig_content:
            print(f"\n  {R}✗ Could not locate original code in {rel_path} — skipping.{N}")
            continue
        new_content = orig_content.replace(original, replacement, 1)
        diff = list(difflib.unified_diff(
            orig_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        ))
        print(f"\n  File: {rel_path}")
        for line in diff[:80]:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"  {G}{line.rstrip()}{N}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"  {R}{line.rstrip()}{N}")
            elif line.startswith("@@"):
                print(f"  {C}{line.rstrip()}{N}")
            else:
                print(f"  {DIM}{line.rstrip()}{N}")
        if len(diff) > 80:
            print(f"  {DIM}  ... ({len(diff) - 80} more lines){N}")
        valid_patches.append((abs_path, rel_path, orig_content, new_content))

    if not valid_patches:
        print()
        return

    print()
    try:
        ans = input(f"  {Y}Apply these changes? [y/N]: {N}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if ans not in ("y", "yes"):
        print(f"  {DIM}Changes not applied.{N}\n")
        return

    # ── Apply with backup ────────────────────────────────────────────────────
    applied = 0
    for abs_path, rel_path, orig_content, new_content in valid_patches:
        backup_path = abs_path + ".rocpd.bak"
        try:
            shutil.copy2(abs_path, backup_path)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            print(f"  {G}✓ Applied: {rel_path}  (backup: {_os.path.basename(backup_path)}){N}")
            applied += 1
        except OSError as exc:
            print(f"  {R}✗ Failed to write {rel_path}: {exc}{N}")

    if applied:
        print(f"\n  {G}✓ {applied} file(s) modified. Rebuild your application to test.{N}\n")
        return True
    else:
        print(f"  {Y}No files were modified.{N}\n")
        return False


def _get_app_path_from_db(database_path: str) -> str:
    """
    Extract the profiled application's executable path from a rocpd database.

    rocprofv3 writes the process command into rocpd_info_process_<uuid>.command.
    Returns the path string, or "" if the database cannot be read or has no entry.
    """
    if not database_path:
        return ""
    try:
        import sqlite3 as _sqlite3
        con = _sqlite3.connect(database_path)
        # Find all rocpd_info_process_* tables
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_info_process_%'"
        ).fetchall()
        for (tname,) in tables:
            row = con.execute(f'SELECT command FROM "{tname}" WHERE command IS NOT NULL LIMIT 1').fetchone()
            if row and row[0]:
                return row[0].strip()
        con.close()
    except Exception:
        pass
    return ""


def _run_interactive_session(
    recommendations: List[Dict[str, Any]],
    tier0_result: Optional[Any] = None,
    database_path: str = "",
    source_dir: str = "",
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_local: Optional[str] = None,
    llm_local_model: Optional[str] = None,
    resume_session: Optional[str] = None,
) -> None:
    """Thin shim: delegates to InteractiveSession in ai_analysis/interactive.py."""
    from rocpd.ai_analysis.interactive import InteractiveSession, SessionStore
    InteractiveSession(
        source_dir=source_dir,
        tier0_result=tier0_result,
        recommendations=recommendations,
        database_path=database_path,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_local=llm_local,
        llm_local_model=llm_local_model,
        session_store=SessionStore(),
        resume_session_id=resume_session,
    ).run()


def add_args(parser: argparse.ArgumentParser):
    """
    Add command-line arguments for AI analysis.

    Args:
        parser: Argument parser to add arguments to

    Returns:
        Function to process parsed arguments
    """
    analysis_options = parser.add_argument_group("Analysis options")

    analysis_options.add_argument(
        "--source-dir",
        type=str,
        default=None,
        dest="source_dir",
        help=(
            "Path to GPU application source directory for Tier 0 static analysis. "
            "Scans .hip/.cpp/.cu files and generates a profiling plan. "
            "Can be used alone (no -i required) or alongside -i for combined analysis."
        ),
    )

    analysis_options.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom analysis prompt/question to guide analysis (e.g., 'Why is my matmul kernel slow?')",
    )

    analysis_options.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="Number of top kernels to analyze (default: 10)",
    )

    analysis_options.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown", "webview"],
        default="text",
        help="Output format: text, json, markdown, or webview (default: text). "
             "File extension is set automatically: .txt, .json, .md, .html",
    )

    analysis_options.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum kernel duration threshold in microseconds (filter out short kernels)",
    )

    # LLM Enhancement Options
    llm_options = parser.add_argument_group(
        "LLM enhancement options (optional)",
        "Enable natural language explanations via Anthropic Claude or OpenAI GPT. "
        "Requires API key - see https://console.anthropic.com/ or https://platform.openai.com/api-keys"
    )

    llm_options.add_argument(
        "--llm",
        type=str,
        choices=["anthropic", "openai"],
        default=None,
        help="Enable LLM-powered analysis enhancement. Choices: 'anthropic' (Claude) or 'openai' (GPT). "
             "Requires API key set via environment variable or --llm-api-key option. "
             "Local analysis always runs first; LLM provides additional natural language insights.",
    )

    llm_options.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM provider. Alternatively, set environment variable: "
             "ANTHROPIC_API_KEY for Anthropic Claude, or OPENAI_API_KEY for OpenAI GPT. "
             "Example: --llm anthropic --llm-api-key sk-ant-... "
             "Or: export ANTHROPIC_API_KEY='sk-ant-...' && rocpd analyze --llm anthropic",
    )

    llm_options.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM model name. Defaults to claude-sonnet-4-20250514 for Anthropic "
             "and gpt-4-turbo-preview for OpenAI. Can also be set via ROCPD_LLM_MODEL environment "
             "variable (--llm-model takes precedence). "
             "Examples: --llm-model claude-opus-4-6, --llm-model gpt-4o",
    )

    llm_options.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (shows LLM API calls, reference guide loading, etc.)",
    )

    analysis_options.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        default=False,
        help=(
            "After showing analysis results, enter interactive mode: presents the "
            "recommended profiling commands as a numbered menu and runs whichever "
            "one you select. Useful for iterating through the profiling workflow "
            "without copy-pasting commands manually."
        ),
    )

    analysis_options.add_argument(
        "--resume-session",
        type=str,
        default=None,
        dest="resume_session",
        help=(
            "Resume a previous interactive session by session ID or file path. "
            "Example: --resume-session 2026-03-10_14-23-01_myapp"
        ),
    )

    llm_options.add_argument(
        "--llm-local",
        type=str,
        choices=["ollama"],
        default=None,
        dest="llm_local",
        help=(
            "Local LLM provider for Stage 1 source summarization (before online LLM). "
            "Choices: 'ollama'. Requires Ollama running at localhost:11434. "
            "Set ROCPD_LLM_LOCAL_URL to override endpoint."
        ),
    )

    llm_options.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        dest="llm_local_model",
        help=(
            "Model name for local LLM (default: codellama:13b). "
            "Can also be set via ROCPD_LLM_LOCAL_MODEL environment variable."
        ),
    )

    def process_args(input: RocpdImportData, args: argparse.Namespace):
        """Process and return valid arguments as dictionary."""
        valid_args = ["source_dir", "prompt", "top_kernels", "format", "min_duration",
                      "llm", "llm_api_key", "llm_model", "verbose", "interactive",
                      "resume_session", "llm_local", "llm_local_model"]
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        # Convert min_duration from microseconds to nanoseconds
        if "min_duration" in ret:
            ret["min_duration"] = ret["min_duration"] * 1000
        return ret

    return process_args


def execute(input: Optional[RocpdImportData], config: Optional[output_config.output_config] = None, **kwargs: Any) -> Optional[RocpdImportData]:
    """
    Execute AI analysis on rocpd database and/or source directory.

    Args:
        input: RocpdImportData object with database connection, or None for source-only mode
        config: Optional output configuration
        **kwargs: Analysis parameters (may include source_dir for Tier 0)

    Returns:
        The input RocpdImportData object (for chaining), or None in source-only mode
    """
    # Update config if provided
    if config is not None:
        config = config.update(**kwargs)
    else:
        config = output_config.output_config(**kwargs)

    # Get database path for display
    database_path = ""
    if input is not None and hasattr(input, "_paths") and input._paths:
        database_path = input._paths[0] if isinstance(input._paths, list) else str(input._paths)

    # Pop interactive before passing to analyze_performance (it doesn't accept it)
    interactive = kwargs.pop("interactive", False)

    # Map 'format' CLI key → 'output_format' parameter expected by analyze_performance
    if "format" in kwargs:
        kwargs["output_format"] = kwargs.pop("format")

    # Collect structured results so interactive mode can build its command menu
    result_store: Dict[str, Any] = {}

    # Run analysis
    output = analyze_performance(
        connection=input,
        database_path=database_path,
        _collect_result=result_store,
        **kwargs,
    )

    # Determine file extension based on output format
    _ext_map = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}
    _fmt = kwargs.get("output_format", "text")
    _ext = _ext_map.get(_fmt, ".txt")

    # Handle output
    if config and config.output_file and config.output_path:
        base = config.output_file
        # Append the format extension if the base name doesn't already have it
        if not base.endswith(_ext):
            base = base + _ext
        output_file = os.path.join(config.output_path, base)
        os.makedirs(config.output_path, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Analysis written to: {output_file}")
        if _fmt == "text":
            print("Tip: use --format webview for an interactive HTML report, "
                  "--format json for machine-readable output, "
                  "or --format markdown for Markdown.")
    else:
        print(output)

    # ── Interactive mode ─────────────────────────────────────────────────────
    if interactive:
        _run_interactive_session(
            recommendations=result_store.get("recommendations", []),
            tier0_result=result_store.get("tier0_result"),
            database_path=result_store.get("database_path", database_path),
            source_dir=kwargs.get("source_dir", ""),
            llm_provider=kwargs.get("llm"),
            llm_api_key=kwargs.get("llm_api_key"),
            llm_model=kwargs.get("llm_model"),
            llm_local=kwargs.get("llm_local"),
            llm_local_model=kwargs.get("llm_local_model"),
            resume_session=kwargs.get("resume_session"),
        )

    return input


def main(argv=None) -> int:
    """
    Main entry point for standalone execution.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="rocpd.analyze",
        description="AI-powered performance analysis for GPU traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        required=True,
        help="Input rocpd database file(s)",
    )

    # Add output config args
    output_config.add_args(parser)

    # Add analysis args
    process_args = add_args(parser)

    # Parse arguments
    args = parser.parse_args(argv)

    try:
        # Create database connection
        input_data = RocpdImportData(args.input)

        # Process arguments
        analysis_args = process_args(input_data, args)

        # Execute analysis
        execute(input_data, **analysis_args)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
