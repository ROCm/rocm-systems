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
Recommendation engine: detect already-collected data, filter redundant commands,
generate prioritised recommendations, and classify code-change suggestions.
"""

import re
import shlex
from typing import Any, Dict, List, Optional

from ..connection import RocinsightConnection as RocpdImportData, execute_statement
from .pmc import _SYS_TRACE_IMPLIED, _OUTPUT_ONLY_ARGS
from .att import _ATT_MIN_HITCOUNT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INIT_OVERHEAD_MAX_KERNEL_PCT = 5.0
_INIT_OVERHEAD_MAX_RUNTIME_NS = 1_000_000_000  # 1 second


# ---------------------------------------------------------------------------
# Collection-context detection
# ---------------------------------------------------------------------------


def _detect_already_collected(connection: RocpdImportData) -> frozenset:
    """
    Inspect the database to infer which rocprofv3 flags were used during
    the original profiling run.

    Returns a frozenset of flag strings that are already covered by the
    existing trace, so recommendations can avoid suggesting redundant
    re-collection steps.

    Detection heuristics:
    - ``kernels`` rows    -> ``--kernel-trace`` (or ``--sys-trace``) was used
    - ``regions`` rows    -> ``--hip-trace`` / ``--hsa-trace`` API spans were
      captured (HIP/HSA API region data, a proxy for sys-trace level coverage)
    - ``memory_copies`` rows -> ``--memory-copy-trace`` was used
    - kernels + regions together -> full ``--sys-trace`` implied; all flags
      in ``_SYS_TRACE_IMPLIED`` are marked as already collected
    - ``pmc_events`` rows -> per-counter names stored as ``"pmc:<NAME>"``
      (e.g. ``"pmc:GRBM_COUNT"``) so ``_filter_rec_commands`` can strip
      already-collected counters from ``--pmc`` recommendation commands
    """
    has_kernels = False
    has_api_regions = False  # 'regions' view = HIP/HSA API spans -> hip/hsa-trace
    has_memcpy = False

    checks = (
        ("kernels", "kernels"),
        ("regions", "api_regions"),
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
    - ``rocprof-compute`` commands are always kept -- they perform a deep
      hardware counter analysis that neither rocprofv3 nor rocprof-sys covers.
    - A short note is appended to ``description`` when flags/counters are
      stripped so the user knows why the command looks different from the docs.
    """
    if not already_collected:
        return commands

    has_sys_trace = "--sys-trace" in already_collected

    # Args that are scope filters or output-only -- they don't represent new
    # data collection on their own.
    _NON_DATA_ARGS = _OUTPUT_ONLY_ARGS | frozenset(
        {
            "--kernel-names",
            "--include-names",
            "--exclude-names",
        }
    )

    filtered = []
    for cmd in commands:
        tool = cmd.get("tool", "")
        flags = cmd.get("flags", [])
        args = cmd.get("args", [])

        # -- rocprof-sys -------------------------------------------------------
        if tool == "rocprof-sys" and has_sys_trace:
            # --trace alone ~ rocprofv3 --sys-trace; drop if it adds nothing new
            extra_flags = [f for f in flags if f != "--trace"]
            meaningful_args = [
                a for a in args if a.get("name", "") not in _OUTPUT_ONLY_ARGS
            ]
            if not extra_flags and not meaningful_args:
                continue  # equivalent to already-collected sys-trace data
            # Has meaningful extra flags (e.g. --trace-gpu-memory) -> keep as-is
            filtered.append(cmd)
            continue

        # -- rocprof-compute ---------------------------------------------------
        if tool == "rocprof-compute":
            filtered.append(cmd)  # always keep -- deep hardware counter analysis
            continue

        # -- rocprofv3 ---------------------------------------------------------
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
                    # All counters already collected -- drop arg and flag entirely
                    new_args.pop(pmc_idx)
                    new_flags = [f for f in new_flags if f != "--pmc"]

        nothing_changed = not redundant and not removed_pmc
        if nothing_changed:
            filtered.append(cmd)
            continue

        # Meaningful args: anything that isn't an output path or a scope filter.
        # --kernel-names scopes collection but doesn't collect new data itself.
        meaningful_args = [a for a in new_args if a.get("name", "") not in _NON_DATA_ARGS]
        if not new_flags and not meaningful_args:
            continue  # nothing new to collect -- drop the command entirely

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
    short_kernels: Optional[Dict[str, Any]] = None,  # TraceLens
    interval_timeline: Optional[Dict[str, Any]] = None,  # TraceLens
    att_analysis: Optional[Dict[str, Any]] = None,  # Tier 3 ATT
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
                                {
                                    "name": "--pmc",
                                    "value": "SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY",
                                },
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

    # Tier 3: ATT (Advanced Thread Trace) recommendations
    if att_analysis and att_analysis.get("has_att_data"):
        _pre_att_rec_count = len(recommendations)
        _CATEGORY_LABELS: Dict[str, str] = {
            "att_vmem_latency": "VMEM Latency",
            "att_lds_conflict": "LDS Bank Conflict",
            "att_dependency_chain": "Dependency Chain",
            "att_divergence": "Branch Divergence",
        }
        _CATEGORY_ACTIONS: Dict[str, List[str]] = {
            "att_vmem_latency": [
                "Improve data locality to increase L1/L2 cache hit rate",
                "Reorder memory access pattern for better coalescing (consecutive threads access consecutive addresses)",
                "Prefetch data into LDS before the compute loop",
                "Reduce working-set size to fit in L2 cache if possible",
            ],
            "att_lds_conflict": [
                "Pad LDS arrays by 1 element per row to eliminate 32-way bank conflicts",
                "Use XOR swizzle pattern for b128 LDS reads (avoids bank conflicts in 2D tiles)",
                "Reduce LDS usage per workgroup to allow more waves per CU",
                "Replace DS_READ with GLOBAL_LOAD when per-thread access is non-conflicting",
            ],
            "att_dependency_chain": [
                "Interleave independent instructions to break dependency chains (ILP)",
                "Unroll loops to expose independent iterations for out-of-order issue",
                "Reduce the number of outstanding S_WAITCNT points by batching loads",
                "Move S_WAITCNT as late as possible (just before the dependent instruction)",
            ],
            "att_divergence": [
                "Minimize branch divergence: reorganize data so threads in a wavefront take the same branch",
                "Replace conditional branches with branchless select (V_CNDMASK_B32 or ternary)",
                "Ensure frequently-taken branches are aligned to avoid i-cache pressure",
                "Check for high-frequency V_CMP -> EXEC mask changes that serialize execution",
            ],
        }

        for kernel in att_analysis.get("kernels", []):
            top_instrs = kernel.get("top_stalling_instructions", [])
            if not top_instrs:
                continue
            top = top_instrs[0]
            stall_ratio = top.get("stall_ratio", 0.0)
            hitcount = top.get("hitcount", 0)
            stall_cycles = top.get("stall_cycles", 0)
            total_latency = top.get("total_latency_cycles", 1)
            pc_offset = top.get("pc_offset", "?")
            src_line = top.get("source_line", "")
            kernel_name = kernel.get("name", "unknown")
            category = kernel.get("stall_category", "att_vmem_latency")

            # Only report if statistically meaningful
            if hitcount < _ATT_MIN_HITCOUNT:
                continue

            stall_pct = stall_ratio * 100.0
            if stall_ratio >= 0.60:
                priority = "HIGH"
            elif stall_ratio >= 0.40:
                priority = "MEDIUM"
            else:
                continue  # Low stall ratio -- not worth reporting

            src_part = f" (line {src_line})" if src_line else ""
            issue = (
                f"[ATT] Kernel '{kernel_name}': instruction at {pc_offset}{src_part} "
                f"has {stall_pct:.0f}% stall ratio "
                f"({stall_cycles} stall / {total_latency} total cycles)"
            )
            label = _CATEGORY_LABELS.get(category, "Stall")
            weighted = kernel.get("total_weighted_stall", 0)
            impact = (
                f"Weighted stall: {weighted:,} cycles x threads. "
                f"Addressing this could reduce kernel latency by up to {stall_pct:.0f}%."
            )

            recommendations.append(
                {
                    "priority": priority,
                    "category": f"ATT {label}",
                    "issue": issue,
                    "suggestion": f"Reduce {label.lower()} stalls in kernel '{kernel_name}'",
                    "actions": _CATEGORY_ACTIONS.get(category, []),
                    "estimated_impact": impact,
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Re-collect ATT trace with larger buffer to capture full trace",
                            "flags": ["--att"],
                            "args": [
                                {"name": "--att-library-path", "value": "/opt/rocm/lib"},
                                {"name": "--att-target-cu", "value": "0"},
                                {"name": "--att-buffer-size", "value": "256"},
                                {"name": "-d", "value": "./att_output"},
                                {"name": "-o", "value": "results"},
                            ],
                            "full_command": (
                                "rocprofv3 --att --att-library-path /opt/rocm/lib "
                                "--att-target-cu 0 "
                                "--att-buffer-size 256 -d ./att_output -o results -- ./app"
                            ),
                        },
                    ],
                }
            )

        # If ATT data was present but no HIGH/MEDIUM stalls found, emit an informational rec
        # so the user knows ATT ran successfully and kernels look clean at instruction level.
        if len(recommendations) == _pre_att_rec_count:
            k_count = att_analysis.get("summary", {}).get("kernel_count", 0)
            recommendations.append(
                {
                    "priority": "INFO",
                    "category": "ATT Analysis",
                    "issue": (
                        f"ATT analysis complete: {k_count} kernel(s) traced -- "
                        "no significant instruction-level stalls detected (stall ratio < 40%)"
                    ),
                    "suggestion": "GPU kernels appear well-optimized at the instruction level",
                    "actions": [
                        "Consider profiling with hardware counters (--pmc) to check memory bandwidth utilization",
                        "Use rocprof-compute for full roofline model and micro-architecture metrics",
                    ],
                    "estimated_impact": "N/A -- no significant stalls found",
                    "commands": [],
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

    # Rule 2a: Init-overhead guard -- suppress generic API overhead advice
    # when the workload is so short that ROCm runtime initialization dominates
    kernel_pct = time_breakdown.get("kernel_percent", 0)
    total_runtime_ns = time_breakdown.get("total_runtime", 0)
    _is_init_overhead = (
        kernel_pct < _INIT_OVERHEAD_MAX_KERNEL_PCT
        and 0 < total_runtime_ns < _INIT_OVERHEAD_MAX_RUNTIME_NS
    )

    if _is_init_overhead and not recommendations:
        recommendations.append(
            {
                "priority": "INFO",
                "category": "Runtime Initialization",
                "issue": f"Short workload ({total_runtime_ns/1e6:.0f} ms) with {kernel_pct:.1f}% GPU compute — overhead is dominated by ROCm runtime initialization",
                "suggestion": "GPU code is already well-optimized for this workload size — remaining overhead is one-time initialization cost",
                "actions": [
                    "Run a longer workload or increase iteration count so GPU compute dominates total runtime",
                    "For micro-benchmarks, run one warmup iteration before measurement to amortize init cost",
                    "Place roctxRangeStart/Stop markers around the hot loop only to exclude init from analysis",
                    "If this IS the full workload, consider lazy initialization or pre-compiled kernels (hiprtc)",
                ],
                "estimated_impact": "N/A — GPU compute itself appears efficient; runtime init cost is one-time",
                "commands": [],
            }
        )

    # Rule 2: High API overhead
    overhead_percent = time_breakdown.get("overhead_percent", 0)
    if overhead_percent > 15 and not _is_init_overhead:
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
            # When counters are available, provide specific guidance based on metrics
            _hc_metrics = (hardware_counters or {}).get("metrics", {})
            _gpu_util = _hc_metrics.get("gpu_utilization_percent")
            _avg_w = _hc_metrics.get("avg_waves")
            _has_ctr = bool((hardware_counters or {}).get("has_counters", False))
            if _has_ctr and _gpu_util is not None:
                if _gpu_util > 90:
                    _suggestion = (
                        f"GPU utilization is {_gpu_util:.1f}% — kernel is compute-bound. "
                        "Optimize the kernel algorithm or reduce instruction count"
                    )
                    _actions = [
                        f"GPU utilization: {_gpu_util:.1f}%, wave occupancy: {_avg_w:.0f} avg — hardware is well-utilized",
                        "Focus on algorithmic optimization: reduce transcendental calls, use FMA",
                        "Consider ATT (--att) for instruction-level stall analysis",
                        "Run rocprof-compute for roofline model and instruction mix breakdown",
                    ]
                elif _gpu_util > 70:
                    _suggestion = (
                        f"GPU utilization is {_gpu_util:.1f}% — moderate. "
                        "Check for memory-bound behavior or occupancy limits"
                    )
                    _actions = [
                        f"GPU utilization: {_gpu_util:.1f}%, wave occupancy: {_avg_w:.0f} avg",
                        "Collect FETCH_SIZE and WRITE_SIZE to check memory bandwidth",
                        "Check for memory coalescing issues",
                        "Run rocprof-compute for roofline analysis",
                    ]
                else:
                    _suggestion = (
                        f"GPU utilization is only {_gpu_util:.1f}% — significant room for improvement"
                    )
                    _actions = [
                        f"GPU utilization: {_gpu_util:.1f}%, wave occupancy: {_avg_w:.0f} avg",
                        "Low utilization suggests memory-bound or stalled execution",
                        "Collect FETCH_SIZE and WRITE_SIZE for memory bandwidth analysis",
                        "Check for excessive synchronization or launch overhead",
                    ]
            else:
                _suggestion = "Profile this kernel with hardware counters to identify its specific bottleneck"
                _actions = [
                    "Collect hardware counters to classify compute vs memory bound",
                    "Check memory access patterns for coalescing issues",
                    "Analyze instruction mix: VALU, MFMA, load/store ratios",
                    "Tune occupancy: balance registers, LDS, and block size",
                ]
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Compute Bottleneck",
                    "issue": f"Kernel '{kernel_name}' consumes {percent:.1f}% of GPU time",
                    "suggestion": _suggestion,
                    "actions": _actions,
                    "estimated_impact": "Highly dependent on bottleneck type; 20-50% improvement possible",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Collect GPU hardware counters scoped to the dominant kernel",
                            "flags": ["--sys-trace"],
                            "args": [
                                {
                                    "name": "--pmc",
                                    "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES",
                                },
                                {
                                    "name": "--kernel-names",
                                    "value": kernel_name,
                                },  # display only; full_command uses shlex.quote
                                {"name": "-d", "value": "./kernel_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": (
                                f"rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"
                                f" --kernel-names {shlex.quote(kernel_name)}"
                                f" -d ./kernel_output -o profile -- ./app"
                            ),
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "Roofline model, instruction mix, and memory bottleneck analysis for this kernel",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {
                                    "name": "--kernel",
                                    "value": kernel_name,
                                },  # display only; full_command uses shlex.quote
                            ],
                            "full_command": f"rocprof-compute profile --kernel {shlex.quote(kernel_name)} -- ./app",
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
                        "issue": f"Many small kernels detected: {total_calls} launches, avg {avg_duration / 1000:.1f} \u03bcs each",
                        "suggestion": "Fuse kernels or batch work to amortize per-launch overhead (~5-10 \u03bcs each)",
                        "actions": [
                            "Combine sequential element-wise kernels (e.g., add + multiply) into a single fused kernel",
                            "Increase problem size per launch to push avg duration above 50 \u03bcs",
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
                    "issue": f"{direction} copies achieving only {bandwidth_gbps:.2f} GB/s (avg transfer: {avg_bytes / 1024:.1f} KB)",
                    "suggestion": "Increase transfer size per operation to reach PCIe or HBM saturation bandwidth",
                    "actions": [
                        f"Consolidate many {avg_bytes / 1024:.1f} KB transfers into fewer large transfers (>1 MB each)",
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

    # Rule 9 -- SHORT KERNELS (TraceLens-derived)
    if short_kernels and short_kernels.get("wasted_pct_of_kernel_time", 0) > 5.0:
        wasted_pct = short_kernels["wasted_pct_of_kernel_time"]
        count = short_kernels.get("short_kernel_count", 0)
        threshold = short_kernels.get("threshold_us", 10)
        recommendations.append(
            {
                "priority": "MEDIUM",
                "category": "Launch Efficiency",
                "issue": f"Short kernel overhead: {count} kernels below {threshold}\u03bcs consume {wasted_pct:.1f}% of kernel time",
                "suggestion": "Reduce kernel launch overhead by fusing small kernels or using persistent kernel patterns",
                "actions": [
                    "- Fuse consecutive elementwise ops into a single kernel",
                    "- Use hipGraph to batch kernel launches and reduce launch latency",
                    "- Consider persistent kernels for kernels called >1000\u00d7/sec",
                    "- Profile with rocprofv3 --hip-trace to measure queue latency vs. execution time",
                ],
                "estimated_impact": "5\u201315% reduction in total kernel time if short kernels are dominant",
                "commands": [],
            }
        )

    # Rule 10 -- GPU IDLE TIME (TraceLens interval arithmetic, more accurate than overhead%)
    if interval_timeline and interval_timeline.get("idle_pct", 0) > 20.0:
        idle_pct = interval_timeline["idle_pct"]
        recommendations.append(
            {
                "priority": "HIGH",
                "category": "GPU Utilization",
                "issue": f"High GPU idle time detected: {idle_pct:.1f}% of wall time the GPU is idle",
                "suggestion": "Overlap CPU dispatch work with GPU execution to reduce idle gaps",
                "actions": [
                    "- Use async HIP API calls (hipMemcpyAsync, kernel launches without hipDeviceSynchronize)",
                    "- Introduce hipStream_t streams to overlap independent kernels and transfers",
                    "- Check for unnecessary hipDeviceSynchronize() calls in hot loops",
                    "- Use rocprofv3 --hip-trace to identify synchronization points causing stalls",
                ],
                "estimated_impact": f"Up to {idle_pct:.0f}% improvement in wall-time throughput if idle is CPU-bound dispatch",
                "commands": [],
            }
        )

    # Rule 6: Default if no issues found (MUST be last -- sentinel for "nothing actionable")
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
                            {
                                "name": "--pmc",
                                "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES",
                            },
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
            rec["commands"] = _filter_rec_commands(
                rec.get("commands", []), already_collected
            )

    return recommendations


def _is_code_change_rec(rec: Dict[str, Any]) -> bool:
    """Return True if this recommendation suggests source-code modifications."""
    CODE_CHANGE_KEYWORDS = (
        "replace ",
        "convert ",
        "add ",
        "insert ",
        "remove ",
        "delete ",
        "change ",
        "modify ",
        "update ",
        "use hip",
        "hipstream",
        "hipmemcpy",
        "hiplaunchkernel",
        "block size",
        "blockdim",
        "thread block",
        "merge kernel",
        "fuse kernel",
        "combine kernel",
        "async",
        "hipstreamcreate",
        "batch ",
        "coalesce",
        "stride",
        "unroll",
        "pragma ",
        "#pragma",
        "__launch_bounds__",
        "wave32",
        "wave64",
    )
    for action in rec.get("actions", []):
        al = action.lower()
        if any(kw in al for kw in CODE_CHANGE_KEYWORDS):
            return True
    return False
