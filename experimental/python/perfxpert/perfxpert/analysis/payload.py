###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
build_analysis_payload — deterministic pass over a rocpd database.

Returns the full dataset every formatter (text / json / markdown / webview)
needs so the agentic pipeline is NOT forced to produce structurally-thin
reports when an LLM is selected. The LLM supplies narrative +
primary_bottleneck + recommendations; this pass supplies the tables + stats
that belong in every report regardless of LLM availability.

The deterministic pass is cheap (single-digit milliseconds on typical
fixtures): each helper is a small SQL query against the already-open
``RocpdImportData`` connection plus some Python post-processing. There is no
LLM involvement here.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..connection import PerfxpertConnection as RocpdImportData
from .core import (
    analyze_api_overhead,
    analyze_hardware_counters,
    analyze_kernel_resources,
    analyze_memory_copies,
    compute_time_breakdown,
    detect_warmup_issues,
    identify_hotspots,
)
from .att import analyze_thread_trace
from .recommendations import _detect_already_collected, generate_recommendations


__all__ = [
    "build_analysis_payload",
    "merge_recommendations",
    "scan_tier0_sources",
    "tier0_dict_to_ns",
]


def tier0_dict_to_ns(d: Dict[str, Any]) -> Any:
    """Wrap a tier0 dict into an attribute-access object.

    The legacy tier0 formatters (``_format_tier0_text`` /
    ``_format_tier0_markdown`` / ``_format_tier0_webview`` /
    ``_format_tier0_json``) read via ``tier0_result.<field>`` because the
    pre-refactor source analyzer returned a dataclass. We now return a dict
    from ``scan_tier0_sources``; this helper lets the formatters work
    unchanged without re-plumbing every attribute access.
    """
    from types import SimpleNamespace
    return SimpleNamespace(**(d or {}))


# ---------------------------------------------------------------------------
# Tier-0 source scanner (lightweight)
# ---------------------------------------------------------------------------
#
# The pre-refactor source_analyzer module was removed along with
# ``ai_analysis/``. This inline scanner preserves the contract callers rely on:
#
#   {
#     "source_dir": str,
#     "programming_model": str,
#     "files_scanned": int,
#     "files_skipped": int,
#     "kernel_count": int,
#     "detected_kernels": [{"name": str, "file": str, "line": int}],
#     "detected_patterns": [{"pattern_id": str, "count": int, "locations": [..]}],
#     "risk_areas": [str],
#     "recommendations": [ ... ],
#     "suggested_counters": [str],
#     "suggested_first_command": str,
#   }
#
# This is the dict shape the formatters' tier0 helpers (already present in
# ``perfxpert/formatters/``) expect when a tier-0-capable result is passed in.

_GPU_EXTS = frozenset(
    {".hip", ".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cl", ".py"}
)
_SKIP_DIRS = frozenset({
    ".git", ".svn", "build", "_build", ".build", "__pycache__",
    "node_modules", ".cache", ".tox", ".mypy_cache", ".pytest_cache",
    "CMakeFiles", "dist", "vendor", "third_party", "thirdparty", "extern",
})
_MAX_FILES = 500
_MAX_FILE_SIZE = 512 * 1024

_KERNEL_RE = re.compile(r"\b__global__\s+\w[\w\s\*&<>,:]*\s+(\w+)\s*\(")
_HIP_LAUNCH_RE = re.compile(r"hipLaunchKernelGGL\s*\(\s*(\w+)")
_TRIPLE_LAUNCH_RE = re.compile(r"(\w+)\s*<<<[^>]*>>>\s*\(")
_SYNCHRONIZE_RE = re.compile(r"hipDeviceSynchronize|cudaDeviceSynchronize|hipStreamSynchronize")
_HIPMEMCPY_RE = re.compile(r"\bhipMemcpy\b|\bcudaMemcpy\b")
_ASYNC_MEMCPY_RE = re.compile(r"\bhipMemcpyAsync\b|\bcudaMemcpyAsync\b")
_STREAM_CREATE_RE = re.compile(
    r"\b(?:hip|cuda)Stream(?:Create|CreateWithFlags|CreateWithPriority)\b"
)
_ROCTX_RE = re.compile(r"roctxRangePush|roctxRangePop|roctxMark")


def scan_tier0_sources(source_dir: str) -> Optional[Dict[str, Any]]:
    """Scan ``source_dir`` for GPU programming patterns.

    Returns ``None`` when ``source_dir`` is falsy. Returns a dict with the
    tier0 contract (see module docstring) on success; returns a minimal dict
    with ``files_scanned=0`` when the directory contains no GPU source files.
    """
    if not source_dir:
        return None
    root = Path(source_dir)
    if not root.exists() or not root.is_dir():
        return {
            "source_dir": str(source_dir),
            "analysis_timestamp": datetime.now().isoformat(),
            "programming_model": "Unknown",
            "files_scanned": 0,
            "files_skipped": 0,
            "kernel_count": 0,
            "detected_kernels": [],
            "detected_patterns": [],
            "risk_areas": [f"Source directory not found: {source_dir}"],
            "recommendations": [],
            "suggested_counters": [],
            "suggested_first_command": "",
            "already_instrumented": False,
            "roctx_marker_count": 0,
            "llm_explanation": None,
        }

    detected_kernels: List[Dict[str, Any]] = []
    pattern_counts: Dict[str, int] = {}
    pattern_locations: Dict[str, List[str]] = {}
    files_scanned = 0
    files_skipped = 0
    roctx_count = 0
    has_hip = False
    has_opencl = False
    launch_locations: List[tuple[str, int]] = []
    saw_async_memcpy = False
    saw_stream_api = False

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip unwanted directories in place
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if files_scanned >= _MAX_FILES:
                files_skipped += 1
                continue
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            if ext not in _GPU_EXTS:
                continue
            try:
                if fpath.stat().st_size > _MAX_FILE_SIZE:
                    files_skipped += 1
                    continue
                text = fpath.read_text(errors="ignore")
            except OSError:
                files_skipped += 1
                continue
            files_scanned += 1
            rel = str(fpath.relative_to(root))
            saw_async_memcpy = saw_async_memcpy or bool(_ASYNC_MEMCPY_RE.search(text))
            saw_stream_api = saw_stream_api or bool(_STREAM_CREATE_RE.search(text))

            # detect kernels
            for m in _KERNEL_RE.finditer(text):
                line = text[: m.start()].count("\n") + 1
                detected_kernels.append({"name": m.group(1), "file": rel, "line": line, "launch_type": "GLOBAL_KERNEL_DEF"})
                has_hip = True
            for m in _HIP_LAUNCH_RE.finditer(text):
                line = text[: m.start()].count("\n") + 1
                detected_kernels.append({"name": m.group(1), "file": rel, "line": line, "launch_type": "HIP_KERNEL_LAUNCH"})
                launch_locations.append((rel, line))
                has_hip = True
            for m in _TRIPLE_LAUNCH_RE.finditer(text):
                line = text[: m.start()].count("\n") + 1
                detected_kernels.append({"name": m.group(1), "file": rel, "line": line, "launch_type": "TRIPLE_ANGLE_LAUNCH"})
                launch_locations.append((rel, line))
                has_hip = True
            if ".cl" == ext:
                has_opencl = True

            # detect patterns
            def _bump(pid: str, line: int) -> None:
                pattern_counts[pid] = pattern_counts.get(pid, 0) + 1
                pattern_locations.setdefault(pid, []).append(f"{rel}:{line}")

            for m in _HIPMEMCPY_RE.finditer(text):
                _bump("memcpy_sync", text[: m.start()].count("\n") + 1)
            for m in _SYNCHRONIZE_RE.finditer(text):
                _bump("device_sync", text[: m.start()].count("\n") + 1)
            for m in _ROCTX_RE.finditer(text):
                roctx_count += 1

    if launch_locations and not saw_async_memcpy and not saw_stream_api:
        rel, line = launch_locations[0]
        pattern_counts["default_stream"] = 1
        pattern_locations["default_stream"] = [f"{rel}:{line}"]

    # derive programming model
    if has_opencl:
        model = "OpenCL"
    elif has_hip or detected_kernels:
        model = "HIP"
    else:
        model = "Unknown"

    patterns = []
    risks = []
    for pid, cnt in pattern_counts.items():
        cat = {
            "memcpy_sync": ("medium", "memory_transfer", "Synchronous hipMemcpy calls may block the CPU"),
            "device_sync": ("low", "synchronization", "Explicit device synchronization points"),
            "default_stream": ("medium", "launch_overhead", "Kernel launches stay on the default stream"),
        }.get(pid, ("info", "pattern", pid.replace("_", " ")))
        patterns.append({
            "pattern_id": pid,
            "severity": cat[0],
            "category": cat[1],
            "description": cat[2],
            "count": cnt,
            "locations": pattern_locations.get(pid, [])[:10],
        })
        if cat[0] in ("high", "medium"):
            risks.append(cat[2])

    suggested_counters = ["SQ_WAVES", "GRBM_GUI_ACTIVE", "GRBM_COUNT"]
    suggested_cmd = (
        "rocprofv3 --sys-trace -d ./profile_out -- ./your_app"
        if detected_kernels else ""
    )

    # Bug 3 — two-bucket separation:
    #
    #   profiling_plan_actions : instrumentation advice (what counters to
    #     collect, what rocprofv3 command to run first). These NEVER flow
    #     into the main recommendations list — they belong under the
    #     dedicated "Tier-0: Source Scan" section.
    #
    #   code_patterns          : actual code-level perf issues inferred
    #     from the source (e.g. multiple synchronous hipMemcpy calls).
    #     These DO feed recommendations_deterministic because they describe
    #     a real performance issue in the user's source.
    profiling_plan_actions: List[Dict[str, Any]] = []
    code_patterns: List[Dict[str, Any]] = []

    if detected_kernels:
        profiling_plan_actions.append({
            "priority": "INFO",
            "category": "Tier-0 Profiling Plan",
            "issue": f"Found {len(detected_kernels)} GPU kernel(s) in {files_scanned} source file(s)",
            "suggestion": "Start with a --sys-trace baseline then add --pmc for hardware counters.",
            "actions": [suggested_cmd] if suggested_cmd else [],
        })
    if pattern_counts.get("memcpy_sync", 0) >= 1:
        code_patterns.append({
            "priority": "MEDIUM",
            "category": "Memory Transfer",
            "issue": "Multiple synchronous hipMemcpy calls detected — may bottleneck end-to-end time.",
            "suggestion": "Consider pinned memory + hipMemcpyAsync with an explicit stream.",
        })
    if pattern_counts.get("device_sync", 0) >= 1:
        code_patterns.append({
            "priority": "MEDIUM",
            "category": "Synchronization",
            "issue": "Repeated hipDeviceSynchronize calls serialize work and add avoidable latency.",
            "suggestion": "Remove device-wide syncs from the hot path and synchronize only at true dependencies.",
        })
    if pattern_counts.get("default_stream", 0) >= 1:
        code_patterns.append({
            "priority": "MEDIUM",
            "category": "No Streams",
            "issue": "Kernel launches appear to stay on the default stream, limiting overlap opportunities.",
            "suggestion": "Move copies and launches onto explicit non-default streams so transfer and compute can overlap.",
        })

    # Build the ``profiling_plan`` dict surfaced under
    # ``tier0_findings.profiling_plan`` — the dedicated section formatters
    # render side-by-side with detected code patterns.
    profiling_plan = {
        "suggested_first_command": suggested_cmd,
        "suggested_counters": list(suggested_counters),
        "actions": [suggested_cmd] if suggested_cmd else [],
        "kernel_count": len(detected_kernels),
        "programming_model": model,
        "description": (
            "Start with a --sys-trace baseline then add --pmc for hardware "
            "counters."
        ),
    }

    return {
        "source_dir": str(root),
        "analysis_timestamp": datetime.now().isoformat(),
        "programming_model": model,
        "files_scanned": files_scanned,
        "files_skipped": files_skipped,
        "kernel_count": len(detected_kernels),
        "detected_kernels": detected_kernels[:100],
        "detected_patterns": patterns,
        "risk_areas": risks,
        # ``recommendations`` is the legacy contract — now carries ONLY
        # code-level perf issues (profiling-plan entries live under
        # ``profiling_plan`` so they land in the dedicated Tier-0 section,
        # not the main recommendations list). Bug 3.
        "recommendations": code_patterns,
        "profiling_plan": profiling_plan,
        "profiling_plan_actions": profiling_plan_actions,
        "code_patterns": code_patterns,
        "suggested_counters": suggested_counters,
        "suggested_first_command": suggested_cmd,
        "already_instrumented": roctx_count > 0,
        "roctx_marker_count": roctx_count,
        "llm_explanation": None,
    }


# ---------------------------------------------------------------------------
# Deterministic payload builder
# ---------------------------------------------------------------------------


def build_analysis_payload(
    connection: Optional[RocpdImportData],
    *,
    source_dir: Optional[str] = None,
    att_dir: Optional[str] = None,
    top_kernels: int = 10,
    min_duration: float = 0.0,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run every deterministic analysis pass over ``connection`` and return a
    rendering-ready dict.

    Args:
        connection: Open rocpd DB connection, or ``None`` when running in
            tier-0-only mode (no DB, only ``source_dir``).
        source_dir: Optional path to a source tree — triggers the tier-0 scan.
        att_dir: Optional path to an ATT stats directory — triggers the
            Advanced Thread Trace section (Tier 3).
        top_kernels: Number of hotspot kernels to surface.
        min_duration: Minimum kernel duration (ns) filter for hotspots.
        progress_callback: Optional ``Callable[[str], None]`` used to emit
            ``"deterministic analysis: …"`` status lines around each phase.

    Returns:
        A dict with every section the formatters expect:
            ``database_path``, ``time_breakdown``, ``hotspots``,
            ``memory_analysis``, ``hardware_counters``, ``kernel_resources``,
            ``api_overhead``, ``warmup_issues``, ``thread_trace``,
            ``tier0_findings``, ``recommendations_deterministic``.

        Sections are empty / ``None`` when the input (connection or option)
        was not provided so formatters can short-circuit those blocks.
    """

    def _progress(msg: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(msg)
            except Exception:
                # progress feedback is best-effort; never fail analysis for a
                # spinner glitch
                pass

    _progress("Running deterministic analysis (hotspots, memory, counters…)")

    database_path = ""
    if connection is not None and hasattr(connection, "_paths") and connection._paths:
        paths = connection._paths
        database_path = str(paths[0] if isinstance(paths, list) else paths)

    payload: Dict[str, Any] = {
        "database_path": database_path,
        "time_breakdown": {},
        "hotspots": [],
        "memory_analysis": {},
        "hardware_counters": {"has_counters": False, "metrics": {}, "counters": {}},
        "kernel_resources": {},
        "api_overhead": {},
        "warmup_issues": {},
        "thread_trace": None,
        "tier0_findings": None,
        "recommendations_deterministic": [],
        "metadata": {},
        # Phase 10 — RCCL / NIC communication section. Populated only when
        # the trace actually contains RCCL spans; absent otherwise so
        # formatters can short-circuit the block.
        # ``communication`` shape is:
        #   { "collectives": [...], "summary": {...} }
        # See perfxpert/tools/rccl_analysis.py for the exact schema.
        "communication": None,
        # Phase 10 — Live Roofline points (per-kernel AI + achieved rate +
        # ridge point). Populated only when Tier-2 hardware counters are
        # available in the DB. See perfxpert/tools/roofline.py::plot_points
        # for the exact schema.
        "roofline": None,
    }

    if connection is not None:
        # 1. Time breakdown
        try:
            payload["time_breakdown"] = compute_time_breakdown(connection) or {}
        except Exception:
            payload["time_breakdown"] = {}

        # 2. Hotspots
        try:
            payload["hotspots"] = identify_hotspots(
                connection, top_n=top_kernels, min_duration=min_duration
            ) or []
        except Exception:
            payload["hotspots"] = []

        # 3. Memory copy analysis
        try:
            payload["memory_analysis"] = analyze_memory_copies(connection) or {}
        except Exception:
            payload["memory_analysis"] = {}

        # 4. Hardware counters (Tier 2)
        try:
            payload["hardware_counters"] = analyze_hardware_counters(connection) or {
                "has_counters": False,
                "metrics": {},
                "counters": {},
            }
        except Exception:
            payload["hardware_counters"] = {"has_counters": False, "metrics": {}, "counters": {}}

        # 4b. RCCL / NIC communication analysis (Phase 10). Populated only
        # when the trace actually contains RCCL spans OR RCCL-named kernels
        # (fallback). Key is left as ``None`` otherwise so formatters can
        # skip the block entirely without having to check for empty dicts.
        if database_path:
            try:
                from perfxpert.tools.rccl_analysis import analyze_collectives

                # Try to read the gfx_id from rocpd_info_agent. Best-effort;
                # communication analysis still produces a useful shape
                # without a peak (efficiency_pct just stays at 0).
                gfx_id = None
                try:
                    cur = connection.execute(
                        "SELECT name FROM rocpd_info_agent WHERE type='GPU' LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        gfx_id = str(row[0])
                except Exception:
                    gfx_id = None

                comm = analyze_collectives(database_path, gfx_id=gfx_id)
                if comm and comm.get("collectives"):
                    payload["communication"] = comm
            except Exception:
                payload["communication"] = None

        # 4c. Live Roofline points (Phase 10). Deterministic per-kernel
        # arithmetic-intensity + achieved-FLOPs/s lookup straight from
        # pmc_events. Only populated when hardware counters are actually
        # available so the webview doesn't render an empty chart for
        # Tier-1 traces.
        has_counters = bool(
            payload.get("hardware_counters", {}).get("has_counters", False)
        )
        if has_counters and database_path:
            try:
                from perfxpert.tools.roofline import plot_points

                rf = plot_points(database_path, top_k=top_kernels)
                if rf and rf.get("kernels"):
                    payload["roofline"] = rf
            except Exception:
                payload["roofline"] = None

        # 5. Kernel resources / occupancy (best-effort — requires kernel symbols)
        try:
            payload["kernel_resources"] = analyze_kernel_resources(
                connection, payload["hotspots"]
            ) or {}
        except Exception:
            payload["kernel_resources"] = {}

        # 6. API overhead breakdown
        try:
            payload["api_overhead"] = analyze_api_overhead(connection) or {}
        except Exception:
            payload["api_overhead"] = {}

        # 7. Warmup outlier detection
        try:
            payload["warmup_issues"] = detect_warmup_issues(
                connection, payload["hotspots"]
            ) or {"has_warmup_issues": False, "outliers": []}
        except Exception:
            payload["warmup_issues"] = {"has_warmup_issues": False, "outliers": []}

    # 8. Thread trace (Tier 3) — only when --att-dir is supplied
    if att_dir:
        try:
            payload["thread_trace"] = analyze_thread_trace(att_dir)
        except Exception:
            payload["thread_trace"] = {"has_att_data": False, "reason": "parse error"}

    # 9. Tier-0 source scan — only when --source-dir is supplied
    if source_dir:
        payload["tier0_findings"] = scan_tier0_sources(source_dir)

    # 10. Deterministic recommendations
    try:
        already_collected = frozenset()
        if connection is not None:
            try:
                already_collected = _detect_already_collected(connection) or frozenset()
            except Exception:
                already_collected = frozenset()
        payload["recommendations_deterministic"] = generate_recommendations(
            time_breakdown=payload.get("time_breakdown") or {},
            hotspots=payload.get("hotspots") or [],
            memory_analysis=payload.get("memory_analysis") or {},
            hardware_counters=payload.get("hardware_counters") or {},
            already_collected=already_collected,
            att_analysis=payload.get("thread_trace"),
            warmup_issues=payload.get("warmup_issues"),
            api_overhead=payload.get("api_overhead"),
        ) or []
    except Exception:
        payload["recommendations_deterministic"] = []

    # Bug 3 — only ``code_patterns`` (real code-level perf issues) feed the
    # main deterministic recommendations list. Profiling-plan instrumentation
    # advice lives under ``tier0_findings.profiling_plan`` and is surfaced
    # in the dedicated "Tier-0: Source Scan" section, never in the main
    # recommendations table.
    if payload.get("tier0_findings"):
        # ``code_patterns`` is the post-Bug-3 field; ``recommendations`` is
        # kept in sync for legacy callers but now also holds only code-level
        # items. Falling back keeps pre-refactor tier0 dicts working.
        code_recs = list(
            payload["tier0_findings"].get("code_patterns")
            or payload["tier0_findings"].get("recommendations")
            or []
        )
        payload["recommendations_deterministic"] = (
            list(payload["recommendations_deterministic"]) + code_recs
        )

    _progress("Deterministic analysis done")

    return payload


# ---------------------------------------------------------------------------
# Recommendation merger
# ---------------------------------------------------------------------------


def _rec_key(rec: Dict[str, Any]) -> tuple:
    """Build a de-duplication key for a recommendation.

    Prefers ``(type, target)`` if the LLM schema fields are present,
    otherwise falls back to ``(category, issue)``. Both forms are
    case-insensitive and whitespace-normalised.
    """
    def _norm(v: Any) -> str:
        return str(v or "").strip().lower()

    # LLM shape
    if rec.get("type") or rec.get("target"):
        return ("tt", _norm(rec.get("type")), _norm(rec.get("target")))
    return ("ci", _norm(rec.get("category")), _norm(rec.get("issue")))


def merge_recommendations(
    llm_recs: List[Dict[str, Any]],
    det_recs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge LLM and deterministic recommendation lists.

    De-dup rule: when the same logical recommendation appears in both
    sources (same ``(type, target)`` or ``(category, issue)`` key), prefer
    the LLM entry but carry over any citation / code_snippet_before /
    code_snippet_after fields from the deterministic counterpart so the
    merged rec retains deterministic evidence.

    LLM entries that only have the short agentic schema (``type`` /
    ``target`` / ``summary``) are normalised onto the legacy keys
    (``category`` / ``issue`` / ``suggestion``) so downstream formatters
    render them correctly.
    """
    merged: List[Dict[str, Any]] = []
    seen: Dict[tuple, int] = {}

    def _normalise(rec: Dict[str, Any]) -> Dict[str, Any]:
        r = dict(rec or {})
        r.setdefault("priority", "INFO")
        r.setdefault("category", r.get("type") or "analysis")
        r.setdefault("issue", r.get("summary") or r.get("issue") or "")
        r.setdefault("suggestion", r.get("summary") or r.get("suggestion") or "")
        return r

    for rec in llm_recs or []:
        if not isinstance(rec, dict):
            continue
        r = _normalise(rec)
        key = _rec_key(r)
        if key in seen:
            continue
        seen[key] = len(merged)
        merged.append(r)

    for rec in det_recs or []:
        if not isinstance(rec, dict):
            continue
        r = _normalise(rec)
        key = _rec_key(r)
        if key in seen:
            # LLM version wins, but carry forward deterministic citation /
            # code snippets so we don't lose ground-truth evidence.
            idx = seen[key]
            for field in ("citation", "code_snippet_before", "code_snippet_after",
                          "actions", "commands", "estimated_impact"):
                if r.get(field) and not merged[idx].get(field):
                    merged[idx][field] = r[field]
            continue
        seen[key] = len(merged)
        merged.append(r)

    return merged
