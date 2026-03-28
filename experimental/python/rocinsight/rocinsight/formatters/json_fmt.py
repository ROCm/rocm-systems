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
JSON formatting functions for ROCInsight analysis results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ._common import _CATEGORY_IDS, _ROCINSIGHT_VERSION


def _format_as_json(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    interval_timeline=None,
    kernel_categories=None,
    short_kernels=None,
    att_analysis: Optional[Dict[str, Any]] = None,
    custom_prompt: Optional[str] = None,
) -> str:
    """Serialize analysis results to JSON conforming to the current schema version (v0.3.0 when TraceLens fields are present, v0.1.0 otherwise).

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
    idle_time_ns = max(
        0, total_runtime_ns - kernel_time_ns - memcpy_time_ns - api_overhead_ns
    )
    idle_pct = (
        float(idle_time_ns / total_runtime_ns * 100.0) if total_runtime_ns > 0 else 0.0
    )

    # --- metadata ---
    has_counters = bool(hw.get("has_counters", False))
    doc: Dict[str, Any] = {
        "schema_version": "0.1.0",
        "metadata": {
            "rocpd_version": _ROCINSIGHT_VERSION,
            "analysis_version": "0.1.0",  # schema version, not module version
            "database_file": database_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_duration_ms": 0,
            "custom_prompt": None,
        },
        # --- profiling_info ---
        "profiling_info": {
            "total_duration_ns": total_runtime_ns,
            "profiling_mode": (
                "sys_trace_with_counters" if has_counters else "sys_trace_only"
            ),
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

    # TraceLens-derived fields (schema v0.3.0)
    if interval_timeline:
        doc["interval_timeline"] = interval_timeline
    if kernel_categories:
        doc["kernel_categories"] = kernel_categories
    if short_kernels:
        doc["short_kernels"] = short_kernels
    if att_analysis and att_analysis.get("has_att_data"):
        doc["att_trace"] = att_analysis

    # Bump schema version when new fields are present
    if interval_timeline or kernel_categories or short_kernels:
        doc["schema_version"] = "0.3.0"
        doc["metadata"]["analysis_version"] = "0.3.0"
    if att_analysis and att_analysis.get("has_att_data"):
        doc["schema_version"] = "0.4.0"
        doc["metadata"]["analysis_version"] = "0.4.0"
        doc["profiling_info"]["analysis_tier"] = 3

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


def _build_recommendations_json(
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map internal recommendation dicts to the schema v0.1.0 format."""
    out = []
    seen_ids: Dict[str, int] = {}
    for rec in recommendations:
        category = rec.get("category", "General")
        base_id = _CATEGORY_IDS.get(
            category, f"ROCPD-{category.upper().replace(' ', '-')[:12]}-001"
        )
        count = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = count
        rec_id = base_id if count == 1 else f"{base_id[:-3]}{count:03d}"

        out.append(
            {
                "id": rec_id,
                "priority": rec.get("priority", "INFO"),
                "category": category,
                "issue": rec.get("issue", ""),
                "suggestion": rec.get("suggestion", ""),
                "actions": rec.get("actions", []),
                "estimated_impact": rec.get("estimated_impact", ""),
                "commands": rec.get("commands", []),
            }
        )
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
            "rocpd_version": _ROCINSIGHT_VERSION,
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
