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

Pure analysis logic lives in the ``analysis/`` sub-package; this file is the
thin orchestration and CLI layer.
"""

import argparse
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    from importlib.metadata import version as _pkg_version

    _PERFXPERT_VERSION = _pkg_version("perfxpert")
except Exception:
    _PERFXPERT_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)

from .connection import PerfxpertConnection as RocpdImportData, execute_statement
from .tracelens_port import (
    compute_interval_timeline,
    analyze_kernels_by_category,
    analyze_short_kernels,
)
from . import output_config

# ---------------------------------------------------------------------------
# Re-export analysis functions from the analysis/ sub-package so that
# ``from perfxpert.analyze import compute_time_breakdown`` (etc.) keeps
# working for all existing callers.
# ---------------------------------------------------------------------------
from .analysis import (  # noqa: F401 -- re-exports for backward compat
    identify_hotspots,
    analyze_memory_copies,
    analyze_hardware_counters,
    detect_warmup_issues,
    analyze_kernel_resources,
    analyze_api_overhead,
    analyze_thread_trace,
    generate_recommendations,
    _split_pmc_into_passes,
    _detect_already_collected,
    _filter_rec_commands,
    _is_code_change_rec,
    _ATT_STALL_CATEGORY_MAP,
    _ATT_MIN_HITCOUNT,
    _att_stall_category,
    _SYS_TRACE_IMPLIED,
    _OUTPUT_ONLY_ARGS,
    _PMC_BLOCK_LIMIT_DEFAULT,
    _PMC_BLOCK_LIMITS,
    _TCC_DERIVED_COUNTERS,
    _pmc_block,
    _pmc_block_limit,
    _INIT_OVERHEAD_MAX_KERNEL_PCT,
    _INIT_OVERHEAD_MAX_RUNTIME_NS,
)
from .analysis import core as _analysis_core


def compute_time_breakdown(connection: RocpdImportData) -> Dict[str, Any]:
    """Backward-compat shim — delegates to ``analysis.core`` but uses
    this module's ``execute_statement`` so that
    ``mock.patch("perfxpert.analyze.execute_statement")`` keeps working."""
    import perfxpert.analysis.core as _m

    _saved = _m.execute_statement
    try:
        _m.execute_statement = execute_statement  # pick up any mock on this module
        return _analysis_core.compute_time_breakdown(connection)
    finally:
        _m.execute_statement = _saved

__all__ = [
    "compute_time_breakdown",
    "identify_hotspots",
    "analyze_memory_copies",
    "analyze_hardware_counters",
    "generate_recommendations",
    "format_analysis_output",
    "add_args",
    "execute",
    "main",
]


# ---------------------------------------------------------------------------
# Output formatting functions (extracted to formatters.py)
# ---------------------------------------------------------------------------
from .formatters import (  # noqa: F401 -- re-exports for backward compat
    _format_as_json,
    _build_summary,
    _build_hw_counters_json,
    _build_recommendations_json,
    _build_warnings_json,
    _format_as_markdown,
    _format_as_webview,
    _tier0_recommendations_text,
    _format_tier0_text,
    _tier0_to_dict,
    _format_tier0_json,
    _format_tier0_markdown,
    _format_tier0_webview,
    format_analysis_output,
    _CATEGORY_IDS,
)




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
        dest="output_format",
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
        "Requires API key - see https://console.anthropic.com/ or https://platform.openai.com/api-keys",
    )

    llm_options.add_argument(
        "--llm",
        type=str,
        dest="llm_provider",
        choices=["anthropic", "openai", "opencode", "claude-code"],
        default=None,
        help=(
            "Enable LLM-powered analysis enhancement. "
            "'anthropic' uses the Anthropic API (requires ANTHROPIC_API_KEY). "
            "'openai' uses the OpenAI API (requires OPENAI_API_KEY). "
            "'opencode' uses the local opencode CLI installed on this machine — "
            "no API key needed, uses existing opencode credentials. "
            "'claude-code' is accepted as a deprecated alias for 'opencode'. "
            "Local analysis always runs first; LLM provides additional natural language insights."
        ),
    )

    llm_options.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM provider. Alternatively, set environment variable: "
        "PERFXPERT_LLM_ANTHROPIC_KEY or ANTHROPIC_API_KEY for Anthropic Claude, "
        "or PERFXPERT_LLM_OPENAI_KEY or OPENAI_API_KEY for OpenAI GPT. "
        "Example: --llm anthropic --llm-api-key sk-ant-... "
        "Or: export PERFXPERT_LLM_ANTHROPIC_KEY='sk-ant-...' && perfxpert analyze --llm anthropic",
    )

    llm_options.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM model name. Defaults to claude-sonnet-4-20250514 for Anthropic "
        "and gpt-4-turbo-preview for OpenAI. Can also be set via PERFXPERT_LLM_MODEL environment "
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
        "--att-dir",
        type=str,
        default=None,
        dest="att_dir",
        help=(
            "Path to directory containing ATT stats_*.csv files from rocprofv3 --att. "
            "Enables Tier 3 Advanced Thread Trace analysis: per-instruction stall ratios "
            "and bottleneck classification (VMEM latency, LDS bank conflict, dependency chains, "
            "branch divergence). Requires rocprof-trace-decoder to be installed. "
            "Example: --att-dir ./att_output"
        ),
    )


    llm_options.add_argument(
        "--llm-thinking",
        metavar="TOKENS",
        type=int,
        default=None,
        dest="llm_thinking",
        help=(
            "Enable extended thinking for deeper LLM analysis. Specify the thinking "
            "budget in tokens (e.g. --llm-thinking 8000). Only available with the "
            "Anthropic provider and compatible models (claude-opus-4, "
            "claude-sonnet-4-5, claude-3-7-sonnet). Adds latency but improves "
            "analysis quality for complex traces with multiple interacting "
            "bottlenecks. Requires --llm anthropic. Also configurable via the "
            "PERFXPERT_LLM_THINKING environment variable (set to token count)."
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
            "Set PERFXPERT_LLM_LOCAL_URL to override endpoint."
        ),
    )

    llm_options.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        dest="llm_local_model",
        help=(
            "Model name for local LLM (default: codellama:13b). "
            "Can also be set via PERFXPERT_LLM_LOCAL_MODEL environment variable."
        ),
    )

    def process_args(input: RocpdImportData, args: argparse.Namespace):
        """Process and return valid arguments as dictionary.

        Arg names are chosen to match `_execute_agentic`'s kwarg
        expectations directly (review E2E bug 1): ``output_format`` and
        ``llm_provider`` are wired via argparse ``dest=`` overrides on the
        `--format` / `--llm` flags. ``enable_llm`` is derived from
        ``llm_provider`` being truthy so the agentic path activates the
        live LLM session without a separate boolean flag.
        """
        valid_args = [
            "source_dir",
            "att_dir",
            "prompt",
            "top_kernels",
            "output_format",
            "min_duration",
            "llm_provider",
            "llm_api_key",
            "llm_model",
            "llm_thinking",
            "verbose",
            "llm_local",
            "llm_local_model",
        ]
        # Argparse defaults argparse emits for flags not passed by the
        # user; we skip these so kwargs do not carry noise that the
        # downstream agentic runtime has to special-case. Example: the
        # `--verbose` store_true flag defaults to False, and the
        # `--top-kernels` integer flag defaults to 10; passing them
        # unconditionally would mask "user did not set this" from
        # `_execute_agentic`.
        _cli_defaults = {
            "verbose": False,
            "top_kernels": 10,
            "min_duration": 0.0,
        }
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is None:
                    continue
                # Drop pure-default values so kwargs reflect what the user
                # actually set on the CLI.
                if itr in _cli_defaults and val == _cli_defaults[itr]:
                    continue
                ret[itr] = val
        # Convert min_duration from microseconds to nanoseconds
        if "min_duration" in ret:
            ret["min_duration"] = ret["min_duration"] * 1000
        # Derive enable_llm: non-None llm_provider means the user asked for LLM
        if ret.get("llm_provider"):
            ret["enable_llm"] = True
        return ret

    return process_args


def execute(
    input: Optional[RocpdImportData],
    config: Optional[output_config.output_config] = None,
    **kwargs: Any,
) -> Optional[RocpdImportData]:
    """
    Public CLI entry point — delegates to the agentic implementation.

    Args:
        input: RocpdImportData object with database connection, or None for source-only mode
        config: Optional output configuration
        **kwargs: Analysis parameters (may include source_dir for Tier 0)

    Returns:
        The input RocpdImportData object (for chaining), or None in source-only mode
    """
    return _execute_agentic(input, config=config, **kwargs)


def _normalize_agentic_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)
    if "format" in normalized and "output_format" not in normalized:
        normalized["output_format"] = normalized.pop("format")
    if "llm" in normalized and "llm_provider" not in normalized:
        normalized["llm_provider"] = normalized.pop("llm")
    if "custom_prompt" in normalized and "prompt" not in normalized:
        normalized["prompt"] = normalized["custom_prompt"]
    if "llm_thinking_tokens" in normalized and "llm_thinking" not in normalized:
        normalized["llm_thinking"] = normalized["llm_thinking_tokens"]
    if normalized.get("llm_provider") == "claude-code":
        normalized["llm_provider"] = "opencode"
    normalized["enable_llm"] = bool(normalized.get("llm_provider"))
    return normalized


def _render_agentic_result(result: Any, output_format: str) -> str:
    serializer_name = {
        "json": "to_json",
        "markdown": "to_markdown",
        "webview": "to_webview",
        "text": "to_text",
    }.get(output_format, "to_text")
    serializer = getattr(result, serializer_name, None)
    if callable(serializer):
        return serializer()

    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    else:
        payload = {
            "narrative": getattr(result, "narrative", ""),
            "recommendations": list(getattr(result, "recommendations", []) or []),
            "primary_bottleneck": getattr(result, "primary_bottleneck", "mixed"),
            "warnings": list(getattr(result, "warnings", []) or []),
            "metadata": dict(getattr(result, "metadata", {}) or {}),
        }

    narrative = str(payload.get("narrative", "") or "").strip()
    recommendations = list(payload.get("recommendations", []) or [])
    warnings = list(payload.get("warnings", []) or [])
    metadata = dict(payload.get("metadata", {}) or {})
    primary_bottleneck = str(payload.get("primary_bottleneck", "mixed"))

    if output_format == "json":
        import json

        return json.dumps(
            {
                "narrative": narrative,
                "recommendations": recommendations,
                "primary_bottleneck": primary_bottleneck,
                "warnings": warnings,
                "metadata": metadata,
            },
            indent=2,
        )

    if output_format == "markdown":
        lines = ["# PerfXpert Analysis", ""]
        lines.append(f"- Primary bottleneck: `{primary_bottleneck}`")
        for key in ("intent", "routed_to"):
            if metadata.get(key):
                lines.append(f"- {key.replace('_', ' ').title()}: `{metadata[key]}`")
        if narrative:
            lines.extend(["", "## Narrative", "", narrative])
        if recommendations:
            lines.extend(["", "## Recommendations", ""])
            for rec in recommendations:
                priority = rec.get("priority", "INFO")
                issue = rec.get("issue") or rec.get("suggestion") or rec.get("category") or "Recommendation"
                lines.append(f"- [{priority}] {issue}")
        if warnings:
            lines.extend(["", "## Warnings", ""])
            for warning in warnings:
                lines.append(f"- {warning}")
        return "\n".join(lines)

    if output_format == "webview":
        from html import escape

        rec_items = "".join(
            f"<li><strong>{escape(str(rec.get('priority', 'INFO')))}</strong>: "
            f"{escape(str(rec.get('issue') or rec.get('suggestion') or rec.get('category') or 'Recommendation'))}</li>"
            for rec in recommendations
        )
        warn_items = "".join(f"<li>{escape(str(warning))}</li>" for warning in warnings)
        narrative_html = escape(narrative).replace("\n", "<br />")
        return (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            "<title>PerfXpert Analysis</title></head><body>"
            f"<h1>PerfXpert Analysis</h1><p><strong>Primary bottleneck:</strong> {escape(primary_bottleneck)}</p>"
            f"<p>{narrative_html or 'Analysis complete.'}</p>"
            f"<h2>Recommendations</h2><ul>{rec_items or '<li>None</li>'}</ul>"
            f"<h2>Warnings</h2><ul>{warn_items or '<li>None</li>'}</ul>"
            "</body></html>"
        )

    lines = []
    if narrative:
        lines.append(narrative)
    else:
        lines.append("Analysis complete.")
    lines.append(f"Primary bottleneck: {primary_bottleneck}")
    if recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for rec in recommendations:
            priority = rec.get("priority", "INFO")
            issue = rec.get("issue") or rec.get("suggestion") or rec.get("category") or "Recommendation"
            lines.append(f"- [{priority}] {issue}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


class _AgenticAnalysisReport:
    """Formatter-backed wrapper for agentic Analysis/Recommendation outputs."""

    def __init__(
        self,
        *,
        analysis_output: Any,
        recommendation_output: Any,
        database_path: str,
        custom_prompt: Optional[str] = None,
    ) -> None:
        self._time_breakdown = _normalize_agentic_breakdown(
            getattr(analysis_output, "time_breakdown", {}) or {}
        )
        self._hotspots = _normalize_agentic_hotspots(
            list(getattr(analysis_output, "hot_kernels", []) or [])
        )
        if self._time_breakdown["total_runtime"] <= 0:
            estimated_total = _estimate_total_runtime_ns(self._hotspots)
            if estimated_total > 0:
                kernel_pct = self._time_breakdown["kernel_percent"] / 100.0
                memcpy_pct = self._time_breakdown["memcpy_percent"] / 100.0
                self._time_breakdown["total_runtime"] = estimated_total
                self._time_breakdown["total_kernel_time"] = int(estimated_total * kernel_pct)
                self._time_breakdown["total_memcpy_time"] = int(estimated_total * memcpy_pct)
        self._recommendations = _normalize_agentic_recommendations(
            list(getattr(recommendation_output, "recommendations", []) or []),
            getattr(analysis_output, "primary_bottleneck", "mixed"),
        )
        has_counters = bool(getattr(analysis_output, "counter_data_available", False))
        self._hardware_counters = {
            "has_counters": has_counters,
            "metrics": {},
            "counters": {},
        }
        self._database_path = database_path
        self._custom_prompt = custom_prompt

    def _format(self, output_format: str) -> str:
        return format_analysis_output(
            time_breakdown=self._time_breakdown,
            hotspots=self._hotspots,
            memory_analysis={},
            recommendations=self._recommendations,
            hardware_counters=self._hardware_counters,
            database_path=self._database_path,
            output_format=output_format,
            custom_prompt=self._custom_prompt,
        )

    def to_json(self) -> str:
        return self._format("json")

    def to_markdown(self) -> str:
        return self._format("markdown")

    def to_webview(self) -> str:
        return self._format("webview")

    def to_text(self) -> str:
        return self._format("text")


def _scale_agentic_pct(value: Any) -> float:
    pct = float(value or 0.0)
    return pct * 100.0 if 0.0 <= pct <= 1.0 else pct


def _normalize_agentic_breakdown(time_breakdown: Dict[str, Any]) -> Dict[str, Any]:
    kernel_pct = _scale_agentic_pct(
        time_breakdown.get("kernel_pct", time_breakdown.get("kernel_percent", 0.0))
    )
    memcpy_pct = _scale_agentic_pct(
        time_breakdown.get("memcpy_pct", time_breakdown.get("memcpy_percent", 0.0))
    )
    overhead_pct = _scale_agentic_pct(
        time_breakdown.get("api_pct", time_breakdown.get("overhead_percent", 0.0))
    )
    idle_pct = _scale_agentic_pct(time_breakdown.get("idle_pct", 0.0))
    total_pct = kernel_pct + memcpy_pct + overhead_pct + idle_pct
    if total_pct > 100.0:
        idle_pct = max(0.0, 100.0 - kernel_pct - memcpy_pct - overhead_pct)
    return {
        "kernel_percent": kernel_pct,
        "memcpy_percent": memcpy_pct,
        "overhead_percent": overhead_pct,
        "idle_percent": idle_pct,
        "total_runtime": int(time_breakdown.get("total_runtime", 0) or 0),
        "total_kernel_time": int(time_breakdown.get("total_kernel_time", 0) or 0),
        "total_memcpy_time": int(time_breakdown.get("total_memcpy_time", 0) or 0),
    }


def _normalize_agentic_hotspots(hot_kernels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for kernel in hot_kernels:
        calls = int(kernel.get("calls", 1) or 1)
        total_duration = int(
            kernel.get(
                "total_duration_ns",
                kernel.get("duration_ns", kernel.get("total_duration", 0)),
            )
        )
        avg_duration = int(
            kernel.get("avg_duration_ns", kernel.get("avg_duration", 0))
            or (total_duration / calls if calls else total_duration)
        )
        pct = kernel.get(
            "percent_of_total",
            kernel.get("pct_of_total", kernel.get("pct", 0.0)),
        )
        normalized.append(
            {
                "name": kernel.get("name", "unknown"),
                "calls": calls,
                "total_duration": total_duration,
                "avg_duration": avg_duration,
                "min_duration": int(kernel.get("min_duration_ns", avg_duration)),
                "max_duration": int(kernel.get("max_duration_ns", total_duration)),
                "percent_of_total": _scale_agentic_pct(pct),
            }
        )
    return normalized


def _format_agentic_impact(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, (int, float)):
        if 0.0 <= value <= 1.0:
            return f"{value:.0%} improvement"
        return str(value)
    return str(value)


def _agentic_priority(index: int, recommendation: Dict[str, Any]) -> str:
    explicit = str(recommendation.get("priority", "")).strip().upper()
    if explicit in {"HIGH", "MEDIUM", "LOW", "INFO"}:
        return explicit
    impact = recommendation.get("expected_impact")
    if isinstance(impact, (int, float)):
        if impact >= 0.4:
            return "HIGH"
        if impact >= 0.15:
            return "MEDIUM"
        return "LOW"
    if index == 0:
        return "HIGH"
    if index < 3:
        return "MEDIUM"
    return "LOW"


def _normalize_agentic_recommendations(
    recommendations: List[Dict[str, Any]],
    default_category: str,
) -> List[Dict[str, Any]]:
    raw_recommendations = []

    for index, rec in enumerate(recommendations):
        priority = _agentic_priority(index, rec)
        category = str(rec.get("category") or default_category or "general")
        title = str(rec.get("title") or rec.get("name") or "Optimization opportunity")
        description = str(
            rec.get("description") or rec.get("rationale") or rec.get("issue") or ""
        )
        impact = _format_agentic_impact(rec.get("expected_impact"))

        next_steps = list(rec.get("actions") or [])
        if not next_steps:
            effort = rec.get("effort")
            effort_factor = rec.get("effort_factor")
            risk = rec.get("risk")
            if effort:
                next_steps.append(f"Estimated effort: {effort}")
            elif effort_factor is not None:
                next_steps.append(f"Effort factor: {effort_factor}")
            if risk:
                next_steps.append(f"Risk: {risk}")

        raw_recommendations.append(
            {
                "priority": priority,
                "category": category,
                "issue": title,
                "suggestion": description,
                "actions": next_steps,
                "estimated_impact": impact,
                "confidence": rec.get("confidence"),
                "commands": rec.get("commands", []),
            }
        )

    return raw_recommendations


def _estimate_total_runtime_ns(hotspots: List[Dict[str, Any]]) -> int:
    estimates = []
    for kernel in hotspots:
        total_duration = int(kernel.get("total_duration", 0) or 0)
        pct = float(kernel.get("percent_of_total", 0.0) or 0.0)
        if total_duration > 0 and pct > 0.0:
            estimates.append(total_duration / (pct / 100.0))
    if not estimates:
        return 0
    estimates.sort()
    return int(estimates[len(estimates) // 2])


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _provider_env_overrides(
    llm_provider: Optional[str],
    normalized_kwargs: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    overrides: Dict[str, Optional[str]] = {}
    llm_model = normalized_kwargs.get("llm_model")
    llm_api_key = normalized_kwargs.get("llm_api_key")

    if llm_model:
        overrides["PERFXPERT_LLM_MODEL"] = str(llm_model)

    if llm_api_key and llm_provider == "anthropic":
        overrides["PERFXPERT_LLM_ANTHROPIC_KEY"] = str(llm_api_key)
    elif llm_api_key and llm_provider == "openai":
        overrides["PERFXPERT_LLM_OPENAI_KEY"] = str(llm_api_key)

    return overrides


# -- known kwargs accepted by `_execute_agentic` ---------------------------
# Any kwarg not in this set that is forwarded from `execute()` will emit a
# WARNING so future argparse additions cannot silently drop through the
# agentic pipeline.
_KNOWN_EXECUTE_KWARGS = frozenset({
    # Output routing
    "format",
    "output_format",
    "output_file",
    "output_path",
    # LLM provider wiring
    "llm",
    "enable_llm",
    "llm_provider",
    "llm_api_key",
    "llm_model",
    "llm_thinking",
    "llm_thinking_tokens",
    "llm_local",
    "llm_local_model",
    # Analysis options forwarded through AnalysisInput.analysis_options
    "source_dir",
    "att_dir",
    "prompt",
    "custom_prompt",
    "top_kernels",
    "min_duration",
    # Execution flags
    "verbose",
})


def _execute_agentic(
    input: Optional[RocpdImportData],
    config: Optional[output_config.output_config] = None,
    **kwargs: Any,
) -> Optional[RocpdImportData]:
    """Agentic path: delegates to the agents session API.

    Database-backed analysis runs Analysis -> Recommendation and then
    renders through the canonical formatter stack. Source-only mode
    falls back to RootOutput rendering until Tier 0 regains a dedicated
    agent-backed implementation.
    """
    try:
        from perfxpert.agents import runtime, schemas
    except ImportError as e:
        raise RuntimeError(
            "Agent runtime is not available. "
            "perfxpert.agents must be importable for the agentic path."
        ) from e
    normalized_kwargs = _normalize_agentic_kwargs(kwargs)

    _unused = set(kwargs) - _KNOWN_EXECUTE_KWARGS
    if _unused:
        import warnings

        warnings.warn(
            "perfxpert.analyze: unused kwargs ignored by agentic runtime: "
            f"{sorted(_unused)}. Wire them in _execute_agentic or drop the "
            "corresponding --flag.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Update config if provided
    if config is not None:
        config = config.update(**normalized_kwargs)
    else:
        config = output_config.output_config(**normalized_kwargs)

    # Get database path for display
    database_path = ""
    if input is not None and hasattr(input, "_paths") and input._paths:
        database_path = str(
            input._paths[0] if isinstance(input._paths, list) else input._paths
        )

    # Get source_dir if provided (for Tier 0 analysis)
    source_dir = normalized_kwargs.get("source_dir")

    # Get custom prompt if provided. CLI emits `prompt` (argparse dest);
    # accept `custom_prompt` as a back-compat alias for library callers.
    custom_prompt = normalized_kwargs.get("prompt") or normalized_kwargs.get("custom_prompt")

    # Build session
    enable_llm = normalized_kwargs.get("enable_llm", False)
    llm_provider = normalized_kwargs.get("llm_provider")
    session = runtime.build_session(
        provider=llm_provider if enable_llm else None,
        airgap=(not enable_llm),
    )

    # Collect downstream analysis options as a side-channel dict on
    # AnalysisInput so the LLM-facing analysis agent can still observe
    # non-schema flags without blocking the deterministic path.
    analysis_options: Dict[str, Any] = {}
    for key in (
        "top_kernels",
        "att_dir",
        "min_duration",
        "llm_model",
        "llm_thinking",
        "llm_local",
        "llm_local_model",
        "verbose",
    ):
        val = normalized_kwargs.get(key)
        if val is not None:
            analysis_options[key] = val

    output_format = normalized_kwargs.get("output_format", "text")
    env_overrides = _provider_env_overrides(llm_provider if enable_llm else None, normalized_kwargs)
    with _temporary_env(env_overrides):
        if input is None:
            root_input = schemas.RootInput(
                user_query=custom_prompt or "Analyze this GPU performance trace.",
                database_path=None,
                source_dir=source_dir,
                provider=llm_provider if enable_llm else None,
                airgap=(not enable_llm),
                session_id=session.session_id,
                analysis_options=analysis_options,
            )
            output = _render_agentic_result(
                session.run_root(root_input),
                output_format,
            )
        else:
            analysis_input = schemas.AnalysisInput(
                database_path=database_path,
                top_kernels=normalized_kwargs.get("top_kernels", 10),
                att_dir=normalized_kwargs.get("att_dir"),
                min_duration=float(normalized_kwargs.get("min_duration", 0.0) or 0.0),
                analysis_options=analysis_options,
            )
            analysis_output = session.run_analysis(analysis_input)
            recommendation_output = session.run_recommendation(
                schemas.RecommendationInput(findings=analysis_output)
            )
            report = _AgenticAnalysisReport(
                analysis_output=analysis_output,
                recommendation_output=recommendation_output,
                database_path=database_path,
                custom_prompt=custom_prompt,
            )
            output = _render_agentic_result(report, output_format)

    # Handle output writing
    _ext_map = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}
    _ext = _ext_map.get(output_format, ".txt")

    if config and config.output_path and not config.output_file:
        if database_path:
            config.output_file = os.path.splitext(os.path.basename(database_path))[0]
        else:
            config.output_file = "analysis"

    if config and config.output_file and config.output_path:
        base = config.output_file
        if not base.endswith(_ext):
            base = base + _ext
        output_file = os.path.join(config.output_path, base)
        os.makedirs(config.output_path, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Analysis written to: {output_file}")
        if output_format == "text":
            print(
                "Tip: use --format webview for an interactive HTML report, "
                "--format json for machine-readable output, "
                "or --format markdown for Markdown."
            )
    else:
        print(output)

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
        prog="perfxpert analyze",
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
