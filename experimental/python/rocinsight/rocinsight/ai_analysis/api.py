#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Public Python API for rocpd AI analysis.

This module provides a simple function-based API for programmatic access
to AI-powered GPU performance analysis. Designed for integration with
tools like Optiq.

Example:
    from rocinsight.ai_analysis import analyze_database
    from pathlib import Path

    result = analyze_database(Path("output.db"))
    print(result.summary.overall_assessment)

    for rec in result.recommendations.high_priority:
        print(f"- {rec.title}")
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from importlib.metadata import version as _pkg_version

    _ROCINSIGHT_VERSION = _pkg_version("rocinsight")
except Exception:
    _ROCINSIGHT_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)

from ..analysis import (
    compute_time_breakdown,
    identify_hotspots,
    analyze_memory_copies,
    analyze_hardware_counters,
    generate_recommendations,
    _detect_already_collected,
)
from ..formatters import format_analysis_output
from ..tracelens_port import (
    compute_interval_timeline,
    analyze_kernels_by_category,
    analyze_short_kernels,
)
from .llm_analyzer import AnalysisContext, LLMAnalyzer
from .exceptions import (
    DatabaseNotFoundError,
    DatabaseCorruptedError,
    LLMAuthenticationError,
    LLMRateLimitError,
    SourceDirectoryNotFoundError,
)


class OutputFormat(Enum):
    """Output format options"""

    PYTHON_OBJECT = "python_object"  # Returns dataclass
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    WEBVIEW = "webview"  # Self-contained interactive HTML


@dataclass
class AnalysisMetadata:
    """Metadata about the analysis"""

    rocpd_version: str
    analysis_version: str = "0.1.0"
    database_file: str = ""
    analysis_timestamp: str = ""
    analysis_duration_ms: int = 0
    custom_prompt: Optional[str] = None


@dataclass
class GPUInfo:
    """GPU device information"""

    name: str
    architecture: str
    agent_id: int = 0


@dataclass
class ProfilingInfo:
    """Profiling session information"""

    total_duration_ns: int
    profiling_mode: str  # "sys_trace_only", "sys_trace_with_counters", "pc_sampling"
    analysis_tier: int  # 1=trace, 2=counters, 3=pc_sampling
    gpus: List[GPUInfo] = field(default_factory=list)


@dataclass
class AnalysisSummary:
    """High-level summary of analysis"""

    overall_assessment: str
    primary_bottleneck: str  # "compute", "memory", "latency", "mixed", "unknown"
    confidence: float  # 0.0 to 1.0
    key_findings: List[str] = field(default_factory=list)


@dataclass
class ExecutionBreakdown:
    """Time distribution breakdown"""

    kernel_time_ns: int
    kernel_time_pct: float
    memcpy_time_ns: int
    memcpy_time_pct: float
    api_overhead_ns: int = 0
    api_overhead_pct: float = 0.0
    idle_time_ns: int = 0
    idle_time_pct: float = 0.0


@dataclass
class Recommendation:
    """Single recommendation"""

    id: str
    priority: str  # "high", "medium", "low"
    category: str  # "memory", "compute", "occupancy", "memory_transfer", etc.
    title: str
    description: str
    estimated_impact: str
    next_steps: List[str] = field(default_factory=list)


@dataclass
class RecommendationSet:
    """Prioritized recommendations"""

    high_priority: List[Recommendation] = field(default_factory=list)
    medium_priority: List[Recommendation] = field(default_factory=list)
    low_priority: List[Recommendation] = field(default_factory=list)


@dataclass
class AnalysisWarning:
    """Warning message"""

    severity: str  # "warning", "info"
    message: str
    recommendation: Optional[str] = None


@dataclass
class SourceAnalysisResult:
    """
    Tier 0 analysis result from static source code scanning.

    Produced by analyze_source() and attached to AnalysisResult.tier0
    when --source-dir is provided alongside -i.
    """

    source_dir: str
    analysis_timestamp: str
    programming_model: str  # "HIP", "HIP+ROCm_Libraries", "OpenCL", "PyTorch_HIP", etc.

    files_scanned: int
    files_skipped: int

    detected_kernels: List[Dict[str, Any]]  # {name, file, line, launch_type}
    kernel_count: int

    detected_patterns: List[
        Dict[str, Any]
    ]  # {pattern_id, severity, category, description, count, locations}
    risk_areas: List[str]

    already_instrumented: bool
    roctx_marker_count: int

    recommendations: List[Dict[str, Any]]  # same structure as generate_recommendations()
    suggested_counters: List[str]
    suggested_first_command: str

    llm_explanation: Optional[str] = None


def _plan_to_source_result(plan) -> "SourceAnalysisResult":
    """Convert a ProfilingPlan to a SourceAnalysisResult dataclass.

    Centralizes the conversion logic so both api.py:analyze_source() and
    analyze.py:analyze_source_code() produce identical SourceAnalysisResult
    objects without duplicating the field-mapping code.
    """
    return SourceAnalysisResult(
        source_dir=plan.source_dir,
        analysis_timestamp=plan.analysis_timestamp,
        programming_model=plan.programming_model,
        files_scanned=plan.files_scanned,
        files_skipped=plan.files_skipped,
        detected_kernels=[
            {
                "name": k.name,
                "file": k.file,
                "line": k.line,
                "launch_type": k.launch_type,
            }
            for k in plan.detected_kernels
        ],
        kernel_count=plan.kernel_count,
        detected_patterns=[
            {
                "pattern_id": p.pattern_id,
                "severity": p.severity,
                "category": p.category,
                "description": p.description,
                "count": p.count,
                "locations": p.locations,
            }
            for p in plan.detected_patterns
        ],
        risk_areas=plan.risk_areas,
        already_instrumented=plan.already_instrumented,
        roctx_marker_count=plan.roctx_marker_count,
        recommendations=plan.recommendations,
        suggested_counters=plan.suggested_counters,
        suggested_first_command=plan.suggested_first_command,
    )


@dataclass
class AnalysisResult:
    """
    Complete analysis result structure.

    This is the main return type for analyze_database().
    Contains all analysis data and can be serialized to JSON/text/markdown.
    """

    metadata: AnalysisMetadata
    profiling_info: ProfilingInfo
    summary: AnalysisSummary
    execution_breakdown: ExecutionBreakdown
    recommendations: RecommendationSet
    warnings: List[AnalysisWarning] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Optional LLM-enhanced natural language explanation
    llm_enhanced_explanation: Optional[str] = None

    # Tier 0 source code analysis (populated when analyze_source() is also run)
    tier0: Optional[SourceAnalysisResult] = None

    # TraceLens-derived analysis (Phase 1)
    kernel_categories: List[dict] = field(default_factory=list)
    short_kernels: dict = field(default_factory=dict)
    interval_timeline: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to schema-conformant JSON (analysis-output.schema.json v0.1.0).

        Delegates to format_analysis_output() to ensure the output matches the
        normative JSON schema. Raises RuntimeError if raw analysis data is not
        available — use analyze_database() to obtain a fully-populated result,
        or use to_dict() for a non-schema-conformant dict.
        """
        raw = getattr(self, "_raw", None)
        if raw is not None:
            return format_analysis_output(
                time_breakdown=raw["time_breakdown"],
                hotspots=raw["hotspots"],
                memory_analysis=raw["memory_analysis"],
                recommendations=raw["recommendations_raw"],
                hardware_counters=raw["hardware_counters"],
                database_path=raw["database_path"],
                output_format="json",
                interval_timeline=raw.get("interval_timeline"),  # NEW
                kernel_categories=raw.get("kernel_categories"),  # NEW
                short_kernels=raw.get("short_kernels"),  # NEW
            )
        raise RuntimeError(
            "Raw analysis data not available. "
            "Use analyze_database() to create the result, "
            "or use to_dict() for a non-schema-conformant dict."
        )

    def to_webview(self) -> str:
        """Generate self-contained interactive HTML report.

        Returns the same AMD-themed webview HTML produced by the rocpd CLI
        ``--format webview`` option. Requires that the result was created via
        :func:`analyze_database` (which populates the raw data cache).

        Raises:
            RuntimeError: If the result was not created via analyze_database().
        """
        raw = getattr(self, "_raw", None)
        if raw is None:
            raise RuntimeError(
                "Raw analysis data not available. "
                "Use analyze_database() to create the result."
            )
        return format_analysis_output(
            time_breakdown=raw["time_breakdown"],
            hotspots=raw["hotspots"],
            memory_analysis=raw["memory_analysis"],
            recommendations=raw["recommendations_raw"],
            hardware_counters=raw["hardware_counters"],
            database_path=raw["database_path"],
            output_format="webview",
            interval_timeline=raw.get("interval_timeline"),  # NEW
            kernel_categories=raw.get("kernel_categories"),  # NEW
            short_kernels=raw.get("short_kernels"),  # NEW
        )

    def to_text(self) -> str:
        """Generate plain text report.

        Works without ``_raw`` attached; renders from dataclass fields directly.
        Does NOT guarantee schema conformance (use ``to_json()`` for that).
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("GPU PERFORMANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Database: {self.metadata.database_file}")
        lines.append(f"Analysis Date: {self.metadata.analysis_timestamp}")
        lines.append(f"Analysis Tier: {self.profiling_info.analysis_tier}")
        if self.metadata.custom_prompt:
            lines.append(f"Custom Prompt: {self.metadata.custom_prompt}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(self.summary.overall_assessment)
        lines.append(f"Primary Bottleneck: {self.summary.primary_bottleneck}")
        lines.append(f"Confidence: {self.summary.confidence:.0%}")
        lines.append("")

        # Key findings
        if self.summary.key_findings:
            lines.append("Key Findings:")
            for finding in self.summary.key_findings:
                lines.append(f"  • {finding}")
            lines.append("")

        # Execution breakdown
        lines.append("EXECUTION BREAKDOWN")
        lines.append("-" * 80)
        lines.append(
            f"Kernel Execution:  {self.execution_breakdown.kernel_time_pct:6.1f}%"
        )
        lines.append(
            f"Memory Copies:     {self.execution_breakdown.memcpy_time_pct:6.1f}%"
        )
        lines.append(
            f"API Overhead:      {self.execution_breakdown.api_overhead_pct:6.1f}%"
        )
        lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)

        for priority, recs in [
            ("HIGH PRIORITY", self.recommendations.high_priority),
            ("MEDIUM PRIORITY", self.recommendations.medium_priority),
            ("LOW PRIORITY", self.recommendations.low_priority),
        ]:
            if recs:
                lines.append(f"\n{priority}:")
                for rec in recs:
                    lines.append(f"\n  {rec.title}")
                    lines.append(f"  {rec.description}")
                    lines.append(f"  Estimated Impact: {rec.estimated_impact}")
                    if rec.next_steps:
                        lines.append("  Next Steps:")
                        for step in rec.next_steps:
                            lines.append(f"    - {step}")

        # LLM-enhanced explanation (if available)
        if self.llm_enhanced_explanation:
            lines.append("\n")
            lines.append("=" * 80)
            lines.append("AI-ENHANCED EXPLANATION")
            lines.append("=" * 80)
            lines.append(self.llm_enhanced_explanation)

        # Warnings
        if self.warnings:
            lines.append("\n")
            lines.append("WARNINGS")
            lines.append("-" * 80)
            for warning in self.warnings:
                lines.append(f"⚠️  {warning.message}")
                if warning.recommendation:
                    lines.append(f"   Recommendation: {warning.recommendation}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown report.

        Works without ``_raw`` attached; renders from dataclass fields directly.
        Does NOT guarantee schema conformance (use ``to_json()`` for that).
        """
        lines = []

        # Header
        lines.append("# GPU Performance Analysis Report")
        lines.append("")
        lines.append(f"**Database:** `{self.metadata.database_file}`")
        lines.append(f"**Analysis Date:** {self.metadata.analysis_timestamp}")
        lines.append(f"**Analysis Tier:** {self.profiling_info.analysis_tier}")
        if self.metadata.custom_prompt:
            lines.append(f"**Custom Prompt:** _{self.metadata.custom_prompt}_")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(self.summary.overall_assessment)
        lines.append("")
        lines.append(f"- **Primary Bottleneck:** {self.summary.primary_bottleneck}")
        lines.append(f"- **Confidence:** {self.summary.confidence:.0%}")
        lines.append("")

        # Key findings
        if self.summary.key_findings:
            lines.append("### Key Findings")
            lines.append("")
            for finding in self.summary.key_findings:
                lines.append(f"- {finding}")
            lines.append("")

        # Execution breakdown
        lines.append("## Execution Breakdown")
        lines.append("")
        lines.append("| Category | Percentage |")
        lines.append("|----------|------------|")
        lines.append(
            f"| Kernel Execution | {self.execution_breakdown.kernel_time_pct:.1f}% |"
        )
        lines.append(
            f"| Memory Copies | {self.execution_breakdown.memcpy_time_pct:.1f}% |"
        )
        lines.append(
            f"| API Overhead | {self.execution_breakdown.api_overhead_pct:.1f}% |"
        )
        lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        for priority, recs, emoji in [
            ("High Priority", self.recommendations.high_priority, "🔴"),
            ("Medium Priority", self.recommendations.medium_priority, "🟡"),
            ("Low Priority", self.recommendations.low_priority, "🟢"),
        ]:
            if recs:
                lines.append(f"### {emoji} {priority}")
                lines.append("")
                for rec in recs:
                    lines.append(f"#### {rec.title}")
                    lines.append("")
                    lines.append(rec.description)
                    lines.append("")
                    lines.append(f"**Estimated Impact:** {rec.estimated_impact}")
                    lines.append("")
                    if rec.next_steps:
                        lines.append("**Next Steps:**")
                        for step in rec.next_steps:
                            lines.append(f"- {step}")
                        lines.append("")

        # LLM-enhanced explanation
        if self.llm_enhanced_explanation:
            lines.append("---")
            lines.append("")
            lines.append("## AI-Enhanced Explanation")
            lines.append("")
            lines.append(self.llm_enhanced_explanation)
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"⚠️ **{warning.severity.upper()}:** {warning.message}")
                if warning.recommendation:
                    lines.append(f"  - Recommendation: {warning.recommendation}")
                lines.append("")

        return "\n".join(lines)


def analyze_database(
    database_path: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_thinking_tokens: Optional[int] = None,
    output_format: OutputFormat = OutputFormat.PYTHON_OBJECT,
    verbose: bool = False,
    top_kernels: int = 10,
    att_dir: Optional[str] = None,
) -> AnalysisResult:
    """
    Analyze a rocpd database file and return AI-powered insights.

    This is the main entry point for programmatic analysis.
    Performs local analysis (always) and optional LLM enhancement.

    Args:
        database_path: Path to .rpd or .db file
        custom_prompt: Optional user question to guide analysis
        enable_llm: Enable LLM-powered natural language enhancement
        llm_provider: LLM provider ("anthropic", "openai")
        llm_api_key: API key for LLM provider (or set env var)
        llm_thinking_tokens: Enable extended thinking with this token budget.
            Only supported with the Anthropic provider and compatible models
            (claude-opus-4, claude-sonnet-4-5, claude-3-7-sonnet).
        output_format: Desired output format
        verbose: Enable verbose logging
        top_kernels: Number of top kernels to analyze

    Returns:
        AnalysisResult object with complete analysis

    Raises:
        DatabaseNotFoundError: Database file doesn't exist
        DatabaseCorruptedError: Database schema is invalid
        MissingDataError: Required tables are missing

    Example:
        >>> from rocinsight.ai_analysis import analyze_database
        >>> from pathlib import Path
        >>>
        >>> result = analyze_database(Path("output.db"))
        >>> print(result.summary.overall_assessment)
        >>> for rec in result.recommendations.high_priority:
        ...     print(f"- {rec.title}")
    """
    # Validate database exists
    if not database_path.exists():
        raise DatabaseNotFoundError(f"Database file not found: {database_path}")

    if verbose:
        print(f"[Analysis] Analyzing database: {database_path}")
        print(f"[Analysis] Enable LLM: {enable_llm}")
        if custom_prompt:
            print(f"[Analysis] Custom prompt: {custom_prompt}")

    att_analysis: dict = {}  # populated inside try block if att_dir is provided

    # Perform local analysis by calling individual analysis functions directly.
    # NOTE: We do NOT call analyze_performance() — it returns a formatted str,
    # not a dict. We need raw data to build the AnalysisResult dataclass.
    try:
        from ..connection import RocinsightConnection as RocpdImportData

        connection = RocpdImportData(str(database_path))

        time_breakdown = compute_time_breakdown(connection)
        hotspots = identify_hotspots(connection, top_n=top_kernels)
        memory_analysis = analyze_memory_copies(connection)
        hardware_counters = analyze_hardware_counters(connection)
        already_collected = _detect_already_collected(connection)

        # TraceLens-derived analysis
        interval_timeline = compute_interval_timeline(connection)
        kernel_categories = analyze_kernels_by_category(
            connection, interval_timeline["total_wall_ns"]
        )
        short_kernels_data = analyze_short_kernels(connection)

        # Tier 3: ATT (optional)
        att_analysis: dict = {}
        if att_dir:
            from ..analysis import analyze_thread_trace

            att_analysis = analyze_thread_trace(att_dir)
            if verbose and not att_analysis.get("has_att_data"):
                print(f"[ATT] {att_analysis.get('reason', 'No ATT data')}")

        recommendations = generate_recommendations(
            time_breakdown,
            hotspots,
            memory_analysis,
            hardware_counters,
            already_collected,
            short_kernels=short_kernels_data,
            interval_timeline=interval_timeline,
            att_analysis=att_analysis if att_dir else None,
        )

        if verbose:
            print("[Analysis] Local analysis complete")

    except Exception as e:
        raise DatabaseCorruptedError(f"Failed to analyze database: {e}")

    # Build AnalysisResult from raw analysis payloads
    result = _build_analysis_result(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=recommendations,
        hardware_counters=hardware_counters,
        database_path=database_path,
        custom_prompt=custom_prompt,
    )

    result.kernel_categories = kernel_categories
    result.short_kernels = short_kernels_data
    result.interval_timeline = interval_timeline

    # Also write into _raw so to_json() / to_webview() include them
    result._raw["interval_timeline"] = interval_timeline
    result._raw["kernel_categories"] = kernel_categories
    result._raw["short_kernels"] = short_kernels_data
    if att_dir and att_analysis.get("has_att_data"):
        result._raw["att_trace"] = att_analysis
        result._raw["att_analysis"] = att_analysis
        result.profiling_info.analysis_tier = 3
        result.profiling_info.profiling_mode = "thread_trace"

    # Optional LLM enhancement
    if enable_llm and llm_provider:
        try:
            if verbose:
                print(f"[Analysis] Enhancing with {llm_provider} LLM...")

            analyzer = LLMAnalyzer(
                provider=llm_provider,
                api_key=llm_api_key,
                verbose=verbose,
                thinking_budget_tokens=llm_thinking_tokens,
            )

            # Convert result to dict for LLM
            analysis_data = _convert_result_to_llm_format(result)

            # Build AnalysisContext so _select_tags() gates reference guide sections
            # (including tracelens_metrics when TraceLens data is present)
            has_counters = hardware_counters.get("has_counters", False)
            analysis_tier = 2 if has_counters else 1
            context = AnalysisContext(
                tier=analysis_tier,
                has_counters=has_counters,
                custom_prompt=custom_prompt,
                kernel_categories=result.kernel_categories or [],
                interval_timeline={
                    k: v
                    for k, v in result.interval_timeline.items()
                    if k.endswith("_pct")
                },
                short_kernel_summary=(
                    {
                        "threshold_us": result.short_kernels.get("threshold_us", 10),
                        "short_kernel_count": result.short_kernels.get(
                            "short_kernel_count", 0
                        ),
                        "wasted_pct_of_kernel_time": result.short_kernels.get(
                            "wasted_pct_of_kernel_time", 0
                        ),
                    }
                    if result.short_kernels
                    else None
                ),
            )

            # Get LLM enhancement
            llm_explanation = analyzer.analyze_with_llm(
                analysis_data,
                custom_prompt=custom_prompt,
                context=context,
            )

            result.llm_enhanced_explanation = llm_explanation

            if verbose:
                print("[Analysis] LLM enhancement complete")

        except (LLMAuthenticationError, LLMRateLimitError):
            # Auth and rate-limit errors must propagate — the caller needs to
            # know their credentials are invalid or exhausted.
            raise
        except Exception as e:
            # Other LLM errors are non-critical: add a warning and continue
            # with local-only results.
            result.warnings.append(
                AnalysisWarning(
                    severity="warning",
                    message=f"LLM enhancement failed: {e}",
                    recommendation="Analysis continues with local-only results",
                )
            )

            if verbose:
                print(f"[Analysis] LLM enhancement failed: {e}")

    return result


def _build_analysis_result(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Dict[str, Any],
    database_path: Path,
    custom_prompt: Optional[str],
) -> AnalysisResult:
    """Build AnalysisResult from raw analysis payloads returned by analyze.py functions.

    Key mapping from generate_recommendations() output:
      rec["issue"]            → Recommendation.title
      rec["suggestion"]       → Recommendation.description
      rec["estimated_impact"] → Recommendation.estimated_impact
      rec["actions"]          → Recommendation.next_steps
      rec["priority"]         → "HIGH"/"MEDIUM"/"INFO" (uppercase) → normalized to lowercase
    """
    from datetime import datetime

    # Build metadata
    metadata = AnalysisMetadata(
        rocpd_version=_ROCINSIGHT_VERSION,
        analysis_version="0.1.0",  # schema version, not module version
        database_file=str(database_path),
        analysis_timestamp=datetime.now().isoformat(),
        custom_prompt=custom_prompt,
    )

    # Build profiling info
    has_counters = hardware_counters.get("has_counters", False)
    profiling_mode = "sys_trace_with_counters" if has_counters else "sys_trace_only"
    analysis_tier = 2 if has_counters else 1

    profiling_info = ProfilingInfo(
        total_duration_ns=int(time_breakdown.get("total_runtime", 0)),
        profiling_mode=profiling_mode,
        analysis_tier=analysis_tier,
        gpus=[],
    )

    # Build summary — mirrors _build_summary() logic in analyze.py
    primary_bottleneck = "mixed"
    confidence = 0.50

    memcpy_pct = time_breakdown.get("memcpy_percent", 0)
    kernel_pct = time_breakdown.get("kernel_percent", 0)
    overhead_pct = time_breakdown.get("overhead_percent", 0)
    if memcpy_pct > 30:
        primary_bottleneck = "memory_transfer"
        confidence = 0.85
    elif memcpy_pct > 20:
        primary_bottleneck = "memory_transfer"
        confidence = 0.70
    elif overhead_pct > 25:
        primary_bottleneck = "latency"
        confidence = 0.75
    elif kernel_pct > 70 and has_counters:
        primary_bottleneck = "compute"
        confidence = 0.80
    elif kernel_pct > 70:
        primary_bottleneck = "compute"
        confidence = 0.60

    summary = AnalysisSummary(
        overall_assessment=f"Analysis complete. {len(hotspots)} kernels analyzed.",
        primary_bottleneck=primary_bottleneck,
        confidence=confidence,
        key_findings=[
            f"Total kernel execution time: {kernel_pct:.1f}%",
            f"Memory copy overhead: {memcpy_pct:.1f}%",
            f"Top kernel: {hotspots[0]['name'] if hotspots else 'N/A'}",
        ],
    )

    # Build execution breakdown
    execution_breakdown = ExecutionBreakdown(
        kernel_time_ns=int(time_breakdown.get("total_kernel_time", 0)),
        kernel_time_pct=kernel_pct,
        memcpy_time_ns=int(time_breakdown.get("total_memcpy_time", 0)),
        memcpy_time_pct=memcpy_pct,
        api_overhead_pct=time_breakdown.get("overhead_percent", 0.0),
    )

    # Build recommendations — map keys from generate_recommendations() output.
    # generate_recommendations() uses: issue, suggestion, estimated_impact, actions,
    # priority (uppercase: "HIGH"/"MEDIUM"/"INFO"), category, commands.
    rec_set = RecommendationSet()
    for i, rec in enumerate(recommendations, 1):
        priority_upper = rec.get("priority", "MEDIUM").upper()
        recommendation = Recommendation(
            id=f"rec_{i:03d}",
            priority=priority_upper.lower(),
            category=rec.get("category", "general"),
            title=rec.get("issue", "Optimization opportunity"),
            description=rec.get("suggestion", ""),
            estimated_impact=rec.get("estimated_impact", "Unknown"),
            next_steps=rec.get("actions", []),
        )

        if priority_upper == "HIGH":
            rec_set.high_priority.append(recommendation)
        elif priority_upper == "MEDIUM":
            rec_set.medium_priority.append(recommendation)
        else:
            rec_set.low_priority.append(recommendation)

    # Build warnings
    warnings = []
    if not has_counters:
        warnings.append(
            AnalysisWarning(
                severity="warning",
                message="No hardware counters collected. Analysis limited to Tier 1 (trace data only).",
                recommendation="Collect counters with: rocprofv3 --pmc GRBM_COUNT SQ_WAVES -- ./app",
            )
        )

    result = AnalysisResult(
        metadata=metadata,
        profiling_info=profiling_info,
        summary=summary,
        execution_breakdown=execution_breakdown,
        recommendations=rec_set,
        warnings=warnings,
    )

    # Attach raw payloads as a dynamic attribute so to_json()/to_webview() can
    # delegate serialization to format_analysis_output() for schema conformance.
    result._raw = {
        "time_breakdown": time_breakdown,
        "hotspots": hotspots,
        "memory_analysis": memory_analysis,
        "recommendations_raw": recommendations,
        "hardware_counters": hardware_counters,
        "database_path": str(database_path),
    }

    return result


def _convert_result_to_llm_format(result: AnalysisResult) -> Dict[str, Any]:
    """Convert AnalysisResult to the format expected by LLMAnalyzer._sanitize_data().

    Populates all sections from the raw analysis payloads stored on the result
    so the LLM receives real profiling data rather than empty placeholders.
    """
    raw = getattr(result, "_raw", {})
    hotspots = raw.get("hotspots", [])
    memory_analysis = raw.get("memory_analysis", {})
    hardware_counters = raw.get("hardware_counters", {})

    return {
        # GPU info — arch not currently stored in the DB views; keep as generic
        "gpu": {"name": "AMD GPU", "arch": "unknown"},
        "execution_breakdown": {
            "kernel_time_pct": result.execution_breakdown.kernel_time_pct,
            "memcpy_time_pct": result.execution_breakdown.memcpy_time_pct,
            "api_overhead_pct": result.execution_breakdown.api_overhead_pct,
        },
        # Real kernel hotspot data
        "kernels": [
            {
                "name": k.get("name"),
                "calls": k.get("calls"),
                "total_duration_ns": k.get("total_duration"),
                "avg_duration_ns": k.get("avg_duration"),
                "percent_of_total": k.get("percent_of_total"),
            }
            for k in hotspots
        ],
        # Real memory transfer data keyed by direction
        "memory_ops": {
            direction: {
                "count": info.get("count"),
                "total_bytes": info.get("total_bytes"),
                "avg_duration_ns": info.get("avg_duration"),
            }
            for direction, info in memory_analysis.items()
        },
        "has_counters": hardware_counters.get("has_counters", False),
        # Derived hardware metrics (gpu_utilization_percent, avg_waves, etc.)
        "hardware_metrics": hardware_counters.get("metrics", {}),
        "has_pc_sampling": result.profiling_info.analysis_tier >= 3,
        "interval_timeline": {
            k: v
            for k, v in result.interval_timeline.items()
            if k.endswith("_pct")  # pct fields only — omit _ns fields to reduce tokens
        },
        "kernel_categories": [
            {k: v for k, v in c.items() if k != "total_ns" and k != "avg_duration_ns"}
            for c in result.kernel_categories
        ],
        "short_kernel_summary": {
            "threshold_us": result.short_kernels.get("threshold_us", 10),
            "short_kernel_count": result.short_kernels.get("short_kernel_count", 0),
            "wasted_pct_of_kernel_time": result.short_kernels.get(
                "wasted_pct_of_kernel_time", 0
            ),
        },
    }


def analyze_database_to_json(
    database_path: Path,
    output_json_path: Optional[Path] = None,
    **kwargs,
) -> str:
    """
    Analyze database and return/save JSON output.

    Args:
        database_path: Path to .rpd or .db file
        output_json_path: Optional path to save JSON file
        **kwargs: Additional arguments passed to analyze_database()

    Returns:
        JSON string

    Example:
        >>> json_output = analyze_database_to_json(
        ...     Path("output.db"),
        ...     output_json_path=Path("analysis.json")
        ... )
    """
    result = analyze_database(database_path, **kwargs)
    json_output = result.to_json()

    if output_json_path:
        output_json_path.write_text(json_output)

    return json_output


def get_kernel_analysis(database_path: Path, kernel_name: str, **kwargs) -> Dict:
    """
    Get analysis for a specific kernel.

    Args:
        database_path: Path to .rpd or .db file
        kernel_name: Exact kernel name or pattern
        **kwargs: Additional arguments

    Returns:
        Kernel analysis data
    """
    # TODO: Implement kernel-specific analysis
    raise NotImplementedError("Kernel-specific analysis not yet implemented")


def get_recommendations(
    database_path: Path,
    priority_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    **kwargs,
) -> List[Recommendation]:
    """
    Get filtered recommendations from analysis.

    Args:
        database_path: Path to .rpd or .db file
        priority_filter: Filter by priority ("high", "medium", "low")
        category_filter: Filter by category
        **kwargs: Additional arguments

    Returns:
        List of Recommendation objects
    """
    result = analyze_database(database_path, **kwargs)

    recommendations = []
    if priority_filter == "high" or priority_filter is None:
        recommendations.extend(result.recommendations.high_priority)
    if priority_filter == "medium" or priority_filter is None:
        recommendations.extend(result.recommendations.medium_priority)
    if priority_filter == "low" or priority_filter is None:
        recommendations.extend(result.recommendations.low_priority)

    if category_filter:
        recommendations = [
            rec for rec in recommendations if rec.category == category_filter
        ]

    return recommendations


def analyze_source(
    source_dir: Path,
    *,
    custom_prompt: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    verbose: bool = False,
) -> SourceAnalysisResult:
    """
    Analyze a source code directory and return a Tier 0 profiling plan.

    No database file is required. Scans .hip, .cpp, .cu, .cl, .py, .h,
    .hpp files for GPU programming patterns and generates structured
    recommendations for what to profile and with which commands.

    Args:
        source_dir: Path to source code directory
        custom_prompt: Optional user question to guide LLM analysis
        enable_llm: Enable LLM-powered explanation of the profiling plan
        llm_provider: LLM provider ("anthropic", "openai")
        llm_api_key: API key for LLM provider (or set env var)
        verbose: Enable verbose logging

    Returns:
        SourceAnalysisResult with profiling plan

    Raises:
        SourceDirectoryNotFoundError: Source directory doesn't exist
        SourceAnalysisError: Error during source scanning

    Example:
        >>> from rocinsight.ai_analysis import analyze_source
        >>> from pathlib import Path
        >>>
        >>> result = analyze_source(Path("./my_app/src"))
        >>> print(result.programming_model)
        >>> print(result.suggested_first_command)
        >>> for rec in result.recommendations:
        ...     print(f"[{rec['priority']}] {rec['category']}: {rec['issue']}")
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise SourceDirectoryNotFoundError(
            f"Source directory not found or not a directory: {source_dir}"
        )

    if verbose:
        print(f"[Tier0] Scanning source directory: {source_dir}")

    from .source_analyzer import SourceAnalyzer

    scanner = SourceAnalyzer(source_dir, verbose=verbose)
    plan = scanner.analyze()

    if verbose:
        print(
            f"[Tier0] Scanned {plan.files_scanned} files, "
            f"found {plan.kernel_count} kernels, "
            f"programming model: {plan.programming_model}"
        )

    # Convert ProfilingPlan to SourceAnalysisResult dataclass
    result = _plan_to_source_result(plan)

    # Optional LLM enhancement
    if enable_llm and llm_provider:
        try:
            if verbose:
                print(f"[Tier0] Enhancing with {llm_provider} LLM...")

            analyzer = LLMAnalyzer(
                provider=llm_provider,
                api_key=llm_api_key,
                verbose=verbose,
            )
            context = AnalysisContext(tier=0, custom_prompt=custom_prompt)
            result.llm_explanation = analyzer.analyze_source_with_llm(
                result, custom_prompt=custom_prompt, context=context
            )

            if verbose:
                print("[Tier0] LLM enhancement complete")

        except (LLMAuthenticationError, LLMRateLimitError):
            raise
        except Exception as e:
            if verbose:
                print(f"[Tier0] LLM enhancement failed: {e}")

    return result


def validate_database(database_path: Path) -> Dict[str, Any]:
    """
    Validate database schema and contents without performing analysis.

    Args:
        database_path: Path to .rpd or .db file

    Returns:
        Validation result dictionary

    Example:
        >>> validation = validate_database(Path("output.db"))
        >>> print(f"Valid: {validation['is_valid']}")
        >>> print(f"Analysis tier: {validation['tier']}")
    """
    if not database_path.exists():
        raise DatabaseNotFoundError(f"Database not found: {database_path}")

    try:
        from ..connection import RocinsightConnection as RocpdImportData, execute_statement

        connection = RocpdImportData(str(database_path))

        # Check for required tables AND views (kernels/memory_copies are views,
        # not raw tables, in rocprofv3 databases created by the rocpd importer)
        tables_query = "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        tables = [
            row[0] for row in execute_statement(connection, tables_query).fetchall()
        ]

        has_kernels = "kernels" in tables
        has_memory_copies = "memory_copies" in tables
        has_counters = "pmc_events" in tables
        has_pc_sampling = "pc_sampling" in tables

        # Determine tier
        tier = 1
        if has_counters:
            tier = 2
        if has_pc_sampling:
            tier = 3

        return {
            "is_valid": has_kernels,
            "tier": tier,
            "has_kernels": has_kernels,
            "has_memory_copies": has_memory_copies,
            "has_counters": has_counters,
            "has_pc_sampling": has_pc_sampling,
            "tables": tables,
        }

    except Exception as e:
        raise DatabaseCorruptedError(f"Database validation failed: {e}")
