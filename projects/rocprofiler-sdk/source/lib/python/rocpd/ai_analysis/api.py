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
    from rocpd.ai_analysis import analyze_database
    from pathlib import Path

    result = analyze_database(Path("output.db"))
    print(result.summary.overall_assessment)

    for rec in result.recommendations.high_priority:
        print(f"- {rec.title}")
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..analyze import (
    analyze_performance as _analyze_performance_internal,
)
from .llm_analyzer import LLMAnalyzer, get_reference_guide_path
from .exceptions import (
    DatabaseNotFoundError,
    DatabaseCorruptedError,
    MissingDataError,
)


class OutputFormat(Enum):
    """Output format options"""

    PYTHON_OBJECT = "python_object"  # Returns dataclass
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Generate plain text report"""
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
        """Generate markdown report"""
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
    output_format: OutputFormat = OutputFormat.PYTHON_OBJECT,
    verbose: bool = False,
    top_kernels: int = 10,
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
        >>> from rocpd.ai_analysis import analyze_database
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

    # Perform local analysis using existing analyze module
    # This calls the analyze_performance function from analyze.py
    try:
        from ..importer import RocpdImportData
        connection = RocpdImportData(str(database_path))

        analysis_result_dict = _analyze_performance_internal(
            connection=connection,
            top_n=top_kernels,
            format_output=False,  # Get raw data, not formatted text
        )

        if verbose:
            print("[Analysis] Local analysis complete")

    except Exception as e:
        raise DatabaseCorruptedError(f"Failed to analyze database: {e}")

    # Build AnalysisResult from local analysis
    result = _build_analysis_result(
        analysis_result_dict,
        database_path,
        custom_prompt,
    )

    # Optional LLM enhancement
    if enable_llm and llm_provider:
        try:
            if verbose:
                print(f"[Analysis] Enhancing with {llm_provider} LLM...")

            analyzer = LLMAnalyzer(
                provider=llm_provider,
                api_key=llm_api_key,
                verbose=verbose,
            )

            # Convert result to dict for LLM
            analysis_data = _convert_result_to_llm_format(result)

            # Get LLM enhancement
            llm_explanation = analyzer.analyze_with_llm(
                analysis_data,
                custom_prompt=custom_prompt,
            )

            result.llm_enhanced_explanation = llm_explanation

            if verbose:
                print("[Analysis] LLM enhancement complete")

        except Exception as e:
            # LLM enhancement is optional - don't fail if it errors
            warning = AnalysisWarning(
                severity="warning",
                message=f"LLM enhancement failed: {e}",
                recommendation="Analysis continues with local-only results",
            )
            result.warnings.append(warning)

            if verbose:
                print(f"[Analysis] LLM enhancement failed: {e}")

    return result


def _build_analysis_result(
    analysis_dict: Dict[str, Any],
    database_path: Path,
    custom_prompt: Optional[str],
) -> AnalysisResult:
    """Build AnalysisResult from raw analysis dictionary"""
    from datetime import datetime

    # Extract data from analysis_dict (from analyze.py)
    breakdown = analysis_dict.get("time_breakdown", {})
    hotspots = analysis_dict.get("hotspots", [])
    recommendations = analysis_dict.get("recommendations", [])

    # Build metadata
    metadata = AnalysisMetadata(
        rocpd_version="6.3.0",
        analysis_version="0.1.0",
        database_file=str(database_path),
        analysis_timestamp=datetime.now().isoformat(),
        custom_prompt=custom_prompt,
    )

    # Build profiling info
    has_counters = analysis_dict.get("has_counters", False)
    profiling_mode = (
        "sys_trace_with_counters" if has_counters else "sys_trace_only"
    )
    analysis_tier = 2 if has_counters else 1

    profiling_info = ProfilingInfo(
        total_duration_ns=int(breakdown.get("total_runtime", 0)),
        profiling_mode=profiling_mode,
        analysis_tier=analysis_tier,
        gpus=[],  # TODO: Extract from database
    )

    # Build summary
    primary_bottleneck = "unknown"
    confidence = 0.5

    # Simple bottleneck classification
    memcpy_pct = breakdown.get("memcpy_percent", 0)
    if memcpy_pct > 30:
        primary_bottleneck = "memory_transfer"
        confidence = 0.8
    elif has_counters:
        # Could use counter data for better classification
        primary_bottleneck = "compute"
        confidence = 0.7

    summary = AnalysisSummary(
        overall_assessment=f"Analysis complete. {len(hotspots)} kernels analyzed.",
        primary_bottleneck=primary_bottleneck,
        confidence=confidence,
        key_findings=[
            f"Total kernel execution time: {breakdown.get('kernel_percent', 0):.1f}%",
            f"Memory copy overhead: {memcpy_pct:.1f}%",
            f"Top kernel: {hotspots[0]['name'] if hotspots else 'N/A'}",
        ],
    )

    # Build execution breakdown
    execution_breakdown = ExecutionBreakdown(
        kernel_time_ns=int(breakdown.get("total_kernel_time", 0)),
        kernel_time_pct=breakdown.get("kernel_percent", 0),
        memcpy_time_ns=int(breakdown.get("total_memcpy_time", 0)),
        memcpy_time_pct=memcpy_pct,
    )

    # Build recommendations
    rec_set = RecommendationSet()
    for i, rec in enumerate(recommendations, 1):
        recommendation = Recommendation(
            id=f"rec_{i:03d}",
            priority=rec.get("priority", "medium"),
            category=rec.get("category", "general"),
            title=rec.get("title", "Optimization opportunity"),
            description=rec.get("description", ""),
            estimated_impact=rec.get("impact", "Unknown"),
            next_steps=[],
        )

        if rec.get("priority") == "high":
            rec_set.high_priority.append(recommendation)
        elif rec.get("priority") == "medium":
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

    return AnalysisResult(
        metadata=metadata,
        profiling_info=profiling_info,
        summary=summary,
        execution_breakdown=execution_breakdown,
        recommendations=rec_set,
        warnings=warnings,
    )


def _convert_result_to_llm_format(result: AnalysisResult) -> Dict[str, Any]:
    """Convert AnalysisResult to format expected by LLM analyzer"""
    return {
        "gpu": {"name": "AMD GPU", "arch": "gfx90a"},  # TODO: Extract from DB
        "execution_breakdown": {
            "kernel_time_pct": result.execution_breakdown.kernel_time_pct,
            "memcpy_time_pct": result.execution_breakdown.memcpy_time_pct,
            "api_overhead_pct": result.execution_breakdown.api_overhead_pct,
        },
        "kernels": [],  # TODO: Add kernel data
        "memory_ops": {},  # TODO: Add memory ops
        "has_counters": result.profiling_info.analysis_tier >= 2,
        "has_pc_sampling": result.profiling_info.analysis_tier >= 3,
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
        from ..importer import RocpdImportData, execute_statement

        connection = RocpdImportData(str(database_path))

        # Check for required tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
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
