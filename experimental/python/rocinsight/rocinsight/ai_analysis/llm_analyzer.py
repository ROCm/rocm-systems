#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
LLM-powered analysis with reference guide ("fence") implementation.

The reference guide is a user-modifiable markdown file that defines:
- GPU hardware specifications
- Performance analysis models and formulas
- Bottleneck classification guidelines
- AMD-specific optimization techniques
- Recommendation quality standards
- Output format requirements

This guide is loaded from llm-reference-guide.md and included in every
LLM request to ensure consistent, high-quality analysis.

To modify LLM behavior, edit: share/rocprofiler-sdk/llm-reference-guide.md
No code changes required - the guide is loaded dynamically.
"""

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from .exceptions import (
    AnalysisError,
    LLMAuthenticationError,
    LLMRateLimitError,
    ReferenceGuideNotFoundError,
)

# Regex to match Unix and Windows file paths that may appear in profiling data
_PATH_PATTERN = re.compile(
    r'(/home/[^\s,"\';>]+|/opt/[^\s,"\';>]+|/root/[^\s,"\';>]+|'
    r'/tmp/[^\s,"\';>]+|/var/[^\s,"\';>]+|[A-Za-z]:\\[^\s,"\';>]+)'
)

# Regex to match relative path traversal (e.g. ../../secret/file)
_RELATIVE_PATH_PATTERN = re.compile(r'(?:\.\./)+\S+')

# Regex to match rocinsight-context tag comments in the reference guide
_TAG_RE = re.compile(r"<!--\s*rocinsight-context:\s*([^-]+?)\s*-->")


def _redact_paths(value: str) -> str:
    """Replace file system paths in a string with [REDACTED]."""
    result = _PATH_PATTERN.sub("[REDACTED]", value)
    result = _RELATIVE_PATH_PATTERN.sub("[REDACTED_PATH]", result)
    return result


# Default location for the reference guide (relative to package installation)
# Users can override with ROCINSIGHT_LLM_REFERENCE_GUIDE environment variable
DEFAULT_REFERENCE_GUIDE_NAME = "llm-reference-guide.md"

# Default model names — override at runtime with ROCINSIGHT_LLM_MODEL env var
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4-turbo-preview"

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Maps provider name → human-readable description.
# Used for --llm CLI help text, error messages, and validation.
#
# To add a new provider:
#   1. Add an entry here: PROVIDER_REGISTRY["my-provider"] = "My Provider description"
#   2. Add the call method to LLMAnalyzer: _call_my_provider(self, system, user) -> str
#   3. Route it in analyze_with_llm's _dispatch() and summarize_source_file()
#   4. Add streaming to LLMConversation._stream_response() if multi-turn is needed
#   5. Add it to the --llm choices in analyze.py add_args()
#   6. Optionally add it to LLMConversation's valid set (for interactive mode)
#
PROVIDER_REGISTRY: Dict[str, str] = {
    "anthropic": "Anthropic Claude API (requires ANTHROPIC_API_KEY)",
    "openai": "OpenAI API (requires OPENAI_API_KEY)",
    "private": "Private/enterprise OpenAI-compatible endpoint (requires ROCINSIGHT_LLM_PRIVATE_URL)",
    "local": "Local Ollama or OpenAI-compatible server (ROCINSIGHT_LLM_LOCAL_URL)",
    "claude-code": "Claude Code — ANTHROPIC_API_KEY (primary) or 'claude' CLI stored credentials (fallback)",
}


def get_reference_guide_path() -> Path:
    """
    Get the path to the LLM reference guide.

    Priority order:
    1. ROCINSIGHT_LLM_REFERENCE_GUIDE environment variable
    2. Relative to this module (ai_analysis/share/)
    3. /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md

    Returns:
        Path to reference guide file

    Raises:
        ReferenceGuideNotFoundError: If guide file not found (lists all attempted paths)
    """
    attempted = []

    # Check environment variable first
    env_path = os.environ.get("ROCINSIGHT_LLM_REFERENCE_GUIDE")
    if env_path:
        guide_path = Path(env_path)
        if guide_path.exists():
            return guide_path
        attempted.append(str(guide_path))

    # Check relative to this module (preferred for development and installation)
    module_path = Path(__file__).parent / "share" / DEFAULT_REFERENCE_GUIDE_NAME
    if module_path.exists():
        return module_path
    attempted.append(str(module_path))

    # Check ROCm installation directory (legacy)
    rocm_path = Path("/opt/rocm/share/rocprofiler-sdk") / DEFAULT_REFERENCE_GUIDE_NAME
    if rocm_path.exists():
        return rocm_path
    attempted.append(str(rocm_path))

    # Not found — report all attempted paths
    raise ReferenceGuideNotFoundError(attempted)


def load_reference_guide() -> str:
    """Load the LLM fence document.

    Same path lookup order as get_reference_guide_path():
    ROCINSIGHT_LLM_REFERENCE_GUIDE env var → module share/ dir → /opt/rocm/share/...

    Raises:
        ReferenceGuideNotFoundError: If guide file not found.
    """
    return get_reference_guide_path().read_text()


# ---------------------------------------------------------------------------
# Context-aware guide filtering
# ---------------------------------------------------------------------------


@dataclass
class AnalysisContext:
    """
    Describes the current analysis state so LLMAnalyzer can select only the
    relevant sections of the reference guide, reducing prompt token cost by
    18–51% depending on the scenario.

    Fields:
        tier: Analysis tier — 0=source-only, 1=trace, 2=hardware counters.
        has_counters: True when PMC counter data is present in the database.
            When True, tier2-tagged sections are loaded even if tier==1.
        bottleneck_type: Primary bottleneck from _build_summary() —
            "compute", "memory_transfer", "latency", or "mixed".
            "compute" and "memory"/"memory_transfer" both trigger the compiler tag.
        gpu_arch: Detected GPU architecture string e.g. "gfx942".
            Reserved for future per-GPU section filtering.
        custom_prompt: The user's --prompt text, if any.
            Triggers compiler tag when it contains compiler/flag/build/compile.
    """

    tier: int = 1
    has_counters: bool = False
    bottleneck_type: Optional[str] = None
    gpu_arch: Optional[str] = None
    custom_prompt: Optional[str] = None

    # TraceLens-derived metrics (used by _select_tags() to gate reference guide section)
    kernel_categories: Optional[list] = (
        None  # [{category, count, pct_of_kernel_time, ...}]
    )
    short_kernel_summary: Optional[dict] = (
        None  # {threshold_us, short_kernel_count, wasted_pct}
    )
    interval_timeline: Optional[dict] = (
        None  # {true_compute_pct, exposed_memcpy_pct, idle_pct}
    )


def _select_tags(ctx: AnalysisContext) -> set:
    """
    Map an AnalysisContext to the minimum set of section tags needed for the
    current analysis, minimising prompt token cost.

    Tag vocabulary:
        always   — critical rules, role, output format, what not to do, summary
        tier1    — profiling workflow, tool reference, common bottleneck types
        tier2    — hardware counters, memory hierarchy, GPU specs, perf models
        compiler — compiler flags section (HIPCC, LLVM AMDGPU, register control)
        source   — reserved for future Tier 0-specific guidance sections

    Selection strategy (most selective first):
    - Tier 1 with a clear bottleneck and no counters → ``always`` only.
      The bottleneck is already identified; the LLM just needs formatting rules
      and output constraints — not the full profiling workflow or GPU spec tables.
    - Tier 1 without a clear bottleneck → ``always + tier1`` so the LLM can
      reason about the pattern from first principles.
    - Tier 2 (counters available) → ``always + tier2``.  The tier2 section
      covers hardware specs and roofline; tier1 workflow adds limited value once
      counter data is present and just inflates the prompt.
    - ``compiler`` is only added when the bottleneck is compute-bound or the
      user explicitly mentions compiler/build topics.
    - ``tier0`` / ``source`` / ``tracelens_metrics`` are additive when the
      matching data is present.

    Fallback: sections with no tag comment are always included.
    """
    tags = {"always"}

    _bt = ctx.bottleneck_type or ""
    _cp = (ctx.custom_prompt or "").lower()

    if ctx.tier == 0:
        # Source-only: source-specific guidance + compiler flags (common optimization path)
        tags.add("source")
        tags.add("compiler")
    elif ctx.has_counters or ctx.tier >= 2:
        # Counter data available: include both tier1 (workflow context) and
        # tier2 (GPU specs + roofline) for full analysis coverage.
        tags.add("tier1")
        tags.add("tier2")
    else:
        # Tier 1 trace, no counters.
        # If the bottleneck is already clear, always-only suffices.
        # Otherwise include tier1 so the LLM can reason about the pattern.
        if not _bt or _bt == "mixed":
            tags.add("tier1")

    # compiler: when compute- or memory-bound, or user asks about build/compile topics
    if _bt in ("compute", "memory", "memory_transfer") or any(
        w in _cp for w in ("compiler", "flag", "build", "compile", "register")
    ):
        tags.add("compiler")

    # tracelens_metrics: include when TraceLens analysis data is available
    if ctx.kernel_categories or ctx.interval_timeline:
        tags.add("tracelens_metrics")

    return tags


def _filter_guide(guide_text: str, tags: set) -> str:
    """
    Return only the sections of the reference guide whose
    ``<!-- rocinsight-context: TAG [, TAG ...] -->`` comment matches one of the
    requested *tags*.

    Parsing rules:
    - Split on newline + "## " to find section boundaries.
    - Scan only the first 3 lines of each section for the tag comment.
    - A section with *no* tag comment is always included (safe fallback for
      user-added sections that lack a tag).
    - Multiple tags in a single comment are comma-separated; any match
      is sufficient to include the section.
    - Tags in the comment are stripped of surrounding whitespace before
      comparison.

    Args:
        guide_text: Full reference guide markdown content.
        tags:       Set of tag strings that should be included
                    (e.g. {"always", "tier1"}).

    Returns:
        Filtered guide text.  Empty string if *guide_text* is empty.
    """
    if not guide_text:
        return ""

    # Split on section boundaries. The intro block (before first ##) is
    # kept as-is (it has no tag → always included).
    raw_sections = re.split(r"\n(?=## )", guide_text)

    included = []
    for section in raw_sections:
        # Examine only the first 3 lines for a tag comment.
        head_lines = section.splitlines()[:3]
        head = "\n".join(head_lines)
        match = _TAG_RE.search(head)

        if match is None:
            # No tag → always include (safe fallback).
            included.append(section)
        else:
            section_tags = {t.strip() for t in match.group(1).split(",")}
            if section_tags & tags:
                included.append(section)

    return "\n".join(included)


class LLMAnalyzer:
    """
    Handles LLM-powered analysis enhancements.

    The reference guide acts as the "fence" - it's loaded once and included
    in every LLM request to ensure consistent, high-quality analysis.

    Example:
        >>> analyzer = LLMAnalyzer(provider="anthropic")
        >>> result = analyzer.analyze_with_llm(analysis_data, custom_prompt="Why is kernel X slow?")
        >>> print(result)
    """

    def __init__(
        self,
        provider: str = "anthropic",  # "anthropic", "openai", or "local"
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reference_guide_path: Optional[Path] = None,
        verbose: bool = False,
        thinking_budget_tokens: Optional[int] = None,
    ):
        """
        Initialize LLM analyzer.

        Args:
            provider: LLM provider ("anthropic", "openai", or "local")
            api_key: API key (if None, reads from environment)
            model: Override model name (if None, uses default for provider)
            reference_guide_path: Path to reference guide (if None, uses default location)
            verbose: Enable verbose logging
            thinking_budget_tokens: Enable extended thinking with this token budget.
                Only supported with the Anthropic provider and compatible models
                (claude-opus-4, claude-sonnet-4-5, claude-3-7-sonnet).
                Can also be set via ROCINSIGHT_LLM_THINKING environment variable.
        """
        if provider not in PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Must be one of: {', '.join(sorted(PROVIDER_REGISTRY))}"
            )
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.api_key = api_key or self._get_api_key_from_env(raise_if_missing=False)

        # Extended thinking budget: explicit parameter takes precedence over env var
        if thinking_budget_tokens is not None:
            self.thinking_budget_tokens = thinking_budget_tokens
        else:
            _env_thinking = os.environ.get("ROCINSIGHT_LLM_THINKING")
            if _env_thinking:
                try:
                    self.thinking_budget_tokens = int(_env_thinking)
                except ValueError:
                    import warnings

                    warnings.warn(
                        f"ROCINSIGHT_LLM_THINKING={_env_thinking!r} is not a valid integer; "
                        "extended thinking disabled.",
                        stacklevel=2,
                    )
                    self.thinking_budget_tokens = None
            else:
                self.thinking_budget_tokens = None

        # Load reference guide (the "fence")
        if reference_guide_path:
            self.reference_guide_path = reference_guide_path
        else:
            self.reference_guide_path = get_reference_guide_path()

        self.reference_guide = self._load_reference_guide()

        if self.verbose:
            print(f"[LLM] Loaded reference guide from: {self.reference_guide_path}")
            print(f"[LLM] Reference guide size: {len(self.reference_guide)} characters")

    def _get_api_key_from_env(self, raise_if_missing: bool = True) -> str:
        """Get API key from environment"""
        if self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "local":
            return os.environ.get("ROCINSIGHT_LLM_LOCAL_API_KEY", "ignored")
        elif self.provider == "private":
            return os.environ.get("ROCINSIGHT_LLM_PRIVATE_API_KEY", "dummy")
        elif self.provider == "claude-code":
            # No dedicated API key — uses Claude Code auth; may fall back to ANTHROPIC_API_KEY.
            return os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        if not key and raise_if_missing:
            raise LLMAuthenticationError(
                f"No API key found for {self.provider}. "
                f"Set {'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'} "
                "environment variable."
            )

        return key

    def _load_reference_guide(self) -> str:
        """
        Load the reference guide from file.

        This makes it easy to modify the guide without changing code.
        The guide is the "fence" that constrains LLM behavior.

        Returns:
            Reference guide content as string

        Raises:
            ReferenceGuideNotFoundError: If guide file doesn't exist
        """
        if not self.reference_guide_path.exists():
            raise ReferenceGuideNotFoundError([str(self.reference_guide_path)])

        return self.reference_guide_path.read_text()

    def _sanitize_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize sensitive data before sending to LLM.

        Privacy rules:
        - Kernel names → [KERNEL_1], [KERNEL_2], etc.
        - Grid dimensions → [GRID_SIZE]
        - Workgroup sizes → [WORKGROUP_SIZE]
        - File paths → [REDACTED]

        Preserved data (aggregated/classified):
        - Bottleneck classifications
        - Aggregated metrics (time percentages, utilization)
        - GPU architecture identifiers

        Args:
            analysis_data: Raw analysis data

        Returns:
            Sanitized copy of analysis data
        """
        sanitized = {}

        # Copy top-level non-sensitive fields, redacting any embedded paths
        for key in ["execution_breakdown", "gpu", "profiling_info"]:
            if key in analysis_data:
                section = analysis_data[key].copy()
                # Redact path-like strings in nested string values
                for k, v in section.items():
                    if isinstance(v, str):
                        section[k] = _redact_paths(v)
                sanitized[key] = section

        # Sanitize kernel information
        if "kernels" in analysis_data:
            sanitized["kernels"] = []
            for i, kernel in enumerate(analysis_data["kernels"], 1):
                sanitized_kernel = {
                    "kernel_id": f"[KERNEL_{i}]",
                    "dispatch_count": kernel.get("dispatch_count"),
                    "pct_total_time": kernel.get("pct_total_time"),
                    "avg_duration_ns": kernel.get("avg_duration_ns"),
                }

                # Include counter data but redact sizes
                if "vgpr_count" in kernel:
                    sanitized_kernel["vgpr_count"] = kernel["vgpr_count"]
                if "occupancy_pct" in kernel:
                    sanitized_kernel["occupancy_pct"] = kernel["occupancy_pct"]
                if "valu_util_pct" in kernel:
                    sanitized_kernel["valu_util_pct"] = kernel["valu_util_pct"]
                if "hbm_util_pct" in kernel:
                    sanitized_kernel["hbm_util_pct"] = kernel["hbm_util_pct"]

                # Redact grid/workgroup sizes
                if "grid_size" in kernel:
                    sanitized_kernel["grid_size"] = "[GRID_SIZE]"
                if "workgroup_size" in kernel:
                    sanitized_kernel["workgroup_size"] = "[WORKGROUP_SIZE]"

                sanitized["kernels"].append(sanitized_kernel)

        # Keep memory operations (aggregated, no sensitive data)
        if "memory_ops" in analysis_data:
            sanitized["memory_ops"] = analysis_data["memory_ops"]

        # Keep data availability flags
        sanitized["has_counters"] = analysis_data.get("has_counters", False)
        sanitized["has_pc_sampling"] = analysis_data.get("has_pc_sampling", False)

        # Pass through TraceLens-derived metrics (already safe: pct values and category strings only)
        for tl_key in (
            "interval_timeline",
            "kernel_categories",
            "short_kernel_summary",
        ):
            if tl_key in analysis_data:
                sanitized[tl_key] = analysis_data[tl_key]

        return sanitized

    def _build_system_prompt(self, context: Optional[AnalysisContext] = None) -> str:
        """
        Build the system prompt with the reference guide.

        When *context* is provided, only the guide sections relevant to the
        current analysis are included (see _filter_guide and _select_tags).
        When *context* is None the full guide is used — preserving backward
        compatibility for callers that do not yet provide context.

        Args:
            context: Optional AnalysisContext describing tier, bottleneck, etc.

        Returns:
            System prompt string with embedded (possibly filtered) reference guide.
        """
        if context is not None:
            guide = _filter_guide(self.reference_guide, _select_tags(context))
            if self.verbose:
                full_len = len(self.reference_guide)
                filt_len = len(guide)
                print(
                    f"[LLM] Guide filtered: {filt_len} / {full_len} chars "
                    f"({(100 * filt_len // full_len if full_len else 0)}% of full guide)"
                )
        else:
            guide = self.reference_guide

        return f"""You are an expert GPU performance analyst specializing in AMD GPUs.

{guide}

CRITICAL: Follow these guidelines strictly:
1. Use ONLY current generation tools (rocprofv3, rocprof-compute, rocprof-sys), NEVER rocprof or rocprof-v2
2. Output plain text ONLY - no markdown headers (###), no **bold**, no special formatting
3. Structure your response exactly as specified in the reference guide
4. Choose the appropriate profiling tool based on the analysis need per documentation
5. Maintain consistent format regardless of analysis complexity
6. All commands and options must match the official documentation exactly
"""

    def _build_user_prompt(
        self,
        analysis_data: Dict[str, Any],
        custom_prompt: Optional[str] = None,
        bottleneck_type: Optional[str] = None,
    ) -> str:
        """
        Build user prompt with profiling data.

        Args:
            analysis_data: Sanitized profiling data
            custom_prompt: Optional user question
            bottleneck_type: Primary bottleneck (drives data section selection)

        Returns:
            User prompt string
        """
        # Format data as structured text for LLM — only include sections
        # relevant to the detected bottleneck to minimise token usage.
        data_summary = self._format_data_for_llm(analysis_data, bottleneck_type=bottleneck_type)

        if custom_prompt:
            return f"""User Question: {custom_prompt}

Profiling Data:
{data_summary}

Please analyze this data and answer the user's question, following the
reference guide. Provide specific, actionable recommendations.

IMPORTANT FORMAT REQUIREMENTS:
- Use PLAIN TEXT only - no markdown headers (###, ##, #)
- Use ONLY current generation tools (rocprofv3, rocprof-compute, rocprof-sys) in profiling suggestions
- NEVER suggest deprecated tools like rocprof or rocprof-v2
- All commands must match official documentation exactly
- Structure recommendations with: Priority, Issue, Suggestion, Actionable Steps
- Be consistent with the output format"""
        else:
            return f"""Profiling Data:
{data_summary}

Please analyze this GPU profiling data and provide:
1. Executive summary (2-3 sentences)
2. Primary bottleneck identification with confidence level
3. Top 3-5 actionable recommendations (prioritized High/Medium/Low)
4. Suggested next profiling steps (if applicable)

IMPORTANT FORMAT REQUIREMENTS:
- Use PLAIN TEXT only - no markdown headers (###, ##, #)
- Use ONLY current generation tools (rocprofv3, rocprof-compute, rocprof-sys) in profiling suggestions
- NEVER suggest deprecated tools like rocprof or rocprof-v2
- All commands must match official documentation exactly
- Structure each recommendation with: Priority, Issue, Suggestion, Actionable Steps, Expected Impact
- Be consistent with the output format regardless of your model

Follow the reference guide strictly for analysis methodology and output format."""

    def _format_data_for_llm(
        self,
        data: Dict[str, Any],
        bottleneck_type: Optional[str] = None,
    ) -> str:
        """Format analysis data as readable text for LLM.

        Only sections relevant to the detected bottleneck are included to
        minimise prompt token cost and speed up inference.

        Section selection by bottleneck:
        - memory_transfer : execution breakdown + memory ops + top-3 kernels (no counter fields)
        - latency / api   : execution breakdown + top-3 kernels (no memory or counter detail)
        - compute         : execution breakdown + top-3 kernels with counter fields
        - mixed / None    : all sections, top-5 kernels
        """
        lines = []
        _bt = bottleneck_type or ""
        _include_memory = _bt in ("", "memory_transfer", "mixed")
        _include_counter_fields = _bt in ("", "compute", "mixed")
        _top_n = 3 if _bt and _bt != "mixed" else 5

        # GPU info — always included (small)
        if "gpu" in data:
            lines.append("## GPU Information")
            lines.append(f"- Name: {data['gpu'].get('name', 'Unknown')}")
            lines.append(f"- Architecture: {data['gpu'].get('arch', 'Unknown')}")
            lines.append("")

        # Execution breakdown — always included
        if "execution_breakdown" in data:
            lines.append("## Execution Breakdown")
            breakdown = data["execution_breakdown"]
            lines.append(f"- Kernel Time: {breakdown.get('kernel_time_pct', 0):.1f}%")
            lines.append(
                f"- Memory Copy Time: {breakdown.get('memcpy_time_pct', 0):.1f}%"
            )
            lines.append(f"- API Overhead: {breakdown.get('api_overhead_pct', 0):.1f}%")
            lines.append("")

        # Top kernels — limit count and fields by bottleneck
        if "kernels" in data:
            lines.append(f"## Top Kernels (top {_top_n})")
            for kernel in data["kernels"][:_top_n]:
                lines.append(f"- {kernel.get('kernel_id', 'Unknown')}")
                lines.append(f"  - Time: {kernel.get('pct_total_time', 0):.1f}% of total")
                lines.append(f"  - Dispatches: {kernel.get('dispatch_count', 'N/A')}")

                if _include_counter_fields:
                    if "vgpr_count" in kernel:
                        lines.append(f"  - VGPR Usage: {kernel.get('vgpr_count')}")
                    if "occupancy_pct" in kernel:
                        lines.append(
                            f"  - Wave Occupancy: {kernel.get('occupancy_pct'):.1f}%"
                        )
                    if "valu_util_pct" in kernel:
                        lines.append(
                            f"  - VALU Utilization: {kernel.get('valu_util_pct'):.1f}%"
                        )
                    if "hbm_util_pct" in kernel:
                        lines.append(
                            f"  - HBM Utilization: {kernel.get('hbm_util_pct'):.1f}%"
                        )
                lines.append("")

        # Memory operations — omit for pure compute / latency bottlenecks
        if _include_memory and "memory_ops" in data:
            lines.append("## Memory Operations")
            mem = data["memory_ops"]
            if "h2d" in mem:
                lines.append(
                    f"- H2D: {mem['h2d'].get('count', 0)} transfers, "
                    f"{mem['h2d'].get('total_bytes', 0) / 1e9:.2f} GB, "
                    f"{mem['h2d'].get('bandwidth_gbps', 0):.1f} GB/s"
                )
            if "d2h" in mem:
                lines.append(
                    f"- D2H: {mem['d2h'].get('count', 0)} transfers, "
                    f"{mem['d2h'].get('total_bytes', 0) / 1e9:.2f} GB, "
                    f"{mem['d2h'].get('bandwidth_gbps', 0):.1f} GB/s"
                )
            lines.append("")

        # TraceLens-derived metrics (present when _convert_result_to_llm_format() included them)
        tracelens_parts = []
        if data.get("interval_timeline"):
            tracelens_parts.append(
                "interval_timeline: " + json.dumps(data["interval_timeline"])
            )
        if data.get("kernel_categories"):
            tracelens_parts.append(
                "kernel_categories: " + json.dumps(data["kernel_categories"])
            )
        if data.get("short_kernel_summary"):
            tracelens_parts.append(
                "short_kernels: " + json.dumps(data["short_kernel_summary"])
            )
        if tracelens_parts:
            lines.append("=== TraceLens-Derived Metrics ===")
            lines.extend(tracelens_parts)
            lines.append("")

        # Data availability note
        lines.append("## Data Availability")
        if data.get("has_counters"):
            lines.append("✅ Hardware counters available (Tier 2 analysis possible)")
        else:
            lines.append("⚠️  No hardware counters (Tier 1 trace analysis only)")

        if data.get("has_pc_sampling"):
            lines.append("✅ PC sampling data available (Tier 3 analysis possible)")

        return "\n".join(lines)

    def analyze_with_llm(
        self,
        analysis_data: Dict[str, Any],
        custom_prompt: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
    ) -> str:
        """
        Send analysis data to LLM for enhanced explanation.

        The LLM receives:
        1. System prompt with reference guide (the "fence")
        2. Sanitized profiling data
        3. Optional custom user prompt

        Args:
            analysis_data: Profiling data and basic analysis results
            custom_prompt: User's custom question (e.g., "Why is kernel X slow?")
            context: Optional AnalysisContext for guide section filtering.
                When provided, only guide sections relevant to the current
                analysis tier/bottleneck are included, reducing token cost.

        Returns:
            LLM-generated natural language analysis

        Raises:
            LLMAuthenticationError: Invalid API key
            LLMRateLimitError: API rate limit exceeded
        """
        # Sanitize data (privacy protection)
        sanitized_data = self._sanitize_data(analysis_data)

        # Build prompts (includes reference guide as "fence")
        _bottleneck = context.bottleneck_type if context else None
        system_prompt = self._build_system_prompt(context=context)
        user_prompt = self._build_user_prompt(
            sanitized_data, custom_prompt, bottleneck_type=_bottleneck
        )

        if self.verbose:
            print(f"[LLM] Calling {self.provider} API...")
            print(f"[LLM] System prompt length: {len(system_prompt)} chars")
            print(f"[LLM] User prompt length: {len(user_prompt)} chars")

        # Extended thinking is only supported by Anthropic
        if self.thinking_budget_tokens is not None and self.provider != "anthropic":
            raise ValueError(
                "Extended thinking is only supported with the Anthropic provider. "
                f"Current provider: {self.provider!r}. "
                "Remove --llm-thinking or switch to --llm anthropic."
            )

        # Reasoning models (gpt-5, o1, o3) consume thinking tokens against
        # max_completion_tokens, leaving nothing for actual output at 4096.
        # Use a higher ceiling that gives room for both thinking + response.
        _RETRY_MAX_TOKENS = 16384

        def _dispatch(sp: str, up: str, max_tokens: int = 4096) -> str:
            if self.provider == "anthropic":
                return self._call_anthropic(sp, up, max_tokens=max_tokens)
            elif self.provider == "openai":
                return self._call_openai(sp, up, max_tokens=max_tokens)
            elif self.provider == "local":
                return self._call_local(sp, up)
            elif self.provider == "private":
                return self._call_private(sp, up)
            elif self.provider == "claude-code":
                return self._call_claude_code(sp, up)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        # Call appropriate LLM API; on context-overflow retry with a smaller
        # guide and a higher token budget (needed for reasoning models).
        try:
            return _dispatch(system_prompt, user_prompt)
        except AnalysisError as _ae:
            if "Retrying with a smaller context" not in str(_ae):
                raise
            # Rebuild system prompt using only the mandatory "always" sections.
            _compact_guide = _filter_guide(self.reference_guide, {"always"})
            _compact_prompt = (
                f"You are an expert GPU performance analyst specializing in AMD GPUs.\n\n"
                f"{_compact_guide}\n\n"
                "CRITICAL: Use ONLY current generation tools (rocprofv3, rocprof-compute, "
                "rocprof-sys). Output plain text only — no markdown headers."
            )
            if self.verbose:
                print(
                    f"[LLM] Retrying with always-only guide "
                    f"({len(_compact_prompt)} chars, was {len(system_prompt)}) "
                    f"and max_tokens={_RETRY_MAX_TOKENS}"
                )
            try:
                return _dispatch(_compact_prompt, user_prompt, max_tokens=_RETRY_MAX_TOKENS)
            except AnalysisError as _ae2:
                # Retry also failed — raise a clean user-facing error.
                raise AnalysisError(
                    f"OpenAI response empty even after reducing context "
                    f"(max_completion_tokens={_RETRY_MAX_TOKENS}). "
                    "The model may not support this token budget — try a different model."
                ) from _ae2

    def _call_anthropic(
        self, system_prompt: str, user_prompt: str, timeout: int = 120,
        max_tokens: int = 4096,
    ) -> str:
        """Call Anthropic Claude API"""
        if not self.api_key:
            raise LLMAuthenticationError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY environment variable."
            )
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            model = (
                self.model or os.environ.get("ROCINSIGHT_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
            )

            # Build base API call kwargs
            create_kwargs: Dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=timeout,
            )

            # Extended thinking support (Anthropic-only)
            if self.thinking_budget_tokens is not None:
                # Warn if model may not support extended thinking
                _thinking_models = (
                    "claude-opus-4",
                    "claude-sonnet-4-5",
                    "claude-3-7-sonnet",
                )
                if not any(m in model for m in _thinking_models):
                    import warnings

                    warnings.warn(
                        f"Extended thinking requested but model {model!r} may not support it. "
                        "Compatible models: claude-opus-4, claude-sonnet-4-5, claude-3-7-sonnet. "
                        "Attempting anyway — API will return an error if unsupported.",
                        stacklevel=3,
                    )

                create_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
                # claude-3-7-sonnet uses the older beta header
                if "3-7" in model:
                    create_kwargs["betas"] = ["thinking-2025-02-19"]

                if self.verbose:
                    print(
                        f"[LLM] Extended thinking enabled: budget={self.thinking_budget_tokens} tokens"
                    )

            response = client.messages.create(**create_kwargs)

            # Extract only text content blocks (skip thinking blocks)
            text_parts = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            return "\n".join(text_parts) if text_parts else ""

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except Exception as e:
            raise AnalysisError(f"Anthropic API error: {e}")

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> str:
        """Call OpenAI GPT API"""
        if not self.api_key:
            raise LLMAuthenticationError(
                "No OpenAI API key. Set OPENAI_API_KEY environment variable."
            )
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        try:
            client = openai.OpenAI(api_key=self.api_key)

            model = (
                self.model or os.environ.get("ROCINSIGHT_LLM_MODEL") or DEFAULT_OPENAI_MODEL
            )
            _messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            # Newer OpenAI models (gpt-5, o1, o3, gpt-4o-2024-11-20+) require
            # max_completion_tokens; older models use max_tokens.  Try the new
            # parameter first and fall back transparently.
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=_messages,
                    max_completion_tokens=max_tokens,
                    timeout=timeout,
                )
            except openai.BadRequestError as _br:
                if "max_completion_tokens" in str(_br):
                    response = client.chat.completions.create(
                        model=model,
                        messages=_messages,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                else:
                    raise

            msg = response.choices[0].message
            content = msg.content

            # Handle content-parts list (newer OpenAI API format)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if hasattr(part, "text"):
                        text_parts.append(str(part.text))
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)

            finish = getattr(response.choices[0], "finish_reason", "unknown")

            # Non-empty but truncated: return what we have (partial > nothing)
            if content and finish == "length":
                import warnings as _w
                _w.warn(
                    f"[LLMAnalyzer] OpenAI response truncated at {max_tokens} tokens "
                    "— returning partial output.",
                    stacklevel=3,
                )
                return content

            if content is None or content == "":
                # Check for explicit refusal
                refusal = getattr(msg, "refusal", None)
                if refusal:
                    raise AnalysisError(f"OpenAI refused request: {refusal}")
                # Empty content with finish_reason="length" means the prompt consumed
                # all available tokens — caller should retry with a smaller guide.
                if finish == "length":
                    raise AnalysisError(
                        f"OpenAI response truncated at token limit "
                        f"(max_completion_tokens={max_tokens}). "
                        "Retrying with a smaller context."
                    )
                raise AnalysisError(
                    f"OpenAI returned empty content (finish_reason={finish!r}). "
                    "Try a different model or simplify the request."
                )
            return content

        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except Exception as e:
            raise AnalysisError(f"OpenAI API error: {e}")

    def _call_local(self, system_prompt: str, user_prompt: str) -> str:
        """Call a local OpenAI-compatible LLM endpoint (e.g. Ollama)."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for local LLM: pip install openai")
        base_url = os.environ.get("ROCINSIGHT_LLM_LOCAL_URL", "http://localhost:11434/v1")
        client = openai.OpenAI(base_url=base_url, api_key="ignored")
        model = self.model or os.environ.get("ROCINSIGHT_LLM_LOCAL_MODEL", "codellama:13b")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                timeout=60,
            )
            return resp.choices[0].message.content
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"Local LLM authentication failed: {e}")
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"Local LLM rate limit exceeded: {e}")
        except Exception as exc:
            raise RuntimeError(
                f"Local LLM request failed ({base_url}). "
                f"Is Ollama running? Set ROCINSIGHT_LLM_LOCAL_URL to override endpoint. "
                f"Error: {exc}"
            ) from exc

    def _call_private(self, system_prompt: str, user_prompt: str) -> str:
        """Call a private/enterprise OpenAI-compatible LLM server.

        Configuration via environment variables:
            ROCINSIGHT_LLM_PRIVATE_URL        Base URL (required)
            ROCINSIGHT_LLM_PRIVATE_MODEL      Model name (required)
            ROCINSIGHT_LLM_PRIVATE_API_KEY    API key (default: "dummy")
            ROCINSIGHT_LLM_PRIVATE_HEADERS    JSON object of extra request headers
                                         (the "user" header defaults to os.getlogin())
            ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL Set to "0" or "false" to disable SSL
                                         certificate verification (requires httpx).
        """
        try:
            import openai
            import json as _json
        except ImportError:
            raise ImportError(
                "openai package required for private LLM: pip install openai"
            )

        base_url = os.environ.get("ROCINSIGHT_LLM_PRIVATE_URL", "")
        if not base_url:
            raise ValueError(
                "ROCINSIGHT_LLM_PRIVATE_URL is not set. "
                "Export it to point at your private LLM server."
            )
        model = self.model or os.environ.get("ROCINSIGHT_LLM_PRIVATE_MODEL", "")
        if not model:
            raise ValueError(
                "No model specified for private provider. "
                "Set ROCINSIGHT_LLM_PRIVATE_MODEL or pass --llm-private-model."
            )
        key = self.api_key or os.environ.get("ROCINSIGHT_LLM_PRIVATE_API_KEY", "dummy")

        headers: dict = {}
        try:
            headers["user"] = os.getlogin()
        except OSError:
            pass
        raw_headers = os.environ.get("ROCINSIGHT_LLM_PRIVATE_HEADERS", "")
        if raw_headers:
            # Try strict JSON first; only normalize single-quotes as a fallback.
            # The replace-based normalization would corrupt values with apostrophes.
            parsed_h = None
            try:
                parsed_h = _json.loads(raw_headers)
            except _json.JSONDecodeError:
                try:
                    parsed_h = _json.loads(raw_headers.replace("'", '"'))
                except _json.JSONDecodeError as e:
                    raise ValueError(f"ROCINSIGHT_LLM_PRIVATE_HEADERS is not valid JSON: {e}")
            if not isinstance(parsed_h, dict):
                raise ValueError(
                    "ROCINSIGHT_LLM_PRIVATE_HEADERS must be a JSON object of header "
                    'key/value pairs (e.g. {"X-My-Header": "value"}), '
                    f"got {type(parsed_h).__name__}"
                )
            headers.update(parsed_h)

        verify_ssl_env = os.environ.get("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", "1").lower()
        verify_ssl = verify_ssl_env not in ("0", "false", "no")
        http_client = None
        if not verify_ssl:
            try:
                import httpx as _httpx

                http_client = _httpx.Client(verify=False)
            except ImportError:
                import warnings

                warnings.warn(
                    "ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL=0 requested but httpx is not installed. "
                    "SSL verification will remain enabled. Run: pip install httpx",
                    stacklevel=2,
                )

        client_kwargs: dict = dict(
            api_key=key, base_url=base_url, default_headers=headers
        )
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        client = openai.OpenAI(**client_kwargs)

        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=4096,
                )
                return resp.choices[0].message.content or ""
            except openai.BadRequestError as e:
                if "max_completion_tokens" in str(e):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4096,
                    )
                    return resp.choices[0].message.content or ""
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Private LLM request failed ({base_url}). "
                    f"Check ROCINSIGHT_LLM_PRIVATE_URL, ROCINSIGHT_LLM_PRIVATE_HEADERS. "
                    f"Error: {exc}"
                ) from exc
        finally:
            if http_client is not None:
                http_client.close()

    # Mapping from Claude Code CLI model aliases to full Anthropic API model IDs.
    # Used by the ANTHROPIC_API_KEY fallback path.
    _CLAUDE_CODE_ALIAS_MAP: Dict[str, str] = {
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
        "haiku": "claude-haiku-4-5",
    }

    def _call_claude_code(self, system_prompt: str, user_prompt: str) -> str:
        """Call the ``claude-code`` provider with a two-tier auth chain.

        Priority:
        1. ``claude-agent-sdk`` with ``ANTHROPIC_API_KEY`` — direct Anthropic API
           call via the SDK, no CLI required.
        2. ``claude -p`` subprocess — falls back to the locally installed Claude
           Code CLI, which uses stored OAuth/session credentials.  The system
           prompt is placed in a CLAUDE.md file so the CLI loads it as project
           context (avoids command-line argument-length limits).

        The model can be overridden with ``--llm-model`` (accepts Claude Code
        aliases: sonnet, opus, haiku, or a full model id).
        """
        model = self.model or os.environ.get("ROCINSIGHT_LLM_MODEL") or "sonnet"

        # ── Tier 1: Agent SDK with ANTHROPIC_API_KEY ─────────────────────────
        try:
            result = self._call_claude_code_api_fallback(system_prompt, user_prompt, model)
            if result:
                return result
        except LLMAuthenticationError:
            pass  # No API key — fall through to CLI
        except (LLMRateLimitError, AnalysisError):
            raise  # Propagate real errors upward

        # ── Tier 2: claude -p subprocess (stored CLI credentials) ────────────
        try:
            cli_result = self._call_claude_cli_subprocess(system_prompt, user_prompt, model)
            if cli_result:
                return cli_result
        except AnalysisError as _cli_err:
            raise LLMAuthenticationError(
                f"Claude Code: no ANTHROPIC_API_KEY and CLI also failed ({_cli_err}). "
                "Set ANTHROPIC_API_KEY or ensure 'claude' CLI is authenticated."
            ) from _cli_err

        raise LLMAuthenticationError(
            "Claude Code: both API-key and CLI tiers returned empty. "
            "Set ANTHROPIC_API_KEY or ensure 'claude' CLI is working."
        )

    def _call_claude_cli_subprocess(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> str:
        """Call ``claude -p`` as a subprocess using Claude Code's stored credentials.

        This is a tier-2 fallback for when the Agent SDK returns empty but the
        ``claude`` CLI is available on PATH.  No API key is required — the CLI
        uses the same stored OAuth session as Claude Code.

        The ``system_prompt`` is written as a CLAUDE.md file in a temporary
        working directory so that Claude Code picks it up as project context —
        this avoids command-line length limits and mirrors how Claude Code
        naturally loads project instructions.

        Raises:
            AnalysisError: If the CLI is not found, times out, or returns an error.
        """
        import subprocess as _subprocess
        import json as _json
        import tempfile as _tempfile
        import pathlib as _pathlib

        with _tempfile.TemporaryDirectory(prefix="rocinsight_claude_") as _tmpdir:
            # Write the system prompt as CLAUDE.md so the CLI picks it up as
            # project-level context (no --system-prompt arg needed, no arg-length
            # limits).  Do NOT pass --bare: that flag prevents Claude Code from
            # reading CLAUDE.md.
            (_pathlib.Path(_tmpdir) / "CLAUDE.md").write_text(system_prompt, encoding="utf-8")

            cmd = [
                "claude",
                "-p", user_prompt,
                "--output-format", "json",
                "--no-session-persistence",
                "--model", model,
            ]
            try:
                proc = _subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=_tmpdir,
                )
            except FileNotFoundError:
                raise AnalysisError("claude CLI not found on PATH")
            except _subprocess.TimeoutExpired:
                raise AnalysisError("claude CLI timed out after 300 s")

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise AnalysisError(
                f"claude CLI exited with code {proc.returncode}"
                + (f": {detail}" if detail else "")
            )

        try:
            data = _json.loads(proc.stdout)
        except _json.JSONDecodeError:
            # Plain text output (older CLI versions): return as-is
            return proc.stdout.strip()

        if data.get("is_error"):
            raise AnalysisError(data.get("result") or "claude CLI reported an error")

        return (data.get("result") or "").strip()

    def _call_claude_code_api_fallback(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> str:
        """Fallback: call Anthropic API directly using ANTHROPIC_API_KEY."""
        try:
            import anthropic as _anthropic
        except ImportError:
            raise AnalysisError(
                "Neither claude-agent-sdk nor anthropic package is installed. "
                "Install one: pip install 'rocinsight[llm]'  or  pip install claude-agent-sdk"
            )

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMAuthenticationError(
                "Claude Code auth unavailable and ANTHROPIC_API_KEY is not set. "
                "Either install Claude Code (https://claude.ai/code) or set ANTHROPIC_API_KEY."
            )

        api_model = self._CLAUDE_CODE_ALIAS_MAP.get(model, model)
        client = _anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=api_model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except _anthropic.AuthenticationError:
            raise LLMAuthenticationError("ANTHROPIC_API_KEY is invalid or expired.")
        except _anthropic.RateLimitError:
            raise LLMRateLimitError("Anthropic API rate limit reached (claude-code fallback).")

        return next((b.text for b in response.content if b.type == "text"), "").strip()

    def summarize_source_file(self, filename: str, content: str) -> str:
        """Stage 1: summarize a GPU source file to its key patterns (local LLM)."""
        system = (
            "You are a GPU code analyst. Given a source file, extract the key information "
            "relevant to GPU performance: kernel definitions, memory access patterns, "
            "synchronization calls, and potential bottlenecks. "
            "Respond in plain text, max 600 words."
        )
        user = f"File: {filename}\n\n```\n{content[:8000]}\n```"
        if self.provider == "local":
            return self._call_local(system, user)
        elif self.provider == "anthropic":
            return self._call_anthropic(system, user)
        elif self.provider == "openai":
            return self._call_openai(system, user)
        elif self.provider == "private":
            return self._call_private(system, user)
        elif self.provider == "claude-code":
            return self._call_claude_code(system, user)
        return ""

    def annotate_profiling_plan(self, metadata: dict) -> str:
        """Annotate profiling plan metadata with LLM advice (no source text)."""
        import json as _json

        system = (
            "You are an expert AMD GPU performance analyst. "
            "Given a structured profiling plan (no source code), "
            "explain what the profiling commands are measuring and why, "
            "and suggest any adjustments. Be concise (max 200 words)."
        )
        user = f"Profiling plan metadata:\n{_json.dumps(metadata, indent=2)}"
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        elif self.provider == "openai":
            return self._call_openai(system, user)
        elif self.provider == "local":
            return self._call_local(system, user)
        elif self.provider == "private":
            return self._call_private(system, user)
        elif self.provider == "claude-code":
            return self._call_claude_code(system, user)
        return ""

    def _sanitize_source_data(self, source_result: Any) -> Dict[str, Any]:
        """
        Sanitize SourceAnalysisResult before sending to LLM.

        Privacy rules for source mode:
        - Kernel names → [KERNEL_1], [KERNEL_2], etc.
        - File paths → [FILE_1], [FILE_2], etc.
        - Pattern IDs, severities, categories, counts → kept (not sensitive)
        - Suggested counters → kept (generic AMD counter names)
        - Risk areas → kept (no proprietary info; redact any embedded paths)
        """
        # Build path → placeholder mapping
        kernel_name_map: Dict[str, str] = {}
        file_path_map: Dict[str, str] = {}

        for i, k in enumerate(source_result.detected_kernels, 1):
            name = k.get("name") if isinstance(k, dict) else k.name
            if name and name not in kernel_name_map:
                kernel_name_map[name] = f"[KERNEL_{i}]"

        all_files = list(
            {
                (k.get("file") if isinstance(k, dict) else k.file)
                for k in source_result.detected_kernels
            }
        )
        for i, fp in enumerate(sorted(all_files), 1):
            if fp:
                file_path_map[fp] = f"[FILE_{i}]"

        def _redact_file(f: str) -> str:
            return file_path_map.get(f, _redact_paths(f))

        def _redact_kernel(n: str) -> str:
            return kernel_name_map.get(n, n)

        sanitized_kernels = []
        for k in source_result.detected_kernels[:10]:
            name = k.get("name") if isinstance(k, dict) else k.name
            fpath = k.get("file") if isinstance(k, dict) else k.file
            line = k.get("line") if isinstance(k, dict) else k.line
            launch = k.get("launch_type") if isinstance(k, dict) else k.launch_type
            sanitized_kernels.append(
                {
                    "name": _redact_kernel(name or ""),
                    "file": _redact_file(fpath or ""),
                    "line": line,
                    "launch_type": launch,
                }
            )

        sanitized_patterns = []
        for p in source_result.detected_patterns:
            pd = (
                p
                if isinstance(p, dict)
                else {
                    "pattern_id": p.pattern_id,
                    "severity": p.severity,
                    "category": p.category,
                    "description": p.description,
                    "count": p.count,
                    "locations": p.locations,
                }
            )
            sanitized_patterns.append(
                {
                    "pattern_id": pd["pattern_id"],
                    "severity": pd["severity"],
                    "category": pd["category"],
                    "description": pd["description"],
                    "count": pd["count"],
                    # Redact locations (may contain file paths)
                    "locations": [
                        _redact_paths(loc) for loc in pd.get("locations", [])[:3]
                    ],
                }
            )

        sanitized_risks = [_redact_paths(r) for r in source_result.risk_areas]

        return {
            "programming_model": source_result.programming_model,
            "files_scanned": source_result.files_scanned,
            "kernel_count": source_result.kernel_count,
            "already_instrumented": source_result.already_instrumented,
            "roctx_marker_count": source_result.roctx_marker_count,
            "detected_kernels": sanitized_kernels,
            "detected_patterns": sanitized_patterns,
            "risk_areas": sanitized_risks,
            "suggested_counters": source_result.suggested_counters,
        }

    def _build_source_user_prompt(
        self,
        sanitized: Dict[str, Any],
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Build user prompt for Tier 0 source code analysis."""
        lines = []

        lines.append("CONTEXT: This is a PRE-PROFILING source code analysis (Tier 0).")
        lines.append("No runtime profiling data has been collected yet.")
        lines.append("Goal: produce a prioritized profiling plan.")
        lines.append("")

        lines.append("## Source Code Summary")
        lines.append(f"- Programming model: {sanitized['programming_model']}")
        lines.append(f"- Files scanned: {sanitized['files_scanned']}")
        lines.append(f"- GPU kernels found: {sanitized['kernel_count']}")
        lines.append(
            f"- Already instrumented with ROCTx: {sanitized['already_instrumented']}"
        )
        lines.append("")

        if sanitized["detected_kernels"]:
            lines.append("## Detected Kernels (names redacted)")
            for k in sanitized["detected_kernels"][:5]:
                lines.append(
                    f"  - {k['name']} ({k['launch_type']}) at {k['file']}:{k['line']}"
                )
            lines.append("")

        if sanitized["detected_patterns"]:
            lines.append("## Detected Patterns")
            for p in sanitized["detected_patterns"]:
                lines.append(
                    f"  - [{p['severity'].upper()}] {p['category']}: "
                    f"{p['description']} (count: {p['count']})"
                )
            lines.append("")

        if sanitized["risk_areas"]:
            lines.append("## Risk Areas")
            for r in sanitized["risk_areas"]:
                lines.append(f"  - {r}")
            lines.append("")

        lines.append("## Suggested Counters")
        lines.append(f"  {', '.join(sanitized['suggested_counters'])}")
        lines.append("")

        if custom_prompt:
            lines.append("## User Question")
            lines.append(custom_prompt)
            lines.append("")
            lines.append(
                "Please answer the user's question and provide a prioritized profiling plan "
                "based on the source code analysis above. Use the reference guide for "
                "AMD GPU profiling methodology. Use PLAIN TEXT only — no markdown headers."
            )
        else:
            lines.append(
                "Based on the source code analysis above, provide:\n"
                "1. Assessment of likely performance risks (2-3 sentences)\n"
                "2. Recommended first profiling step with rationale\n"
                "3. Top 3 things to look for when the first trace comes back\n"
                "4. Any source-level patterns that suggest architectural issues\n\n"
                "Use PLAIN TEXT only — no markdown headers (###, ##, #).\n"
                "Use ONLY current generation tools (rocprofv3, rocprof-compute, rocprof-sys)."
            )

        return "\n".join(lines)

    def suggest_optimizations(
        self,
        summaries: List[tuple],
        custom_prompt: str = "",
    ) -> str:
        """
        Request per-file GPU code optimization suggestions.

        Uses a focused system prompt (NOT the full profiling reference guide) so
        the LLM responds with concrete code-level advice, not profiling guidance.

        Each file's section in the response starts with "FILE: <filename>".

        Args:
            summaries: List of (filename, content_or_summary) pairs.
            custom_prompt: Optional extra instructions prepended to the user turn.

        Returns:
            LLM response text (plain text, FILE: sections per file).

        Raises:
            LLMAuthenticationError, LLMRateLimitError, AnalysisError
        """
        system = (
            "You are an expert AMD GPU performance engineer specializing in HIP/CUDA "
            "code optimization. Review the provided GPU source files and give concrete, "
            "actionable optimization suggestions.\n\n"
            "REQUIRED RESPONSE FORMAT:\n"
            "Start each file's section with exactly:\n"
            "FILE: <filename>\n"
            "Then list specific suggestions for that file.\n\n"
            "Focus on: memory coalescing, wave occupancy, unnecessary "
            "hipDeviceSynchronize calls, blocking hipMemcpy, MFMA usage, "
            "LDS utilization, loop structure, and kernel launch parameters.\n"
            "Be specific — reference actual patterns visible in the code.\n"
            "Use plain text only — no markdown headers or bold."
        )
        combined = "\n\n".join(
            f"=== {name} ===\n{content}" for name, content in summaries
        )
        user = (
            f"{custom_prompt}\n\nSource files:\n\n{combined}"
            if custom_prompt
            else combined
        )

        if self.verbose:
            print(
                f"[LLM] suggest_optimizations: {len(summaries)} file(s), "
                f"user prompt {len(user)} chars"
            )

        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        elif self.provider == "openai":
            return self._call_openai(system, user, max_tokens=3000)
        elif self.provider == "local":
            return self._call_local(system, user)
        elif self.provider == "private":
            return self._call_private(system, user)
        elif self.provider == "claude-code":
            return self._call_claude_code(system, user)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def analyze_source_with_llm(
        self,
        source_result: Any,
        custom_prompt: Optional[str] = None,
        context: Optional[AnalysisContext] = None,
    ) -> str:
        """
        Send Tier 0 source analysis to LLM for enhanced profiling guidance.

        Args:
            source_result: SourceAnalysisResult (from api.py)
            custom_prompt: Optional user question
            context: Optional AnalysisContext for guide section filtering.
                When provided, only guide sections relevant to the current
                analysis tier/bottleneck are included, reducing token cost.

        Returns:
            LLM-generated profiling guidance as plain text

        Raises:
            LLMAuthenticationError: Invalid API key
            LLMRateLimitError: API rate limit exceeded
        """
        sanitized = self._sanitize_source_data(source_result)
        system_prompt = self._build_system_prompt(context=context)
        user_prompt = self._build_source_user_prompt(sanitized, custom_prompt)

        if self.verbose:
            print(f"[LLM] Calling {self.provider} API for Tier 0 source analysis...")
            print(f"[LLM] User prompt length: {len(user_prompt)} chars")

        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "local":
            return self._call_local(system_prompt, user_prompt)
        elif self.provider == "private":
            return self._call_private(system_prompt, user_prompt)
        elif self.provider == "claude-code":
            return self._call_claude_code(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
