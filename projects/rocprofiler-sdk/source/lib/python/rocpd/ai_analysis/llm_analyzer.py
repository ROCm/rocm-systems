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
from pathlib import Path
from typing import Optional, Dict, Any, List

# Regex to match Unix and Windows file paths that may appear in profiling data
_PATH_PATTERN = re.compile(
    r'(/home/[^\s,"\';>]+|/opt/[^\s,"\';>]+|/root/[^\s,"\';>]+|'
    r'/tmp/[^\s,"\';>]+|/var/[^\s,"\';>]+|[A-Za-z]:\\[^\s,"\';>]+)'
)


def _redact_paths(value: str) -> str:
    """Replace file system paths in a string with [REDACTED]."""
    return _PATH_PATTERN.sub("[REDACTED]", value)
from .exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    ReferenceGuideNotFoundError,
)


# Default location for the reference guide (relative to package installation)
# Users can override with ROCPD_LLM_REFERENCE_GUIDE environment variable
DEFAULT_REFERENCE_GUIDE_NAME = "llm-reference-guide.md"

# Default model names — override at runtime with ROCPD_LLM_MODEL env var
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4-turbo-preview"


def get_reference_guide_path() -> Path:
    """
    Get the path to the LLM reference guide.

    Priority order:
    1. ROCPD_LLM_REFERENCE_GUIDE environment variable
    2. Relative to this module (ai_analysis/share/)
    3. /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md

    Returns:
        Path to reference guide file

    Raises:
        ReferenceGuideNotFoundError: If guide file not found (lists all attempted paths)
    """
    attempted = []

    # Check environment variable first
    env_path = os.environ.get("ROCPD_LLM_REFERENCE_GUIDE")
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
        provider: str = "anthropic",  # "anthropic" or "openai"
        api_key: Optional[str] = None,
        reference_guide_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        """
        Initialize LLM analyzer.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            api_key: API key (if None, reads from environment)
            reference_guide_path: Path to reference guide (if None, uses default location)
            verbose: Enable verbose logging
        """
        self.provider = provider
        self.verbose = verbose
        self.api_key = api_key or self._get_api_key_from_env()

        # Load reference guide (the "fence")
        if reference_guide_path:
            self.reference_guide_path = reference_guide_path
        else:
            self.reference_guide_path = get_reference_guide_path()

        self.reference_guide = self._load_reference_guide()

        if self.verbose:
            print(f"[LLM] Loaded reference guide from: {self.reference_guide_path}")
            print(f"[LLM] Reference guide size: {len(self.reference_guide)} characters")

    def _get_api_key_from_env(self) -> str:
        """Get API key from environment"""
        if self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        if not key:
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
            raise ReferenceGuideNotFoundError(str(self.reference_guide_path))

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

        return sanitized

    def _build_system_prompt(self) -> str:
        """
        Build system prompt that includes the reference guide.

        This is where the "fence" is applied - the LLM gets the reference
        guide as context for every request, ensuring it follows the guidelines.

        Returns:
            System prompt with embedded reference guide
        """
        return f"""You are an expert GPU performance analyst specializing in AMD GPUs.

{self.reference_guide}

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
    ) -> str:
        """
        Build user prompt with profiling data.

        Args:
            analysis_data: Sanitized profiling data
            custom_prompt: Optional user question

        Returns:
            User prompt string
        """
        # Format data as structured text for LLM
        data_summary = self._format_data_for_llm(analysis_data)

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

    def _format_data_for_llm(self, data: Dict[str, Any]) -> str:
        """Format analysis data as readable text for LLM"""
        lines = []

        # GPU info
        if "gpu" in data:
            lines.append("## GPU Information")
            lines.append(f"- Name: {data['gpu'].get('name', 'Unknown')}")
            lines.append(f"- Architecture: {data['gpu'].get('arch', 'Unknown')}")
            lines.append("")

        # Execution breakdown
        if "execution_breakdown" in data:
            lines.append("## Execution Breakdown")
            breakdown = data["execution_breakdown"]
            lines.append(f"- Kernel Time: {breakdown.get('kernel_time_pct', 0):.1f}%")
            lines.append(
                f"- Memory Copy Time: {breakdown.get('memcpy_time_pct', 0):.1f}%"
            )
            lines.append(
                f"- API Overhead: {breakdown.get('api_overhead_pct', 0):.1f}%"
            )
            lines.append("")

        # Top kernels
        if "kernels" in data:
            lines.append("## Top Kernels")
            for kernel in data["kernels"][:5]:  # Top 5
                lines.append(f"- {kernel.get('kernel_id', 'Unknown')}")
                lines.append(
                    f"  - Time: {kernel.get('pct_total_time', 0):.1f}% of total"
                )
                lines.append(
                    f"  - Dispatches: {kernel.get('dispatch_count', 'N/A')}"
                )

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

        # Memory operations
        if "memory_ops" in data:
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

        # Data availability note
        lines.append("## Data Availability")
        if data.get("has_counters"):
            lines.append("✅ Hardware counters available (Tier 2 analysis possible)")
        else:
            lines.append(
                "⚠️  No hardware counters (Tier 1 trace analysis only)"
            )

        if data.get("has_pc_sampling"):
            lines.append("✅ PC sampling data available (Tier 3 analysis possible)")

        return "\n".join(lines)

    def analyze_with_llm(
        self, analysis_data: Dict[str, Any], custom_prompt: Optional[str] = None
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

        Returns:
            LLM-generated natural language analysis

        Raises:
            LLMAuthenticationError: Invalid API key
            LLMRateLimitError: API rate limit exceeded
        """
        # Sanitize data (privacy protection)
        sanitized_data = self._sanitize_data(analysis_data)

        # Build prompts (includes reference guide as "fence")
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(sanitized_data, custom_prompt)

        if self.verbose:
            print(f"[LLM] Calling {self.provider} API...")
            print(f"[LLM] System prompt length: {len(system_prompt)} chars")
            print(f"[LLM] User prompt length: {len(user_prompt)} chars")

        # Call appropriate LLM API
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            model = os.environ.get("ROCPD_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except Exception as e:
            raise AnalysisError(f"Anthropic API error: {e}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI GPT API"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        try:
            client = openai.OpenAI(api_key=self.api_key)

            model = os.environ.get("ROCPD_LLM_MODEL") or DEFAULT_OPENAI_MODEL
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
                    max_completion_tokens=4096,
                )
            except openai.BadRequestError as _br:
                if "max_completion_tokens" in str(_br):
                    response = client.chat.completions.create(
                        model=model,
                        messages=_messages,
                        max_tokens=4096,
                    )
                else:
                    raise

            return response.choices[0].message.content

        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except Exception as e:
            raise AnalysisError(f"OpenAI API error: {e}")


from .exceptions import AnalysisError  # Needed for _call methods