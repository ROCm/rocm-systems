#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
AI Analysis Module for perfxpert

This module provides AI-powered GPU performance analysis with optional
LLM enhancement. The analysis is guided by a user-modifiable reference
guide (the "fence") that ensures high-quality, actionable insights.

Key Features:
- Local-first analysis (always available, no internet required)
- Optional LLM enhancement (Anthropic Claude, OpenAI GPT)
- User-modifiable reference guide for customizing LLM behavior
- Data sanitization for privacy in LLM mode
- JSON, text, and markdown output formats

Usage:
    from perfxpert.ai_analysis import analyze_database

    result = analyze_database(
        database_path=Path("output.db"),
        enable_llm=True,
        llm_provider="anthropic"
    )

    print(result.summary.overall_assessment)
"""

from .api import (
    analyze_database,
    analyze_database_to_json,
    analyze_source,
    get_kernel_analysis,
    get_recommendations,
    validate_database,
    AnalysisResult,
    SourceAnalysisResult,
    OutputFormat,
)

from .exceptions import (
    AnalysisError,
    AnalysisTimeoutError,
    DatabaseNotFoundError,
    DatabaseCorruptedError,
    MissingDataError,
    UnsupportedGPUError,
    LLMAuthenticationError,
    LLMRateLimitError,
    ReferenceGuideNotFoundError,
    SourceDirectoryNotFoundError,
    SourceAnalysisError,
)

from .llm_analyzer import LLMAnalyzer  # deprecated stub
from .source_analyzer import SourceAnalyzer

# LLMConversation, InteractiveSession, WorkflowSession removed in Phase 6 — use perfxpert.agents.runtime
# load_reference_guide moved to perfxpert.providers._reference_guide
# PROVIDER_REGISTRY moved to agents.runtime (fallback in place)


__all__ = [
    # Main API functions
    "analyze_database",
    "analyze_database_to_json",
    "analyze_source",
    "get_kernel_analysis",
    "get_recommendations",
    "validate_database",
    # Data classes
    "AnalysisResult",
    "SourceAnalysisResult",
    "OutputFormat",
    # Exceptions
    "AnalysisError",
    "AnalysisTimeoutError",
    "DatabaseNotFoundError",
    "DatabaseCorruptedError",
    "MissingDataError",
    "UnsupportedGPUError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "ReferenceGuideNotFoundError",
    "SourceDirectoryNotFoundError",
    "SourceAnalysisError",
    # LLM integration
    "LLMAnalyzer",  # deprecated, retained for back-compat imports
    # Source analysis
    "SourceAnalyzer",
]

__version__ = "0.1.0"
