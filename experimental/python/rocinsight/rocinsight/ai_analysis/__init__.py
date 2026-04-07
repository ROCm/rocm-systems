#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
AI Analysis Module for rocinsight

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
    from rocinsight.ai_analysis import analyze_database

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

from .llm_analyzer import LLMAnalyzer, AnalysisContext, load_reference_guide, PROVIDER_REGISTRY
from .llm_conversation import LLMConversation
from .source_analyzer import SourceAnalyzer


def _get_interactive():
    from .interactive import InteractiveSession, SessionStore, SessionData

    return InteractiveSession, SessionStore, SessionData


def __getattr__(name):
    if name in ("InteractiveSession", "SessionStore", "SessionData"):
        InteractiveSession, SessionStore, SessionData = _get_interactive()
        # Cache in module globals to avoid repeated import on subsequent accesses
        import sys

        mod = sys.modules[__name__]
        mod.InteractiveSession = InteractiveSession
        mod.SessionStore = SessionStore
        mod.SessionData = SessionData
        return getattr(mod, name)
    raise AttributeError(f"module 'rocinsight.ai_analysis' has no attribute {name!r}")


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
    # Interactive session
    "InteractiveSession",
    "SessionStore",
    "SessionData",
    # LLM integration
    "LLMAnalyzer",
    "AnalysisContext",
    "LLMConversation",
    "load_reference_guide",
    "PROVIDER_REGISTRY",
    # Source analysis
    "SourceAnalyzer",
]

__version__ = "0.1.0"
