#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
AI Analysis Module for rocpd

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
    from rocpd.ai_analysis import analyze_database

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
    get_kernel_analysis,
    get_recommendations,
    validate_database,
    AnalysisResult,
    OutputFormat,
)

from .exceptions import (
    AnalysisError,
    DatabaseNotFoundError,
    DatabaseCorruptedError,
    MissingDataError,
    UnsupportedGPUError,
    LLMAuthenticationError,
    LLMRateLimitError,
    ReferenceGuideNotFoundError,
)

__all__ = [
    # Main API functions
    "analyze_database",
    "analyze_database_to_json",
    "get_kernel_analysis",
    "get_recommendations",
    "validate_database",
    # Data classes
    "AnalysisResult",
    "OutputFormat",
    # Exceptions
    "AnalysisError",
    "DatabaseNotFoundError",
    "DatabaseCorruptedError",
    "MissingDataError",
    "UnsupportedGPUError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "ReferenceGuideNotFoundError",
]

__version__ = "0.1.0"
