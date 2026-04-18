#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Thin LLMAnalyzer shell retained for backward-compat imports.

The LLM entry points live in the agent runtime (perfxpert/agents/runtime.py)
and in individual provider adapters (perfxpert/providers/<name>_model.py).
This file exists so third-party code importing `from perfxpert.ai_analysis.llm_analyzer import LLMAnalyzer`
doesn't break. The class is deprecated and will be removed in vX.Y+2.
"""

import warnings
from typing import Optional


class LLMAnalyzer:
    """DEPRECATED — use perfxpert.agents.runtime.create_session instead."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        warnings.warn(
            "LLMAnalyzer is deprecated; use perfxpert.agents.runtime.build_session instead. "
            "Will be removed in vX.Y+2.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.provider = provider
        self.api_key = api_key
