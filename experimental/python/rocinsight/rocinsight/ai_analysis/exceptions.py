#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Exception classes for AI analysis module.
"""

from typing import List, Optional


class AnalysisError(Exception):
    """Base exception for AI analysis errors"""

    pass


class DatabaseNotFoundError(AnalysisError):
    """Database file not found"""

    pass


class DatabaseCorruptedError(AnalysisError):
    """Database schema invalid or corrupted"""

    pass


class MissingDataError(AnalysisError):
    """Required data missing from database"""

    def __init__(self, message: str, missing_tables: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_tables = missing_tables or []


class UnsupportedGPUError(AnalysisError):
    """GPU architecture not supported"""

    def __init__(self, message: str, gpu_arch: Optional[str] = None):
        super().__init__(message)
        self.gpu_arch = gpu_arch


class LLMAuthenticationError(AnalysisError):
    """LLM API authentication failed"""

    pass


class LLMRateLimitError(AnalysisError):
    """LLM API rate limit exceeded"""

    pass


class AnalysisTimeoutError(AnalysisError):
    """Analysis took too long"""

    pass


class ReferenceGuideNotFoundError(AnalysisError):
    """LLM reference guide file not found"""

    def __init__(self, attempted_paths: List[str]):
        paths_str = "\n  - ".join(attempted_paths)
        super().__init__(
            f"LLM reference guide not found. Attempted locations:\n  - {paths_str}\n"
            "This file is required for LLM-enhanced analysis.\n"
            "See documentation for how to create or restore this file."
        )
        self.attempted_paths = attempted_paths


class SourceDirectoryNotFoundError(AnalysisError):
    """Source code directory not found or not a directory"""

    pass


class SourceAnalysisError(AnalysisError):
    """Error during source code analysis"""

    pass
