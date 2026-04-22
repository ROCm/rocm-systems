#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
perfxpert — standalone AI-powered GPU performance analysis.

Reads rocprofiler-sdk trace databases (.db files) and provides
human-readable insights, bottleneck detection, and optimization
recommendations without any C++ library dependency.

CLI
---
    python -m perfxpert analyze -i trace.db
    perfxpert analyze -i trace.db --format json
"""

__version__ = "0.1.0"
__author__ = "Advanced Micro Devices, Inc."

from .connection import PerfxpertConnection, execute_statement, merge_sqlite_dbs

__all__ = [
    "__version__",
    "PerfxpertConnection",
    "execute_statement",
    "merge_sqlite_dbs",
]
