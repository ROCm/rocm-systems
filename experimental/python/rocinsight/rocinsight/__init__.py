#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
rocinsight — standalone AI-powered GPU performance analysis.

Reads rocprofiler-sdk trace databases (.db files) and provides
human-readable insights, bottleneck detection, and optimization
recommendations without any C++ library dependency.

Quick start
-----------
    from pathlib import Path
    from rocinsight.ai_analysis import analyze_database

    result = analyze_database(Path("trace.db"))
    print(result.summary.overall_assessment)

CLI
---
    python -m rocinsight analyze -i trace.db
    rocinsight analyze -i trace.db --format json
"""

__version__ = "0.1.0"
__author__ = "Advanced Micro Devices, Inc."

from .connection import RocinsightConnection, execute_statement, merge_sqlite_dbs

__all__ = [
    "__version__",
    "RocinsightConnection",
    "execute_statement",
    "merge_sqlite_dbs",
]
