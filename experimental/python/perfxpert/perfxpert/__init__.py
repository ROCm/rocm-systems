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

import re
from importlib.metadata import PackageNotFoundError, version as _metadata_version
from pathlib import Path


def _source_tree_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "0+unknown"
    match = re.search(r"(?m)^version\s*=\s*\"([^\"]+)\"", text)
    return match.group(1) if match else "0+unknown"


try:
    __version__ = _metadata_version("perfxpert")
except PackageNotFoundError:
    __version__ = _source_tree_version()

__author__ = "Advanced Micro Devices, Inc."

from .connection import PerfxpertConnection, execute_statement, merge_sqlite_dbs

__all__ = [
    "__version__",
    "PerfxpertConnection",
    "execute_statement",
    "merge_sqlite_dbs",
]
