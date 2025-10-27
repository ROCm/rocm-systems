###############################################################################
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
ROCm Profiling Data (rocpd) - Pure Python Implementation

This module provides a 100% pure Python implementation of rocpd functionality,
eliminating the need for version-specific compiled extensions.

Works with Python 3.6+ without any compilation required.
"""

import sys
import os

# Ensure sqlite3 is available (built-in to Python)
try:
    import sqlite3

    sqlite3.connect(":memory:")  # Test if sqlite3 is available
except Exception as e:
    print(f"ERROR: sqlite3 not available: {e}", file=sys.stderr)
    print("Python must be compiled with sqlite3 support", file=sys.stderr)

# Import pure Python implementation of libpyrocpd
from . import libpyrocpd
from .importer import RocpdImportData

__all__ = [
    "connect",
    "execute",
    "read_agents",
    "read_nodes",
    "read_processes",
    "read_threads",
    "write_perfetto",
    "write_csv",
    "write_otf2",
    "RocpdImportData",
    "version_info",
]

version_info = {
    "version": "@PROJECT_VERSION@",
    "major": int("@PROJECT_VERSION_MAJOR@"),
    "minor": int("@PROJECT_VERSION_MINOR@"),
    "patch": int("@PROJECT_VERSION_PATCH@"),
    "git_revision": "@ROCPROFILER_SDK_GIT_REVISION@",
    "library_arch": "@CMAKE_LIBRARY_ARCHITECTURE@",
    "system_name": "@CMAKE_SYSTEM_NAME@",
    "system_processor": "@CMAKE_SYSTEM_PROCESSOR@",
    "system_version": "@CMAKE_SYSTEM_VERSION@",
    "compiler_id": "@CMAKE_CXX_COMPILER_ID@",
    "compiler_version": "@CMAKE_CXX_COMPILER_VERSION@",
    "rocm_version": "@rocm_version_FULL_VERSION@",
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "binding_type": "pure_python",
}


def format_path(path, tag=os.path.basename(sys.executable)):
    """Format path with variable substitution"""
    return libpyrocpd.format_path(path, tag)


def connect(input, *args, **kwargs):
    """
    Connect to rocpd database(s).

    Args:
        input: Path to database file, list of paths, or existing connection

    Returns:
        RocpdImportData instance
    """
    return RocpdImportData(input, *args, **kwargs)


def execute(data, *args, **kwargs):
    """Execute SQL query on database"""
    return data.execute(*args, **kwargs)


def read_agents(data, condition=""):
    """Read agent information from database"""
    return libpyrocpd.read_agents(data, condition)


def read_nodes(data, condition=""):
    """Read node information from database"""
    return libpyrocpd.read_nodes(data, condition)


def read_processes(data, condition=""):
    """Read process information from database"""
    return libpyrocpd.read_processes(data, condition)


def read_threads(data, condition=""):
    """Read thread information from database"""
    return libpyrocpd.read_threads(data, condition)


def write_perfetto(connection, config=None, **kwargs):
    """
    Write Perfetto pftrace output file

    Args:
        connection (rocpd.RocpdImportData):
            rocPD instance of database connection(s)
        config (rocpd.output_config.output_config):
            Output specification

    Returns:
        bool: returns True if successful
    """
    from . import output_config

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    return libpyrocpd.write_perfetto(connection, config)


def write_csv(connection, config=None, **kwargs):
    """
    Write CSV output file(s)

    Args:
        connection (rocpd.RocpdImportData):
            rocPD instance of database connection(s)
        config (rocpd.output_config.output_config):
            Output specification

    Returns:
        bool: returns True if successful
    """
    from . import csv as csv_module

    config_obj = None
    if config is None:
        from . import output_config

        config_obj = output_config.output_config(**kwargs)
    else:
        config_obj = config.update(**kwargs)

    # Use the pure Python CSV writer
    return csv_module.write_csv(connection, config_obj)


def write_otf2(connection, config=None, **kwargs):
    """
    Write OTF2 output file

    Args:
        connection (rocpd.RocpdImportData):
            rocPD instance of database connection(s)
        config (rocpd.output_config.output_config):
            Output specification

    Returns:
        bool: returns True if successful
    """
    from . import output_config

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    return libpyrocpd.write_otf2(connection, config)
