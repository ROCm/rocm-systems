###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

import sys
import os

try:
    import ctypes

    sqlite3lib = ctypes.CDLL("libsqlite3.so")
except Exception:
    pass

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
}


def format_path(path, tag=os.path.basename(sys.executable)):
    return libpyrocpd.format_path(path, tag)


def connect(input, *args, **kwargs):
    return RocpdImportData(input, *args, **kwargs)


def execute(data, *args, **kwargs):
    return data.execute(*args, **kwargs)


def read_agents(data, condition=""):
    return libpyrocpd.read_agents(data, condition)


def read_nodes(data, condition=""):
    return libpyrocpd.read_nodes(data, condition)


def read_processes(data, condition=""):
    return libpyrocpd.read_processes(data, condition)


def read_threads(data, condition=""):
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
    from . import output_config

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    return libpyrocpd.write_csv(connection, config)


def write_otf2(connection, config=None, **kwargs):
    """
    Write OTF@ output file

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
