# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""readers subpackage — exports all four format reader classes.

Each import is guarded by try/except ImportError so this init works even when
individual reader modules are not yet implemented (incremental plan delivery).
"""
from __future__ import annotations

try:
    from rocprofsys_validator.readers.rocpd import RocpdReader
except ImportError:
    RocpdReader = None  # type: ignore[assignment,misc]

try:
    from rocprofsys_validator.readers.perfetto import PerfettoReader
except ImportError:
    PerfettoReader = None  # type: ignore[assignment,misc]

try:
    from rocprofsys_validator.readers.timemory import TimemoryReader
except ImportError:
    TimemoryReader = None  # type: ignore[assignment,misc]

try:
    from rocprofsys_validator.readers.timemory_json import TimemoryJsonReader
except ImportError:
    TimemoryJsonReader = None  # type: ignore[assignment,misc]

try:
    from rocprofsys_validator.readers.causal import CausalReader
except ImportError:
    CausalReader = None  # type: ignore[assignment,misc]

try:
    from rocprofsys_validator.readers.rocprofiler_json import RocprofilerJsonReader
except ImportError:
    RocprofilerJsonReader = None  # type: ignore[assignment,misc]

try:
    from rocprofsys_validator.readers.unified_memory import UnifiedMemoryReader
except ImportError:
    UnifiedMemoryReader = None  # type: ignore[assignment,misc]

__all__ = [
    "PerfettoReader",
    "RocpdReader",
    "TimemoryReader",
    "TimemoryJsonReader",
    "CausalReader",
    "RocprofilerJsonReader",
    "UnifiedMemoryReader",
]
