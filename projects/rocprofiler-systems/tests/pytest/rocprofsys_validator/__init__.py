# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rocprofsys_validator — Validation framework for rocprof-sys profiling outputs.

Provides a composable, extensible API for validating Perfetto traces (.pftrace),
RocPD SQLite databases, and timemory JSON outputs, individually and in cross-format
comparisons.
"""

from rocprofsys_validator.core import AssertionBase, FormatReader, CheckResult
from rocprofsys_validator.readers import (
    PerfettoReader,
    RocpdReader,
    TimemoryReader,
    TimemoryJsonReader,
    CausalReader,
    RocprofilerJsonReader,
    UnifiedMemoryReader,
)
from rocprofsys_validator.readers.perfetto import ANYTHING
from rocprofsys_validator.expect import expect, expect_all
from rocprofsys_validator.registry import reader, register_validator, discover_validators
from rocprofsys_validator.baseline import assert_baseline
from rocprofsys_validator.correlation import (
    assert_kernel_correlation,
    assert_hip_correlation,
    assert_temporal_ordering,
    assert_timemory_perfetto_correlation,
    assert_record_count_parity,
)
from rocprofsys_validator.validator import (
    ProfilerOutputValidator,
    SupportsTimeline,
    SupportsCounters,
    SupportsCallTree,
    SupportsAntiPatterns,
)

__version__ = "0.1.0"
__all__ = [
    "CheckResult",
    "FormatReader",
    "AssertionBase",
    "PerfettoReader",
    "RocpdReader",
    "TimemoryReader",
    "TimemoryJsonReader",
    "CausalReader",
    "RocprofilerJsonReader",
    "UnifiedMemoryReader",
    "expect",
    "expect_all",
    "ANYTHING",
    "assert_baseline",
    "assert_kernel_correlation",
    "assert_hip_correlation",
    "assert_temporal_ordering",
    "assert_timemory_perfetto_correlation",
    "assert_record_count_parity",
    "reader",
    "register_validator",
    "discover_validators",
    "ProfilerOutputValidator",
    "SupportsTimeline",
    "SupportsCounters",
    "SupportsCallTree",
    "SupportsAntiPatterns",
]
