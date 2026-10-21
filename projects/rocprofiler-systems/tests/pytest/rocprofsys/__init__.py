# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
rocprofsys testing utilities package.

Provides reusable components for testing rocprofiler-systems:
- Test runners (sampling, binary rewrite, runtime instrument)
- Output validators (perfetto, rocpd, timemory, regex patterns)
- Configuration management
- GPU and system detection utilities
"""

from .config import (
    RocprofsysConfig,
    discover_build_config,
    discover_install_config,
)
from .environment import (
    TestEnvironment,
    TestEnvKind,
)
from .gpu import (
    GPUInfo,
    detect_gpu,
    get_offload_extractor,
    get_rocminfo,
    get_target_gpu_arch,
    get_xnack_support,
    lookup_gpu_category,
)
from .runners import (
    BaselineRunner,
    BinaryRewriteRunner,
    CausalRunner,
    PythonRunner,
    RuntimeInstrumentRunner,
    SamplingRunner,
    SysRunRunner,
    TestResult,
    safe_remove,
)
from .validators import (
    ValidationResult,
    validate_causal_json,
    validate_file_exists,
    validate_file_regex,
    validate_perfetto_trace,
    validate_regex,
    validate_rocpd_database,
    validate_timemory_json,
    validate_unified_memory_outputs,
)

__all__ = [
    # Config
    "RocprofsysConfig",
    "discover_build_config",
    "discover_install_config",
    # Environment
    "TestEnvironment",
    "TestEnvKind",
    # Runners
    "TestResult",
    "BaselineRunner",
    "SamplingRunner",
    "BinaryRewriteRunner",
    "RuntimeInstrumentRunner",
    "SysRunRunner",
    "CausalRunner",
    "PythonRunner",
    "safe_remove",
    # Validators
    "ValidationResult",
    "validate_perfetto_trace",
    "validate_rocpd_database",
    "validate_timemory_json",
    "validate_causal_json",
    "validate_unified_memory_outputs",
    "validate_file_exists",
    "validate_regex",
    "validate_file_regex",
    # GPU
    "GPUInfo",
    "get_rocminfo",
    "detect_gpu",
    "lookup_gpu_category",
    "get_target_gpu_arch",
    "get_offload_extractor",
    "get_xnack_support",
]
