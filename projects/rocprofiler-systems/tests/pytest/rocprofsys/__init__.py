# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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
    discover_install_config,
    discover_build_config,
)

from .runners import (
    TestResult,
    BaselineRunner,
    SamplingRunner,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SysRunRunner,
)
from .validators import (
    ValidationResult,
    validate_perfetto_trace,
    validate_rocpd_database,
    validate_timemory_json,
    validate_causal_json,
    validate_file_exists,
    validate_regex,
)

from .gpu import (
    GPUInfo,
    get_rocminfo,
    detect_gpu,
    lookup_gpu_category,
    get_target_gpu_arch,
    get_offload_extractor,
)

__all__ = [
    # Config
    "RocprofsysConfig",
    "discover_build_config",
    "discover_install_config",
    # Runners
    "TestResult",
    "BaselineRunner",
    "SamplingRunner",
    "BinaryRewriteRunner",
    "RuntimeInstrumentRunner",
    "SysRunRunner",
    # Validators
    "ValidationResult",
    "validate_perfetto_trace",
    "validate_rocpd_database",
    "validate_timemory_json",
    "validate_causal_json",
    "validate_file_exists",
    "validate_regex",
    # GPU
    "GPUInfo",
    "get_rocminfo",
    "detect_gpu",
    "lookup_gpu_category",
    "get_target_gpu_arch",
    "get_offload_extractor",
]
