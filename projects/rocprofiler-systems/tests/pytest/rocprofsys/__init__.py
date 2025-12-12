# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""
rocprofsys testing utilities package.

Provides reusable components for testing rocprofiler-systems:
- Test runners (sampling, binary rewrite, runtime instrument)
- Output validators (perfetto, rocpd, timemory, regex patterns)
- Configuration management
- GPU and system detection utilities
"""


from .config import RocprofsysConfig, discover_install_config, discover_build_config

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
)
from .gpu import GPUInfo, detect_gpu, is_navi_architecture

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
    # GPU
    "GPUInfo",
    "detect_gpu",
    "is_navi_architecture",
    "is_mi300_architecture",
]
