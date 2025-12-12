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

import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class GPUInfo:
    """Information about detected GPU(s)

    Attributes:
        available: Whether any GPU is available
        architectures: List of GPU architectures
        device_count: Number of GPUs detected
        is_navi: Whether the GPU is a NAVI architecture
        is_mi300: Whether the GPU is MI300 series
    """

    available: bool
    architectures: list[str]
    device_count: int
    is_navi: bool
    is_mi300: bool

    @property
    def rocm_events_for_test(self) -> str:
        """Get appropriate ROCm events for testing based on architecture."""
        if self.is_navi:
            return "SQ_WAVES"
        return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0"

    @property
    def counter_names(self) -> list[str]:
        """Get counter names for validation based on architecture"""
        if self.is_navi:
            return ["SQ_WAVES"]
        return ["GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "TA_TA_BUSY"]
    
    @property
    def expected_counter_files(self) -> list[str]:
        """Get expected counter output files based on architecture."""
        return [f"rocprof-device-0-{name}.txt" for name in self.counter_names]

@lru_cache(maxsize=1)
def detect_gpu() -> GPUInfo:
    """Detect available AMD GPUs and their capabilities.

    Uses rocminfo and amd-smi to gather GPU information. Results are cached
    for the lifetime of the process.

    Returns:
        GPUInfo with detected GPU information
    """
    architectures: list[str] = []
    device_count = 0

    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        try:
            result = subprocess.run(
                [rocminfo],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                arch_matches = re.findall(r"gfx([0-9A-Fa-f]+)", result.stdout)
                architectures = list(set(f"gfx{m}" for m in arch_matches))
                device_count = len(architectures)
        except (subprocess.TimeoutExpired, OSError):
            pass

    navi = any(is_navi_architecture(arch) for arch in architectures)
    if (not navi):
        mi300 = any(is_mi300_architecture(arch) for arch in architectures)
    else:
        mi300 = False

    return GPUInfo(
        available = device_count > 0,
        architectures = sorted(architectures),
        device_count = device_count,
        is_navi = navi,
        is_mi300 = mi300,
    )

def is_navi_architecture(arch: str) -> bool:
    """Check if an architecture string represents NAVI GPU.

    NAVI includes gfx10xx, gfx11xx, and gfx12xx architectures.

    Args:
        arch: Architecture string (e.g., 'gfx1100', 'gfx90a')

    Returns:
        True if NAVI architecture
    """
    navi_matches = re.match(r"gfx(10|11|12)[A-Fa-f0-9][A-Fa-f0-9]", arch)
    return navi_matches is not None

def is_mi300_architecture(arch: str) -> bool:
    """Detect if the GPU architecture is MI300 series.

    Args:
        arch: Architecture string (e.g., 'gfx940', 'gfx94a')

    Returns:
        True if MI300 architecture
    """
    mi300_matches = re.match(r"gfx9[4-9][0-9A-Fa-f]", arch)
    return mi300_matches is not None
