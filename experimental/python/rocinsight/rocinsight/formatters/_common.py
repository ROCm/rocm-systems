###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
Shared helpers used by multiple formatter modules.
"""

try:
    from importlib.metadata import version as _pkg_version

    _ROCINSIGHT_VERSION = _pkg_version("rocinsight")
except Exception:
    _ROCINSIGHT_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)


# Stable IDs for known recommendation categories.
_CATEGORY_IDS = {
    "Low Occupancy": "ROCPD-OCCUPANCY-001",
    "GPU Utilization": "ROCPD-UTILIZATION-001",
    "Memory Transfer": "ROCPD-MEMCPY-001",
    "API Overhead": "ROCPD-API-001",
    "Compute Bottleneck": "ROCPD-COMPUTE-001",
    "Launch Overhead": "ROCPD-LAUNCH-001",
    "Launch Efficiency": "ROCPD-LAUNCH-EFFICIENCY-001",
    "Memory Bandwidth": "ROCPD-MEMBW-001",
    "Performance": "ROCPD-INFO-001",
}
