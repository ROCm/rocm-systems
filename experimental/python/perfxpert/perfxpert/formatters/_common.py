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

    _PERFXPERT_VERSION = _pkg_version("perfxpert")
except Exception:
    _PERFXPERT_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)


# Stable IDs for known recommendation categories.
_CATEGORY_IDS = {
    "Low Occupancy": "ROCPD-OCCUPANCY-001",
    "GPU Utilization": "ROCPD-UTILIZATION-001",
    "Memory Transfer": "ROCPD-MEMCPY-001",
    "API Overhead": "ROCPD-API-001",
    "Compute Bottleneck": "ROCPD-COMPUTE-001",
    "Kernel Hotspot": "ROCPD-HOTSPOT-001",
    "Compute-Bound Kernel": "ROCPD-COMPUTE-BOUND-001",
    "Mixed Bottleneck Kernel": "ROCPD-MIXED-001",
    "Memory-Bound Kernel": "ROCPD-MEMORY-BOUND-001",
    "Warmup": "ROCPD-WARMUP-001",
    "Launch Overhead": "ROCPD-LAUNCH-001",
    "Launch Efficiency": "ROCPD-LAUNCH-EFFICIENCY-001",
    "Memory Bandwidth": "ROCPD-MEMBW-001",
    "Performance": "ROCPD-INFO-001",
}


def _communication_capture_incomplete(communication):
    if not communication:
        return False
    summary = communication.get("summary", {}) or {}
    return bool(
        communication.get("capture_incomplete", False)
        or summary.get("capture_incomplete", False)
    )


def _communication_has_measured_metrics(communication):
    if not communication:
        return False
    summary = communication.get("summary", {}) or {}
    if "measured_metrics_available" in summary:
        return bool(summary.get("measured_metrics_available"))

    for collective in communication.get("collectives") or []:
        try:
            msg_bytes = float(collective.get("msg_bytes", 0) or 0)
            duration_ns = float(collective.get("duration_ns", 0) or 0)
        except (TypeError, ValueError):
            continue
        if msg_bytes > 0 and duration_ns > 0:
            return True
    return False


def _communication_event_count(communication):
    if not communication:
        return 0
    summary = communication.get("summary", {}) or {}
    collectives = communication.get("collectives") or []
    if (
        _communication_capture_incomplete(communication)
        and not _communication_has_measured_metrics(communication)
    ):
        return int(
            summary.get("fallback_kernel_count")
            or summary.get("op_count")
            or len(collectives)
            or 0
        )
    return int(summary.get("op_count") or len(collectives) or 0)


def _should_render_communication(communication):
    if not communication:
        return False
    return bool(
        communication.get("collectives")
        or _communication_capture_incomplete(communication)
    )
