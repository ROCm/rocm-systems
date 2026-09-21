# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.system`` shim: only the surface ``nccl/core`` uses.

``nccl/core/communicator.py`` imports ``from cuda.core import system``
and calls :func:`system.get_num_devices` exactly once. Every other
``System`` attribute that cuda-core 0.3.2 ships (``driver_version``,
``devices``, ...) is intentionally left out.
"""

from __future__ import annotations

from ._hip import check_hip, hip


def get_num_devices() -> int:
    """Return the number of HIP devices visible to the current process."""
    return int(check_hip(hip.hipGetDeviceCount(), "hipGetDeviceCount"))
