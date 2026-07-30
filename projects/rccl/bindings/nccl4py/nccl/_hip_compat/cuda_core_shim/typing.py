# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type aliases / protocols mirrored from cuda.core 0.3.2.

``nccl/core`` uses these for static typing only (most references are
guarded by ``if TYPE_CHECKING:``). Runtime definitions are kept minimal
so that ``from cuda.core import IsStreamT, DevicePointerT`` (and the
underscore-prefixed fallback paths) resolve without pulling in any HIP
code.
"""

from __future__ import annotations

from typing import Protocol, Tuple, Union


class IsStreamT(Protocol):
    """Objects exposing the __cuda_stream__ protocol.

    Mirrors ``cuda.core._stream.IsStreamT``. The shim's :class:`Stream`
    implements this protocol; foreign stream objects (e.g. PyTorch
    streams) are accepted by ``Device.create_stream`` as long as they
    return ``(version, ptr)`` from ``__cuda_stream__()``.
    """

    def __cuda_stream__(self) -> Tuple[int, int]:  # pragma: no cover - protocol
        ...


# cuda.core types this as Union[CUdeviceptr, int, None]. We only ever see
# `int` (raw device pointer) on the HIP side, but nccl/core treats this as
# a typing-only alias, so a permissive Union is fine.
DevicePointerT = Union[int, None]
