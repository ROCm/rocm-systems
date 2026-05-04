# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HIP-backed ``cuda.core`` shim shipped by nccl4py.

Re-exports :mod:`nccl._hip_compat.cuda_core_shim` under the ``cuda.core``
namespace so ROCm hosts can ``import cuda.core`` without the upstream
``cuda-core`` / ``cuda-bindings`` / ``nvidia-*`` packages. Importing
eagerly loads hip-python.
"""

from __future__ import annotations

import sys
from types import ModuleType

# Eager imports: pulling cuda.core means HIP is wanted now.
from nccl._hip_compat.cuda_core_shim._device import Device
from nccl._hip_compat.cuda_core_shim._stream import Stream
from nccl._hip_compat.cuda_core_shim._memory import Buffer, MemoryResource
from nccl._hip_compat.cuda_core_shim.typing import IsStreamT, DevicePointerT
from nccl._hip_compat.cuda_core_shim import _system as _system_mod
from nccl._hip_compat.cuda_core_shim import utils as _utils_mod

__all__ = [
    "Device",
    "Stream",
    "Buffer",
    "MemoryResource",
    "IsStreamT",
    "DevicePointerT",
    "system",
    "utils",
]


# `cuda.core.system` and `cuda.core.utils` re-expose the underlying shim
# modules. Importing `from cuda.core import system, utils` and
# `from cuda.core.utils import StridedMemoryView` both resolve.
system = _system_mod
utils = _utils_mod

sys.modules.setdefault(__name__ + ".system", _system_mod)
sys.modules.setdefault(__name__ + ".utils", _utils_mod)


def _make_passthrough(name: str, *, is_pkg: bool = False, **attrs) -> ModuleType:
    mod = ModuleType(name)
    if is_pkg:
        mod.__path__ = []  # type: ignore[attr-defined]
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Fallback paths used by some nccl/core typing imports:
#     from cuda.core._stream import IsStreamT
#     from cuda.core._memory._buffer import DevicePointerT
sys.modules.setdefault(
    __name__ + "._stream",
    _make_passthrough(__name__ + "._stream", IsStreamT=IsStreamT, Stream=Stream),
)

sys.modules.setdefault(
    __name__ + "._memory",
    _make_passthrough(
        __name__ + "._memory",
        is_pkg=True,
        Buffer=Buffer,
        MemoryResource=MemoryResource,
        DevicePointerT=DevicePointerT,
    ),
)

sys.modules.setdefault(
    __name__ + "._memory._buffer",
    _make_passthrough(
        __name__ + "._memory._buffer",
        Buffer=Buffer,
        MemoryResource=MemoryResource,
        DevicePointerT=DevicePointerT,
    ),
)
