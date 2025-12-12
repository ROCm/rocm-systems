# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal HIP-backed shim for the ``cuda.core`` surface used by ``nccl/core``.

Ported (selectively) from cuda-core 0.3.2 (Apache-2.0 licensed):

    https://github.com/NVIDIA/cuda-python/tree/cuda-core-v0.3.2

Driver calls are rewritten on top of ``from hip import hip`` (hip-python),
so the resulting shim has no dependency on cuda-bindings, cuda-core,
``nvidia-*`` packages, or ``hip-python-as-cuda``.

Scope: exactly the symbols imported by ``nccl/core/*.py`` (``Device``,
``Stream``, ``Buffer``, ``MemoryResource``, ``system``, ``IsStreamT``,
``DevicePointerT``, ``StridedMemoryView``,
``args_viewable_as_strided_memory``). Anything outside that surface is
intentionally not provided.

Stop-gap: a future ROCm-native ``hip.core`` will make this layer
unnecessary; until then this package is registered under the
``cuda.core`` namespace via :func:`_register_as_cuda_core` from
``nccl/__init__.py``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable

__all__ = ["_register_as_cuda_core"]


def _make_lazy_module(
    name: str, getter: Callable[[str], Any], *, is_pkg: bool = False
) -> ModuleType:
    """Build a ModuleType whose attribute access is resolved by ``getter``."""
    mod = ModuleType(name)
    mod.__getattr__ = getter  # type: ignore[attr-defined]
    if is_pkg:
        mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def _resolve_cuda_core(name: str) -> Any:
    if name == "Device":
        from ._device import Device

        return Device
    if name == "Stream":
        from ._stream import Stream

        return Stream
    if name == "Buffer":
        from ._memory import Buffer

        return Buffer
    if name == "MemoryResource":
        from ._memory import MemoryResource

        return MemoryResource
    if name == "system":
        return sys.modules["cuda.core.system"]
    if name == "utils":
        return sys.modules["cuda.core.utils"]
    if name == "IsStreamT":
        from .typing import IsStreamT

        return IsStreamT
    if name == "DevicePointerT":
        from .typing import DevicePointerT

        return DevicePointerT
    raise AttributeError(f"module 'cuda.core' has no attribute {name!r} (HIP shim)")


def _resolve_cuda_core_system(name: str) -> Any:
    if name == "get_num_devices":
        from ._system import get_num_devices

        return get_num_devices
    raise AttributeError(f"module 'cuda.core.system' has no attribute {name!r} (HIP shim)")


def _resolve_cuda_core_utils(name: str) -> Any:
    if name == "StridedMemoryView":
        from ._memoryview import StridedMemoryView

        return StridedMemoryView
    if name == "args_viewable_as_strided_memory":
        from ._memoryview import args_viewable_as_strided_memory

        return args_viewable_as_strided_memory
    raise AttributeError(f"module 'cuda.core.utils' has no attribute {name!r} (HIP shim)")


def _resolve_cuda_core_stream(name: str) -> Any:
    # Fallback path used by nccl/core/typing.py (`from cuda.core._stream import IsStreamT`).
    if name == "IsStreamT":
        from .typing import IsStreamT

        return IsStreamT
    if name == "Stream":
        from ._stream import Stream

        return Stream
    raise AttributeError(f"module 'cuda.core._stream' has no attribute {name!r} (HIP shim)")


def _resolve_cuda_core_memory(name: str) -> Any:
    # Fallback path used by nccl/core/memory.py
    # (`from cuda.core._memory._buffer import DevicePointerT`).
    if name == "DevicePointerT":
        from .typing import DevicePointerT

        return DevicePointerT
    if name == "Buffer":
        from ._memory import Buffer

        return Buffer
    if name == "MemoryResource":
        from ._memory import MemoryResource

        return MemoryResource
    raise AttributeError(
        f"module 'cuda.core._memory(._buffer)' has no attribute {name!r} (HIP shim)"
    )


def _register_as_cuda_core() -> None:
    """Install this shim as the cuda.core namespace via sys.modules.

    Idempotent: a second call is a no-op. Must run before any
    ``from cuda.core import ...`` triggered by nccl.core.* or by
    user code that has imported nccl.
    """
    if "cuda.core" in sys.modules:
        return

    if "cuda" not in sys.modules:
        cuda_pkg = ModuleType("cuda")
        cuda_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["cuda"] = cuda_pkg
    else:
        cuda_pkg = sys.modules["cuda"]

    cuda_core = _make_lazy_module("cuda.core", _resolve_cuda_core, is_pkg=True)
    cuda_core_system = _make_lazy_module("cuda.core.system", _resolve_cuda_core_system)
    cuda_core_utils = _make_lazy_module("cuda.core.utils", _resolve_cuda_core_utils)
    cuda_core_stream = _make_lazy_module("cuda.core._stream", _resolve_cuda_core_stream)
    cuda_core_memory = _make_lazy_module(
        "cuda.core._memory", _resolve_cuda_core_memory, is_pkg=True
    )
    cuda_core_memory_buffer = _make_lazy_module(
        "cuda.core._memory._buffer", _resolve_cuda_core_memory
    )

    sys.modules["cuda.core"] = cuda_core
    sys.modules["cuda.core.system"] = cuda_core_system
    sys.modules["cuda.core.utils"] = cuda_core_utils
    sys.modules["cuda.core._stream"] = cuda_core_stream
    sys.modules["cuda.core._memory"] = cuda_core_memory
    sys.modules["cuda.core._memory._buffer"] = cuda_core_memory_buffer

    cuda_pkg.core = cuda_core  # type: ignore[attr-defined]
