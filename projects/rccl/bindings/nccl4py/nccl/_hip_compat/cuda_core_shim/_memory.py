# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.Buffer`` / ``cuda.core.MemoryResource`` shim, ported from cuda-core 0.3.2.

``Buffer`` surface used by ``nccl/core``::

    Buffer.from_handle(ptr, size, mr=None) -> Buffer
    Buffer.allocate(size, stream=None)     -> Buffer
    buf.handle / buf.size / buf.memory_resource
    buf.is_device_accessible / buf.is_host_accessible / buf.device_id
    buf.close(stream=None)
    buf.__dlpack__() / buf.__dlpack_device__()

:class:`MemoryResource` is an :class:`abc.ABC`.
``nccl.core.memory.NcclMemoryResource`` subclasses it and implements
``allocate`` / ``deallocate`` / ``is_device_accessible`` /
``is_host_accessible`` / ``device_id``.
"""

from __future__ import annotations

import abc
import weakref
from typing import Optional, Tuple

from ._hip import check_hip, hip
from .typing import DevicePointerT  # noqa: F401  (re-exported for fallback)


class Buffer:
    """Handle to allocated GPU memory.

    Mirrors cuda-core 0.3.2's ``Buffer``: direct construction is forbidden
    (use :meth:`from_handle`, :meth:`allocate`, or any
    :class:`MemoryResource` subclass). When a :class:`MemoryResource` is
    associated, ``close()`` returns the allocation to that resource;
    when the resource is ``None`` the buffer treats the pointer as
    externally-owned and is a no-op on close.
    """

    class _MNFF:
        """Members Needed For Finalize: separate object so weakref-finalize can
        safely call ``close()`` after the Buffer has been garbage-collected."""

        __slots__ = ("ptr", "size", "mr")

        def __init__(self, buffer_obj, ptr, size, mr):
            self.ptr = ptr
            self.size = size
            self.mr = mr
            weakref.finalize(buffer_obj, self.close)

        def close(self, stream=None):
            if self.ptr and self.mr is not None:
                self.mr.deallocate(self.ptr, self.size, stream)
                self.ptr = 0
                self.mr = None

    __slots__ = ("__weakref__", "_mnff")

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "Buffer objects cannot be instantiated directly. " "Please use MemoryResource APIs."
        )

    @classmethod
    def _init(cls, ptr, size, mr=None) -> "Buffer":
        self = super().__new__(cls)
        self._mnff = Buffer._MNFF(self, ptr, size, mr)
        return self

    @classmethod
    def from_handle(cls, ptr, size, mr=None) -> "Buffer":
        """Wrap an existing device pointer in a Buffer.

        ``mr`` is the resource that owns the allocation; close() will
        delegate to ``mr.deallocate``. Pass ``mr=None`` for borrowed
        pointers (close becomes a no-op).
        """
        return cls._init(ptr, size, mr)

    @classmethod
    def allocate(cls, size: int, stream=None) -> "Buffer":
        """Allocate ``size`` bytes on the current HIP device.

        Convenience entry point backed by a per-device singleton
        :class:`_HipMallocResource` (raw ``hipMalloc`` / ``hipFree``).
        Production code in nccl uses :class:`NcclMemoryResource`
        (``nccl.core.memory``) instead.
        """
        if int(size) <= 0:
            raise ValueError(f"size must be positive, got {size}")
        return _hip_default_resource().allocate(int(size), stream)

    def close(self, stream=None) -> None:
        self._mnff.close(stream)

    @property
    def handle(self):
        return self._mnff.ptr

    @property
    def size(self) -> int:
        return self._mnff.size

    @property
    def memory_resource(self):
        return self._mnff.mr

    @property
    def is_device_accessible(self) -> bool:
        if self._mnff.mr is not None:
            return self._mnff.mr.is_device_accessible
        return True

    @property
    def is_host_accessible(self) -> bool:
        if self._mnff.mr is not None:
            return self._mnff.mr.is_host_accessible
        return False

    @property
    def device_id(self) -> int:
        if self._mnff.mr is not None:
            return self._mnff.mr.device_id
        from ._device import _hip_current_device

        return _hip_current_device()

    def __dlpack__(
        self,
        *,
        stream: Optional[int] = None,
        max_version: Optional[Tuple[int, int]] = None,
        dl_device=None,
        copy: Optional[bool] = None,
    ):
        # Phase 5.3 fills this in (cuda-core 0.3.2 _memory.py / _dlpack.pyx port).
        from ._dlpack import make_py_capsule  # noqa: F401  (lazy until 5.3)

        if dl_device is not None:
            raise BufferError("dl_device override is not supported")
        if copy is True:
            raise BufferError("copy=True is not supported")
        versioned = max_version is not None and tuple(max_version) >= (1, 0)
        return make_py_capsule(self, versioned)

    def __dlpack_device__(self) -> Tuple[int, int]:
        from ._dlpack import DLDeviceType

        d_h = (bool(self.is_device_accessible), bool(self.is_host_accessible))
        if d_h == (True, False):
            return (DLDeviceType.kROCM, self.device_id)
        if d_h == (True, True):
            return (DLDeviceType.kROCMHost, 0)
        if d_h == (False, True):
            return (DLDeviceType.kCPU, 0)
        raise BufferError("buffer is neither device-accessible nor host-accessible")


class MemoryResource(abc.ABC):
    """Abstract base class for memory resources.

    Subclasses must implement ``allocate``, ``deallocate``,
    ``is_device_accessible``, ``is_host_accessible``, and ``device_id``.
    Mirrors cuda-core 0.3.2's MemoryResource so existing nccl-side
    subclasses (``NcclMemoryResource``) work unchanged.
    """

    __slots__ = ("_handle",)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abc.abstractmethod
    def allocate(self, size: int, stream=None) -> Buffer: ...

    @abc.abstractmethod
    def deallocate(self, ptr, size: int, stream=None) -> None: ...

    @property
    @abc.abstractmethod
    def is_device_accessible(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_host_accessible(self) -> bool: ...

    @property
    @abc.abstractmethod
    def device_id(self) -> int: ...


class _HipMallocResource(MemoryResource):
    """Default device memory resource backed by raw hipMalloc / hipFree.

    Used by :meth:`Buffer.allocate` as a drop-in path. NCCL's own
    allocator (``ncclMemAlloc``) lives in :mod:`nccl.core.memory` and
    has its own resource subclass.
    """

    __slots__ = ("_dev_id",)

    def __init__(self, device_id: Optional[int] = None):
        self._handle = None
        if device_id is None:
            from ._device import _hip_current_device

            device_id = _hip_current_device()
        self._dev_id = int(device_id)

    def allocate(self, size: int, stream=None) -> Buffer:
        ptr = int(check_hip(hip.hipMalloc(int(size)), f"hipMalloc({size})"))
        return Buffer._init(ptr, int(size), self)

    def deallocate(self, ptr, size: int, stream=None) -> None:
        if not ptr:
            return
        check_hip(hip.hipFree(int(ptr)), "hipFree")

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._dev_id


_default_resource_cache: dict[int, _HipMallocResource] = {}


def _hip_default_resource() -> _HipMallocResource:
    """Per-device singleton :class:`_HipMallocResource` backing
    :meth:`Buffer.allocate`."""
    from ._device import _hip_current_device

    dev = _hip_current_device()
    mr = _default_resource_cache.get(dev)
    if mr is None:
        mr = _HipMallocResource(dev)
        _default_resource_cache[dev] = mr
    return mr
