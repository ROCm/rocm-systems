# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""NCCL Reshard handle lifecycle and reshard entry points."""

from __future__ import annotations

import threading

from nccl._extensions.bindings import nccl_m2n as _m2n_bindings
from nccl._extensions._runtime import NATIVE_CALL_LOCK
from nccl.core.typing import NcclInvalid, NcclStreamSpec
from nccl.m2n.config import Config
from nccl.m2n.tensor import DistTensor


def _extract_ptr(obj: object, *, attr: str, name: str) -> int:
    if isinstance(obj, int):
        return int(obj)
    value = getattr(obj, attr, None)
    if isinstance(value, int):
        return value
    raise TypeError(f"{name} must be an int or expose .{attr} as an int")


def _stream_ptr(stream: NcclStreamSpec | None) -> int:
    if stream is None:
        return 0
    if isinstance(stream, int):
        return int(stream)
    cuda_stream = getattr(stream, "cuda_stream", None)
    if isinstance(cuda_stream, int):
        return cuda_stream

    try:
        from nccl.core.cuda import get_stream_ptr

        return int(get_stream_ptr(stream))
    except (ImportError, ModuleNotFoundError):
        pass

    protocol = getattr(stream, "__cuda_stream__", None)
    if callable(protocol):
        value = protocol()
        if isinstance(value, tuple):
            return int(value[-1])
        return int(value)
    raise TypeError("stream must be None, an int, a torch CUDA stream, or expose __cuda_stream__")


def _validate_tensors(src: DistTensor, dst: DistTensor) -> None:
    if not isinstance(src, DistTensor) or not isinstance(dst, DistTensor):
        raise TypeError("src and dst must be DistTensor descriptors")
    if src.ndims != dst.ndims:
        raise NcclInvalid(f"src/dst ndims mismatch: {src.ndims} != {dst.ndims}")
    if src.dtype != dst.dtype:
        raise NcclInvalid(f"src/dst dtype mismatch: {src.dtype} != {dst.dtype}")


def _reshard(
    handle: int,
    comm: object,
    src: DistTensor,
    dst: DistTensor,
    stream: NcclStreamSpec | None = None,
) -> None:
    with NATIVE_CALL_LOCK:
        _validate_tensors(src, dst)
        src_prepared = src.as_binding()
        dst_prepared = dst.as_binding()
        _m2n_bindings.reshard(
            handle,
            _extract_ptr(comm, attr="ptr", name="comm"),
            int(src_prepared.struct.ptr),
            int(dst_prepared.struct.ptr),
            _stream_ptr(stream),
        )


def _reshard_with_window(
    handle: int,
    comm: object,
    window: object,
    src: DistTensor,
    dst: DistTensor,
    stream: NcclStreamSpec | None = None,
) -> None:
    with NATIVE_CALL_LOCK:
        _validate_tensors(src, dst)
        src_prepared = src.as_binding()
        dst_prepared = dst.as_binding()
        _m2n_bindings.reshard_with_window(
            handle,
            _extract_ptr(comm, attr="ptr", name="comm"),
            _extract_ptr(window, attr="handle", name="window"),
            int(src_prepared.struct.ptr),
            int(dst_prepared.struct.ptr),
            _stream_ptr(stream),
        )


class Handle:
    """NCCL Reshard runtime handle.

    Construct with :meth:`create` and release with :meth:`destroy`, or use it
    as a context manager. The handle wraps ``ncclM2nHandle_t`` and is invalid
    after destruction. Native calls from this package are serialized
    process-wide because M2N runtime state is shared across handles.

    CUDA work remains asynchronous. Synchronize user streams before calling
    :meth:`destroy` or leaving a handle context.
    """

    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)
        self._lock = threading.RLock()

    @classmethod
    def create(cls, config: Config | None = None) -> "Handle":
        with NATIVE_CALL_LOCK:
            config_ptr = 0
            if config is not None:
                if not isinstance(config, Config):
                    raise TypeError(
                        f"config must be nccl.m2n.Config or None, got {type(config).__name__}"
                    )
                cfg = config.to_binding()
                config_ptr = int(cfg.ptr)
                if config_ptr == 0:
                    raise NcclInvalid("config binding pointer must be non-null")
            handle_ptr = int(_m2n_bindings.init(config_ptr))
            if handle_ptr == 0:
                raise NcclInvalid("ncclM2nInit returned a null handle")
            return cls(handle_ptr)

    @property
    def ptr(self) -> int:
        """Raw opaque ``ncclM2nHandle_t`` token."""
        with self._lock:
            self._check_valid("read ptr")
            return self._ptr

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: Handle is not initialized or has been destroyed"
            )

    def reshard_with_window(
        self,
        comm: object,
        window: object,
        src: DistTensor,
        dst: DistTensor,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """Call ``ncclReshardWithWindow``.

        ``comm`` may be a raw ``ncclComm_t`` integer or an NCCL4Py
        ``Communicator``.  ``window`` may be a raw ``ncclWindow_t`` integer or
        a NCCL4Py ``RegisteredWindowHandle``.

        Buffer lifetime follows :meth:`reshard`.
        """

        with NATIVE_CALL_LOCK, self._lock:
            self._check_valid("reshard_with_window")
            _reshard_with_window(self._ptr, comm, window, src, dst, stream)

    def reshard(
        self,
        comm: object,
        src: DistTensor,
        dst: DistTensor,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """Call copy/staging-backed ``ncclReshard``.

        reshard enqueues asynchronous work on stream; every buffer — including
        PyTorch tensors — must remain allocated and unmodified until that work
        completes. With PyTorch's caching allocator, hold references until you
        synchronize, or call tensor.record_stream(...) yourself with your torch
        stream.
        """

        with NATIVE_CALL_LOCK, self._lock:
            self._check_valid("reshard")
            _reshard(self._ptr, comm, src, dst, stream)

    def destroy(self) -> None:
        """Release the handle. Subsequent operations on this object are invalid."""
        with NATIVE_CALL_LOCK, self._lock:
            if self._ptr:
                _m2n_bindings.finalize(self._ptr)
                self._ptr = 0

    def __enter__(self) -> "Handle":
        with self._lock:
            self._check_valid("enter context manager")
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()

    def __repr__(self) -> str:
        with self._lock:
            if self._ptr:
                return f"<Handle ptr={self._ptr:#x}>"
            return "<Handle destroyed>"


__all__ = ["Handle"]
