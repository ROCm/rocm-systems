# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.utils.StridedMemoryView`` shim, ported from cuda-core 0.3.2.

Pure-Python equivalent of ``_memoryview.pyx``: ctypes-backed DLPack
consumer + plain Python CUDA Array Interface (v3) consumer + the
``@args_viewable_as_strided_memory`` decorator. ``kROCM`` (=10) is
accepted alongside the upstream ``kCUDA`` (=2) so PyTorch-ROCm /
CuPy-ROCm tensors round-trip without mutation, and DLPack capsules
exported from this shim report ``device_type == 10``.
"""

from __future__ import annotations

import functools
import weakref
from ctypes import POINTER
from ctypes import cast as _ctypes_cast
from typing import Any, Optional

import numpy as _np

from . import _dlpack as _dl
from ._hip import check_hip, hip

# ---------------------------------------------------------------------------
# DLDataType -> numpy.dtype mapping
# ---------------------------------------------------------------------------


def _dlpack_dtype_to_numpy(dl_dtype) -> _np.dtype:
    code = int(dl_dtype.code)
    bits = int(dl_dtype.bits)
    lanes = int(dl_dtype.lanes)
    if lanes != 1:
        raise NotImplementedError(f"vector dtypes (lanes={lanes}) are not supported")

    if code == int(_dl.DLDataTypeCode.kDLUInt):
        if bits == 8:
            return _np.dtype(_np.uint8)
        if bits == 16:
            return _np.dtype(_np.uint16)
        if bits == 32:
            return _np.dtype(_np.uint32)
        if bits == 64:
            return _np.dtype(_np.uint64)
        raise TypeError(f"uint{bits} is not supported")
    if code == int(_dl.DLDataTypeCode.kDLInt):
        if bits == 8:
            return _np.dtype(_np.int8)
        if bits == 16:
            return _np.dtype(_np.int16)
        if bits == 32:
            return _np.dtype(_np.int32)
        if bits == 64:
            return _np.dtype(_np.int64)
        raise TypeError(f"int{bits} is not supported")
    if code == int(_dl.DLDataTypeCode.kDLFloat):
        if bits == 16:
            return _np.dtype(_np.float16)
        if bits == 32:
            return _np.dtype(_np.float32)
        if bits == 64:
            return _np.dtype(_np.float64)
        raise TypeError(f"float{bits} is not supported")
    if code == int(_dl.DLDataTypeCode.kDLComplex):
        if bits == 64:
            return _np.dtype(_np.complex64)
        if bits == 128:
            return _np.dtype(_np.complex128)
        raise TypeError(f"complex{bits} is not supported")
    if code == int(_dl.DLDataTypeCode.kDLBool):
        if bits == 8:
            return _np.dtype(_np.bool_)
        raise TypeError(f"{bits}-bit bool is not supported")
    if code == int(_dl.DLDataTypeCode.kDLBfloat):
        # Optional dependency: ml_dtypes.bfloat16. Fall back to a non-fatal
        # tagged numpy dtype so callers can still query bytes/strides.
        try:
            from ml_dtypes import bfloat16  # type: ignore[import-not-found]
        except Exception:
            raise NotImplementedError("bfloat16 dtype requires the optional `ml-dtypes` package")
        return _np.dtype(bfloat16)
    raise TypeError(f"unsupported DLPack dtype code: {code}")


# ---------------------------------------------------------------------------
# StridedMemoryView
# ---------------------------------------------------------------------------


class StridedMemoryView:
    """Metadata for a strided dense array/tensor exchanged via DLPack or CAI.

    Mirrors cuda-core 0.3.2's StridedMemoryView. Construction direct or via
    :func:`args_viewable_as_strided_memory`. Attributes are filled in either
    by :func:`view_as_dlpack` (DLPack source) or :func:`view_as_cai`
    (``__cuda_array_interface__`` source).
    """

    __slots__ = (
        "__weakref__",
        "ptr",
        "shape",
        "strides",
        "dtype",
        "device_id",
        "is_device_accessible",
        "readonly",
        "exporting_obj",
    )

    def __init__(self, obj=None, stream_ptr=None):
        self.ptr = None
        self.shape = None
        self.strides = None
        self.dtype = None
        self.device_id = None
        self.is_device_accessible = None
        self.readonly = None
        self.exporting_obj = None
        if obj is not None:
            if _has_dlpack(obj):
                view_as_dlpack(obj, stream_ptr, self)
            else:
                view_as_cai(obj, stream_ptr, self)

    def __repr__(self) -> str:
        def _short(o):
            cls = o if isinstance(o, type) else type(o)
            mod = cls.__module__
            if mod in (None, "builtins"):
                return cls.__name__
            return f"{mod}.{cls.__name__}"

        return (
            f"StridedMemoryView(ptr={self.ptr},\n"
            f"                  shape={self.shape},\n"
            f"                  strides={self.strides},\n"
            f"                  dtype={_short(self.dtype)},\n"
            f"                  device_id={self.device_id},\n"
            f"                  is_device_accessible={self.is_device_accessible},\n"
            f"                  readonly={self.readonly},\n"
            f"                  exporting_obj={_short(self.exporting_obj)})"
        )


# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------


def _has_dlpack(obj) -> bool:
    if hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__"):
        return True
    if hasattr(obj, "__cuda_array_interface__"):
        return False
    raise RuntimeError(
        "the input object does not support any data exchange protocol "
        "(__dlpack__ / __dlpack_device__ or __cuda_array_interface__)"
    )


# ---------------------------------------------------------------------------
# DLPack consumer
# ---------------------------------------------------------------------------


_DEVICE_ACCESSIBLE_TYPES = (
    int(_dl.DLDeviceType.kCUDA),
    int(_dl.DLDeviceType.kCUDAHost),
    int(_dl.DLDeviceType.kCUDAManaged),
    int(_dl.DLDeviceType.kROCM),
    int(_dl.DLDeviceType.kROCMHost),
)


def _release_dlpack(capsule, used_name):
    """Per DLPack spec: a consumer that renamed the capsule to
    ``used_dltensor[_versioned]`` must invoke ``dl_tensor.deleter`` to
    signal the producer that its pinned state can be released.

    Mirrors ``StridedMemoryView.__dealloc__`` in upstream cuda-core 0.7.0
    ``_memoryview.pyx``: looks up the DLM struct via the renamed capsule
    and calls its ``deleter`` field. Wrapped in ``try/except: pass`` —
    finalizer callbacks are run by ``weakref`` machinery and
    ``sys.unraisablehook``-noisy errors here would surface unrelated
    teardown noise to users.
    """
    try:
        if not _dl.PyCapsule_IsValid(capsule, used_name):
            return
        addr = _dl.PyCapsule_GetPointer(capsule, used_name)
        if not addr:
            return
        if used_name == _dl.DLPACK_VERSIONED_TENSOR_USED_NAME:
            dlm_p = _ctypes_cast(addr, POINTER(_dl.DLManagedTensorVersioned))
        else:
            dlm_p = _ctypes_cast(addr, POINTER(_dl.DLManagedTensor))
        if dlm_p.contents.deleter:
            dlm_p.contents.deleter(dlm_p)
    except BaseException:
        pass


def view_as_dlpack(obj, stream_ptr, view: Optional[StridedMemoryView] = None) -> StridedMemoryView:
    dldevice, device_id = obj.__dlpack_device__()
    dldevice = int(dldevice)
    is_device_accessible = False

    if dldevice == int(_dl.DLDeviceType.kCPU):
        assert int(device_id) == 0
        device_id = -1
        if stream_ptr is None:
            raise BufferError("stream=None is ambiguous with view()")
        elif int(stream_ptr) == -1:
            stream_ptr = None
    elif dldevice == int(_dl.DLDeviceType.kCUDA) or dldevice == int(_dl.DLDeviceType.kROCM):
        assert int(device_id) >= 0
        is_device_accessible = True
        if stream_ptr is None:
            raise BufferError("stream=None is ambiguous with view()")
    elif dldevice in (
        int(_dl.DLDeviceType.kCUDAHost),
        int(_dl.DLDeviceType.kCUDAManaged),
        int(_dl.DLDeviceType.kROCMHost),
    ):
        is_device_accessible = True
    else:
        raise BufferError(f"DLPack device_type={dldevice} is not supported by the HIP shim")

    # Try the versioned protocol first; fall back to legacy if the producer
    # doesn't accept ``max_version``.
    try:
        capsule = obj.__dlpack__(
            stream=int(stream_ptr) if stream_ptr else None,
            max_version=(_dl.DLPACK_MAJOR_VERSION, _dl.DLPACK_MINOR_VERSION),
        )
    except TypeError:
        capsule = obj.__dlpack__(stream=int(stream_ptr) if stream_ptr else None)

    dl_tensor, is_readonly, used_name = _dl.parse_capsule(capsule)

    out = view if view is not None else StridedMemoryView()
    out.ptr = int(dl_tensor.data) if dl_tensor.data else 0
    ndim = int(dl_tensor.ndim)
    out.shape = tuple(int(dl_tensor.shape[i]) for i in range(ndim))
    if dl_tensor.strides:
        out.strides = tuple(int(dl_tensor.strides[i]) for i in range(ndim))
    else:
        # NULL strides => C-contiguous; upstream signals that with ``None``.
        out.strides = None
    out.dtype = _dlpack_dtype_to_numpy(dl_tensor.dtype)
    out.device_id = int(device_id)
    out.is_device_accessible = is_device_accessible
    out.readonly = is_readonly
    out.exporting_obj = obj

    _dl.PyCapsule_SetName(capsule, used_name)

    # Per DLPack spec, renaming to ``used_*`` claims ownership: the
    # consumer must call ``dl_tensor.deleter`` itself when done with
    # the data. Register a weakref.finalize that does so when ``out``
    # is garbage-collected; the finalizer holds a strong reference to
    # ``capsule`` (and ``used_name``) for the duration, pinning the
    # producer-allocated DLM struct alongside the SMV.
    weakref.finalize(out, _release_dlpack, capsule, used_name)
    return out


# ---------------------------------------------------------------------------
# CUDA Array Interface (v3) consumer
# ---------------------------------------------------------------------------


def _hip_pointer_device_id(ptr: int) -> int:
    """Look up the device ordinal that owns ``ptr`` via hipPointerGetAttributes.

    Mirrors the cuda-core 0.3.2 path that calls
    ``cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr)``.
    """
    attrs = check_hip(hip.hipPointerGetAttributes(int(ptr)), "hipPointerGetAttributes")
    # hip-python returns either a struct with `.device` or a tuple-like
    # with the device ordinal at a known index. Try common attribute names.
    for name in ("device", "device_ordinal", "deviceOrdinal"):
        if hasattr(attrs, name):
            return int(getattr(attrs, name))
    raise RuntimeError(
        f"hipPointerGetAttributes returned an unexpected payload type: {type(attrs).__name__}"
    )


def view_as_cai(obj, stream_ptr, view: Optional[StridedMemoryView] = None) -> StridedMemoryView:
    cai = obj.__cuda_array_interface__
    if int(cai.get("version", 0)) < 3:
        raise BufferError("only CUDA Array Interface v3 or above is supported")
    if cai.get("mask") is not None:
        raise BufferError("CAI 'mask' field is not supported")
    if stream_ptr is None:
        raise BufferError("stream=None is ambiguous with view()")

    out = view if view is not None else StridedMemoryView()
    ptr, readonly = cai["data"]
    out.exporting_obj = obj
    out.ptr = int(ptr)
    out.readonly = bool(readonly)
    out.shape = tuple(cai["shape"])
    out.dtype = _np.dtype(cai["typestr"])

    strides = cai.get("strides")
    if strides is not None:
        # CAI publishes strides in bytes; cuda-core stores them in element counts.
        itemsize = out.dtype.itemsize
        out.strides = tuple(int(s) // itemsize for s in strides)
    else:
        out.strides = None

    out.is_device_accessible = True
    out.device_id = _hip_pointer_device_id(out.ptr)

    # CAI stream-ordering across producer/consumer streams. Skipped on HIP
    # for the stop-gap shim: nccl always passes the consumer stream as
    # stream_ptr and the call sites in nccl/core do their own ordering.
    return out


# ---------------------------------------------------------------------------
# Proxy + decorator
# ---------------------------------------------------------------------------


class _StridedMemoryViewProxy:
    """Thin pre-resolved wrapper. ``view(stream_ptr)`` materializes a
    :class:`StridedMemoryView` against the original DLPack/CAI source."""

    __slots__ = ("_obj", "_has_dlpack")

    def __init__(self, obj):
        self._obj = obj
        self._has_dlpack = _has_dlpack(obj)

    def view(self, stream_ptr=None) -> StridedMemoryView:
        if self._has_dlpack:
            return view_as_dlpack(self._obj, stream_ptr)
        return view_as_cai(self._obj, stream_ptr)


def args_viewable_as_strided_memory(arg_indices):
    """Decorator; replaces positional args at ``arg_indices`` with proxies.

    Inside the decorated function each replaced argument has a
    ``.view(stream_ptr)`` method returning a :class:`StridedMemoryView`.
    Mirrors cuda-core 0.3.2's ``utils.args_viewable_as_strided_memory``
    surface 1:1.
    """
    arg_indices = tuple(arg_indices)

    def wrapped_func_with_indices(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = list(args)
            for idx in arg_indices:
                args[idx] = _StridedMemoryViewProxy(args[idx])
            return func(*args, **kwargs)

        return wrapped_func

    return wrapped_func_with_indices
