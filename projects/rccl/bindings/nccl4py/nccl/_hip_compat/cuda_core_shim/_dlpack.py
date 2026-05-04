# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core`` DLPack shim, ported from cuda-core 0.3.2 ``_dlpack.{pxd,pyx}``.

Pure-Python equivalent of the upstream Cython implementation: ctypes is
used for the ``DLManagedTensor`` / ``DLManagedTensorVersioned`` struct
layout and for the CPython ``PyCapsule_*`` C API. The producer side
(:func:`make_py_capsule`) wires ``kROCM`` (=10) wherever upstream wired
``kCUDA``; consumer-side parsing (used by :class:`StridedMemoryView`)
accepts both ``kCUDA`` and ``kROCM`` so PyTorch-ROCm tensors and (in case
of mixed setups) CUDA tensors both round-trip.

Memory lifecycle:

* Each ``DLManagedTensor[Versioned]`` is a Python ctypes ``Structure``
  kept alive in ``_alive_capsules`` keyed by ``id(capsule)`` (i.e. the
  ``PyObject`` address of the capsule that wraps the struct).
* :func:`_pycapsule_destructor` drops the entry when the capsule is
  destroyed; the per-tensor ``deleter`` callback is a no-op (consumers
  that take ownership simply hold the capsule until done with it, and
  the destructor cleans up regardless of consumed/unconsumed state).
"""

from __future__ import annotations

import ctypes
import enum
from ctypes import (
    POINTER,
    Structure,
    c_int32,
    c_int64,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from typing import Tuple

# DLPack 1.0 spec values
DLPACK_MAJOR_VERSION = 1
DLPACK_MINOR_VERSION = 0
DLPACK_FLAG_BITMASK_READ_ONLY = 1 << 0
DLPACK_FLAG_BITMASK_IS_COPIED = 1 << 1


# Capsule names per the DLPack spec
DLPACK_TENSOR_UNUSED_NAME = b"dltensor"
DLPACK_VERSIONED_TENSOR_UNUSED_NAME = b"dltensor_versioned"
DLPACK_TENSOR_USED_NAME = b"used_dltensor"
DLPACK_VERSIONED_TENSOR_USED_NAME = b"used_dltensor_versioned"


class DLDeviceType(enum.IntEnum):
    """DLPack device-type constants, including kROCM (=10)."""

    kCPU = 1
    kCUDA = 2
    kCUDAHost = 3
    kOpenCL = 4
    kVulkan = 7
    kMetal = 8
    kVPI = 9
    kROCM = 10
    kROCMHost = 11
    kExtDev = 12
    kCUDAManaged = 13


# Backwards-compatible aliases that mirror the names used in the upstream
# Cython code (``_kDLCPU`` etc.). Kept private; consumer code should use
# ``DLDeviceType.kROCM`` etc.
_kDLCPU = DLDeviceType.kCPU
_kDLCUDA = DLDeviceType.kCUDA
_kDLCUDAHost = DLDeviceType.kCUDAHost
_kDLCUDAManaged = DLDeviceType.kCUDAManaged
_kDLROCM = DLDeviceType.kROCM
_kDLROCMHost = DLDeviceType.kROCMHost


class DLDataTypeCode(enum.IntEnum):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


# ---------------------------------------------------------------------------
# ctypes struct layout (mirrors dlpack.h v1.0)
# ---------------------------------------------------------------------------


class DLDevice(Structure):
    _fields_ = [
        ("device_type", c_int32),
        ("device_id", c_int32),
    ]


class DLDataType(Structure):
    _fields_ = [
        ("code", c_uint8),
        ("bits", c_uint8),
        ("lanes", c_uint16),
    ]


class DLTensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("device", DLDevice),
        ("ndim", c_int32),
        ("dtype", DLDataType),
        ("shape", POINTER(c_int64)),
        ("strides", POINTER(c_int64)),
        ("byte_offset", c_uint64),
    ]


# Forward-declared for use in _DELETER_FUNCTYPE
class DLManagedTensor(Structure):
    pass


_DLM_DELETER = ctypes.CFUNCTYPE(None, POINTER(DLManagedTensor))

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", c_void_p),
    ("deleter", _DLM_DELETER),
]


class DLPackVersion(Structure):
    _fields_ = [
        ("major", c_uint32),
        ("minor", c_uint32),
    ]


class DLManagedTensorVersioned(Structure):
    pass


_DLMV_DELETER = ctypes.CFUNCTYPE(None, POINTER(DLManagedTensorVersioned))

DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", c_void_p),
    ("deleter", _DLMV_DELETER),
    ("flags", c_uint64),
    ("dl_tensor", DLTensor),
]


# ---------------------------------------------------------------------------
# CPython PyCapsule_* C API via ctypes.pythonapi
# ---------------------------------------------------------------------------

# IMPORTANT: the capsule destructor MUST take ``c_void_p`` rather than
# ``py_object``. CPython invokes the destructor from ``capsule_dealloc``
# at refcount=0; a ``py_object`` callback would INCREF the capsule on
# entry (raising refcount to 1) and DECREF on return, which immediately
# re-enters ``_Py_Dealloc`` -> ``capsule_dealloc`` -> our destructor and
# blows the C stack.  ``c_void_p`` is a raw pointer; ctypes does not
# touch refcounts, matching CPython's expectations.
_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = [c_void_p, ctypes.c_char_p, _CAPSULE_DESTRUCTOR]

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
PyCapsule_IsValid.restype = c_int32
PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
PyCapsule_SetName.restype = c_int32
PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]


# ---------------------------------------------------------------------------
# Producer-side: make_py_capsule(buf, versioned)
# ---------------------------------------------------------------------------

# Module-level registry pinning the ctypes struct, shape array, and the
# originating Buffer for the lifetime of the capsule.  Indexed by the
# capsule's PyObject address (== ``id(capsule)``) so the destructor,
# which only receives the raw capsule pointer, can drop the entry
# without calling any PyCapsule_* C function (those would re-enter the
# refcount machinery on a half-deallocated capsule).
_alive_capsules: "dict[int, tuple]" = {}


def _pycapsule_destructor(capsule_ptr):
    """CPython invokes this at capsule_dealloc time with the raw
    PyObject* of the capsule. Drop the registry entry; that releases the
    last Python reference to the managed-tensor struct, the shape array,
    and the originating Buffer (which in turn frees the device memory
    via its MemoryResource finalizer).
    """
    if capsule_ptr:
        _alive_capsules.pop(int(capsule_ptr), None)


_pycapsule_destructor_cb = _CAPSULE_DESTRUCTOR(_pycapsule_destructor)


def _deleter_noop(tensor_ptr):
    """DLPack-spec consumer-callable deleter.

    Per the DLPack spec a consumer that takes ownership of the capsule
    is required to call ``dl_tensor.deleter(dl_tensor_ptr)`` when it is
    done with the tensor. In this shim the actual cleanup is keyed off
    the capsule's lifetime via :func:`_pycapsule_destructor`, so this
    callback intentionally does nothing — the registry entry survives
    until the capsule is destroyed.
    """
    return None


_deleter_unversioned_cb = _DLM_DELETER(_deleter_noop)
_deleter_versioned_cb = _DLMV_DELETER(_deleter_noop)


def _device_for_buffer(buf) -> Tuple[int, int]:
    """Map (is_device_accessible, is_host_accessible) -> (DLDeviceType, dev_id)
    for a Buffer being exported through DLPack."""
    d = bool(buf.is_device_accessible)
    h = bool(buf.is_host_accessible)
    if d and not h:
        return (int(DLDeviceType.kROCM), int(buf.device_id))
    if d and h:
        return (int(DLDeviceType.kROCMHost), 0)
    if not d and h:
        return (int(DLDeviceType.kCPU), 0)
    raise BufferError("invalid buffer: neither device-accessible nor host-accessible")


def make_py_capsule(buf, versioned: bool):
    """Build a DLPack capsule wrapping ``buf``'s 1-D contiguous bytes view.

    ``versioned=True`` produces a DLPack 1.0 ``DLManagedTensorVersioned``;
    ``False`` produces the legacy ``DLManagedTensor``. The byte-level view
    matches the upstream cuda-core 0.3.2 implementation: ndim=1, dtype=int8,
    shape=[size]. Consumers that need a different dtype/shape are expected
    to reinterpret the buffer themselves.
    """
    # Allocate the shape array (1-D view of byte-sized elements).
    shape_arr = (c_int64 * 1)(int(buf.size))

    if versioned:
        dlm_ver = DLManagedTensorVersioned()
        dlm_ver.version.major = DLPACK_MAJOR_VERSION
        dlm_ver.version.minor = DLPACK_MINOR_VERSION
        dlm_ver.flags = 0
        dlm_ver.deleter = _deleter_versioned_cb
        dlm_ver.manager_ctx = 0
        dl_tensor = dlm_ver.dl_tensor
        struct_ref = dlm_ver
        capsule_name = DLPACK_VERSIONED_TENSOR_UNUSED_NAME
    else:
        dlm = DLManagedTensor()
        dlm.deleter = _deleter_unversioned_cb
        dlm.manager_ctx = 0
        dl_tensor = dlm.dl_tensor
        struct_ref = dlm
        capsule_name = DLPACK_TENSOR_UNUSED_NAME

    dl_tensor.data = c_void_p(int(buf.handle))
    dl_tensor.ndim = 1
    dl_tensor.dtype.code = int(DLDataTypeCode.kDLInt)
    dl_tensor.dtype.bits = 8
    dl_tensor.dtype.lanes = 1
    dl_tensor.shape = ctypes.cast(shape_arr, POINTER(c_int64))
    dl_tensor.strides = ctypes.cast(0, POINTER(c_int64))  # contiguous: NULL
    dl_tensor.byte_offset = 0

    device_type, device_id = _device_for_buffer(buf)
    dl_tensor.device.device_type = device_type
    dl_tensor.device.device_id = device_id

    if versioned:
        # Have to assign back: dl_tensor was a copy; rebind the struct field.
        struct_ref.dl_tensor = dl_tensor
    else:
        struct_ref.dl_tensor = dl_tensor

    addr = ctypes.addressof(struct_ref)
    capsule = PyCapsule_New(addr, capsule_name, _pycapsule_destructor_cb)
    # Pin the ctypes struct, shape array, and originating Buffer for the
    # lifetime of the capsule. Indexed by the capsule's PyObject address
    # (== ``id(capsule)``) so the destructor can drop the entry using
    # only the raw pointer it receives.
    _alive_capsules[id(capsule)] = (struct_ref, shape_arr, buf)
    return capsule


# ---------------------------------------------------------------------------
# Consumer-side helpers used by _memoryview.view_as_dlpack
# ---------------------------------------------------------------------------


def parse_capsule(capsule):
    """Inspect a DLPack capsule and return ``(dl_tensor, is_readonly, used_name)``.

    Raises :class:`BufferError` on an unknown / already-consumed capsule.
    The caller is responsible for calling :func:`PyCapsule_SetName` with
    ``used_name`` once the data has been copied into a StridedMemoryView.
    """
    if PyCapsule_IsValid(capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME):
        addr = PyCapsule_GetPointer(capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME)
        if not addr:
            raise BufferError("DLPack capsule wraps a NULL pointer")
        dlm_ver = ctypes.cast(addr, POINTER(DLManagedTensorVersioned)).contents
        is_readonly = bool(dlm_ver.flags & DLPACK_FLAG_BITMASK_READ_ONLY)
        return dlm_ver.dl_tensor, is_readonly, DLPACK_VERSIONED_TENSOR_USED_NAME

    if PyCapsule_IsValid(capsule, DLPACK_TENSOR_UNUSED_NAME):
        addr = PyCapsule_GetPointer(capsule, DLPACK_TENSOR_UNUSED_NAME)
        if not addr:
            raise BufferError("DLPack capsule wraps a NULL pointer")
        dlm = ctypes.cast(addr, POINTER(DLManagedTensor)).contents
        return dlm.dl_tensor, False, DLPACK_TENSOR_USED_NAME

    raise BufferError(
        "DLPack capsule is invalid or already consumed (expected dltensor / " "dltensor_versioned)"
    )
