# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core`` DLPack shim, ported from cuda-core 0.7.0 ``_dlpack.{pxd,pyx}``.

Pure-Python equivalent of the upstream Cython implementation: ctypes is
used for the ``DLManagedTensor`` / ``DLManagedTensorVersioned`` struct
layout and for the CPython ``PyCapsule_*`` / ``Py_IncRef`` /
``Py_DecRef`` C API. The producer side (:func:`make_py_capsule`) wires
``kROCM`` (=10) wherever upstream wired ``kCUDA``; consumer-side parsing
(used by :class:`StridedMemoryView`) accepts both ``kCUDA`` and
``kROCM`` so PyTorch-ROCm tensors and (in case of mixed setups) CUDA
tensors both round-trip.

Memory lifecycle (mirrors upstream cuda-core 0.7.0):

* Each ``DLManagedTensor[Versioned]`` is allocated on the C heap via
  ``PyMem_RawMalloc``; the ``shape``/``strides`` array is a separate
  ``PyMem_RawMalloc`` allocation. Both are owned by the DLM struct and
  freed by the per-tensor ``deleter`` callback.
* ``manager_ctx`` holds an ``Py_INCREF``-bumped raw ``PyObject*`` for
  the originating Buffer (or any producer state object), pinning it
  until cleanup. The matching ``Py_DECREF`` runs inside the per-tensor
  ``deleter``.
* The PyCapsule destructor (:func:`_pycapsule_deleter`, wired via
  :func:`PyCapsule_New`) checks the capsule name: if it is still
  ``dltensor`` / ``dltensor_versioned`` (consumer never claimed
  ownership), it invokes ``dlm.deleter(dlm)`` to release the producer
  state. If the consumer renamed to ``used_*`` (per DLPack spec,
  claiming ownership), the destructor is a no-op — the consumer is
  responsible for calling ``deleter`` from its own destructor (see
  :class:`StridedMemoryView` and ``_release_dlpack`` in
  ``_memoryview.py``).
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


# Forward-declared for use in _DLM_DELETER
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
# CPython PyCapsule_* / refcount C API via ctypes.pythonapi
# ---------------------------------------------------------------------------

# IMPORTANT: the PyCapsule destructor MUST take ``c_void_p`` rather than
# ``py_object``. CPython invokes the destructor from ``capsule_dealloc``
# at refcount=0; a ``py_object`` callback would INCREF the capsule on
# entry (raising refcount to 1) and DECREF on return, which immediately
# re-enters ``_Py_Dealloc`` -> ``capsule_dealloc`` -> our destructor and
# blows the C stack. ``c_void_p`` is a raw pointer; ctypes does not
# touch refcounts, matching CPython's expectations.
_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = [c_void_p, ctypes.c_char_p, _CAPSULE_DESTRUCTOR]

# Caller-side bindings: take a real Python capsule object.
PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
PyCapsule_IsValid.restype = c_int32
PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
PyCapsule_SetName.restype = c_int32
PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

# Destructor-side bindings: same C functions, but receive the raw
# PyObject* via ``c_void_p``. These avoid the ``py_object`` argtype
# inside ``_pycapsule_deleter`` so ctypes does not synthesize a Python
# reference on a half-deallocated capsule.
_PyCapsule_IsValid_raw = ctypes.PYFUNCTYPE(c_int32, c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_PyCapsule_GetPointer_raw = ctypes.PYFUNCTYPE(c_void_p, c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


# Reference counting: raw-pointer signatures throughout. INCREF/DECREF
# of an arbitrary Python object via ``c_void_p`` argtype keeps the
# bookkeeping symmetric (``id(obj)`` going in, raw PyObject* coming
# out of ``manager_ctx``) and bypasses any ctypes-side ``py_object``
# conversion.
_Py_IncRef_raw = ctypes.PYFUNCTYPE(None, c_void_p)(("Py_IncRef", ctypes.pythonapi))
_Py_DecRef_raw = ctypes.PYFUNCTYPE(None, c_void_p)(("Py_DecRef", ctypes.pythonapi))


# ---------------------------------------------------------------------------
# C heap allocation (PyMem_RawMalloc / PyMem_RawFree)
# ---------------------------------------------------------------------------

# Mirrors upstream ``stdlib.malloc`` / ``stdlib.free`` for the DLM struct
# and shape/strides array. ``PyMem_RawMalloc`` is the GIL-independent
# allocator from the Python C API; on Linux it boils down to plain
# ``malloc`` (matching ``stdlib.malloc``), and unlike ``ctypes.CDLL(None)``
# it does not depend on the host symbol table.
_PyMem_RawMalloc = ctypes.pythonapi.PyMem_RawMalloc
_PyMem_RawMalloc.restype = c_void_p
_PyMem_RawMalloc.argtypes = [ctypes.c_size_t]

_PyMem_RawFree = ctypes.pythonapi.PyMem_RawFree
_PyMem_RawFree.restype = None
_PyMem_RawFree.argtypes = [c_void_p]


# ---------------------------------------------------------------------------
# Producer-side: per-tensor deleters
# ---------------------------------------------------------------------------


def _dlm_deleter_unversioned(tensor_ptr):
    """C ABI: ``void deleter(DLManagedTensor* self)``.

    Mirrors ``cdef void deleter(DLManagedTensor* tensor) noexcept with gil``
    in upstream cuda-core 0.7.0. Frees the shape/strides array, DECREFs
    the pinned ``manager_ctx``, and frees the struct itself. Tolerant
    of a NULL pointer (matches upstream) and any partial-init state
    coming from an aborted ``make_py_capsule``.
    """
    try:
        if not tensor_ptr:
            return
        dlm = tensor_ptr.contents
        if dlm.dl_tensor.shape:
            _PyMem_RawFree(ctypes.cast(dlm.dl_tensor.shape, c_void_p))
            dlm.dl_tensor.shape = ctypes.cast(0, POINTER(c_int64))
        if dlm.manager_ctx:
            _Py_DecRef_raw(dlm.manager_ctx)
            dlm.manager_ctx = 0
        _PyMem_RawFree(ctypes.cast(tensor_ptr, c_void_p))
    except BaseException:
        # Destructors must not propagate exceptions — would corrupt
        # the CPython teardown path. Drop on the floor.
        pass


def _dlm_deleter_versioned(tensor_ptr):
    """C ABI: ``void deleter(DLManagedTensorVersioned* self)``.

    Versioned-DLPack twin of :func:`_dlm_deleter_unversioned`.
    """
    try:
        if not tensor_ptr:
            return
        dlm = tensor_ptr.contents
        if dlm.dl_tensor.shape:
            _PyMem_RawFree(ctypes.cast(dlm.dl_tensor.shape, c_void_p))
            dlm.dl_tensor.shape = ctypes.cast(0, POINTER(c_int64))
        if dlm.manager_ctx:
            _Py_DecRef_raw(dlm.manager_ctx)
            dlm.manager_ctx = 0
        _PyMem_RawFree(ctypes.cast(tensor_ptr, c_void_p))
    except BaseException:
        pass


# Module-level CFUNCTYPE wrappers; their ctypes trampolines must outlive
# every DLM struct that stores their address in the ``deleter`` field.
_dlm_deleter_unversioned_cb = _DLM_DELETER(_dlm_deleter_unversioned)
_dlm_deleter_versioned_cb = _DLMV_DELETER(_dlm_deleter_versioned)


def _pycapsule_deleter(capsule_ptr):
    """C ABI: ``void destructor(PyObject* capsule)``.

    Wired as the PyCapsule destructor; CPython invokes it at
    ``capsule_dealloc`` time with the raw PyObject* of the capsule.
    Mirrors ``cdef void pycapsule_deleter(object capsule) noexcept`` in
    upstream cuda-core 0.7.0: if the capsule still bears the unconsumed
    name (``dltensor`` / ``dltensor_versioned``), invokes the in-struct
    ``dlm.deleter(dlm)`` to release the producer-pinned state. If the
    consumer has renamed to ``used_*`` (per DLPack spec, claiming
    ownership), this is a no-op — the consumer is responsible for
    calling ``deleter`` from its own destructor.

    Uses the raw-pointer ``_PyCapsule_*_raw`` bindings so ctypes does
    not synthesize Python references on the half-deallocated capsule.
    """
    try:
        if not capsule_ptr:
            return
        if _PyCapsule_IsValid_raw(capsule_ptr, DLPACK_TENSOR_UNUSED_NAME):
            addr = _PyCapsule_GetPointer_raw(capsule_ptr, DLPACK_TENSOR_UNUSED_NAME)
            if addr:
                dlm_p = ctypes.cast(addr, POINTER(DLManagedTensor))
                if dlm_p.contents.deleter:
                    dlm_p.contents.deleter(dlm_p)
        elif _PyCapsule_IsValid_raw(capsule_ptr, DLPACK_VERSIONED_TENSOR_UNUSED_NAME):
            addr = _PyCapsule_GetPointer_raw(
                capsule_ptr, DLPACK_VERSIONED_TENSOR_UNUSED_NAME
            )
            if addr:
                dlm_p = ctypes.cast(addr, POINTER(DLManagedTensorVersioned))
                if dlm_p.contents.deleter:
                    dlm_p.contents.deleter(dlm_p)
    except BaseException:
        pass


_pycapsule_deleter_cb = _CAPSULE_DESTRUCTOR(_pycapsule_deleter)


# ---------------------------------------------------------------------------
# Producer-side: make_py_capsule(buf, versioned)
# ---------------------------------------------------------------------------


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
    ``False`` produces the legacy ``DLManagedTensor``. The byte-level
    view matches upstream cuda-core: ndim=1, dtype=int8, shape=[size],
    strides=[1] (DLPack v1.2+ requires non-NULL strides for ndim != 0).

    Memory lifecycle: the DLM struct and its shape/strides array are
    raw-malloc'd; ``manager_ctx`` holds an ``Py_INCREF``-bumped reference
    to ``buf`` for the lifetime of the capsule (or until the consumer
    invokes ``deleter`` per the DLPack spec). Failure paths roll back
    in the reverse order of acquisition.
    """
    dlm_ptr = 0
    shape_ptr = 0
    incref_done = False

    try:
        if versioned:
            dlm_ptr = _PyMem_RawMalloc(ctypes.sizeof(DLManagedTensorVersioned))
            if not dlm_ptr:
                raise MemoryError("DLManagedTensorVersioned allocation failed")
            ctypes.memset(dlm_ptr, 0, ctypes.sizeof(DLManagedTensorVersioned))
            dlm = ctypes.cast(dlm_ptr, POINTER(DLManagedTensorVersioned)).contents
            dlm.version.major = DLPACK_MAJOR_VERSION
            dlm.version.minor = DLPACK_MINOR_VERSION
            dlm.flags = 0
            dlm.deleter = _dlm_deleter_versioned_cb
            capsule_name = DLPACK_VERSIONED_TENSOR_UNUSED_NAME
        else:
            dlm_ptr = _PyMem_RawMalloc(ctypes.sizeof(DLManagedTensor))
            if not dlm_ptr:
                raise MemoryError("DLManagedTensor allocation failed")
            ctypes.memset(dlm_ptr, 0, ctypes.sizeof(DLManagedTensor))
            dlm = ctypes.cast(dlm_ptr, POINTER(DLManagedTensor)).contents
            dlm.deleter = _dlm_deleter_unversioned_cb
            capsule_name = DLPACK_TENSOR_UNUSED_NAME

        # 1-D shape + contiguous strides, both stored in a single
        # 2-element int64 array (matches upstream byte-layout exactly:
        # ``shape = arr; strides = arr + ndim``).
        shape_ptr = _PyMem_RawMalloc(ctypes.sizeof(c_int64) * 2)
        if not shape_ptr:
            raise MemoryError("DLPack shape array allocation failed")
        shape_arr = ctypes.cast(shape_ptr, POINTER(c_int64))
        shape_arr[0] = int(buf.size)
        shape_arr[1] = 1  # contiguous stride

        dlm.dl_tensor.data = c_void_p(int(buf.handle))
        dlm.dl_tensor.ndim = 1
        dlm.dl_tensor.dtype.code = int(DLDataTypeCode.kDLInt)
        dlm.dl_tensor.dtype.bits = 8
        dlm.dl_tensor.dtype.lanes = 1
        dlm.dl_tensor.shape = shape_arr
        dlm.dl_tensor.strides = ctypes.cast(
            shape_ptr + ctypes.sizeof(c_int64), POINTER(c_int64)
        )
        dlm.dl_tensor.byte_offset = 0

        device_type, device_id = _device_for_buffer(buf)
        dlm.dl_tensor.device.device_type = device_type
        dlm.dl_tensor.device.device_id = device_id

        # Pin the originating Buffer (or producer state) by INCREF'ing
        # it and stashing the raw PyObject* in manager_ctx. The matching
        # DECREF runs inside _dlm_deleter_*.
        _Py_IncRef_raw(id(buf))
        incref_done = True
        dlm.manager_ctx = id(buf)

        return PyCapsule_New(dlm_ptr, capsule_name, _pycapsule_deleter_cb)
    except BaseException:
        # Roll back partial state on error before the capsule was created
        # (no PyCapsule destructor will run, so we have to clean up here).
        if incref_done:
            try:
                _Py_DecRef_raw(id(buf))
            except BaseException:
                pass
        if shape_ptr:
            try:
                _PyMem_RawFree(c_void_p(shape_ptr))
            except BaseException:
                pass
        if dlm_ptr:
            try:
                _PyMem_RawFree(c_void_p(dlm_ptr))
            except BaseException:
                pass
        raise


# ---------------------------------------------------------------------------
# Consumer-side helpers used by _memoryview.view_as_dlpack
# ---------------------------------------------------------------------------


def parse_capsule(capsule):
    """Inspect a DLPack capsule and return ``(dl_tensor, is_readonly, used_name)``.

    Raises :class:`BufferError` on an unknown / already-consumed capsule.
    The caller is responsible for calling :func:`PyCapsule_SetName` with
    ``used_name`` once the data has been copied into a StridedMemoryView,
    and for invoking ``dl_tensor.deleter`` from the StridedMemoryView's
    finalizer (see ``_release_dlpack`` in ``_memoryview.py``).
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
        "DLPack capsule is invalid or already consumed (expected dltensor / "
        "dltensor_versioned)"
    )
