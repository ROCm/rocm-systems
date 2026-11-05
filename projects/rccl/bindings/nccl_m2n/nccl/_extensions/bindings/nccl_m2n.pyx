# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 0.2.0. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport intptr_t, uint64_t
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr, get_buffer_pointer,
                              get_resource_ptr, get_nested_resource_ptr)

from enum import IntEnum as _IntEnum

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

from libc.stdlib cimport calloc, free, malloc
from cython cimport view
cimport cpython.buffer
cimport cpython.memoryview
cimport cpython
from libc.string cimport memcmp, memcpy
import numpy as _numpy


cdef __from_data(data, dtype_name, expected_dtype, lowpp_type):
    # _numpy.recarray is a subclass of _numpy.ndarray, so implicitly handled here.
    if isinstance(data, lowpp_type):
        return data
    if not isinstance(data, _numpy.ndarray):
        raise TypeError("data argument must be a NumPy ndarray")
    if data.size != 1:
        raise ValueError("data array must have a size of 1")
    if data.dtype != expected_dtype:
        raise ValueError(f"data array must be of dtype {dtype_name}")
    return lowpp_type.from_ptr(data.ctypes.data, not data.flags.writeable, data)


cdef __from_buffer(buffer, size, lowpp_type):
    cdef Py_buffer view
    if cpython.PyObject_GetBuffer(buffer, &view, cpython.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, view.readonly, buffer)
    finally:
        cpython.PyBuffer_Release(&view)


cdef __getbuffer(object self, cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
    buffer.buf = <char *>ptr
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = 1
    buffer.len = size
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = readonly
    buffer.shape = &buffer.len
    buffer.strides = &buffer.itemsize
    buffer.suboffsets = NULL


###############################################################################
# POD
###############################################################################

cdef _get_mesh_dtype_offsets():
    cdef ncclMesh_t pod = ncclMesh_t()
    return _numpy.dtype({
        'names': ['size', 'version', 'ndims', 'dims', 'startRank'],
        'formats': [_numpy.uintp, _numpy.uint32, _numpy.int32, _numpy.intp, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ndims)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dims)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.startRank)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclMesh_t),
    })

mesh_dtype = _get_mesh_dtype_offsets()

cdef class Mesh:
    """Empty-initialize an instance of `ncclMesh_t`.


    .. seealso:: `ncclMesh_t`
    """
    cdef:
        ncclMesh_t *_ptr
        object _owner
        object _dims_owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclMesh_t *>calloc(1, sizeof(ncclMesh_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Mesh")
        self._owner = None
        self._dims_owner = None
        self._owned = True
        self._readonly = False
        self._ptr[0].size = sizeof(ncclMesh_t)
        self._ptr[0].version = NCCL_M2N_API_VERSION

    def __dealloc__(self):
        cdef ncclMesh_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Mesh object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Mesh other_
        if not isinstance(other, Mesh):
            return False
        other_ = other
        return (
            self._ptr[0].size == other_._ptr[0].size
            and self._ptr[0].version == other_._ptr[0].version
            and self._ptr[0].ndims == other_._ptr[0].ndims
            and self._ptr[0].startRank == other_._ptr[0].startRank
            and self.dims == other_.dims
        )

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclMesh_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclMesh_t *>malloc(sizeof(ncclMesh_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Mesh")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclMesh_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def dims(self):
        cdef int ndims = self._ptr[0].ndims
        if self._ptr[0].dims == NULL:
            return ()
        if ndims < 1 or ndims > NCCL_RESHARD_MAX_MESH_DIMS:
            raise ValueError(
                f"mesh.ndims must be in [1, {NCCL_RESHARD_MAX_MESH_DIMS}], got {ndims}"
            )
        return tuple(self._ptr[0].dims[d] for d in range(ndims))

    @dims.setter
    def dims(self, values):
        if self._readonly:
            raise ValueError("This Mesh instance is read-only")
        if len(values) < 1 or len(values) > NCCL_RESHARD_MAX_MESH_DIMS:
            raise ValueError(f"dims must have length 1..{NCCL_RESHARD_MAX_MESH_DIMS}")
        dims = _numpy.ascontiguousarray(values, dtype=_numpy.int32)
        self._dims_owner = dims
        self._ptr[0].ndims = len(values)
        self._ptr[0].dims = <int*><intptr_t>dims.ctypes.data

    @property
    def ndims(self):
        return self._ptr[0].ndims

    @property
    def startRank(self):
        """int: """
        return self._ptr[0].startRank

    @startRank.setter
    def startRank(self, val):
        if self._readonly:
            raise ValueError("This Mesh instance is read-only")
        self._ptr[0].startRank = val

    @staticmethod
    def from_buffer(buffer):
        """Create an Mesh instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclMesh_t), Mesh)

    @staticmethod
    def from_data(data):
        """Create an Mesh instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `mesh_dtype` holding the data.
        """
        return __from_data(data, "mesh_dtype", mesh_dtype, Mesh)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Mesh instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef int d
        cdef int ndims
        cdef object dims
        cdef Mesh obj = Mesh.__new__(Mesh)
        obj._dims_owner = None
        if owner is None:
            obj._ptr = <ncclMesh_t *>malloc(sizeof(ncclMesh_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Mesh")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclMesh_t))
            obj._owner = None
            obj._owned = True
            ndims = obj._ptr[0].ndims
            if obj._ptr[0].dims != NULL and 1 <= ndims <= NCCL_RESHARD_MAX_MESH_DIMS:
                dims = _numpy.empty(ndims, dtype=_numpy.int32)
                for d in range(ndims):
                    dims[d] = obj._ptr[0].dims[d]
                obj._dims_owner = dims
                obj._ptr[0].dims = <int*><intptr_t>dims.ctypes.data
            elif ndims < 1 or ndims > NCCL_RESHARD_MAX_MESH_DIMS:
                obj._ptr[0].dims = NULL
        else:
            obj._ptr = <ncclMesh_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_config_dtype_offsets():
    cdef ncclM2nConfig_t pod = ncclM2nConfig_t()
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'maxCta'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.maxCta)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclM2nConfig_t),
    })

config_dtype = _get_config_dtype_offsets()

cdef class Config:
    """Initialize an instance of `ncclM2nConfig_t` using configured defaults.


    .. seealso:: `ncclM2nConfig_t`
    """
    cdef:
        ncclM2nConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclM2nConfig_t *>calloc(1, sizeof(ncclM2nConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Config")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclM2nConfig_t)
        self._ptr[0].magic = NCCL_M2N_API_MAGIC
        self._ptr[0].version = NCCL_M2N_API_VERSION
        self._ptr[0].maxCta = NCCL_M2N_CONFIG_UNDEF_INT

    def __dealloc__(self):
        cdef ncclM2nConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Config object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Config other_
        if not isinstance(other, Config):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclM2nConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclM2nConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclM2nConfig_t *>malloc(sizeof(ncclM2nConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Config")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclM2nConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].version = val

    @property
    def maxCta(self):
        """int: """
        return self._ptr[0].maxCta

    @maxCta.setter
    def maxCta(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].maxCta = val

    @staticmethod
    def from_buffer(buffer):
        """Create an Config instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclM2nConfig_t), Config)

    @staticmethod
    def from_data(data):
        """Create an Config instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `config_dtype` holding the data.
        """
        return __from_data(data, "config_dtype", config_dtype, Config)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Config instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Config obj = Config.__new__(Config)
        if owner is None:
            obj._ptr = <ncclM2nConfig_t *>malloc(sizeof(ncclM2nConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Config")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclM2nConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclM2nConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dist_tensor_dtype_offsets():
    cdef ncclDistTensor_t pod = ncclDistTensor_t()
    return _numpy.dtype({
        'names': ['size', 'version', 'dataPtr', 'localShape', 'ndims', 'dtype', 'mesh', 'placements'],
        'formats': [_numpy.uintp, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.int32,
                    _numpy.dtype(('V', sizeof(ncclDataType_t))), _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dataPtr)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.localShape)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ndims)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dtype)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.mesh)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.placements)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclDistTensor_t),
    })

dist_tensor_dtype = _get_dist_tensor_dtype_offsets()

cdef class DistTensor:
    """Empty-initialize an instance of `ncclDistTensor_t`.


    .. seealso:: `ncclDistTensor_t`
    """
    cdef:
        ncclDistTensor_t *_ptr
        object _owner
        object _local_shape_owner
        object _mesh_owner
        object _placements_owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclDistTensor_t *>calloc(1, sizeof(ncclDistTensor_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DistTensor")
        self._owner = None
        self._local_shape_owner = None
        self._mesh_owner = None
        self._placements_owner = None
        self._owned = True
        self._readonly = False
        self._ptr[0].size = sizeof(ncclDistTensor_t)
        self._ptr[0].version = NCCL_M2N_API_VERSION

    def __dealloc__(self):
        cdef ncclDistTensor_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.DistTensor object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DistTensor other_
        if not isinstance(other, DistTensor):
            return False
        other_ = other
        return (
            self._ptr[0].size == other_._ptr[0].size
            and self._ptr[0].version == other_._ptr[0].version
            and self.dataPtr == other_.dataPtr
            and self.localShape == other_.localShape
            and self.ndims == other_.ndims
            and self.dtype == other_.dtype
            and self.mesh == other_.mesh
            and self.placements == other_.placements
        )

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDistTensor_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclDistTensor_t *>malloc(sizeof(ncclDistTensor_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DistTensor")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDistTensor_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def dataPtr(self):
        """int: """
        return <intptr_t>(self._ptr[0].dataPtr)

    @dataPtr.setter
    def dataPtr(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].dataPtr = <void *><intptr_t>val

    @property
    def localShape(self):
        cdef int ndims = self._ptr[0].ndims
        if self._ptr[0].localShape == NULL:
            return ()
        if ndims < 1 or ndims > NCCL_RESHARD_MAX_TENSOR_DIMS:
            raise ValueError(
                f"ndims must be in [1, {NCCL_RESHARD_MAX_TENSOR_DIMS}], got {ndims}"
            )
        return tuple(self._ptr[0].localShape[d] for d in range(ndims))

    @localShape.setter
    def localShape(self, values):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        if len(values) < 1 or len(values) > NCCL_RESHARD_MAX_TENSOR_DIMS:
            raise ValueError(f"localShape must have length 1..{NCCL_RESHARD_MAX_TENSOR_DIMS}")
        local_shape = _numpy.ascontiguousarray(values, dtype=_numpy.uintp)
        self._local_shape_owner = local_shape
        self._ptr[0].localShape = <size_t*><intptr_t>local_shape.ctypes.data
        self._ptr[0].ndims = len(values)

    @property
    def ndims(self):
        """int: """
        return self._ptr[0].ndims

    @ndims.setter
    def ndims(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].ndims = val

    @property
    def dtype(self):
        """: """
        return self._ptr[0].dtype

    @dtype.setter
    def dtype(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].dtype = val

    @property
    def mesh(self):
        return <intptr_t>(self._ptr[0].mesh)

    @mesh.setter
    def mesh(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._mesh_owner = None
        self._ptr[0].mesh = <const ncclMesh_t*><intptr_t>val

    @property
    def placements(self):
        cdef int mesh_ndims
        if self._ptr[0].placements == NULL:
            return ()
        if self._placements_owner is not None:
            return tuple(int(value) for value in self._placements_owner)
        if self._ptr[0].mesh == NULL:
            return ()
        mesh_ndims = self._ptr[0].mesh.ndims
        if mesh_ndims < 1 or mesh_ndims > NCCL_RESHARD_MAX_MESH_DIMS:
            raise ValueError(
                f"mesh.ndims must be in [1, {NCCL_RESHARD_MAX_MESH_DIMS}], got {mesh_ndims}"
            )
        return tuple(self._ptr[0].placements[d] for d in range(mesh_ndims))

    @placements.setter
    def placements(self, values):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        if len(values) < 1 or len(values) > NCCL_RESHARD_MAX_MESH_DIMS:
            raise ValueError(f"placements must have length 1..{NCCL_RESHARD_MAX_MESH_DIMS}")
        placements = _numpy.ascontiguousarray(values, dtype=_numpy.int32)
        self._placements_owner = placements
        self._ptr[0].placements = <int*><intptr_t>placements.ctypes.data

    @staticmethod
    def from_buffer(buffer):
        """Create an DistTensor instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclDistTensor_t), DistTensor)

    @staticmethod
    def from_data(data):
        """Create an DistTensor instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dist_tensor_dtype` holding the data.
        """
        return __from_data(data, "dist_tensor_dtype", dist_tensor_dtype, DistTensor)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DistTensor instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef int d
        cdef int tensor_ndims
        cdef int mesh_ndims
        cdef Mesh mesh_owner
        cdef object local_shape
        cdef object placements
        cdef DistTensor obj = DistTensor.__new__(DistTensor)
        obj._local_shape_owner = None
        obj._mesh_owner = None
        obj._placements_owner = None
        if owner is None:
            obj._ptr = <ncclDistTensor_t *>malloc(sizeof(ncclDistTensor_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DistTensor")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDistTensor_t))
            obj._owner = None
            obj._owned = True
            tensor_ndims = obj._ptr[0].ndims
            if obj._ptr[0].localShape != NULL and 1 <= tensor_ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS:
                local_shape = _numpy.empty(tensor_ndims, dtype=_numpy.uintp)
                for d in range(tensor_ndims):
                    local_shape[d] = obj._ptr[0].localShape[d]
                obj._local_shape_owner = local_shape
                obj._ptr[0].localShape = <size_t*><intptr_t>local_shape.ctypes.data
            elif tensor_ndims < 1 or tensor_ndims > NCCL_RESHARD_MAX_TENSOR_DIMS:
                obj._ptr[0].localShape = NULL
            if obj._ptr[0].mesh != NULL:
                mesh_ndims = obj._ptr[0].mesh.ndims
                if obj._ptr[0].placements != NULL and 1 <= mesh_ndims <= NCCL_RESHARD_MAX_MESH_DIMS:
                    placements = _numpy.empty(mesh_ndims, dtype=_numpy.int32)
                    for d in range(mesh_ndims):
                        placements[d] = obj._ptr[0].placements[d]
                    obj._placements_owner = placements
                    obj._ptr[0].placements = <int*><intptr_t>placements.ctypes.data
                elif mesh_ndims < 1 or mesh_ndims > NCCL_RESHARD_MAX_MESH_DIMS:
                    obj._ptr[0].placements = NULL
                mesh_owner = Mesh.from_ptr(<intptr_t>obj._ptr[0].mesh, readonly, None)
                obj._mesh_owner = mesh_owner
                obj._ptr[0].mesh = mesh_owner._ptr
            else:
                obj._ptr[0].placements = NULL
        else:
            obj._ptr = <ncclDistTensor_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


MESH_NDIMS = NCCL_RESHARD_MAX_MESH_DIMS
MAX_TENSOR_DIMS = NCCL_RESHARD_MAX_TENSOR_DIMS
REPLICATE = NCCL_RESHARD_REPLICATE


###############################################################################
# Error handling
###############################################################################

from nccl.bindings.nccl import NCCLError as _NCCLError
from nccl._extensions._runtime import NATIVE_CALL_LOCK as _NATIVE_CALL_LOCK
from ._internal.utils import FunctionNotFoundError


class NCCLReshardError(_NCCLError):

    def __init__(self, status, detail=None):
        self.status = int(status)
        self.detail = detail
        message = f"NCCL Reshard error code {self.status}"
        if detail:
            message += f": {detail}"
        Exception.__init__(self, message)

    def __reduce__(self):
        return (type(self), (self.status, self.detail))


@cython.profile(False)
cpdef inline check_status(int status):
    cdef const char* detail = NULL
    cdef bytes detail_bytes
    cdef object detail_text = None
    if status != 0:
        detail = ncclM2nGetLastError()
        if detail != NULL:
            detail_bytes = <bytes>detail
            if detail_bytes:
                detail_text = detail_bytes.decode("utf-8", "replace")
        raise NCCLReshardError(status, detail_text)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t init(intptr_t config) except? 0:
    cdef Handle handle
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nInit(&handle, <const ncclM2nConfig_t*>config)
        check_status(status)
    return <intptr_t>handle


cpdef finalize(intptr_t handle):
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nFinalize(<Handle>handle)
        check_status(status)


cpdef group_start():
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nGroupStart()
        check_status(status)


cpdef group_end():
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nGroupEnd()
        check_status(status)


cpdef group_abort():
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nGroupAbort()
        check_status(status)


cpdef reshard_with_window(intptr_t handle, intptr_t comm, intptr_t window, intptr_t src, intptr_t dst, intptr_t stream):
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclReshardWithWindow(<Handle>handle, <Comm>comm, <Window>window, <const ncclDistTensor_t*>src, <const ncclDistTensor_t*>dst, <Stream>stream)
        check_status(status)


cpdef reshard(intptr_t handle, intptr_t comm, intptr_t src, intptr_t dst, intptr_t stream):
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclReshard(<Handle>handle, <Comm>comm, <const ncclDistTensor_t*>src, <const ncclDistTensor_t*>dst, <Stream>stream)
        check_status(status)
