# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Distributed tensor descriptor wrapper for NCCL M2N."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import nccl.core.typing as nccl_typing
from nccl._extensions.bindings import nccl_m2n as _m2n_bindings
from nccl.core.typing import NcclDataType, NcclInvalid
from nccl.m2n.constants import MAX_TENSOR_DIMS, MESH_NDIMS
from nccl.m2n.mesh import Mesh
from nccl.m2n.placement import normalize_placements


_DTYPE_NAMES: dict[str, NcclDataType] = {
    "int8": nccl_typing.INT8,
    "char": nccl_typing.INT8,
    "uint8": nccl_typing.UINT8,
    "byte": nccl_typing.UINT8,
    "int32": nccl_typing.INT32,
    "int": nccl_typing.INT32,
    "uint32": nccl_typing.UINT32,
    "int64": nccl_typing.INT64,
    "long": nccl_typing.INT64,
    "uint64": nccl_typing.UINT64,
    "float16": nccl_typing.FLOAT16,
    "half": nccl_typing.FLOAT16,
    "bfloat16": nccl_typing.BFLOAT16,
    "float32": nccl_typing.FLOAT32,
    "float": nccl_typing.FLOAT32,
    "float64": nccl_typing.FLOAT64,
    "double": nccl_typing.FLOAT64,
    "float8_e4m3fn": nccl_typing.FLOAT8E4M3,
    "float8_e4m3": nccl_typing.FLOAT8E4M3,
    "float8_e5m2": nccl_typing.FLOAT8E5M2,
}

_KIND_SIZE_TO_DTYPE: dict[tuple[str, int], NcclDataType] = {
    ("i", 1): nccl_typing.INT8,
    ("u", 1): nccl_typing.UINT8,
    ("i", 4): nccl_typing.INT32,
    ("u", 4): nccl_typing.UINT32,
    ("i", 8): nccl_typing.INT64,
    ("u", 8): nccl_typing.UINT64,
    ("f", 2): nccl_typing.FLOAT16,
    ("f", 4): nccl_typing.FLOAT32,
    ("f", 8): nccl_typing.FLOAT64,
}


def normalize_dtype(dtype: object) -> NcclDataType:
    if isinstance(dtype, NcclDataType):
        return dtype
    if isinstance(dtype, int):
        try:
            return NcclDataType(dtype)
        except ValueError as exc:
            raise NcclInvalid(f"unsupported NCCL Reshard dtype: {dtype!r}") from exc

    name = str(dtype).replace("torch.", "").lower()
    if name in _DTYPE_NAMES:
        return _DTYPE_NAMES[name]

    try:
        import numpy as np

        np_dtype = np.dtype(dtype)
        if np_dtype.name in _DTYPE_NAMES:
            return _DTYPE_NAMES[np_dtype.name]
        kind_size = (np_dtype.kind, np_dtype.itemsize)
        if kind_size in _KIND_SIZE_TO_DTYPE:
            return _KIND_SIZE_TO_DTYPE[kind_size]
    except (ImportError, TypeError, ValueError):
        pass

    raise NcclInvalid(f"unsupported NCCL Reshard dtype: {dtype!r}")


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(dim) for dim in shape)
    if not 1 <= len(out) <= MAX_TENSOR_DIMS:
        raise ValueError(f"tensor rank must be 1..{MAX_TENSOR_DIMS}, got shape={out}")
    if any(dim < 0 for dim in out):
        raise ValueError(f"local_shape values must be non-negative, got {out}")
    return out


def _cai_is_contiguous(shape: tuple[int, ...], strides: object, typestr: str) -> bool:
    if strides is None or any(dim == 0 for dim in shape):
        return True
    try:
        import numpy as np

        itemsize = int(np.dtype(typestr).itemsize)
        normalized_strides = tuple(int(stride) for stride in strides)
    except (TypeError, ValueError):
        return False
    if len(normalized_strides) != len(shape):
        return False
    expected = itemsize
    for dim, stride in zip(reversed(shape), reversed(normalized_strides), strict=True):
        if dim > 1 and stride != expected:
            return False
        expected *= dim
    return True


def _resolve_buffer(buffer: object) -> tuple[int, tuple[int, ...] | None, NcclDataType | None]:
    if buffer is None:
        return 0, None, None
    if isinstance(buffer, int):
        if buffer < 0:
            raise ValueError(f"raw device pointer must be non-negative, got {buffer}")
        return int(buffer), None, None

    data_ptr = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        is_contiguous = getattr(buffer, "is_contiguous", None)
        if callable(is_contiguous) and not is_contiguous():
            raise ValueError(
                f"{type(buffer).__name__} must be contiguous; "
                "ncclDistTensor_t does not carry stride metadata"
            )
        shape = _shape_tuple(tuple(int(dim) for dim in buffer.shape))
        dtype = normalize_dtype(buffer.dtype)
        ptr = int(data_ptr())
        if ptr < 0:
            raise ValueError(f"device pointer must be non-negative, got {ptr}")
        return ptr, shape, dtype

    cai = getattr(buffer, "__cuda_array_interface__", None)
    if cai is not None:
        ptr = int(cai["data"][0])
        if ptr < 0:
            raise ValueError(f"CUDA array device pointer must be non-negative, got {ptr}")
        shape = _shape_tuple(cai["shape"])
        typestr = cai.get("typestr")
        if typestr is None:
            raise ValueError(
                "__cuda_array_interface__ is missing typestr for "
                f"{type(buffer).__name__}"
            )
        if not _cai_is_contiguous(shape, cai.get("strides"), typestr):
            raise ValueError(
                f"{type(buffer).__name__} must expose a contiguous CUDA array; "
                "ncclDistTensor_t does not carry stride metadata"
            )
        dtype = normalize_dtype(typestr)
        return ptr, shape, dtype

    raise NcclInvalid(
        f"cannot resolve {type(buffer).__name__}: expected None, raw int "
        "device pointer, torch.Tensor, or CUDA Array Interface object"
    )


@dataclass
class _PreparedDistTensor:
    mesh: _m2n_bindings.Mesh
    struct: _m2n_bindings.DistTensor


class DistTensor:
    """Pythonic ``ncclDistTensor_t`` descriptor.

    ``buffer=None`` represents a non-participating side for this rank.  Mesh,
    placements, local shape, and dtype are still required because NCCL M2N
    reads both side descriptors on every rank.
    """

    def __init__(
        self,
        buffer: object,
        local_shape: Sequence[int] | None = None,
        dtype: object | None = None,
        *,
        mesh: Mesh,
        placements: Sequence[object],
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError(f"mesh must be nccl.m2n.Mesh, got {type(mesh).__name__}")
        ptr, inferred_shape, inferred_dtype = _resolve_buffer(buffer)
        if local_shape is None:
            if inferred_shape is None:
                raise ValueError("local_shape is required for None or raw-pointer buffers")
            local_shape = inferred_shape
        if dtype is None:
            if inferred_dtype is None:
                raise ValueError("dtype is required for None or raw-pointer buffers")
            dtype = inferred_dtype

        shape = _shape_tuple(local_shape)
        self._buffer = buffer
        self._data_ptr = int(ptr)
        self._local_shape = shape
        self._dtype = normalize_dtype(dtype)
        self._mesh = mesh
        self._placements = normalize_placements(placements)

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    @property
    def buffer(self) -> object:
        return self._buffer

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self._local_shape

    @property
    def ndims(self) -> int:
        return len(self._local_shape)

    @property
    def dtype(self) -> NcclDataType:
        return self._dtype

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def placements(self) -> tuple[int, ...]:
        return self._placements

    @property
    def is_participating(self) -> bool:
        return self._data_ptr != 0

    def as_binding(self) -> _PreparedDistTensor:
        mesh_struct = self._mesh.to_binding()
        struct = _m2n_bindings.DistTensor()
        struct.dataPtr = int(self._data_ptr)
        struct.localShape = self._local_shape
        struct.ndims = self.ndims
        struct.dtype = int(self._dtype)
        struct.mesh = int(mesh_struct.ptr)
        struct.placements = tuple(
            int(p) for p in self._placements[: self._mesh.ndims]
        )
        return _PreparedDistTensor(mesh=mesh_struct, struct=struct)

    def __repr__(self) -> str:
        role = "data" if self.is_participating else "none"
        return (
            f"<DistTensor role={role} shape={self.local_shape} "
            f"dtype={self.dtype.name} mesh={self.mesh.dims}@{self.mesh.start_rank}>"
        )


__all__ = ["DistTensor", "normalize_dtype"]
